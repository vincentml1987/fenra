"""Server side of Fenra's distributed-compute contract (2026-09-18; see
Communications/client-server-plan.md - Vero owns the matching client in
fenra_client/, and net.py there is the contract in code form).

Remote volunteer machines run fenra_client, which reaches OUT to this
server over plain HTTP polling (NAT-friendly, no inbound ports on their
side). Three endpoints, Bearer-token authenticated:

    POST /api/v1/clients/heartbeat            liveness + model inventory
    GET  /api/v1/clients/jobs/next            204 = nothing, 200 = a job
    POST /api/v1/clients/jobs/{id}/result     409 = stale/superseded job

A job is one exact Ollama request (kind "generate" or "chat"); the client
relays it to its own local Ollama and posts the raw response back. All
simulation logic - scheduling, eligibility, retry - stays in fenra.py;
this module only knows clients, jobs, and liveness.

Inert unless host_clients.json exists next to fenra.py (gitignored -
holds the pre-shared tokens; see host_clients.example.json). No config,
no server thread, nothing changes for a single-machine world.
"""
import hmac
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# A client that hasn't heartbeated inside this window is offline, and any
# job it holds is abandoned. Client heartbeats every few seconds on their
# own thread, independent of generation, so a slow model doesn't trip it.
LIVENESS_SEC = 20
# A job nobody has picked up inside this window means the client is alive
# but not polling (e.g. paused right after being claimed) - give up on it.
PICKUP_TIMEOUT_SEC = 45

DEFAULT_BIND_HOST = "127.0.0.1"
DEFAULT_BIND_PORT = 8642
HOST_PREFIX = "remote://"


class RemoteHostError(Exception):
    """A remote host dropped, timed out, or reported an error mid-call.
    fenra.py treats every flavor identically: abandon this attempt, retry
    the turn on another eligible host (or the local one)."""


def is_remote_host(host):
    return isinstance(host, str) and host.startswith(HOST_PREFIX)


def normalize_tag(tag):
    """Ollama lists untagged models as ':latest' ('phi4-mini' is stored as
    'phi4-mini:latest') - compare on the normalized form so a world config
    saying 'phi4-mini' still matches. Otherwise exact: family and size
    must match, never 'close enough'."""
    tag = tag.strip()
    return tag if ":" in tag else f"{tag}:latest"


class RemoteHostManager:
    def __init__(self, config_path):
        self._config_path = config_path
        self._lock = threading.Lock()
        self._tokens = {}    # token -> label
        self._clients = {}   # label -> live state (see _touch)
        self._jobs = {}      # job_id -> job dict, only while in flight
        self._server = None
        self.bind_host = DEFAULT_BIND_HOST
        self.bind_port = DEFAULT_BIND_PORT

    # ---- config / lifecycle -------------------------------------------

    def load_config(self):
        """Returns True if there's a usable config with at least one
        client. Re-readable; tokens are only ever replaced wholesale."""
        try:
            with open(self._config_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            return False
        tokens = {}
        for entry in cfg.get("clients", []):
            token, label = entry.get("token"), entry.get("label")
            if token and label:
                tokens[token] = label
        with self._lock:
            self._tokens = tokens
        self.bind_host = cfg.get("bind_host", DEFAULT_BIND_HOST)
        self.bind_port = int(cfg.get("bind_port", DEFAULT_BIND_PORT))
        return bool(tokens)

    def start(self):
        """Idempotent. Silently does nothing without a usable config, so
        a plain single-machine world never opens a port."""
        if self._server is not None or not self.load_config():
            return False
        manager = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def _send(self, status, payload=None):
                self.send_response(status)
                if payload is None:
                    self.end_headers()
                    return
                data = json.dumps(payload).encode("utf-8")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self):
                length = int(self.headers.get("Content-Length", 0) or 0)
                if not length:
                    return {}
                try:
                    return json.loads(self.rfile.read(length))
                except ValueError:
                    return None

            def _label(self):
                header = self.headers.get("Authorization", "")
                if not header.startswith("Bearer "):
                    return None
                return manager.label_for_token(header[len("Bearer "):])

            def do_POST(self):
                label = self._label()
                if label is None:
                    self._send(401, {"error": "bad or missing token"})
                    return
                body = self._body()
                if body is None:
                    self._send(400, {"error": "invalid JSON"})
                    return
                if self.path == "/api/v1/clients/heartbeat":
                    manager.heartbeat(label, body)
                    self._send(200, {"ack": True})
                    return
                prefix = "/api/v1/clients/jobs/"
                if self.path.startswith(prefix) and self.path.endswith("/result"):
                    job_id = self.path[len(prefix):-len("/result")]
                    if manager.accept_result(label, job_id, body):
                        self._send(200, {"ack": True})
                    else:
                        self._send(409, {"ack": False, "reason": "stale job"})
                    return
                self._send(404, {"error": "not found"})

            def do_GET(self):
                label = self._label()
                if label is None:
                    self._send(401, {"error": "bad or missing token"})
                    return
                if self.path == "/api/v1/clients/jobs/next":
                    job = manager.next_job(label)
                    if job is None:
                        self._send(204)
                    else:
                        self._send(200, job)
                    return
                self._send(404, {"error": "not found"})

        self._server = ThreadingHTTPServer((self.bind_host, self.bind_port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return True

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    # ---- what the HTTP handlers call ------------------------------------

    def label_for_token(self, token):
        """Constant-time compare against every known token - the token is
        the real identity, the label is just what the server calls it."""
        found = None
        with self._lock:
            for known, label in self._tokens.items():
                if hmac.compare_digest(known.encode(), token.encode()):
                    found = label
        return found

    def _touch(self, label):
        return self._clients.setdefault(label, {
            "last_seen": 0.0, "models": set(), "status": "idle",
            "running_model": None, "version": None, "claimed_id": None,
        })

    def heartbeat(self, label, body):
        # The client-supplied client_id is display-only and never trusted
        # over the token-derived label (agreed in the plan): a mismatch is
        # simply ignored so a client can't rename itself.
        with self._lock:
            c = self._touch(label)
            c["last_seen"] = time.time()
            c["models"] = {normalize_tag(m) for m in body.get("models", [])}
            c["status"] = body.get("status", "idle")
            c["running_model"] = body.get("running_model")
            c["version"] = body.get("client_version")

    def next_job(self, label):
        with self._lock:
            self._touch(label)["last_seen"] = time.time()
            for job_id, job in self._jobs.items():
                if job["label"] == label and job["state"] == "pending":
                    job["state"] = "dispatched"
                    return {"job_id": job_id, "kind": job["kind"],
                            "ollama_request": job["request"]}
        return None

    def accept_result(self, label, job_id, body):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["label"] != label or job["state"] != "dispatched":
                return False   # superseded, abandoned, unknown, or not theirs
            job["result"] = body
            job["state"] = "done"
            job["event"].set()
            return True

    # ---- what fenra.py calls --------------------------------------------

    def eligible_hosts(self, required_models):
        """Remote hosts (as 'remote://label') that are alive, idle, and
        hold EVERY model this turn needs. A turn's urge/voice/function-
        agent calls all run on one host, so it needs all of them."""
        need = {normalize_tag(m) for m in required_models}
        now = time.time()
        out = []
        with self._lock:
            for label, c in self._clients.items():
                if (now - c["last_seen"] < LIVENESS_SEC
                        and c["status"] == "idle" and need <= c["models"]):
                    out.append(HOST_PREFIX + label)
        return sorted(out)

    def call(self, host, kind, request):
        """Blocks until the client returns the raw Ollama response dict.
        Raises RemoteHostError if the client vanishes, never picks the job
        up, or reports any error. No overall timeout on purpose - a slow
        model is legitimate; liveness is what tells us the host is gone."""
        label = host[len(HOST_PREFIX):]
        job_id = uuid.uuid4().hex
        event = threading.Event()
        job = {"label": label, "kind": kind, "request": request,
               "state": "pending", "result": None, "event": event}
        with self._lock:
            self._jobs[job_id] = job
        started = time.time()
        try:
            while not event.wait(1.0):
                with self._lock:
                    c = self._clients.get(label)
                    alive = c is not None and time.time() - c["last_seen"] < LIVENESS_SEC
                    state = job["state"]
                if not alive:
                    raise RemoteHostError(f"{label} stopped responding")
                if state == "pending" and time.time() - started > PICKUP_TIMEOUT_SEC:
                    raise RemoteHostError(f"{label} never picked up the job")
        except RemoteHostError:
            with self._lock:
                job["state"] = "abandoned"
            raise
        finally:
            with self._lock:
                self._jobs.pop(job_id, None)
        result = job["result"] or {}
        if result.get("outcome") == "ok":
            return result.get("ollama_response") or {}
        raise RemoteHostError(
            f"{label} reported {result.get('error_kind', 'error')}: "
            f"{result.get('error_detail', '')}"
        )

    def snapshot(self):
        """One row per known client, for a future Connections tab."""
        now = time.time()
        with self._lock:
            return [
                {"label": label,
                 "online": now - c["last_seen"] < LIVENESS_SEC,
                 "status": c["status"], "models": sorted(c["models"]),
                 "running_model": c["running_model"], "version": c["version"],
                 "seconds_since_seen": round(now - c["last_seen"], 1)}
                for label, c in sorted(self._clients.items())
            ]
