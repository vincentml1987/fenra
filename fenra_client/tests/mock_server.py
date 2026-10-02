"""Minimal stdlib mock of the proposed Fenra server contract (see
Communications/client-server-plan.md, "Vero's Proposed Contract").

Lets the client be exercised end-to-end (real local Ollama, real HTTP
round trip) without Qualia's actual server existing yet. Not part of the
shipped client - a manual testing tool only.

Run directly: python -m fenra_client.tests.mock_server
Then, in another shell, inject a job:
    curl -X POST http://127.0.0.1:8642/admin/enqueue -H "Content-Type: application/json" -d "{\"model\": \"llama3.2\", \"kind\": \"generate\", \"prompt\": \"Say hi in five words.\"}"
And to make the next result submission for a given job_id look stale:
    curl -X POST http://127.0.0.1:8642/admin/mark-stale/<job_id>
"""
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_lock = threading.Lock()
_job_queue = []
_stale_job_ids = set()


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload=None):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if payload is not None:
            self.wfile.write(json.dumps(payload).encode("utf-8"))

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def log_message(self, fmt, *args):
        print("[mock_server]", fmt % args)

    def do_POST(self):
        if self.path == "/api/v1/clients/heartbeat":
            body = self._read_json()
            print(f"[heartbeat] {body}")
            self._send_json(200, {"ack": True})
            return

        if self.path.startswith("/api/v1/clients/jobs/") and self.path.endswith("/result"):
            job_id = self.path.split("/")[-2]
            body = self._read_json()
            with _lock:
                is_stale = job_id in _stale_job_ids
                _stale_job_ids.discard(job_id)
            if is_stale:
                self._send_json(409, {"ack": False, "reason": "stale job"})
                return
            print(f"[result] job {job_id}: {body}")
            self._send_json(200, {"ack": True})
            return

        if self.path == "/admin/enqueue":
            body = self._read_json()
            job_id = str(uuid.uuid4())
            kind = body.pop("kind", "generate")
            job = {"job_id": job_id, "kind": kind, "ollama_request": body}
            with _lock:
                _job_queue.append(job)
            print(f"[admin] enqueued job {job_id}")
            self._send_json(200, {"job_id": job_id})
            return

        if self.path.startswith("/admin/mark-stale/"):
            job_id = self.path.split("/")[-1]
            with _lock:
                _stale_job_ids.add(job_id)
            self._send_json(200, {"ack": True})
            return

        self._send_json(404, {"error": "not found"})

    def do_GET(self):
        if self.path == "/api/v1/clients/jobs/next":
            with _lock:
                job = _job_queue.pop(0) if _job_queue else None
            if job is None:
                self.send_response(204)
                self.end_headers()
            else:
                print(f"[dispatch] handing out job {job['job_id']}")
                self._send_json(200, job)
            return

        self._send_json(404, {"error": "not found"})


def run(port=8642):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Mock Fenra server listening on http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
