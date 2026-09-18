"""Relay to the client's own local Ollama instance.

Zero Fenra domain knowledge: this module does not know what a "voice" or
"urge" is, it just forwards an already-assembled Ollama request body to
localhost and hands back whatever Ollama says (or an error).

Kill support: run_job() performs the actual HTTP call in a child
process, not the calling thread. This was NOT the original design
(a requests.Session closed from another thread was tried first) - that
approach was empirically tested and confirmed NOT to work: calling
Session.close() while a session.post() is blocked waiting on a slow
server response has no effect at all on this platform/library version;
the blocked call runs to completion regardless. A multiprocessing.Process
does not have this problem - Process.terminate() reliably and
immediately kills the child, unblocking the caller, confirmed via direct
testing (killed a 5s-blocking call in ~0.5s). This is the actual
mechanism "true kill" requires.

Ollama's own server-side generation may still keep running after the
child process is terminated, since terminating the client's process only
drops its end of the HTTP connection - that is a documented, accepted
limitation (this module can only stop the client's own participation,
not what a remote Ollama server chooses to do afterward), not a bug to
work around here.
"""
import multiprocessing
import queue as queue_module

import requests


class KilledError(Exception):
    """Raised when a job's child process was terminated by a kill request."""


class MalformedResponseError(Exception):
    """Raised when Ollama's response fails a basic structural check."""


def list_models(ollama_host, timeout=10):
    """Poll local Ollama's model inventory. Never raises - returns []
    on any failure, matching fenra.py's own list_ollama_models()."""
    try:
        resp = requests.get(f"{ollama_host}/api/tags", timeout=timeout)
        resp.raise_for_status()
        return sorted(m["name"] for m in resp.json().get("models", []))
    except requests.RequestException:
        return []


def _endpoint_for_kind(ollama_host, kind):
    if kind == "chat":
        return f"{ollama_host}/api/chat"
    return f"{ollama_host}/api/generate"


def _check_structural(kind, payload):
    """Cheap, structural-only validation - well-formed JSON (already
    guaranteed by resp.json() not raising), expected shape, non-empty.
    No semantic/content validation - that is explicitly out of scope,
    per the agreed plan (the dispatch layer downstream already treats
    all model output as untrusted regardless of source)."""
    if not isinstance(payload, dict):
        raise MalformedResponseError("response body was not a JSON object")
    if kind == "chat":
        message = payload.get("message")
        if not isinstance(message, dict) or not message.get("content"):
            raise MalformedResponseError("chat response missing non-empty message.content")
    else:
        if not payload.get("response"):
            raise MalformedResponseError("generate response missing non-empty 'response'")


def _child_call(ollama_host, kind, request_body, result_queue):
    """Runs in a separate process. Must only communicate back via the
    queue - no shared state with the parent."""
    try:
        resp = requests.post(
            _endpoint_for_kind(ollama_host, kind),
            json=request_body,
            timeout=None,  # a client-side timeout would not cancel server-side
                           # generation anyway - killing this whole process is
                           # what true-kill actually relies on instead.
        )
        resp.raise_for_status()
        result_queue.put(("ok", resp.json()))
    except requests.exceptions.RequestException as exc:
        result_queue.put(("error", str(exc)))
    except Exception as exc:  # keep the child from dying silently on anything else
        result_queue.put(("error", str(exc)))


def run_job(ollama_host, kind, ollama_request, client_state, poll_interval=0.1):
    """Forward ollama_request verbatim to local Ollama in a child
    process, return the raw parsed JSON response.

    Stores the child Process on client_state so a kill request from
    another thread can terminate it. Raises KilledError if terminated
    before completion, MalformedResponseError if Ollama's response fails
    the structural check, or requests.RequestException-shaped failure
    (re-raised as a plain RuntimeError, since the actual exception object
    cannot cross the process boundary) for a real connection-level
    failure.
    """
    request_body = dict(ollama_request)
    request_body["stream"] = False  # forced regardless of what the job specified

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_child_call, args=(ollama_host, kind, request_body, result_queue)
    )

    with client_state.lock:
        if client_state.kill_requested:
            client_state.kill_requested = False
            raise KilledError("kill requested before the call was issued")
        client_state.active_process = process

    process.start()
    try:
        while True:
            with client_state.lock:
                killed = client_state.kill_requested
            if killed:
                process.terminate()
                process.join(timeout=5)
                with client_state.lock:
                    client_state.kill_requested = False
                raise KilledError("kill requested during the call")
            try:
                outcome, payload = result_queue.get(timeout=poll_interval)
                break
            except queue_module.Empty:
                if not process.is_alive():
                    # Child died without putting anything on the queue.
                    raise RuntimeError("Ollama relay process exited unexpectedly")
                continue
    finally:
        with client_state.lock:
            client_state.active_process = None
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)

    if outcome == "error":
        raise RuntimeError(payload)

    _check_structural(kind, payload)
    return payload
