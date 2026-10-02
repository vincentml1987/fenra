"""HTTP calls to the Fenra server. This is the ONLY place the server's
scheme/host/port get concatenated into a URL, so swapping http -> https
later is a config edit here, never a code change anywhere else.

Contract shape matches the proposal posted to
Communications/client-server-plan.md (2026-09-18) - not yet confirmed
by Qualia. Endpoint paths and field names here may still change; keep
this module isolated from worker.py's control flow so a contract
revision only touches this file.
"""
import requests


class StaleJobError(Exception):
    """Server responded 409 - this job was already superseded. Not an
    error worth surfacing to the user, just a no-op signal."""


def base_url(state):
    return f"{state['server_scheme']}://{state['server_host']}:{state['server_port']}"


def _headers(state):
    return {"Authorization": f"Bearer {state['token']}"}


def send_heartbeat(state, models, status, running_model, timeout=10):
    body = {
        "client_id": state["client_id"],
        "models": models,
        "status": status,
        "running_model": running_model,
        "client_version": state["client_version"],
    }
    resp = requests.post(
        f"{base_url(state)}/api/v1/clients/heartbeat",
        json=body,
        headers=_headers(state),
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def poll_work(state, timeout=10):
    """Returns None if nothing is assigned, else a dict with job_id,
    kind, ollama_request."""
    resp = requests.get(
        f"{base_url(state)}/api/v1/clients/jobs/next",
        headers=_headers(state),
        timeout=timeout,
    )
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    return resp.json()


def submit_result(state, job_id, outcome, ollama_response=None,
                   error_kind=None, error_detail=None, timeout=15):
    body = {"job_id": job_id, "outcome": outcome}
    if outcome == "ok":
        body["ollama_response"] = ollama_response
    else:
        body["error_kind"] = error_kind
        body["error_detail"] = error_detail
    resp = requests.post(
        f"{base_url(state)}/api/v1/clients/jobs/{job_id}/result",
        json=body,
        headers=_headers(state),
        timeout=timeout,
    )
    if resp.status_code == 409:
        raise StaleJobError(f"job {job_id} already superseded")
    resp.raise_for_status()
    return resp.json()
