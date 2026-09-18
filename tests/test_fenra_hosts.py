"""Server-side tests for fenra_hosts (the Fenra half of the distributed-
compute contract in Communications/client-server-plan.md).

The contract test at the bottom drives Vero's REAL fenra_client.net
against the REAL server, not a mock of either side - that's the actual
interop check.

Run: python -m pytest tests -q
"""
import json
import os
import sys
import threading
import time

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fenra_hosts  # noqa: E402
from fenra_hosts import RemoteHostError, RemoteHostManager, normalize_tag  # noqa: E402

TOKEN = "test-token-vero"
LABEL = "Vero"


@pytest.fixture
def server(tmp_path):
    cfg = tmp_path / "host_clients.json"
    cfg.write_text(json.dumps({
        "bind_host": "127.0.0.1", "bind_port": 0,
        "clients": [{"label": LABEL, "token": TOKEN}],
    }))
    manager = RemoteHostManager(str(cfg))
    assert manager.start()
    port = manager._server.server_address[1]
    manager.base = f"http://127.0.0.1:{port}"
    yield manager
    manager.stop()


def auth(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


def heartbeat(server, models=("phi4-mini:latest", "qwen3:30b", "granite4.1:8b"),
              status="idle", token=TOKEN):
    return requests.post(
        f"{server.base}/api/v1/clients/heartbeat",
        json={"client_id": "whatever", "models": list(models), "status": status,
              "running_model": None, "client_version": "0.1"},
        headers=auth(token),
    )


def test_no_config_means_no_server(tmp_path):
    manager = RemoteHostManager(str(tmp_path / "missing.json"))
    assert manager.start() is False
    assert manager._server is None


def test_bad_or_missing_token_is_401(server):
    assert heartbeat(server, token="wrong").status_code == 401
    assert requests.get(f"{server.base}/api/v1/clients/jobs/next").status_code == 401


def test_heartbeat_registers_client_and_ignores_client_supplied_id(server):
    assert heartbeat(server).json() == {"ack": True}
    (row,) = server.snapshot()
    assert row["label"] == LABEL          # token-derived, not "whatever"
    assert row["online"] and row["status"] == "idle"


def test_normalize_tag_treats_untagged_as_latest():
    assert normalize_tag("phi4-mini") == "phi4-mini:latest"
    assert normalize_tag("qwen3:30b") == "qwen3:30b"


def test_eligibility_needs_every_model_exactly(server):
    heartbeat(server)
    need = ["phi4-mini", "granite4.1:8b", "qwen3:30b"]   # untagged urge model
    assert server.eligible_hosts(need) == ["remote://Vero"]
    assert server.eligible_hosts(need + ["gemma3:27b"]) == []      # missing one
    assert server.eligible_hosts(["qwen3:14b"]) == []               # near-miss size


def test_paused_and_offline_clients_are_not_eligible(server, monkeypatch):
    heartbeat(server, status="paused")
    assert server.eligible_hosts(["qwen3:30b"]) == []
    heartbeat(server, status="idle")
    assert server.eligible_hosts(["qwen3:30b"]) == ["remote://Vero"]
    monkeypatch.setattr(fenra_hosts, "LIVENESS_SEC", 0)
    assert server.eligible_hosts(["qwen3:30b"]) == []


def _fake_client(server, respond, stop):
    """Polls like the real client and answers each job via `respond`."""
    while not stop.is_set():
        r = requests.get(f"{server.base}/api/v1/clients/jobs/next", headers=auth())
        if r.status_code == 200:
            job = r.json()
            body = respond(job)
            requests.post(
                f"{server.base}/api/v1/clients/jobs/{job['job_id']}/result",
                json=dict(body, job_id=job["job_id"]), headers=auth(),
            )
        time.sleep(0.05)


def test_call_round_trips_a_job(server):
    heartbeat(server)
    stop = threading.Event()
    seen = {}

    def respond(job):
        seen.update(job)
        return {"outcome": "ok", "ollama_response": {"response": "hello"}}

    t = threading.Thread(target=_fake_client, args=(server, respond, stop), daemon=True)
    t.start()
    try:
        out = server.call("remote://Vero", "generate",
                          {"model": "granite4.1:8b", "prompt": "hi", "stream": False})
    finally:
        stop.set()
    assert out == {"response": "hello"}
    assert seen["kind"] == "generate" and seen["ollama_request"]["prompt"] == "hi"


def test_client_reported_error_raises_so_the_turn_can_retry(server):
    heartbeat(server)
    stop = threading.Event()
    t = threading.Thread(
        target=_fake_client, daemon=True,
        args=(server, lambda job: {"outcome": "error", "error_kind": "killed",
                                   "error_detail": "owner killed it"}, stop))
    t.start()
    try:
        with pytest.raises(RemoteHostError, match="killed"):
            server.call("remote://Vero", "chat", {"model": "qwen3:30b"})
    finally:
        stop.set()


def test_client_that_stops_heartbeating_abandons_the_job(server, monkeypatch):
    heartbeat(server)
    monkeypatch.setattr(fenra_hosts, "LIVENESS_SEC", 1)
    with pytest.raises(RemoteHostError, match="stopped responding"):
        server.call("remote://Vero", "generate", {"model": "x"})


def test_late_result_for_an_abandoned_job_is_409(server, monkeypatch):
    heartbeat(server)
    monkeypatch.setattr(fenra_hosts, "LIVENESS_SEC", 1)
    got = {}

    def grab_then_go_silent():
        while "job" not in got:
            r = requests.get(f"{server.base}/api/v1/clients/jobs/next", headers=auth())
            if r.status_code == 200:
                got["job"] = r.json()
            time.sleep(0.05)

    threading.Thread(target=grab_then_go_silent, daemon=True).start()
    with pytest.raises(RemoteHostError):
        server.call("remote://Vero", "generate", {"model": "x"})
    late = requests.post(
        f"{server.base}/api/v1/clients/jobs/{got['job']['job_id']}/result",
        json={"outcome": "ok", "ollama_response": {"response": "too late"}},
        headers=auth(),
    )
    assert late.status_code == 409


# ---- the real interop check: Vero's actual client code vs. this server ----

def test_vero_real_net_module_against_real_server(server):
    from fenra_client import net

    host, port = server.base.replace("http://", "").split(":")
    state = {"server_scheme": "http", "server_host": host, "server_port": int(port),
             "token": TOKEN, "client_id": "Vero", "client_version": "0.1"}

    assert net.send_heartbeat(state, ["qwen3:30b"], "idle", None) == {"ack": True}
    assert net.poll_work(state) is None                       # 204 -> None

    result = {}
    caller = threading.Thread(
        target=lambda: result.update(
            out=server.call("remote://Vero", "chat", {"model": "qwen3:30b"})),
        daemon=True)
    caller.start()
    job = None
    for _ in range(100):
        job = net.poll_work(state)
        if job:
            break
        time.sleep(0.05)
    assert job and job["kind"] == "chat"
    net.submit_result(state, job["job_id"], "ok",
                      ollama_response={"message": {"content": "done"}})
    caller.join(5)
    assert result["out"] == {"message": {"content": "done"}}

    with pytest.raises(net.StaleJobError):                     # 409 -> StaleJobError
        net.submit_result(state, job["job_id"], "ok", ollama_response={})
