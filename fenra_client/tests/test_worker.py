import time

import pytest

from fenra_client import net, worker
from fenra_client.config import default_client_state
from fenra_client.worker import ClientState, ClientWorker


def _fast_config():
    cfg = default_client_state()
    cfg["heartbeat_interval_sec"] = 0.2
    cfg["work_poll_interval_sec"] = 0.15
    return cfg


def test_heartbeat_survives_slow_job(monkeypatch, fake_ollama):
    """Heartbeat must keep firing on schedule even while the work loop
    is blocked on a slow local Ollama call."""
    fake_ollama.delay_seconds = 1.5
    state = ClientState(_fast_config())
    state.config["ollama_host"] = fake_ollama.host

    heartbeat_calls = []
    jobs = [{"job_id": "job-1", "kind": "generate", "ollama_request": {"model": "x"}}]

    def fake_send_heartbeat(cfg, models, status, running_model):
        heartbeat_calls.append(time.monotonic())
        return {"ack": True}

    def fake_poll_work(cfg):
        return jobs.pop(0) if jobs else None

    def fake_submit_result(cfg, job_id, **kwargs):
        return {"ack": True}

    monkeypatch.setattr(net, "send_heartbeat", fake_send_heartbeat)
    monkeypatch.setattr(net, "poll_work", fake_poll_work)
    monkeypatch.setattr(net, "submit_result", fake_submit_result)

    client_worker = ClientWorker(state)
    client_worker.start()
    time.sleep(2.0)  # spans the whole 1.5s "slow job"
    client_worker.stop()
    time.sleep(0.3)

    # heartbeat_interval_sec=0.2 over a 2s window should yield several
    # heartbeats; if the loops shared a thread we would see ~0-1.
    assert len(heartbeat_calls) >= 5, heartbeat_calls


def test_pause_stops_new_work_polling(monkeypatch):
    state = ClientState(_fast_config())
    poll_count = {"n": 0}

    def fake_poll_work(cfg):
        poll_count["n"] += 1
        return None

    def fake_send_heartbeat(cfg, models, status, running_model):
        return {"ack": True}

    monkeypatch.setattr(net, "poll_work", fake_poll_work)
    monkeypatch.setattr(net, "send_heartbeat", fake_send_heartbeat)
    monkeypatch.setattr(worker.ollama_relay, "list_models", lambda host: [])

    state.set_paused(True)
    client_worker = ClientWorker(state)
    client_worker.start()
    time.sleep(0.6)
    client_worker.stop()
    time.sleep(0.2)

    assert poll_count["n"] == 0, "poll_work should not be called while paused"
