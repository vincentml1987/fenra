"""fenra.py's use of fenra_hosts: host claiming (eligibility, exclusion on
retry, local fallback) and routing of the two Ollama call functions to a
remote host. Uses a stand-in manager; the real HTTP path is covered in
test_fenra_hosts.py."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fenra  # noqa: E402
import fenra_hosts  # noqa: E402

LOCAL = "http://localhost:11434"
REMOTE = "remote://Vero"
NEED = ["phi4-mini", "granite4.1:8b", "qwen3:30b"]


class FakeHosts:
    def __init__(self, eligible):
        self.eligible = eligible
        self.calls = []

    def eligible_hosts(self, models):
        return list(self.eligible)

    def call(self, host, kind, request):
        self.calls.append((host, kind, request))
        if kind == "generate":
            return {"response": "remote says hi"}
        return {"message": {"content": "", "tool_calls": []}}


@pytest.fixture(autouse=True)
def clean_claims():
    fenra._host_claims.clear()
    yield
    fenra._host_claims.clear()


@pytest.fixture
def fake(monkeypatch):
    f = FakeHosts([REMOTE])
    monkeypatch.setattr(fenra, "HOSTS", f)
    return f


def test_prefers_an_eligible_remote_host(fake):
    assert fenra.claim_host_for_voice(LOCAL, NEED) == REMOTE


def test_falls_back_to_local_when_nothing_eligible(fake):
    fake.eligible = []
    assert fenra.claim_host_for_voice(LOCAL, NEED) == LOCAL


def test_excluded_remote_falls_back_to_local(fake):
    assert fenra.claim_host_for_voice(LOCAL, NEED, exclude={REMOTE}) == LOCAL


def test_an_already_claimed_remote_is_not_double_booked(fake):
    assert fenra.claim_host_for_voice(LOCAL, NEED) == REMOTE
    assert fenra.claim_host_for_voice(LOCAL, NEED) == LOCAL
    fenra.release_host(REMOTE)
    assert fenra.claim_host_for_voice(LOCAL, NEED) == REMOTE


def test_call_ollama_routes_remote_hosts_through_the_job_queue(fake):
    out = fenra.call_ollama(REMOTE, "granite4.1:8b", "hello", options={"num_predict": 5})
    assert out == "remote says hi"
    host, kind, request = fake.calls[0]
    assert (host, kind) == (REMOTE, "generate")
    assert request == {"model": "granite4.1:8b", "prompt": "hello",
                       "stream": False, "options": {"num_predict": 5}}


def test_call_function_agent_routes_remote_hosts_as_chat(fake):
    msg = fenra.call_function_agent(REMOTE, "qwen3:30b", "system", [{"type": "function"}])
    assert msg == {"content": "", "tool_calls": []}
    host, kind, request = fake.calls[0]
    assert kind == "chat"
    assert request["messages"] == [{"role": "system", "content": "system"}]
    assert request["tools"] == [{"type": "function"}] and request["stream"] is False


def test_function_agent_first_attempt_host_loss_propagates_for_a_turn_retry(monkeypatch):
    def boom(*a, **k):
        raise fenra_hosts.RemoteHostError("Vero stopped responding")
    monkeypatch.setattr(fenra, "call_function_agent", boom)
    monkeypatch.setattr(fenra, "build_function_agent_tools", lambda: [])
    monkeypatch.setattr(fenra, "build_function_agent_prompt", lambda *a, **k: "p")
    with pytest.raises(fenra_hosts.RemoteHostError):
        fenra.run_function_agent_turn(REMOTE, "w", "v", "text", "hud", "m")
