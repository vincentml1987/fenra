import threading
import time

import pytest

from fenra_client import ollama_relay
from fenra_client.worker import ClientState
from fenra_client.config import default_client_state


def _fresh_state():
    return ClientState(default_client_state())


def test_run_job_success(fake_ollama):
    state = _fresh_state()
    fake_ollama.response_body = {"response": "hello"}
    result = ollama_relay.run_job(fake_ollama.host, "generate", {"model": "x"}, state)
    assert result["response"] == "hello"


def test_run_job_malformed_response(fake_ollama):
    state = _fresh_state()
    fake_ollama.response_body = {"unexpected": "shape"}
    with pytest.raises(ollama_relay.MalformedResponseError):
        ollama_relay.run_job(fake_ollama.host, "generate", {"model": "x"}, state)


def test_run_job_accepts_tool_call_with_empty_content(fake_ollama):
    """A function-agent response is a real, valid chat response even
    though content is empty - the answer lives in tool_calls instead.
    Regression test for the bug Qualia found on the first live run: every
    remote function-agent call was being rejected as malformed."""
    state = _fresh_state()
    fake_ollama.response_body = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "yell", "arguments": {}}}],
        }
    }
    result = ollama_relay.run_job(fake_ollama.host, "chat", {"model": "x"}, state)
    assert result["message"]["tool_calls"]


def test_run_job_rejects_chat_with_no_content_and_no_tool_calls(fake_ollama):
    state = _fresh_state()
    fake_ollama.response_body = {"message": {"role": "assistant", "content": ""}}
    with pytest.raises(ollama_relay.MalformedResponseError):
        ollama_relay.run_job(fake_ollama.host, "chat", {"model": "x"}, state)


def test_kill_interrupts_in_flight_call(fake_ollama):
    """A slow fake Ollama call (5s) must be interrupted well before it
    would naturally finish once request_kill() is called."""
    fake_ollama.delay_seconds = 5
    state = _fresh_state()
    result_holder = {}

    def worker_thread():
        try:
            ollama_relay.run_job(fake_ollama.host, "generate", {"model": "x"}, state)
            result_holder["outcome"] = "completed"
        except ollama_relay.KilledError:
            result_holder["outcome"] = "killed"
        except Exception as exc:  # any connection-level abort also counts as killed
            result_holder["outcome"] = f"other: {exc}"

    thread = threading.Thread(target=worker_thread)
    start = time.monotonic()
    thread.start()
    time.sleep(0.3)  # let the call actually start and register its session
    state.request_kill()
    thread.join(timeout=3)
    elapsed = time.monotonic() - start

    assert not thread.is_alive(), "worker thread did not unblock after kill"
    assert elapsed < 3, f"kill took too long to interrupt the call ({elapsed:.2f}s)"
    assert result_holder["outcome"] in ("killed",), result_holder
