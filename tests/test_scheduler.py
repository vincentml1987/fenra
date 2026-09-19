"""Phase B: the concurrent scheduler - capacity claims, fair ordering, and
a real FenraApp offering voices to a local slot and a remote client at
once. Turn threads are faked so no Ollama or clock is involved."""
import os
import sys
import threading
import time
import tkinter as tk

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fenra  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")

LOCAL = "http://localhost:11434"
REMOTE = "remote://Vero"


class FakeHosts:
    """Vero can serve only turns whose models include granite4.1:8b."""
    def __init__(self):
        self.up = True

    def start(self):
        return False

    def status(self):
        return {"listening": False, "bind_host": "127.0.0.1", "bind_port": 0,
                "clients_configured": 0}

    def snapshot(self):
        return []

    def eligible_hosts(self, models):
        return [REMOTE] if self.up and "granite4.1:8b" in models else []


@pytest.fixture(autouse=True)
def clean():
    fenra._host_claims.clear()
    fenra._host_activity.clear()
    yield
    fenra._host_claims.clear()
    fenra._host_activity.clear()


# ---- pure pieces ---------------------------------------------------------

def test_order_candidates_starts_at_rotation_index():
    assert fenra.order_candidates(["A", "B", "C", "D"], 2, {}) == ["C", "D", "A", "B"]


def test_order_candidates_puts_least_recently_started_first():
    got = fenra.order_candidates(["A", "B", "C"], 0, {"A": 30.0, "B": 10.0, "C": 20.0})
    assert got == ["B", "C", "A"]


def test_one_slot_order_is_plain_round_robin():
    started, order = {}, []
    for t in range(1, 7):
        voice = fenra.order_candidates(["A", "B", "C"], 0, started)[0]
        started[voice] = float(t)
        order.append(voice)
    assert order == ["A", "B", "C", "A", "B", "C"]


def test_local_slots_cap_concurrent_local_turns(monkeypatch):
    monkeypatch.setattr(fenra, "HOSTS", FakeHosts())
    need = ["phi4-mini", "qwen2.5:14b", "qwen3:30b"]        # Vero can't serve this
    assert fenra.claim_host_for_voice(LOCAL, need) == LOCAL
    assert fenra.claim_host_for_voice(LOCAL, need) is None   # one slot, taken
    fenra.release_host(LOCAL)
    assert fenra.claim_host_for_voice(LOCAL, need, local_slots=2) == LOCAL
    assert fenra.claim_host_for_voice(LOCAL, need, local_slots=2) == LOCAL
    assert fenra.claim_host_for_voice(LOCAL, need, local_slots=2) is None


def test_remote_holds_one_turn_and_excluded_remote_is_skipped(monkeypatch):
    monkeypatch.setattr(fenra, "HOSTS", FakeHosts())
    need = ["phi4-mini", "granite4.1:8b", "qwen3:30b"]
    assert fenra.claim_host_for_voice(LOCAL, need) == REMOTE
    assert fenra.claim_host_for_voice(LOCAL, need) == LOCAL           # remote busy
    assert fenra.claim_host_for_voice(LOCAL, need) is None            # both busy
    fenra.release_host(REMOTE)
    assert fenra.claim_host_for_voice(LOCAL, need, exclude={REMOTE}) is None


# ---- the scheduler inside a real FenraApp --------------------------------

@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(fenra, "WORLDS_DIR", str(tmp_path / "worlds"))
    monkeypatch.setattr(fenra, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(fenra, "HOSTS", FakeHosts())
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    root.withdraw()
    a = fenra.FenraApp(root)
    a.world_name = "w"
    a.world_voices = ["Ash", "Cove", "Wick"]
    for voice, model in (("Ash", "qwen2.5:14b"), ("Cove", "granite4.1:8b"),
                         ("Wick", "mistral-small:22b")):
        st = fenra.default_voice_state()
        st["model"] = model
        fenra.save_voice_state("w", voice, st)
    a.gates = {v: threading.Event() for v in a.world_voices}
    a.started = []

    def fake_turn(voice, holder):
        a.started.append(voice)
        a.gates[voice].wait(10)
        fenra.release_host(holder["host"], voice)
        holder["host"] = None
        with a._in_flight_lock:
            a._in_flight.pop(voice, None)

    a._run_voice_turn = fake_turn
    yield a
    for g in a.gates.values():
        g.set()
    root.destroy()


def settle():
    time.sleep(0.2)


def test_scheduler_uses_both_hosts_and_skips_the_voice_with_no_free_host(app):
    app._schedule_turns()
    settle()
    # Ash -> local, Cove -> Vero, Wick has no free host (local slot taken)
    assert sorted(app.started) == ["Ash", "Cove"]
    assert sorted(app._in_flight_names()) == ["Ash", "Cove"]
    assert fenra._host_claims == {LOCAL: 1, REMOTE: 1}


def test_skipped_voice_goes_first_when_a_host_frees_up(app):
    app._schedule_turns()
    settle()
    app.gates["Ash"].set()                 # Ash finishes, local frees
    settle()
    app._schedule_turns()
    settle()
    assert "Wick" in app.started           # waited longest, so first in line
    assert app.started.count("Ash") == 1   # and Ash didn't jump the queue


def test_a_voice_already_in_flight_is_never_started_twice(app):
    app._schedule_turns()
    settle()
    app._schedule_turns()
    app._schedule_turns()
    settle()
    assert sorted(app.started) == ["Ash", "Cove"]


def test_paused_and_piloted_voices_are_skipped(app):
    fenra.set_voice_paused("w", "Ash", True)
    st = fenra.load_voice_state("w", "Wick")
    st["piloted"] = True
    fenra.save_voice_state("w", "Wick", st)
    app._schedule_turns()
    settle()
    assert app.started == ["Cove"]


def test_all_paused_reports_it_and_starts_nothing(app):
    for v in app.world_voices:
        fenra.set_voice_paused("w", v, True)
    app._schedule_turns()
    settle()
    assert app.started == []


def test_two_local_slots_run_two_local_voices_at_once(app):
    app.local_slots_var.set("2")
    fenra.HOSTS.up = False                 # Vero offline: everything is local
    app._schedule_turns()
    settle()
    assert sorted(app.started) == ["Ash", "Cove"]      # 2 slots, 3 voices


# ---- the turn body itself, with Ollama faked ------------------------------

@pytest.fixture
def turn_app(app, monkeypatch):
    """`app` with the real _run_voice_turn back in place and the three Ollama
    call functions faked. Records the host and prompt of every voice call."""
    del app._run_voice_turn                       # drop the instance-level fake
    app.running = True
    app.calls = []

    def fake_call_ollama(host, model, prompt, options=None):
        app.calls.append((host, model, prompt))
        if host == REMOTE and model == "granite4.1:8b":
            raise fenra.fenra_hosts.RemoteHostError("Vero stopped responding")
        return "urge text" if model == app.urge_model_var.get() else "a thought"

    monkeypatch.setattr(fenra, "call_ollama", fake_call_ollama)
    monkeypatch.setattr(
        fenra, "call_function_agent",
        lambda host, *a, **k: {"content": "", "tool_calls": []})
    return app


def test_turn_completes_on_the_local_host_and_releases_everything(turn_app):
    a = turn_app
    holder = {"host": fenra.claim_host_for_voice(LOCAL, ["u", "qwen2.5:14b", "f"])}
    with a._in_flight_lock:
        a._in_flight["Ash"] = None
    a._run_voice_turn("Ash", holder)
    assert len(fenra.load_voice_state("w", "Ash")["thoughts"]) == 1
    assert fenra._host_claims == {} and fenra._host_activity == {}
    assert a._in_flight_names() == [] and holder["host"] is None


def test_remote_failure_retries_locally_once_and_keeps_the_hud_note(turn_app):
    a = turn_app
    st = fenra.load_voice_state("w", "Cove")
    st["last_function_agent_note"] = "REMEMBER-THIS-NOTE"
    fenra.save_voice_state("w", "Cove", st)
    need = [a.urge_model_var.get(), "granite4.1:8b", a.function_agent_model_var.get()]
    holder = {"host": fenra.claim_host_for_voice(LOCAL, need)}
    assert holder["host"] == REMOTE
    with a._in_flight_lock:
        a._in_flight["Cove"] = None
    a._run_voice_turn("Cove", holder)

    voice_calls = [c for c in a.calls if c[1] == "granite4.1:8b"]
    assert [c[0] for c in voice_calls] == [REMOTE, LOCAL]        # failed, then retried
    assert all("REMEMBER-THIS-NOTE" in c[2] for c in voice_calls)  # note survived the retry
    assert len(fenra.load_voice_state("w", "Cove")["thoughts"]) == 1   # not duplicated
    assert fenra._host_claims == {} and fenra._host_activity == {}
    assert holder["host"] is None


def test_an_unexpected_error_still_releases_the_claim(turn_app, monkeypatch):
    a = turn_app
    monkeypatch.setattr(fenra, "build_hud", lambda *x, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    holder = {"host": fenra.claim_host_for_voice(LOCAL, ["u", "qwen2.5:14b", "f"])}
    with a._in_flight_lock:
        a._in_flight["Ash"] = None
    a._run_voice_turn("Ash", holder)             # must not raise
    assert fenra._host_claims == {} and a._in_flight_names() == []
