"""Phase A of the concurrency work: shared state must survive overlapping
turns. These use a temp worlds directory and temp corrections file, never
the real ones."""
import json
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fenra  # noqa: E402

WORLD = "w"
VOICE = "Cove"


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(fenra, "WORLDS_DIR", str(tmp_path / "worlds"))
    monkeypatch.setattr(fenra, "BASE_DIR", str(tmp_path))
    fenra.save_voice_state(WORLD, VOICE, fenra.default_voice_state())
    return tmp_path


def run_threads(n, target):
    threads = [threading.Thread(target=target, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not any(t.is_alive() for t in threads), "a worker hung"


def test_atomic_write_never_exposes_a_partial_file(world):
    path = fenra.voice_state_path(WORLD, VOICE)
    stop = threading.Event()
    bad = []

    def writer(_):
        n = 0
        while not stop.is_set():
            n += 1
            state = fenra.default_voice_state()
            state["thoughts"] = [{"id": i, "text": "x" * 200} for i in range(50 + n % 50)]
            fenra.save_voice_state(WORLD, VOICE, state)

    def reader(_):
        for _i in range(300):
            try:
                with open(path, encoding="utf-8") as f:
                    json.load(f)
            except json.JSONDecodeError as exc:
                bad.append(exc)
            except OSError:
                pass   # Windows can refuse an open mid-replace; not a torn read

    w = threading.Thread(target=writer, args=(0,))
    w.start()
    try:
        run_threads(4, reader)
    finally:
        stop.set()
        w.join(30)
    assert not bad, f"reader saw a partial file {len(bad)} times"


def test_concurrent_append_message_loses_nothing(world):
    per, workers = 25, 8

    def work(i):
        for k in range(per):
            fenra.append_message(WORLD, VOICE, VOICE, f"t{i}-{k}")

    run_threads(workers, work)
    thoughts = fenra.load_voice_state(WORLD, VOICE)["thoughts"]
    assert len(thoughts) == per * workers
    assert len({t["id"] for t in thoughts}) == per * workers      # ids unique


def test_concurrent_dispatch_corrections_lose_nothing(world):
    per, workers = 20, 8

    def work(i):
        for k in range(per):
            fenra.append_dispatch_correction(WORLD, VOICE, f"item{i}-{k}", None, "ok")

    run_threads(workers, work)
    entries = fenra.load_dispatch_corrections()
    assert len(entries) == per * workers
    assert len({e["id"] for e in entries}) == per * workers


def test_update_voice_state_is_atomic_against_concurrent_appends(world):
    def work(i):
        for k in range(20):
            if i % 2:
                fenra.append_message(WORLD, VOICE, VOICE, f"m{i}-{k}")
            else:
                fenra.update_voice_state(
                    WORLD, VOICE, lambda s, k=k: s.update(last_function_agent_note=f"n{k}"))

    run_threads(6, work)
    state = fenra.load_voice_state(WORLD, VOICE)
    assert len(state["thoughts"]) == 3 * 20          # no append lost to a note update
    assert state["last_function_agent_note"].startswith("n")


def test_lock_is_not_held_across_a_slow_llm_call(world, monkeypatch):
    """The whole design depends on the LLM wait happening OUTSIDE the world
    lock. A slow fake function-agent call must not stop another thread."""
    started = threading.Event()

    def slow_agent(*a, **k):
        started.set()
        time.sleep(1.5)
        return {"content": "", "tool_calls": []}

    monkeypatch.setattr(fenra, "call_function_agent", slow_agent)
    monkeypatch.setattr(fenra, "build_function_agent_tools", lambda: [])
    monkeypatch.setattr(fenra, "build_function_agent_prompt", lambda *a, **k: "p")
    t = threading.Thread(
        target=fenra.run_function_agent_turn,
        args=("http://x", WORLD, VOICE, "text", "hud", "m"), daemon=True)
    t.start()
    assert started.wait(5)
    t0 = time.time()
    fenra.append_message(WORLD, VOICE, VOICE, "written while the LLM call is in flight")
    assert time.time() - t0 < 0.5, "world lock was held during the LLM call"
    t.join(10)


def test_world_lock_is_reentrant_for_nested_locked_calls(world):
    # dispatch_one_function_call (locked) -> fn_* -> _log_room_event (locked)
    with fenra.WORLD_LOCK:
        fenra.append_message(WORLD, VOICE, VOICE, "nested")
    assert len(fenra.load_voice_state(WORLD, VOICE)["thoughts"]) == 1
