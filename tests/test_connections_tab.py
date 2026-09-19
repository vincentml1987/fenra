"""The Connections tab, built inside the real FenraApp with a stand-in
host manager. Skipped where Tk can't open a display."""
import os
import sys
import time
import tkinter as tk

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fenra  # noqa: E402

# Tk variables are garbage-collected after root.destroy(), which logs
# "main thread is not in main loop" - teardown noise, not a real failure.
pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")

LOCAL = "http://localhost:11434"


class FakeHosts:
    def __init__(self):
        self.rows = [
            {"label": "Vero", "online": True, "status": "idle",
             "models": ["granite4.1:8b", "qwen3:30b"], "running_model": None,
             "version": "0.1", "seconds_since_seen": 3.2},
            {"label": "Tyler", "online": False, "status": "never connected",
             "models": [], "running_model": None, "version": None,
             "seconds_since_seen": None},
        ]

    def start(self):
        return False

    def status(self):
        return {"listening": True, "bind_host": "0.0.0.0", "bind_port": 8642,
                "clients_configured": 2}

    def snapshot(self):
        return [dict(r) for r in self.rows]

    def eligible_hosts(self, models):
        return []


@pytest.fixture
def app(monkeypatch):
    fake = FakeHosts()
    monkeypatch.setattr(fenra, "HOSTS", fake)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    root.withdraw()
    a = fenra.FenraApp(root)
    a.fake = fake
    a._local_models = ["gemma3:27b"]
    a._local_models_fetched_at = time.time()   # don't hit a real Ollama
    yield a
    fenra._host_activity.clear()
    root.destroy()


def rows(app):
    t = app.connections_tree
    return {i: t.item(i, "values") for i in t.get_children()}


def test_lists_local_and_every_configured_client(app):
    app._render_connections()
    r = rows(app)
    assert list(r) == [LOCAL, "remote://Vero", "remote://Tyler"]
    assert r[LOCAL][1] == "Local" and r["remote://Vero"][1] == "Remote"
    assert r["remote://Tyler"][2] == "never connected" and r["remote://Tyler"][5] == "never"
    assert "listening on 0.0.0.0:8642" in app.connections_status_var.get()


def test_shows_which_voice_and_phase_is_running_where(app):
    fenra.set_host_activity("remote://Vero", "Cove", "voice")
    app._render_connections()
    assert rows(app)["remote://Vero"][3].startswith("Cove - voice (")
    fenra.release_host("remote://Vero", "Cove")
    app._render_connections()
    assert rows(app)["remote://Vero"][3] == "idle"


def test_updates_in_place_and_reflects_paused_and_offline(app):
    app._render_connections()
    app.fake.rows[0]["status"] = "paused"
    app._render_connections()
    assert rows(app)["remote://Vero"][2] == "paused"
    app.fake.rows[0]["online"] = False
    app._render_connections()
    assert rows(app)["remote://Vero"][2] == "offline"


def test_status_line_when_no_client_server(app, monkeypatch):
    monkeypatch.setattr(app.fake, "status", lambda: {
        "listening": False, "bind_host": "127.0.0.1", "bind_port": 8642,
        "clients_configured": 0})
    app._render_connections()
    assert "not running" in app.connections_status_var.get()
