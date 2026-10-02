"""Client config: default/load/save trio, matching fenra.py's own
default_x_state()/load_x_state()/save_x_state() shape (merge on-disk
JSON over defaults, silently fall back to defaults on any read/parse
error, never crash on a missing or corrupt config file)."""
import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "client_config.json")


def default_client_state():
    return {
        "client_id": "unnamed-client",
        "token": "",
        "server_scheme": "http",  # swap to "https" here when ready; never inline elsewhere
        "server_host": "127.0.0.1",  # placeholder for local dogfooding; real volunteer
                                      # setups will point this at Teddy's actual server
        "server_port": 8642,
        "ollama_host": "http://127.0.0.1:11434",  # always local; 127.0.0.1 not "localhost" -
                                                   # resolving the hostname "localhost" was
                                                   # observed to add ~2s per call on some
                                                   # Windows machines, the IP has no such cost
        "heartbeat_interval_sec": 7,
        "work_poll_interval_sec": 2,
        "client_version": "0.1.0",
    }


def load_client_state(path=CONFIG_PATH):
    state = default_client_state()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                state.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return state


def save_client_state(state, path=CONFIG_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
