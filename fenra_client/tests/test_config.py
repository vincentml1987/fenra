import json

from fenra_client import config


def test_load_falls_back_to_defaults_on_corrupt_json(tmp_path):
    path = tmp_path / "client_config.json"
    path.write_text("{ this is not valid json", encoding="utf-8")

    state = config.load_client_state(path=str(path))

    assert state == config.default_client_state()


def test_load_merges_on_disk_values_over_defaults(tmp_path):
    path = tmp_path / "client_config.json"
    path.write_text(json.dumps({"client_id": "teddy-box-2"}), encoding="utf-8")

    state = config.load_client_state(path=str(path))

    assert state["client_id"] == "teddy-box-2"
    assert state["server_port"] == config.default_client_state()["server_port"]


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "client_config.json"
    original = config.default_client_state()
    original["token"] = "abc123"

    config.save_client_state(original, path=str(path))
    reloaded = config.load_client_state(path=str(path))

    assert reloaded == original
