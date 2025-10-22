import sys
import types

import pytest


def _stub_conductor():
    module = types.ModuleType("conductor")
    module._CONFIGS_LOADED = True
    module.AGENTS = [
        {"name": "Agent One", "groups_in": ["Alpha"], "groups_out": []},
        {"name": "Agent Two", "groups_in": [], "groups_out": []},
    ]
    module.AGENTS_BY_NAME = {a["name"]: a for a in module.AGENTS}
    module.AGENTS_BY_GROUP_IN = {"Alpha": {"Agent One"}}
    module.STATE = {"current_agent": "Agent One"}
    module.UI = None
    return module


@pytest.fixture
def conductor_stub(monkeypatch):
    stub = _stub_conductor()
    monkeypatch.setitem(sys.modules, "conductor", stub)
    return stub


@pytest.fixture(autouse=True)
def stub_savers(monkeypatch):
    saved = {"agents": None, "state": None}

    def fake_save_agents(agents):
        saved["agents"] = [dict(a) for a in agents]

    def fake_save_state(state):
        saved["state"] = dict(state)

    monkeypatch.setattr("config_loader.save_agents", fake_save_agents)
    monkeypatch.setattr("config_loader.save_state", fake_save_state)
    return saved


def test_rename_current_agent_updates_indexes(conductor_stub, stub_savers):
    from fenra_functions import rename_agent

    result = rename_agent("Agent_Prime")

    assert result == "Renamed Agent One → Agent Prime"
    assert conductor_stub.AGENTS[0]["name"] == "Agent Prime"
    assert "Agent Prime" in conductor_stub.AGENTS_BY_NAME
    assert "Agent One" not in conductor_stub.AGENTS_BY_NAME
    assert conductor_stub.AGENTS_BY_GROUP_IN == {"Alpha": {"Agent Prime"}}
    assert conductor_stub.STATE["current_agent"] == "Agent Prime"
    assert stub_savers["agents"][0]["name"] == "Agent Prime"
    assert stub_savers["state"]["current_agent"] == "Agent Prime"


def test_rename_specific_agent_with_two_arguments(conductor_stub, stub_savers):
    from fenra_functions import rename_agent

    result = rename_agent("Agent_Two", "Agent_Second")

    assert result == "Renamed Agent Two → Agent Second"
    assert conductor_stub.AGENTS[1]["name"] == "Agent Second"
    assert "Agent Second" in conductor_stub.AGENTS_BY_NAME
    assert "Agent Two" not in conductor_stub.AGENTS_BY_NAME
    assert conductor_stub.STATE["current_agent"] == "Agent One"
    assert stub_savers["agents"][1]["name"] == "Agent Second"
