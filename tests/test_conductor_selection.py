import json
import conductor


def test_select_next_agent():
    conductor.save_agents = lambda *a, **k: None
    conductor.PDVS = {"talk": 0.9}
    conductor.CLASSES = {
        "ruminator": {"triggering_pdv": "talk", "pdv_adjustments": []},
        "speaker": {"triggering_pdv": "talk", "pdv_adjustments": []},
    }
    ava = {
        "name": "Ava",
        "agent_class": "ruminator",
        "groups_out": ["G"],
        "groups_in": [],
    }
    echo = {
        "name": "Echo",
        "agent_class": "speaker",
        "groups_in": ["G"],
        "groups_out": ["Core"],
    }
    sink = {
        "name": "Sink",
        "agent_class": "speaker",
        "groups_in": ["Core"],
        "groups_out": [],
    }
    conductor.AGENTS = [ava, echo, sink]
    conductor.AGENTS_BY_NAME = {"Ava": ava, "Echo": echo, "Sink": sink}
    conductor.AGENTS_BY_GROUP_IN = {"G": {"Echo"}, "Core": {"Sink"}}
    nxt = conductor.select_next_agent("Ava")
    assert nxt["name"] == "Echo"


def test_dead_end_flagging():
    conductor.save_agents = lambda *a, **k: None
    conductor.PDVS = {"talk": 0.5}
    conductor.CLASSES = {"ruminator": {"triggering_pdv": "talk", "pdv_adjustments": []}}
    solo = {
        "name": "Solo",
        "agent_class": "ruminator",
        "groups_out": ["G2"],
        "groups_in": [],
    }
    conductor.AGENTS = [solo]
    conductor.AGENTS_BY_NAME = {"Solo": solo}
    conductor.AGENTS_BY_GROUP_IN = {}
    res = conductor.select_next_agent("Solo")
    assert res is None
    assert solo.get("flag_no_downstream") is True
    assert solo.get("missing_out_groups") == ["G2"]


def test_select_next_agent_prefers_listener(tmp_path):
    conductor.save_agents = lambda *a, **k: None
    queue_path = tmp_path / "queued_messages.json"
    conductor.QUEUE_PATH = str(queue_path)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps([{"message": "hi"}]))
    orig_has_queue = conductor.has_queued_messages
    conductor.has_queued_messages = lambda path=str(queue_path): orig_has_queue(str(queue_path))

    conductor.PDVS = {"talk": 0.5}
    conductor.CLASSES = {
        "base": {"triggering_pdv": "talk", "pdv_adjustments": []},
        "listener": {"triggering_pdv": "talk", "pdv_adjustments": [], "reads_message_queue": True},
    }
    src = {"name": "Src", "agent_class": "base", "groups_out": ["G"], "groups_in": []}
    non_listener = {
        "name": "NonListener",
        "agent_class": "base",
        "groups_in": ["G"],
        "groups_out": [],
    }
    listener = {
        "name": "Listener",
        "agent_class": "listener",
        "groups_in": ["G"],
        "groups_out": ["Out"],
    }
    non_listener["groups_out"] = ["Out"]
    sink = {
        "name": "Sink",
        "agent_class": "base",
        "groups_in": ["Out"],
        "groups_out": [],
    }
    conductor.AGENTS = [src, non_listener, listener, sink]
    conductor.AGENTS_BY_NAME = {a["name"]: a for a in conductor.AGENTS}
    conductor.AGENTS_BY_GROUP_IN = {"G": {"NonListener", "Listener"}, "Out": {"Sink"}}

    nxt = conductor.select_next_agent("Src")
    assert nxt["name"] == "Listener"


def test_select_next_agent_queue_no_listeners(tmp_path):
    conductor.save_agents = lambda *a, **k: None
    queue_path = tmp_path / "queued_messages.json"
    conductor.QUEUE_PATH = str(queue_path)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps([{"message": "hi"}]))
    orig_has_queue = conductor.has_queued_messages
    conductor.has_queued_messages = lambda path=str(queue_path): orig_has_queue(str(queue_path))

    conductor.PDVS = {"talk": 0.5}
    conductor.CLASSES = {
        "base": {"triggering_pdv": "talk", "pdv_adjustments": []},
    }
    src = {"name": "Src", "agent_class": "base", "groups_out": ["G"], "groups_in": []}
    dst = {
        "name": "Dst",
        "agent_class": "base",
        "groups_in": ["G"],
        "groups_out": ["Out"],
    }
    sink = {
        "name": "Sink",
        "agent_class": "base",
        "groups_in": ["Out"],
        "groups_out": [],
    }
    conductor.AGENTS = [src, dst, sink]
    conductor.AGENTS_BY_NAME = {a["name"]: a for a in conductor.AGENTS}
    conductor.AGENTS_BY_GROUP_IN = {"G": {"Dst"}, "Out": {"Sink"}}

    nxt = conductor.select_next_agent("Src")
    assert nxt["name"] == "Dst"
