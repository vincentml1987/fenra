"""v0.21.0: the currencies -> per-voice inventory refactor. Uses a temp
worlds directory, never the real one."""
import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fenra  # noqa: E402

W = "w"


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(fenra, "WORLDS_DIR", str(tmp_path / "worlds"))
    monkeypatch.setattr(fenra, "BASE_DIR", str(tmp_path))
    return tmp_path


def make_voice(name, inventory=None, world=W):
    st = fenra.default_voice_state()
    st["inventory"] = dict(inventory or {})
    fenra.save_voice_state(world, name, st)


def set_items(items):
    st = fenra.default_world_state()
    st["items"] = items
    fenra.save_world_state(W, st)


def inv(name):
    return fenra.load_voice_state(W, name)["inventory"]


# ---- catalog + starting draw ------------------------------------------------

def test_no_items_listed_means_empty_inventory(world):
    fenra.save_world_state(W, fenra.default_world_state())
    assert fenra.item_catalog(W) == []
    assert fenra.default_voice_state(W)["inventory"] == {}
    assert fenra.default_voice_state()["inventory"] == {}


def test_starting_draw_follows_the_worlds_item_list(world):
    set_items([{"name": "zib", "min": 3, "max": 3}, {"name": "quorp", "min": 2, "max": 5},
               {"name": "none", "min": 0, "max": 0}])
    for _ in range(20):
        drawn = fenra.default_voice_state(W)["inventory"]
        assert drawn["zib"] == 3
        assert 2 <= drawn["quorp"] <= 5
        assert "none" not in drawn        # a 0 draw means "not owned"


def test_catalog_skips_malformed_and_duplicate_entries(world):
    set_items([{"name": "a", "min": 1, "max": 2}, {"name": "A"}, {"name": ""},
               "junk", {"name": "b", "min": "x"}, {"name": "c"}])
    assert [n for n, _, _ in fenra.item_catalog(W)] == ["a", "c"]


def test_saving_world_controls_keeps_the_items_list(world):
    set_items([{"name": "zib", "min": 1, "max": 2}])
    state = fenra.load_world_state(W)
    state.update({"interval": "5"})          # what _save_world_controls now does
    fenra.save_world_state(W, state)
    assert [n for n, _, _ in fenra.item_catalog(W)] == ["zib"]


# ---- text helpers -----------------------------------------------------------

def test_inventory_text_is_alphabetical_and_hides_zeroes():
    assert fenra.inventory_text({"b": 2, "A": 1, "z": 0}) == "A: 1, b: 2"
    assert fenra.inventory_text({}) == "empty"


def test_parse_inventory_text_round_trip_and_errors():
    assert fenra.parse_inventory_text("zib=3, quorp = 12,") == {"zib": 3, "quorp": 12}
    assert fenra.parse_inventory_text("zib=0") == {}
    for bad in ("zib", "zib=x", "zib=-1", "=3", "zib=1.5"):
        with pytest.raises(ValueError):
            fenra.parse_inventory_text(bad)


# ---- HUD --------------------------------------------------------------------

def test_hud_shows_only_the_voices_own_inventory(world):
    make_voice("Ash", {"zib": 3})
    make_voice("Cove", {"quorp": 9})
    hud = fenra.build_hud(W, "Ash")
    assert "Inventory: zib: 3" in hud
    assert "quorp" not in hud and "Cove (" not in hud
    assert "urrenc" not in hud


def test_hud_with_nothing_owned_says_empty(world):
    make_voice("Ash", {})
    assert "Inventory: empty" in fenra.build_hud(W, "Ash")


# ---- give_item --------------------------------------------------------------

def test_registry_has_give_item_and_no_give_currency():
    assert "give_item" in fenra.FUNCTION_REGISTRY
    assert "give_currency" not in fenra.FUNCTION_REGISTRY
    assert fenra.FUNCTION_REGISTRY["give_item"]["params"] == "target|item|amount"
    assert "give_item" in fenra.URGE_FUNCTIONS


def test_give_item_moves_items_and_is_case_insensitive(world):
    make_voice("Ash", {"Zib": 5})
    make_voice("Cove", {})
    out = fenra.fn_give_item(W, "Ash", "Cove|zib|2")
    assert out == "sent 2 Zib to Cove"
    assert inv("Ash") == {"Zib": 3} and inv("Cove") == {"Zib": 2}
    fenra.fn_give_item(W, "Ash", "Cove|ZIB|1")
    assert inv("Cove") == {"Zib": 3}          # merged into the existing key


def test_giving_everything_removes_the_item(world):
    make_voice("Ash", {"zib": 2})
    make_voice("Cove", {})
    fenra.fn_give_item(W, "Ash", "Cove|zib|2")
    assert inv("Ash") == {} and inv("Cove") == {"zib": 2}


@pytest.mark.parametrize("args,message", [
    ("Cove|zib|9", "you only have 5 zib, can't send 9"),
    ("Cove|quorp|1", "you don't have any quorp"),
    ("Ash|zib|1", "can't give_item to yourself"),
    ("Nobody|zib|1", "isn't a voice in this world"),
    ("Cove|zib|0", "amount must be positive"),
    ("Cove|zib|-2", "amount must be positive"),
    ("Cove|zib|1.5", "amount must be a whole number"),
    ("Cove|zib|lots", "isn't a number"),
    ("Cove|zib", "expected 'target|item|amount'"),
])
def test_give_item_rejections_change_nothing(world, args, message):
    make_voice("Ash", {"zib": 5})
    make_voice("Cove", {})
    with pytest.raises(ValueError, match=message):
        fenra.fn_give_item(W, "Ash", args)
    assert inv("Ash") == {"zib": 5} and inv("Cove") == {}


def test_dispatch_and_self_error_text_for_give_item(world):
    make_voice("Ash", {"zib": 1})
    make_voice("Cove", {})
    args = {"target": "Cove", "item": "quorp", "amount": "1"}
    outcome, detail = fenra.dispatch_one_function_call(W, "Ash", "give_item", args)
    assert outcome == "error" and "don't have any" in detail
    assert fenra._self_error_text("give_item", args, detail) == "You reach for quorp, but you don't have any."
    args = {"target": "Cove", "item": "zib", "amount": "5"}
    outcome, detail = fenra.dispatch_one_function_call(W, "Ash", "give_item", args)
    assert "come up short" in fenra._self_error_text("give_item", args, detail)


def test_concurrent_gives_conserve_the_total(world):
    make_voice("Ash", {"zib": 200})
    make_voice("Cove", {"zib": 200})

    def work(i):
        giver, taker = ("Ash", "Cove") if i % 2 == 0 else ("Cove", "Ash")
        for _ in range(20):
            fenra.dispatch_one_function_call(
                W, giver, "give_item", {"target": taker, "item": "zib", "amount": "1"})

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not any(t.is_alive() for t in threads)
    assert sum(inv("Ash").values()) + sum(inv("Cove").values()) == 400


# ---- old worlds (deliberately not upgraded) ----------------------------------

def test_old_currency_worlds_load_without_upgrade_and_are_flagged(world):
    path = fenra.voice_state_path(W, "Ash")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": "m", "currencies": {"Air": 3.0, "Water": 9.0}}, f)
    assert fenra.old_currency_format_voices(W) == ["Ash"]
    st = fenra.load_voice_state(W, "Ash")
    assert st["inventory"] == {}                        # nothing invented
    fenra.save_voice_state(W, "Ash", st)
    with open(path, encoding="utf-8") as f:
        assert json.load(f)["currencies"] == {"Air": 3.0, "Water": 9.0}   # rides along
    make_voice("Cove", {"zib": 1})
    assert fenra.old_currency_format_voices(W) == []   # Ash was re-saved, so now has an inventory field


# ---- v0.21.1: no timestamps in the model-facing history ----------------------

def test_model_history_has_no_timestamps_but_the_gui_render_keeps_them():
    thoughts = [
        {"id": 1, "timestamp": "2026-09-19T03:05:47", "speaker": "Ash", "text": "first"},
        {"id": 2, "timestamp": "2026-09-19T03:15:47", "speaker": "Ash", "text": "second"},
    ]
    assert fenra.render_thoughts(thoughts) == "Ash: first\nAsh: second"
    assert "2026-09-19T03:05:47" in fenra.render_thoughts_for_display(thoughts)
    assert thoughts[0]["timestamp"] == "2026-09-19T03:05:47"      # stored field untouched
