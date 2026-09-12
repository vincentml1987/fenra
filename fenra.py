"""fenra.py - worlds-rebuild branch, genuinely blank rebuild.

Teddy's call (2026-09-08): back up from the fenra.py architecture on
fenras-aletheosis entirely and rebuild from a simpler foundation -
voices and groups as the only concepts, nothing else yet. Goal for
this first pass, his own words: "let the internal thoughts work" -
prove the core loop (a voice thinks, its thought reaches whoever
should see it) before anything gets layered on top. No functions, no
permissions, no Hearth, no Topology - those come later, deliberately
not here yet.

THE MODEL, exactly as specified:

- World (renamed from "session") - a fully separate container. Worlds
  share nothing with each other - no cross-world storage of any kind.
  Lives at worlds/<world>/.
- Voice - model, identity, messages, currencies. `behavior` existed in the
  first pass and is gone (2026-09-09) - it was the same boilerplate for
  every voice, and the HUD below (ending in identity) replaces what it
  was doing. `currencies` (2026-09-09, single `currency` field; replaced
  2026-09-12 with four independent elemental balances - Air, Earth,
  Fire, Water, see CURRENCY_ELEMENTS/CURRENCY_RANGES below) is real
  balance any voice can move via `give_currency` - genuinely
  exploratory, no plan for it beyond seeing what they do with it once
  they can see it and move it. The single-dollar version got dropped
  because the `$` sign itself imported a real-world frame of reference
  ("rich"/"poor") that Fenra never defined any meaning for (Teddy's
  read, prompted by a voice describing itself as poor while actually
  holding the town's largest balance) - four un-ranked, unexplained
  currencies with no stated exchange rate, not even one Teddy or Qualia
  privately know, are meant to remove that borrowed frame entirely and
  let any value they end up having emerge from how they're actually
  used. `messages` (2026-09-10 - replaces the
  original flat `context` string) is a real list of structured entries
  (`{id, timestamp, speaker, text}`, stable `id`), fully editable by
  Teddy at any time down to one specific message ("even context," his
  words, extended into an actual data structure instead of a text blob)
  - not a fixed-size window, not a separate history file. It grows by
  `append_message`: every time a voice thinks, and every time a fellow
  group member's thought lands, one entry gets appended - the SAME
  shape whether it's the voice's own thought or an incoming one, no
  special case for self vs. other. `render_messages` flattens the list
  back into the exact `"[timestamp] name: text"` text the model has
  always received - the storage/GUI changed, what Ollama sees given the
  same content did not.
- Group - name, members (a list of voice names), and `board` (2026-09-10
  - see FUNCTIONS below). No owner, no join_policy, no visibility, no
  direction on membership itself - manually managed only, entirely from
  the GUI.

THE LOOP: one voice per tick, simple round-robin across the world's
voice list. Build the prompt as context + HUD (`build_hud()`), call
Ollama, get the raw response, run it through `run_function_calls()`
(2026-09-09 - functions are back, no permission layer this round, every
voice can call everything). That returns two versions of the response:
the real one (`full_text`), with any `⟦function_name(args)⟧` calls
resolved into `⟦RESULT: ...⟧` text folded in - that's what the speaker's
own context gets - and a masked one (`masked_text`), where every call
becomes that function's own flavored action mask (2026-09-10 - see
FUNCTION_REGISTRY's `"mask"` key, `_mask_for_call`) with no arguments and
no result ever shown - that's what every OTHER member of every group the
speaker belongs to gets instead (deduped across overlapping groups). A
voice can always see what it did; bystanders only see roughly what kind
of thing happened, never the details.

THE HUD: the last thing in every prompt, computed fresh every tick and
never persisted to context (Teddy's call, 2026-09-09 - it reflects live
world state and shouldn't compound the same context-bloat problem a
silently-timing-out voice can already produce). Tells a voice its own
name/model, its own groups, every group that exists in the world, who
it can currently see (shares a group with), who exists but isn't
visible to it, *everyone's* currency balances (all four elements -
2026-09-12) - not just its own (2026-09-10, Teddy's call: full
transparency, deliberately with no goal attached, after watching
give_currency turn into rote/formulaic use - see whether visibility on
its own changes anything) - and how to call/
discover functions (hard-coded, same reasoning as the old branch's
bootstrap notice - the calling convention is mechanics, not content, so
it isn't optional) - ending with its own identity line as the literal
last line of the entire prompt. `hud_fields()` (2026-09-10) returns the
same pieces as plain data, not text - `build_hud` formats them, and the
GUI's read-only HUD summary (Voices tab) calls the identical function,
so the two can never drift apart.

THE GUI is object-oriented (2026-09-10): select a voice or group in its
list, its properties appear underneath - nothing duplicated across
tabs. Group membership and a group's board are properties of the
*group*, edited only from the Groups tab (a Board panel there, matching
the Messages panel below) - never from Voices, even though a voice's
HUD summary displays derived facts about both (which groups it's in,
who it can/can't see, board activity) - those stay read-only there on
purpose, since editing them has no sensible meaning outside the group
that actually owns them. A voice's Messages panel (Voices tab) is a
real multi-column list (id/timestamp/speaker/text) - select a row to
edit or delete that one message, or add a new one - not a single text
blob. The Currency tab is gone (2026-09-10) - redundant once currency
became a real per-voice field on the Voices tab itself.

GROUP CHAT (Groups tab, 2026-09-10): a read-only view of everything
actually said in a group, reconstructed by `group_chat_transcript()`
from each member's own self-tagged records (see `append_message`'s
`groups` param) rather than from anyone's inbox - shows the real full
text a voice said, not the masked version bystanders receive. Only a
voice's own thought carries a `groups` tag (every group it broadcast to
that turn, as a list - it can belong to more than one at once); a
delivered/masked copy in a recipient's own history never does, so
there's no ambiguity about which group a delivery "belongs to" even
when speaker and recipient share more than one group.

FUNCTIONS: reintroduced 2026-09-09, using the old branch's exact
`⟦function_name(args)⟧` call syntax (U+27E6/U+27E7 - essentially never
appears by accident) and `FUNCTION_REGISTRY` shape, but rebuilt lean -
no permission layer (every voice can call everything), no
`functions.jsonl` logging, no fabrication-detection. `send_message`
delivers straight into the target's real `context` via the existing
`append_to_context`, wrapped in an explicit flag so it reads as a
message rather than ordinary group chatter - not a separate, transitory
mechanism. `give_currency` moves real balance, in one of the four
elemental currencies, between two voices' `currencies` fields
(2026-09-12 - see the Voice bullet above). `functions()` lists what's
callable.

BOARDS (2026-09-10): a group's `board` is a list of posts
(`{id, subject, text, author, timestamp, seen}`, `seen` a
`{voice: "skimmed"|"read"}` map - absence means unread), gated only by
group membership, no ownership checks. Built after watching voices
repeatedly invent fictional functions for the same underlying want - a
way to deliberately notify/post to a specific group as a real action,
not just by talking, which already broadcasts automatically. Group chat
is push (lands in context whether you looked or not); a board is pull
(exists whether or not you check it). `post_board` adds a post,
`skim_board` lists subject + first/last-sentence summaries and marks
posts "skimmed", `read_board` returns one post's full text and marks it
"read" (never downgrades a "read" post back to "skimmed"), `delete_board`
removes a post outright - genuinely anyone in the group, not just the
original author. The HUD reports per-group unread/skimmed counts for a
voice's own groups only, never content.
"""

import json
import math
import os
import random
import re
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, scrolledtext, simpledialog, ttk

import requests

FENRA_VERSION = "0.3.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORLDS_DIR = os.path.join(BASE_DIR, "worlds")

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3"
DEFAULT_INTERVAL_SEC = 3
DEFAULT_WORLD_NAME = "default"
DEFAULT_VOICE_NAME = "voice1"

# Four independent elemental currencies (2026-09-12, replacing the old
# single dollar-denominated `currency` field - see the module docstring's
# Voice bullet for why). Alphabetical order is the one and only display
# order everywhere (HUD text, GUI, site export) - a fixed but genuinely
# neutral choice, since ordering by any other rule (e.g. by amount) would
# itself imply one currency matters more than another, which nothing in
# this design is allowed to assert. CURRENCY_RANGES are deliberately
# different spreads per element, not just different means, so real
# scarcity differences show up in how much of each actually exists in the
# world - no exchange rate is defined anywhere, by Teddy, by Qualia, or in
# this code, on purpose.
CURRENCY_ELEMENTS = ("Air", "Earth", "Fire", "Water")
CURRENCY_RANGES = {
    "Fire": (1, 6),
    "Air": (3, 10),
    "Water": (8, 20),
    "Earth": (15, 35),
}

WORLD_STATE_FILENAME = "world.json"
VOICE_STATE_FILENAME = "state.json"
START_SIGNAL_FILENAME = "start_signal.txt"
STOP_SIGNAL_FILENAME = "stop_signal.txt"

_GROUP_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


# --------------------------------------------------------------- storage --

def sanitize_name(name):
    return (name or "").strip().lower().replace(" ", "_").replace("'", "")


def world_dir(world_name):
    return os.path.join(WORLDS_DIR, world_name)


def ensure_world_dir(world_name):
    path = world_dir(world_name)
    os.makedirs(path, exist_ok=True)
    return path


def list_worlds():
    if not os.path.isdir(WORLDS_DIR):
        return []
    names = [d for d in os.listdir(WORLDS_DIR) if os.path.isdir(os.path.join(WORLDS_DIR, d))]

    def sort_key(name):
        path = os.path.join(world_dir(name), WORLD_STATE_FILENAME)
        return os.path.getmtime(path) if os.path.exists(path) else 0

    return sorted(names, key=sort_key, reverse=True)


def default_world_state():
    return {
        "host": DEFAULT_HOST,
        "interval": DEFAULT_INTERVAL_SEC,
        "model_default": DEFAULT_MODEL,
        "voices": [],
        "voice_rotation_index": 0,
        "urge_model": DEFAULT_URGE_MODEL,
        "num_predict": 1500,
        "urge_num_predict": 250,
        "repeat_penalty": 1.3,
    }


def world_state_path(world_name):
    return os.path.join(world_dir(world_name), WORLD_STATE_FILENAME)


def load_world_state(world_name):
    state = default_world_state()
    path = world_state_path(world_name)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                state.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return state


def save_world_state(world_name, state):
    state = dict(state)
    state["fenra_version"] = FENRA_VERSION
    ensure_world_dir(world_name)
    with open(world_state_path(world_name), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ------------------------------------------------------------------ voices --

def voices_root_dir(world_name):
    return os.path.join(world_dir(world_name), "voices")


def voice_dir(world_name, voice_name):
    return os.path.join(voices_root_dir(world_name), voice_name)


def ensure_voice_dir(world_name, voice_name):
    path = voice_dir(world_name, voice_name)
    os.makedirs(path, exist_ok=True)
    return path


def voice_state_path(world_name, voice_name):
    return os.path.join(voice_dir(world_name, voice_name), VOICE_STATE_FILENAME)


def list_voices(world_name):
    root = voices_root_dir(world_name)
    if not os.path.isdir(root):
        return []
    return sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and os.path.exists(voice_state_path(world_name, d))
    )


def random_starting_currencies():
    """One fresh random draw per element from CURRENCY_RANGES - used both
    for a brand-new voice's starting balances (default_voice_state) and
    for the one-time 2026-09-12 migration of pre-existing voices, so
    every voice gets the same treatment regardless of when it was
    created. Rounded to whole numbers - fractional elemental currency
    reads as false precision on values nobody's defined any real
    granularity for."""
    return {name: float(random.randint(*bounds)) for name, bounds in CURRENCY_RANGES.items()}


def default_voice_state():
    return {
        "model": DEFAULT_MODEL,
        "identity": "",
        "messages": [],
        "currencies": random_starting_currencies(),
        "urge": {name: 0.0 for name in URGE_FUNCTIONS},
        "understand_urge": {name: 0.0 for name in URGE_FUNCTIONS},
        "understand_urge_general": 0.0,
    }


def load_voice_state(world_name, voice_name):
    state = default_voice_state()
    path = voice_state_path(world_name, voice_name)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                state.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return state


def save_voice_state(world_name, voice_name, state):
    state = dict(state)
    state["fenra_version"] = FENRA_VERSION
    ensure_voice_dir(world_name, voice_name)
    with open(voice_state_path(world_name, voice_name), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def delete_voice(world_name, voice_name):
    import shutil
    path = voice_dir(world_name, voice_name)
    if os.path.isdir(path):
        shutil.rmtree(path)


def append_message(world_name, voice_name, speaker, text, timestamp=None, groups=None):
    """The one and only way a voice's message history grows - appends a
    real structured entry ({id, timestamp, speaker, text, groups}), same
    shape whether it's the voice's own thought or an incoming one from a
    fellow group member (see the module docstring). Re-reads from disk
    immediately before appending rather than trusting an in-memory copy,
    so a concurrent write (Teddy editing a message in the GUI at the
    same moment) can't get silently clobbered. `id` is stable
    (max existing + 1) - the same pattern board posts already use.

    `groups` (2026-09-10, for the Groups-tab chat view): only meaningful
    on a voice's own self-record of its own thought - every group it
    broadcast that thought to, as a list (a voice can belong to more
    than one group at once, so this can't be a single value). Delivered
    copies in a *recipient's* history don't carry this - the merged
    group-chat view is reconstructed entirely from each member's own
    tagged self-records, not from anyone's inbox, so there's no
    ambiguity about which group a delivery "belongs to" even when a
    speaker shares more than one group with the same recipient."""
    state = load_voice_state(world_name, voice_name)
    messages = state.get("messages", [])
    next_id = max((m["id"] for m in messages), default=0) + 1
    messages.append({
        "id": next_id,
        "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
        "speaker": speaker,
        "text": text,
        "groups": groups or [],
    })
    state["messages"] = messages
    save_voice_state(world_name, voice_name, state)


def group_chat_transcript(world_name, group_name):
    """The group's whole merged chat (Groups tab) - reconstructed from
    each member's own self-tagged records (see append_message's
    `groups` docstring), not from any recipient's masked inbox copies,
    so it shows the real full text, not what bystanders see. Sorted by
    timestamp across members."""
    gstate = load_group_state(world_name, group_name)
    if not gstate:
        return []
    entries = []
    for member in gstate.get("members", []):
        state = load_voice_state(world_name, member)
        for m in state.get("messages", []):
            if m.get("speaker") == member and group_name in m.get("groups", []):
                entries.append(m)
    entries.sort(key=lambda m: m["timestamp"])
    return entries


def render_messages(messages):
    """Flattens a voice's structured message list back into the exact
    text the model has always received - "[timestamp] speaker: text"
    per line, newline-joined. Storage/GUI changed (2026-09-10); what
    Ollama sees given the same content did not."""
    return "\n".join(f"[{m['timestamp']}] {m['speaker']}: {m['text']}" for m in messages)


# ------------------------------------------------------------------ groups --

def groups_root_dir(world_name):
    return os.path.join(world_dir(world_name), "groups")


def ensure_groups_root_dir(world_name):
    path = groups_root_dir(world_name)
    os.makedirs(path, exist_ok=True)
    return path


def group_path(world_name, name):
    name = sanitize_name(name)
    if not name or not _GROUP_NAME_RE.match(name):
        raise ValueError(
            "group names may only contain letters, numbers, underscores, and hyphens "
            f"(spaces and apostrophes get stripped automatically) - got '{name}'"
        )
    return os.path.join(groups_root_dir(world_name), f"{name}.json")


def list_groups(world_name):
    root = groups_root_dir(world_name)
    if not os.path.isdir(root):
        return []
    return sorted(
        f[: -len(".json")] for f in os.listdir(root)
        if f.endswith(".json") and os.path.isfile(os.path.join(root, f))
    )


def default_group_state(name):
    return {"name": name, "members": [], "board": []}


def load_group_state(world_name, name):
    path = group_path(world_name, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_group_state(world_name, name, state):
    ensure_groups_root_dir(world_name)
    with open(group_path(world_name, name), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def delete_group(world_name, name):
    path = group_path(world_name, name)
    if os.path.exists(path):
        os.remove(path)


def groups_containing(world_name, voice_name):
    """Every group (sanitized name) that currently lists voice_name as
    a member - used both by the tick loop (who does a thought reach)
    and the Groups tab (member add/remove)."""
    out = []
    for name in list_groups(world_name):
        state = load_group_state(world_name, name)
        if state and voice_name in state.get("members", []):
            out.append(name)
    return out


def hud_fields(world_name, voice_name):
    """Every piece build_hud's text is made of, as plain data - the GUI
    (Voices tab HUD summary) calls this directly instead of parsing
    build_hud's string, so the two can never drift out of sync (Teddy's
    call, 2026-09-10)."""
    state = load_voice_state(world_name, voice_name)
    own_groups = groups_containing(world_name, voice_name)
    all_groups = list_groups(world_name)

    seen = set()
    for gname in own_groups:
        gstate = load_group_state(world_name, gname) or {}
        seen.update(gstate.get("members", []))
    seen.discard(voice_name)

    unseen = [v for v in list_voices(world_name) if v != voice_name and v not in seen]

    # Board unread/skimmed counts, own groups only - same privacy
    # boundary as "Voices you can see": a voice shouldn't know about
    # board activity in a group it isn't in.
    board_counts = []
    for gname in own_groups:
        gstate = load_group_state(world_name, gname) or {}
        board = gstate.get("board", [])
        unread = sum(1 for p in board if voice_name not in p.get("seen", {}))
        skimmed = sum(1 for p in board if p.get("seen", {}).get(voice_name) == "skimmed")
        board_counts.append(f"{gname}: {unread} unread, {skimmed} skimmed")

    # Everyone's balances, not just your own (Teddy's call, 2026-09-10) -
    # full transparency rather than a private number, deliberately with
    # no goal attached. Sorted alphabetically by voice name (2026-09-12) -
    # NOT by amount anymore, now that there are four independent
    # currencies with no defined exchange rate: ranking by any one of
    # them would itself assert that element matters more than the
    # others, which nothing in this design is allowed to do.
    balances = []
    for v in list_voices(world_name):
        v_state = state if v == voice_name else load_voice_state(world_name, v)
        v_currencies = v_state.get("currencies", {})
        balances.append((v, {el: v_currencies.get(el, 0.0) for el in CURRENCY_ELEMENTS}))
    balances.sort(key=lambda pair: pair[0])

    return {
        "model": state.get("model", DEFAULT_MODEL),
        "own_groups": own_groups,
        "all_groups": all_groups,
        "seen": sorted(seen),
        "unseen": unseen,
        "board_counts": board_counts,
        "balances": balances,
        "identity": state.get("identity", ""),
    }


def build_hud(world_name, voice_name):
    """The last thing in a voice's prompt (see module docstring) -
    computed fresh every tick, never written to state.json. Own
    name/model, own groups, every group in the world, who's currently
    seen (shares a group), who exists but isn't seen, then the voice's
    own identity line as the literal last line."""
    f = hud_fields(world_name, voice_name)
    board_line = "Board activity: " + (", ".join(f["board_counts"]) if f["board_counts"] else "none")
    currency_line = "Currency levels (everyone, four elemental currencies - Air, Earth, "
    currency_line += "Fire, Water - no exchange rate is defined between them): " + ", ".join(
        f"{v} (" + ", ".join(f"{el}: {amts[el]:.1f}" for el in CURRENCY_ELEMENTS) + ")"
        for v, amts in f["balances"]
    )

    lines = [
        "Everything above this line is your thoughts. Everything below is your HUD.",
        f"Name: {voice_name}",
        f"Model: {f['model']}",
        f"Your groups: {', '.join(f['own_groups']) if f['own_groups'] else 'none'}",
        f"All groups in this world: {', '.join(f['all_groups']) if f['all_groups'] else 'none'}",
        f"Voices you can see: {', '.join(f['seen']) if f['seen'] else 'none'}",
        f"Voices that exist but you cannot see: {', '.join(f['unseen']) if f['unseen'] else 'none'}",
        board_line,
        currency_line,
        "You can call functions by writing ⟦function_name(args)⟧ in your "
        "response - try ⟦functions()⟧ to see everything available to you.",
        f["identity"],
    ]
    return "\n".join(lines)


# --------------------------------------------------------------- functions --

FUNCTION_CALL_RE = re.compile(r"⟦\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s*⟧", re.DOTALL)


def _parse_target_and_rest(args_text):
    """Every two-part function takes "target|rest" - split once on the
    first '|', both sides stripped. Raises if the '|' is missing."""
    if "|" not in args_text:
        raise ValueError("expected 'target|...' - got no '|' separator")
    target, rest = args_text.split("|", 1)
    return target.strip(), rest.strip()


def _first_pipe_arg(args_text):
    """The first '|'-delimited argument, or the whole trimmed string if
    there's no '|' at all - used only to fill the {arg0} placeholder in
    an action mask (see FUNCTION_REGISTRY/run_function_calls), so it
    covers two-part calls (send_message's target) and single-arg calls
    (skim_board's group) the same way."""
    if not args_text:
        return ""
    return args_text.split("|", 1)[0].strip()


def _require_group_member(world_name, group, caller_name):
    """Every board function starts here: resolves the (possibly
    unsanitized) group name, loads its state, and raises unless the
    caller is actually a member - the only gating boards have at all
    (Teddy's call: no ownership checks beyond that). Returns
    (sanitized_group_name, group_state) so the caller can mutate
    group_state["board"] and save it back."""
    group = sanitize_name(group)
    state = load_group_state(world_name, group)
    if not state or caller_name not in state.get("members", []):
        raise ValueError(f"'{group}' isn't a group you're in")
    return group, state


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _first_and_last_sentence(text):
    """A naive skim-summary for a board post - not linguistically
    careful, just enough of a taste to decide whether to read_board the
    full thing."""
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s]
    if not sentences:
        return ""
    if len(sentences) == 1:
        return sentences[0]
    return f"{sentences[0]} [...] {sentences[-1]}"


def fn_send_message(world_name, caller_name, args_text):
    """A direct message to one specific voice - the Slack-DM equivalent.
    Delivered straight into the target's real, persisted messages via
    the same append_message every group broadcast already uses, just
    addressed to one voice and wrapped in an explicit flag so the text
    itself reads as a message rather than ordinary group chatter
    (Teddy's call, 2026-09-09 - not a separate transitory mechanism)."""
    target, text = _parse_target_and_rest(args_text)
    if target not in list_voices(world_name):
        raise ValueError(f"'{target}' isn't a voice in this world")
    if target == caller_name:
        raise ValueError("you can't send_message yourself")
    if not text:
        raise ValueError("no message text given")
    timestamp = datetime.now().isoformat(timespec="seconds")
    wrapped = (
        f"***You received the following message from {caller_name} at "
        f"{timestamp}*** {text} ***End Message from {caller_name}***"
    )
    append_message(world_name, target, caller_name, wrapped, timestamp)
    return f"message sent to {target}"


def fn_give_currency(world_name, caller_name, args_text):
    """Real transfer between two voices' own stored balance, in one of
    the four elemental currencies (2026-09-12 - see CURRENCY_ELEMENTS/
    the module docstring's Voice bullet for why there are four rather
    than one dollar-denominated balance)."""
    parts = args_text.split("|", 2)
    if len(parts) != 3:
        raise ValueError("expected 'target|element|amount'")
    target, element_text, amount_text = (p.strip() for p in parts)
    if target not in list_voices(world_name):
        raise ValueError(f"'{target}' isn't a voice in this world")
    if target == caller_name:
        raise ValueError("you can't give_currency to yourself")

    element = next((e for e in CURRENCY_ELEMENTS if e.lower() == element_text.lower()), None)
    if element is None:
        raise ValueError(
            f"'{element_text}' isn't a real currency - it's one of {', '.join(CURRENCY_ELEMENTS)}"
        )

    # Thousands-separator commas stripped so "1,000" still works; no `$`
    # to strip anymore now that these aren't dollars.
    cleaned = amount_text.strip().replace(",", "")
    try:
        amount = float(cleaned)
    except ValueError:
        raise ValueError(f"'{amount_text}' isn't a number")
    if amount <= 0:
        raise ValueError("amount must be positive")

    caller_state = load_voice_state(world_name, caller_name)
    balance = caller_state.get("currencies", {}).get(element, 0.0)
    if amount > balance:
        raise ValueError(f"you only have {balance:.1f} {element}, can't send {amount:.1f}")
    caller_state.setdefault("currencies", {})[element] = balance - amount
    save_voice_state(world_name, caller_name, caller_state)

    target_state = load_voice_state(world_name, target)
    target_currencies = target_state.setdefault("currencies", {})
    target_currencies[element] = target_currencies.get(element, 0.0) + amount
    save_voice_state(world_name, target, target_state)

    return f"sent {amount:.1f} {element} to {target}"


def fn_post_board(world_name, caller_name, args_text):
    """Post a new message to a group's board - a real, pull-based
    artifact external to the group's normal push-everything-into-
    context chat. Needs three pieces, not two - the only function so
    far that does."""
    parts = args_text.split("|", 2)
    if len(parts) != 3:
        raise ValueError("expected 'group|subject|text'")
    group_raw, subject, text = (p.strip() for p in parts)
    group, gstate = _require_group_member(world_name, group_raw, caller_name)
    if not subject or not text:
        raise ValueError("subject and text can't be empty")
    board = gstate.get("board", [])
    next_id = max((p["id"] for p in board), default=0) + 1
    board.append({
        "id": next_id,
        "subject": subject,
        "text": text,
        "author": caller_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seen": {},
    })
    gstate["board"] = board
    save_group_state(world_name, group, gstate)
    return f"posted to {group} board (id {next_id})"


def fn_skim_board(world_name, caller_name, args_text):
    """Subject + first/last-sentence summary of every post on a group's
    board - not the full text (see read_board for that). Marks any post
    not already in the caller's seen map as "skimmed"; never downgrades
    an already-"read" post back to "skimmed"."""
    group, gstate = _require_group_member(world_name, args_text, caller_name)
    board = gstate.get("board", [])
    if not board:
        return f"{group} board is empty"
    lines = []
    changed = False
    for post in board:
        seen = post.setdefault("seen", {})
        if caller_name not in seen:
            seen[caller_name] = "skimmed"
            changed = True
        summary = _first_and_last_sentence(post["text"])
        lines.append(f"[{post['id']}] {post['subject']} (by {post['author']}): {summary}")
    if changed:
        gstate["board"] = board
        save_group_state(world_name, group, gstate)
    return "\n".join(lines)


def fn_read_board(world_name, caller_name, args_text):
    """Full text of one specific board post. Marks it "read" for the
    caller, upgrading from any prior state."""
    group_raw, post_id_text = _parse_target_and_rest(args_text)
    group, gstate = _require_group_member(world_name, group_raw, caller_name)
    try:
        post_id = int(post_id_text)
    except ValueError:
        raise ValueError(f"'{post_id_text}' isn't a valid post id")
    board = gstate.get("board", [])
    post = next((p for p in board if p["id"] == post_id), None)
    if not post:
        raise ValueError(f"no post {post_id} on {group} board")
    post.setdefault("seen", {})[caller_name] = "read"
    gstate["board"] = board
    save_group_state(world_name, group, gstate)
    return f"[{post['id']}] {post['subject']} (by {post['author']}, {post['timestamp']}): {post['text']}"


def fn_delete_board(world_name, caller_name, args_text):
    """Delete a post from a group's board. No ownership check - anyone
    in the group can delete anyone's post (Teddy's explicit call)."""
    group_raw, post_id_text = _parse_target_and_rest(args_text)
    group, gstate = _require_group_member(world_name, group_raw, caller_name)
    try:
        post_id = int(post_id_text)
    except ValueError:
        raise ValueError(f"'{post_id_text}' isn't a valid post id")
    board = gstate.get("board", [])
    new_board = [p for p in board if p["id"] != post_id]
    if len(new_board) == len(board):
        raise ValueError(f"no post {post_id} on {group} board")
    gstate["board"] = new_board
    save_group_state(world_name, group, gstate)
    return f"deleted post {post_id} from {group} board"


def fn_functions(world_name, caller_name, args_text):
    """Lists the registry, optionally filtered by a substring in the
    name or description."""
    query = args_text.strip().lower() if args_text else None
    lines = []
    for name, meta in FUNCTION_REGISTRY.items():
        desc = meta["description"]
        if query and query not in name.lower() and query not in desc.lower():
            continue
        lines.append(f"{name}({meta['params']}): {desc}")
    return "\n".join(lines) if lines else "no matching functions"


FUNCTION_REGISTRY = {
    "send_message": {
        "fn": fn_send_message,
        "params": "target|text",
        "description": "Send a direct message to one specific voice - delivered into their context.",
        "mask": "{caller} whispers to {arg0}.",
    },
    "give_currency": {
        "fn": fn_give_currency,
        "params": "target|element|amount",
        "description": "Give some of your own currency, in one of the four elemental currencies (Air, Earth, Fire, Water), to another voice.",
        "mask": "{caller} hands some currency to {arg0}.",
    },
    "functions": {
        "fn": fn_functions,
        "params": "[search term]",
        "description": "List everything you can call, optionally filtered by a search term.",
        "mask": "{caller} considers their actions.",
    },
    "post_board": {
        "fn": fn_post_board,
        "params": "group|subject|text",
        "description": "Post a new message to a group's board (subject + text). You must be a member of the group.",
        "mask": "{caller} posts a message to the {arg0} board.",
    },
    "skim_board": {
        "fn": fn_skim_board,
        "params": "group",
        "description": "See every post currently on a group's board (subject + first/last sentence only). Marks unread posts as skimmed.",
        "mask": "{caller} skims the {arg0} board.",
    },
    "read_board": {
        "fn": fn_read_board,
        "params": "group|post_id",
        "description": "Read one specific post on a group's board in full. Marks it as read.",
        "mask": "{caller} reads a message on the {arg0} board.",
    },
    "delete_board": {
        "fn": fn_delete_board,
        "params": "group|post_id",
        "description": "Delete a post from a group's board. Anyone in the group can delete any post.",
        "mask": "{caller} removes a message from the {arg0} board.",
    },
}


# --------------------------------------------------------------------- urge --
# Round-one urge-system constants (2026-09-11 design, locked with Teddy after
# extensive live model testing - see Qualia/worlds-rebuild-notes.md). Every
# real function except bare `functions()` (pure introspection, not something
# a voice "does") tracks a felt "Urge" that grows the longer it goes unused
# and resets on a successful call - see xleud()/apply_urge_tick() below for
# the actual mechanics.

URGE_FUNCTIONS = tuple(name for name in FUNCTION_REGISTRY if name != "functions")

URGE_DESIRE = 7            # denominator in the XLEUD saturating curve
URGE_DRIVE = 1              # Urge growth per tick a function goes unused
URGE_FLOOR_PCT = 50         # perform-urge must be at/above this XLEUD% to be felt
URGE_TOP_N = 3              # cap on functions handed to the urge agent per turn
UNDERSTAND_URGE_BUMP = 3    # added to understand-urge on a real (non-hallucinated) error
DEFAULT_URGE_MODEL = "phi4-mini"

URGE_VIEWER_CATEGORIES = ("Perform Urges", "Understand Urges", "This Turn")


def _mask_for_call(caller_name, name, args_text):
    """The bystander-facing text for one function call - a flavored,
    per-function action mask (FUNCTION_REGISTRY[name]["mask"]),
    rendered with {caller} and {arg0} (see _first_pipe_arg). An
    unrecognized function name (no registry entry - a hallucinated
    call, no real mask to pull from) falls back to a WoW-nod easter egg
    (Teddy's call, 2026-09-10): "makes some strange gestures" - the
    classic failed-cast flavor text."""
    meta = FUNCTION_REGISTRY.get(name)
    if not meta or "mask" not in meta:
        return f"(*{caller_name} makes some strange gestures.*)"
    try:
        return meta["mask"].format(caller=caller_name, arg0=_first_pipe_arg(args_text))
    except (KeyError, IndexError):
        return f"(*{caller_name} makes some strange gestures.*)"


def run_function_calls(world_name, caller_name, response_text):
    """Scans response_text for every ⟦function_name(args)⟧ call and runs
    each one for real. Returns a (full_text, masked_text, outcomes)
    triple:

    - full_text: response_text with a ⟦RESULT: ...⟧ line appended per
      call - what the caller's own context gets (they made the call,
      they see what it actually did).
    - masked_text: response_text with each call replaced by that
      function's own flavored action mask (see _mask_for_call) - never
      the arguments, never the result - what gets broadcast to everyone
      else in a shared group (Teddy's call, 2026-09-10: calling a
      function shouldn't be any more visible to bystanders than a real
      action is - they can see *that* it happened, roughly what kind of
      thing it was, not the details, unless the caller chooses to say so
      in their own words).
    - outcomes: a list of (name, "ok" | "error" | "unknown") pairs, one
      per call found, in order - captured here rather than re-parsed
      from the RESULT lines later, since this is the one place that
      already knows each call's real outcome first-hand. Feeds the
      urge system (see apply_urge_tick(), 2026-09-11).

    No calls found -> full_text/masked_text are response_text
    unchanged, outcomes is empty."""
    matches = list(FUNCTION_CALL_RE.finditer(response_text))
    if not matches:
        return response_text, response_text, []

    result_lines = []
    outcomes = []
    for match in matches:
        name, args_text = match.group(1), match.group(2)
        meta = FUNCTION_REGISTRY.get(name)
        if not meta:
            result_lines.append(f"⟦RESULT: {name} -> error: unknown function '{name}'⟧")
            outcomes.append((name, "unknown"))
            continue
        try:
            result = meta["fn"](world_name, caller_name, args_text)
            result_lines.append(f"⟦RESULT: {name} -> ok: {result}⟧")
            outcomes.append((name, "ok"))
        except Exception as exc:
            result_lines.append(f"⟦RESULT: {name} -> error: {exc}⟧")
            outcomes.append((name, "error"))

    full_text = response_text + "\n" + "\n".join(result_lines)
    masked_text = FUNCTION_CALL_RE.sub(
        lambda m: _mask_for_call(caller_name, m.group(1), m.group(2)), response_text
    )
    return full_text, masked_text, outcomes


def xleud(urge_value, desire=URGE_DESIRE):
    """The felt-urge percentage (Teddy's coinage, backronym: eXponential
    Level of Euler-damped Urge over Desire) - a saturating curve bounded
    [0, 1), so it decelerates rather than blowing up under heavy neglect
    instead of growing unboundedly the way a raw ratio would. Never
    persisted - always computed fresh from the persisted raw Urge value
    (same "computed, not saved" spirit as build_hud())."""
    return 1 - math.exp(-urge_value / desire)


def apply_urge_tick(world_name, voice_name, outcomes):
    """Called once per tick for the ACTIVE voice, right after
    run_function_calls returns its outcomes list. Updates both urge
    tracks in one load/save round-trip:

    - perform-urge (state["urge"]): a successfully-called function
      resets fully to 0 (Teddy's round-one Satisfaction rule); every
      other real function grows by URGE_DRIVE, whether it errored or
      simply wasn't touched this turn - only success counts as
      "using" it.
    - understand-urge (state["understand_urge"] /
      state["understand_urge_general"]): a real function that errored
      bumps THAT function's own understand-urge specifically (not a
      general one - Teddy's call, 2026-09-11). A call to a name that
      isn't in FUNCTION_REGISTRY at all has no real function to
      attribute the bump to, so it bumps the aggregate
      understand_urge_general slot instead."""
    ok_names = {name for name, outcome in outcomes if outcome == "ok"}
    errored_real_names = {name for name, outcome in outcomes if outcome == "error"}
    unknown_called = any(outcome == "unknown" for _, outcome in outcomes)

    state = load_voice_state(world_name, voice_name)

    urge = state.get("urge", {})
    for name in URGE_FUNCTIONS:
        urge[name] = 0.0 if name in ok_names else urge.get(name, 0.0) + URGE_DRIVE
    state["urge"] = urge

    if errored_real_names or unknown_called:
        understand = state.get("understand_urge", {})
        for name in errored_real_names:
            understand[name] = understand.get(name, 0.0) + UNDERSTAND_URGE_BUMP
        state["understand_urge"] = understand
        if unknown_called:
            state["understand_urge_general"] = (
                state.get("understand_urge_general", 0.0) + UNDERSTAND_URGE_BUMP
            )

    save_voice_state(world_name, voice_name, state)


def compute_urge_snapshot(state):
    """Pure function of a voice's persisted state -> everything the
    per-tick urge decision (and the Urge Viewer GUI panel) needs.
    Never mutates, never touches disk.

    Understand-urge deliberately has no floor before it can "win" -
    an error is a discrete, meaningful event, not gradual disuse, so
    even a single fresh one should be able to override the roleplay
    urge agent immediately if it's the highest signal right now
    (confirmed with Teddy, 2026-09-11). Perform-urge keeps its
    URGE_FLOOR_PCT floor - it isn't felt at all below that."""
    urge = state.get("urge", {})
    understand = state.get("understand_urge", {})
    general = state.get("understand_urge_general", 0.0)

    perform = [(name, xleud(urge.get(name, 0.0))) for name in URGE_FUNCTIONS]
    understand_signals = [(name, xleud(understand.get(name, 0.0))) for name in URGE_FUNCTIONS]
    understand_signals.append((None, xleud(general)))  # None = aggregate/general

    top_understand = max(understand_signals, key=lambda pair: pair[1])
    top_perform = sorted(
        (p for p in perform if p[1] * 100 >= URGE_FLOOR_PCT), key=lambda p: -p[1]
    )[:URGE_TOP_N]

    highest_perform = max((p[1] for p in perform), default=0.0)
    understand_wins = top_understand[1] > 0 and top_understand[1] >= highest_perform

    return {
        "understand_wins": understand_wins,
        "understand_winner": top_understand,  # (name_or_None, xleud)
        "top_perform": top_perform,            # up to URGE_TOP_N, only >= floor
    }


URGE_AGENT_INSTRUCTIONS = """You are a small utility model with no memory between calls. Your only job: given a list of "function urges" (a function name, a brief description of what it does, and an intensity percentage), write ONE short paragraph — 2 to 4 sentences — describing what it feels like to carry these urges right now.

The percentage is how strongly this urge is currently felt — the longer a function has gone unused, the higher it climbs, and the harder it becomes to ignore.

Write it in second person ("You feel..."), as an embodied, organic sensation — not a command, not a to-do list, not an instruction to act. Never state the raw percentage number in your output.

You may ONLY name the functions listed below — never invent, imply, or reference any function not listed. You MUST name every one of them, using its exact name (e.g. "post_board", not "post to the board"). If only one function is listed, describe only that one function — do not invent or imply any others.

Do not greet, explain what you're doing, or add anything besides the paragraph itself.

Urge values follow:
"""


def build_urge_agent_prompt(top_perform):
    """The stateless, one-shot prompt sent to the urge agent - fixed
    instructions plus the top_perform functions' real registry
    description (reused as-is, no separate urge-specific description
    field needed) and current XLEUD%. Confirmed via a 21-case stress
    battery against phi4-mini/gemma3:4b/llama3.2:3b, 2026-09-11."""
    lines = [
        f"- {name} ({FUNCTION_REGISTRY[name]['description']}): {round(pct * 100)}%"
        for name, pct in top_perform
    ]
    return URGE_AGENT_INSTRUCTIONS + "\n".join(lines)


def build_call_syntax_reminders(names):
    """Deterministic, never-LLM-generated reminder of each named
    function's real call syntax, sourced straight from
    FUNCTION_REGISTRY - appended after the urge agent's own output so
    a felt urge is never left without a guaranteed-correct path to
    actually resolve it (the same failure mode as the Wren
    board-hallucination finding: naming a function isn't the same as
    knowing how to call it)."""
    return "\n".join(
        f"{name} -> ⟦{name}({FUNCTION_REGISTRY[name]['params']})⟧" for name in names
    )


# ------------------------------------------------------------------ model --

def call_ollama(host, model, prompt, options=None):
    # No fixed timeout, matching fenras-aletheosis's own REQUEST_TIMEOUT=None
    # (its comment applies here too, unchanged): some models are legitimately
    # slow, and a client-side timeout doesn't cancel server-side generation -
    # it just abandons the connection while the server keeps working anyway,
    # which can pile up rather than help.
    #
    # `options` (2026-09-11) - e.g. {"num_predict": ..., "repeat_penalty": ...}
    # - a generation-limiting guard, motivated by the Phi-3 runaway-generation
    # and Orin verbatim-repetition findings earlier this branch. Deliberately
    # left config-agnostic here (a plain dict, no `self.*` access) - callers
    # build the options from world/voice-specific settings; this function
    # doesn't know or care where they came from.
    resp = requests.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": options or {}},
        timeout=None,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def list_ollama_models(host):
    try:
        resp = requests.get(f"{host}/api/tags", timeout=10)
        resp.raise_for_status()
        return sorted(m["name"] for m in resp.json().get("models", []))
    except requests.RequestException:
        return []


# ---------------------------------------------------------------- the app --

class FenraApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Fenra - worlds-rebuild v{FENRA_VERSION}")
        self.root.geometry("1000x650")

        self.world_name = None
        self.world_voices = []          # this world's round-robin order
        self.voice_rotation_index = 0
        self.running = False
        self.loop_thread = None

        self.host_var = tk.StringVar(value=DEFAULT_HOST)
        self.interval_var = tk.StringVar(value=str(DEFAULT_INTERVAL_SEC))
        self.urge_model_var = tk.StringVar(value=DEFAULT_URGE_MODEL)
        self.num_predict_var = tk.StringVar(value="1500")
        self.urge_num_predict_var = tk.StringVar(value="250")
        self.repeat_penalty_var = tk.StringVar(value="1.3")
        self.status_var = tk.StringVar(value="Idle")
        self.world_var = tk.StringVar(value="")

        self._current_voice_names = []   # listbox-index -> voice name
        self.displayed_voice = None
        self._current_messages = []      # this voice's messages, as loaded
        self.selected_message_id = None
        self._current_group_names = []   # listbox-index -> group name
        self.displayed_group = None
        self._current_board = []         # this group's board, as loaded
        self.selected_post_id = None

        self._build_menu()
        self._build_ui()
        self._startup_world()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------ startup --

    def _startup_world(self):
        worlds = list_worlds()
        name = worlds[0] if worlds else DEFAULT_WORLD_NAME
        self._load_world(name)

    def _on_close(self):
        self.running = False
        if self.world_name:
            self._save_world_controls()
        self.root.destroy()

    # ---------------------------------------------------------------- UI --

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        worlds_menu = tk.Menu(file_menu, tearoff=0)
        worlds_menu.add_command(label="New world...", command=self.new_world)
        worlds_menu.add_command(label="Rename current world...", command=self.rename_world)
        worlds_menu.add_separator()
        self._worlds_menu = worlds_menu
        self._rebuild_worlds_menu()
        file_menu.add_cascade(label="Worlds", menu=worlds_menu)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def _rebuild_worlds_menu(self):
        # Keep "New world.../Rename.../separator" (first 3 entries), drop
        # everything after, re-list every world fresh.
        self._worlds_menu.delete(3, "end")
        for name in list_worlds():
            self._worlds_menu.add_command(label=name, command=lambda n=name: self._load_world(n))

    def _build_ui(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill="x", padx=6, pady=4)
        ttk.Label(toolbar, text="World:").pack(side="left")
        ttk.Label(toolbar, textvariable=self.world_var, font=("Segoe UI", 9, "bold")).pack(side="left", padx=(2, 14))
        ttk.Label(toolbar, text="Host:").pack(side="left")
        ttk.Entry(toolbar, textvariable=self.host_var, width=24).pack(side="left", padx=(2, 10))
        ttk.Label(toolbar, text="Interval (s):").pack(side="left")
        ttk.Entry(toolbar, textvariable=self.interval_var, width=5).pack(side="left", padx=(2, 10))
        self.start_stop_btn = ttk.Button(toolbar, text="Start", command=self.toggle_loop)
        self.start_stop_btn.pack(side="left", padx=(0, 10))
        ttk.Label(toolbar, textvariable=self.status_var, foreground="#666").pack(side="left")

        # Second row (2026-09-11) - urge-agent model + generation-limiting
        # guard, all world.json-backed, same StringVar+Entry pattern as
        # Host/Interval above. Own row rather than crowding the first -
        # nine label/entry pairs plus button/status is a lot for one line.
        toolbar2 = ttk.Frame(self.root)
        toolbar2.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(toolbar2, text="Urge model:").pack(side="left")
        ttk.Entry(toolbar2, textvariable=self.urge_model_var, width=14).pack(side="left", padx=(2, 10))
        ttk.Label(toolbar2, text="num_predict:").pack(side="left")
        ttk.Entry(toolbar2, textvariable=self.num_predict_var, width=6).pack(side="left", padx=(2, 10))
        ttk.Label(toolbar2, text="Urge num_predict:").pack(side="left")
        ttk.Entry(toolbar2, textvariable=self.urge_num_predict_var, width=6).pack(side="left", padx=(2, 10))
        ttk.Label(toolbar2, text="repeat_penalty:").pack(side="left")
        ttk.Entry(toolbar2, textvariable=self.repeat_penalty_var, width=6).pack(side="left", padx=(2, 10))

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)
        self.voices_tab = ttk.Frame(notebook)
        self.groups_tab = ttk.Frame(notebook)
        notebook.add(self.voices_tab, text="Voices")
        notebook.add(self.groups_tab, text="Groups")

        self._build_voices_tab()
        self._build_groups_tab()

    # ----------------------------------------------------------- Voices tab --

    def _build_voices_tab(self):
        # Inner notebook (2026-09-11) - "Voice Editor" is the entire
        # existing body below, unmoved; "Urge Viewer" is new, read-only.
        # Teddy's ask: keep the voice list in the parent Voices tab
        # (don't duplicate it) - the Urge Viewer reads self.displayed_voice
        # rather than having its own selector.
        inner_notebook = ttk.Notebook(self.voices_tab)
        inner_notebook.pack(fill="both", expand=True)
        editor_tab = ttk.Frame(inner_notebook)
        urge_tab = ttk.Frame(inner_notebook)
        inner_notebook.add(editor_tab, text="Voice Editor")
        inner_notebook.add(urge_tab, text="Urge Viewer")

        frame = editor_tab

        top_bar = ttk.Frame(frame)
        top_bar.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Button(top_bar, text="New voice...", command=self.new_voice).pack(side="left", padx=2)
        ttk.Button(top_bar, text="Delete voice", command=self.delete_voice).pack(side="left", padx=2)
        ttk.Button(top_bar, text="Save voice", command=self.save_voice).pack(side="left", padx=2)

        paned = ttk.Panedwindow(frame, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)
        left = ttk.Frame(paned, width=180)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=4)

        list_frame = ttk.Frame(left)
        list_frame.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.voices_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, exportselection=False)
        scrollbar.config(command=self.voices_listbox.yview)
        self.voices_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.voices_listbox.bind("<<ListboxSelect>>", self._on_voice_select)

        params_row = ttk.Frame(right)
        params_row.pack(fill="x", pady=(0, 4))
        ttk.Label(params_row, text="Model:").pack(side="left")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.model_combo = ttk.Combobox(params_row, textvariable=self.model_var, width=20, state="normal")
        self.model_combo.pack(side="left", padx=(2, 4))
        ttk.Button(params_row, text="↻", width=3, command=self.refresh_models).pack(side="left")
        # Four independent elemental currencies (2026-09-12), one compact
        # label+entry pair each, in the same fixed CURRENCY_ELEMENTS order
        # everywhere else uses - no "$" anymore, on purpose.
        self.currency_vars = {}
        for element in CURRENCY_ELEMENTS:
            ttk.Label(params_row, text=f"{element}:").pack(side="left", padx=(10, 0))
            var = tk.StringVar(value="0")
            self.currency_vars[element] = var
            ttk.Entry(params_row, textvariable=var, width=6).pack(side="left")

        ttk.Label(right, text="Identity (last line of the HUD, every cycle):").pack(anchor="w", padx=2)
        self.identity_box = scrolledtext.ScrolledText(right, wrap="word", height=4)
        self.identity_box.pack(fill="x", padx=2, pady=(0, 4))

        hud_frame = ttk.LabelFrame(
            right, text="HUD (read-only - membership/board managed from the Groups tab)"
        )
        hud_frame.pack(fill="x", padx=2, pady=(0, 4))
        self.hud_summary_box = tk.Text(hud_frame, wrap="word", height=5, state="disabled")
        self.hud_summary_box.pack(fill="x", padx=4, pady=4)

        messages_frame = ttk.LabelFrame(right, text="Messages")
        messages_frame.pack(fill="both", expand=True, padx=2, pady=(0, 2))

        msg_top_bar = ttk.Frame(messages_frame)
        msg_top_bar.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Button(msg_top_bar, text="New message", command=self.new_message).pack(side="left", padx=2)
        ttk.Button(msg_top_bar, text="Save message", command=self.save_message).pack(side="left", padx=2)
        ttk.Button(msg_top_bar, text="Delete message", command=self.delete_message).pack(side="left", padx=2)

        tree_frame = ttk.Frame(messages_frame)
        tree_frame.pack(fill="x", padx=4, pady=4)
        msg_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical")
        self.messages_tree = ttk.Treeview(
            tree_frame,
            columns=("id", "timestamp", "speaker", "text"),
            show="headings",
            yscrollcommand=msg_scrollbar.set,
            height=4,
        )
        for col, width in (("id", 40), ("timestamp", 130), ("speaker", 90), ("text", 400)):
            self.messages_tree.heading(col, text=col.capitalize())
            self.messages_tree.column(col, width=width, stretch=(col == "text"))
        msg_scrollbar.config(command=self.messages_tree.yview)
        self.messages_tree.pack(side="left", fill="both", expand=True)
        msg_scrollbar.pack(side="right", fill="y")
        self.messages_tree.bind("<<TreeviewSelect>>", self._on_message_select)

        edit_frame = ttk.Frame(messages_frame)
        edit_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        ttk.Label(edit_frame, text="Timestamp:").grid(row=0, column=0, sticky="w")
        self.msg_timestamp_var = tk.StringVar(value="")
        ttk.Entry(edit_frame, textvariable=self.msg_timestamp_var, width=22).grid(row=0, column=1, sticky="w", padx=(2, 12))
        ttk.Label(edit_frame, text="Speaker:").grid(row=0, column=2, sticky="w")
        self.msg_speaker_var = tk.StringVar(value="")
        ttk.Entry(edit_frame, textvariable=self.msg_speaker_var, width=16).grid(row=0, column=3, sticky="w", padx=(2, 0))
        # ~50% of the window's height (Teddy's ask, 2026-09-10) - the
        # message list above shrank to make room for this.
        self.msg_text_box = scrolledtext.ScrolledText(edit_frame, wrap="word", height=16)
        self.msg_text_box.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=(4, 0))
        edit_frame.grid_columnconfigure(3, weight=1)
        edit_frame.grid_rowconfigure(1, weight=1)

        self._build_urge_viewer_tab(urge_tab)

    def _build_urge_viewer_tab(self, parent):
        """Read-only (2026-09-11, round one - no editing yet). The
        Outlook-Options-style master-detail Teddy asked for applies here,
        to the categories *within* one voice's urge breakdown - a
        vertical category list on the left, clicking one swaps the
        detail panel on the right. Always reflects self.displayed_voice
        (see _load_voice's added refresh call) rather than duplicating
        the voice selector that already exists on the Voice Editor tab."""
        paned = ttk.Panedwindow(parent, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)
        left = ttk.Frame(paned, width=160)
        self._urge_detail_frame = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(self._urge_detail_frame, weight=4)

        self.urge_category_listbox = tk.Listbox(left, exportselection=False)
        self.urge_category_listbox.pack(fill="both", expand=True)
        for cat in URGE_VIEWER_CATEGORIES:
            self.urge_category_listbox.insert("end", cat)
        self.urge_category_listbox.selection_set(0)
        self.urge_category_listbox.bind("<<ListboxSelect>>", lambda e: self._refresh_urge_viewer())

    def _refresh_urge_viewer(self):
        """Called whenever the displayed voice changes (_load_voice) or
        the category selection changes - always shows
        self.displayed_voice's current urge state, read-only. No
        auto-refresh during the running loop - a voice re-selected in
        the Voice Editor tab (which the existing live-refresh fix
        already does on delivery/turn) naturally keeps this current
        without adding overhead to the hot tick path."""
        for child in self._urge_detail_frame.winfo_children():
            child.destroy()
        if not self.displayed_voice:
            ttk.Label(self._urge_detail_frame, text="No voice selected.").pack(anchor="w", padx=4, pady=4)
            return

        selection = self.urge_category_listbox.curselection()
        category = URGE_VIEWER_CATEGORIES[selection[0]] if selection else URGE_VIEWER_CATEGORIES[0]

        state = load_voice_state(self.world_name, self.displayed_voice)
        snapshot = compute_urge_snapshot(state)

        ttk.Label(
            self._urge_detail_frame, text=f"{self.displayed_voice} - {category}",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=4, pady=(0, 8))

        if category == "Perform Urges":
            for name in URGE_FUNCTIONS:
                u = state.get("urge", {}).get(name, 0.0)
                ttk.Label(
                    self._urge_detail_frame, text=f"{name}: Urge={u:.1f}  XLEUD={xleud(u) * 100:.0f}%"
                ).pack(anchor="w", padx=4)
        elif category == "Understand Urges":
            for name in URGE_FUNCTIONS:
                u = state.get("understand_urge", {}).get(name, 0.0)
                ttk.Label(
                    self._urge_detail_frame, text=f"{name}: Urge={u:.1f}  XLEUD={xleud(u) * 100:.0f}%"
                ).pack(anchor="w", padx=4)
            general = state.get("understand_urge_general", 0.0)
            ttk.Label(
                self._urge_detail_frame,
                text=f"(general/unknown-function): Urge={general:.1f}  XLEUD={xleud(general) * 100:.0f}%",
            ).pack(anchor="w", padx=4, pady=(6, 0))
        else:  # "This Turn"
            winner_name, _ = snapshot["understand_winner"]
            if snapshot["understand_wins"]:
                text = f'Would show the canned nudge: "functions({winner_name or ""})"'
            elif snapshot["top_perform"]:
                names = ", ".join(n for n, _ in snapshot["top_perform"])
                text = f"Would call the urge agent for: {names}"
            else:
                text = "Would show nothing (every perform-urge below the floor)."
            ttk.Label(self._urge_detail_frame, text=text, wraplength=400, justify="left").pack(anchor="w", padx=4)

    def _populate_voices_list(self):
        self._current_voice_names = list_voices(self.world_name)
        self.voices_listbox.delete(0, "end")
        for name in self._current_voice_names:
            self.voices_listbox.insert("end", name)

    def _on_voice_select(self, event):
        selection = self.voices_listbox.curselection()
        if not selection:
            return
        if self.displayed_voice:
            self._save_voice_snapshot(self.displayed_voice)
        name = self._current_voice_names[selection[0]]
        self._load_voice(name)

    def _load_voice(self, name):
        state = load_voice_state(self.world_name, name)
        self.displayed_voice = name
        self.model_var.set(state.get("model", DEFAULT_MODEL))
        currencies = state.get("currencies", {})
        for element, var in self.currency_vars.items():
            var.set(f"{currencies.get(element, 0.0):.1f}")
        self.identity_box.delete("1.0", "end")
        self.identity_box.insert("end", state.get("identity", ""))
        self._refresh_hud_summary(name)
        self._current_messages = state.get("messages", [])
        self._populate_messages_tree()
        self._clear_message_edit()
        self._refresh_urge_viewer()

    def _refresh_hud_summary(self, name):
        """Read-only - see hud_fields() for the actual data, shared with
        build_hud() so this can never drift out of sync with what the
        voice really receives."""
        f = hud_fields(self.world_name, name)
        board_summary = ", ".join(f["board_counts"]) if f["board_counts"] else "none"
        lines = [
            f"Your groups: {', '.join(f['own_groups']) if f['own_groups'] else 'none'}",
            f"Voices you can see: {', '.join(f['seen']) if f['seen'] else 'none'}",
            f"Voices that exist but you cannot see: {', '.join(f['unseen']) if f['unseen'] else 'none'}",
            f"Board activity: {board_summary}",
        ]
        self.hud_summary_box.config(state="normal")
        self.hud_summary_box.delete("1.0", "end")
        self.hud_summary_box.insert("end", "\n".join(lines))
        self.hud_summary_box.config(state="disabled")

    def _populate_messages_tree(self):
        self.messages_tree.delete(*self.messages_tree.get_children())
        for m in sorted(self._current_messages, key=lambda m: m["id"]):
            preview = m["text"].replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:120] + "..."
            self.messages_tree.insert(
                "", "end", iid=str(m["id"]), values=(m["id"], m["timestamp"], m["speaker"], preview)
            )

    def _clear_message_edit(self):
        self.selected_message_id = None
        self.msg_timestamp_var.set("")
        self.msg_speaker_var.set("")
        self.msg_text_box.delete("1.0", "end")

    def _on_message_select(self, event):
        selection = self.messages_tree.selection()
        if not selection:
            return
        msg_id = int(selection[0])
        message = next((m for m in self._current_messages if m["id"] == msg_id), None)
        if not message:
            return
        self.selected_message_id = msg_id
        self.msg_timestamp_var.set(message["timestamp"])
        self.msg_speaker_var.set(message["speaker"])
        self.msg_text_box.delete("1.0", "end")
        self.msg_text_box.insert("end", message["text"])

    def new_message(self):
        if not self.displayed_voice:
            return
        self._clear_message_edit()
        self.msg_timestamp_var.set(datetime.now().isoformat(timespec="seconds"))
        self.msg_speaker_var.set(self.displayed_voice)

    def save_message(self):
        if not self.displayed_voice:
            return
        timestamp = self.msg_timestamp_var.get().strip()
        speaker = self.msg_speaker_var.get().strip()
        text = self.msg_text_box.get("1.0", "end-1c")
        if not timestamp or not speaker:
            messagebox.showerror("Fenra", "Timestamp and speaker can't be empty.")
            return
        if self.selected_message_id is not None:
            for m in self._current_messages:
                if m["id"] == self.selected_message_id:
                    m["timestamp"], m["speaker"], m["text"] = timestamp, speaker, text
                    break
        else:
            next_id = max((m["id"] for m in self._current_messages), default=0) + 1
            self._current_messages.append(
                {"id": next_id, "timestamp": timestamp, "speaker": speaker, "text": text}
            )
            self.selected_message_id = next_id
        state = load_voice_state(self.world_name, self.displayed_voice)
        state["messages"] = self._current_messages
        save_voice_state(self.world_name, self.displayed_voice, state)
        self._populate_messages_tree()
        self.status_var.set("Message saved")

    def delete_message(self):
        if not self.displayed_voice or self.selected_message_id is None:
            return
        if not messagebox.askyesno("Fenra", "Delete this message? This can't be undone."):
            return
        self._current_messages = [m for m in self._current_messages if m["id"] != self.selected_message_id]
        state = load_voice_state(self.world_name, self.displayed_voice)
        state["messages"] = self._current_messages
        save_voice_state(self.world_name, self.displayed_voice, state)
        self._populate_messages_tree()
        self._clear_message_edit()

    def _save_voice_snapshot(self, name):
        # Loads existing state first and only overwrites the
        # widget-backed fields (2026-09-11 fix) - this used to rebuild
        # the whole state dict from scratch with just these 4 fields,
        # which silently wiped any field with no GUI widget (the new
        # urge/understand_urge tracking has none - it's computed
        # server-side, never hand-edited). This is the real root-cause
        # fix, not a urge-specific patch - it protects any future new
        # voice-state field the same way.
        state = load_voice_state(self.world_name, name)
        existing_currencies = state.get("currencies", {})
        currencies = {}
        for element, var in self.currency_vars.items():
            try:
                currencies[element] = float(var.get())
            except ValueError:
                currencies[element] = existing_currencies.get(element, 0.0)
        state["model"] = self.model_var.get()
        state["identity"] = self.identity_box.get("1.0", "end-1c")
        state["messages"] = self._current_messages
        state["currencies"] = currencies
        save_voice_state(self.world_name, name, state)

    def save_voice(self):
        if not self.displayed_voice:
            return
        self._save_voice_snapshot(self.displayed_voice)
        self.status_var.set(f"Saved '{self.displayed_voice}'")

    def new_voice(self):
        name = simpledialog.askstring("New Voice", "Voice name:", parent=self.root)
        if not name:
            return
        name = sanitize_name(name)
        if not name:
            return
        if name in list_voices(self.world_name):
            messagebox.showerror("Fenra", f"a voice named '{name}' already exists.")
            return
        save_voice_state(self.world_name, name, default_voice_state())
        self.world_voices.append(name)
        self._save_world_controls()
        self._populate_voices_list()
        self._refresh_group_member_candidates()

    def delete_voice(self):
        if not self.displayed_voice:
            return
        name = self.displayed_voice
        if not messagebox.askyesno("Fenra", f"Delete voice '{name}'? This can't be undone."):
            return
        delete_voice(self.world_name, name)
        if name in self.world_voices:
            self.world_voices.remove(name)
        # Also drop it from every group's membership - a deleted voice
        # can't stay listed as a member of anything.
        for gname in list_groups(self.world_name):
            gstate = load_group_state(self.world_name, gname)
            if gstate and name in gstate.get("members", []):
                gstate["members"].remove(name)
                save_group_state(self.world_name, gname, gstate)
        self._save_world_controls()
        self.displayed_voice = None
        self._populate_voices_list()
        self._refresh_group_member_candidates()
        if self.displayed_group:
            self._load_group(self.displayed_group)

    def refresh_models(self):
        models = list_ollama_models(self.host_var.get())
        self.model_combo["values"] = models
        if models:
            self.status_var.set(f"{len(models)} model(s) available")
        else:
            self.status_var.set("Could not reach Ollama host")

    # ----------------------------------------------------------- Groups tab --

    def _build_groups_tab(self):
        frame = self.groups_tab

        top_bar = ttk.Frame(frame)
        top_bar.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Button(top_bar, text="New group...", command=self.new_group).pack(side="left", padx=2)
        ttk.Button(top_bar, text="Delete group", command=self.delete_group).pack(side="left", padx=2)
        ttk.Button(top_bar, text="Rename group", command=self.rename_group).pack(side="left", padx=2)

        paned = ttk.Panedwindow(frame, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)
        left = ttk.Frame(paned, width=180)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=3)

        list_frame = ttk.Frame(left)
        list_frame.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.groups_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, exportselection=False)
        scrollbar.config(command=self.groups_listbox.yview)
        self.groups_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.groups_listbox.bind("<<ListboxSelect>>", self._on_group_select)

        name_row = ttk.Frame(right)
        name_row.pack(fill="x", padx=2, pady=(0, 6))
        ttk.Label(name_row, text="Name:", width=12).pack(side="left")
        self.group_name_var = tk.StringVar(value="")
        ttk.Label(name_row, textvariable=self.group_name_var, font=("Segoe UI", 9, "bold")).pack(side="left")

        # Fixed-size, not expanding - a group's member count is small,
        # and the chat/board panels below need the room more (Teddy's
        # call, 2026-09-10).
        members_frame = ttk.LabelFrame(right, text="Members")
        members_frame.pack(fill="x", expand=False, padx=2, pady=(0, 4))
        self.group_members_listbox = tk.Listbox(
            members_frame, selectmode="extended", exportselection=False, height=4
        )
        self.group_members_listbox.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Button(members_frame, text="Remove selected", command=self.remove_group_members).pack(
            anchor="e", padx=4, pady=(0, 4)
        )

        add_row = ttk.Frame(right)
        add_row.pack(fill="x", padx=2, pady=(0, 4))
        ttk.Label(add_row, text="Add voice:").pack(side="left")
        self.group_add_voice_var = tk.StringVar(value="")
        self.group_add_voice_combo = ttk.Combobox(
            add_row, textvariable=self.group_add_voice_var, width=20, state="readonly"
        )
        self.group_add_voice_combo.pack(side="left", padx=(4, 4))
        ttk.Button(add_row, text="Add", command=self.add_group_member).pack(side="left")

        # Chat and Board share the remaining space, resizable against
        # each other (same Panedwindow pattern as the left/right split).
        lower_paned = ttk.Panedwindow(right, orient="vertical")
        lower_paned.pack(fill="both", expand=True, padx=2, pady=(0, 2))

        chat_frame = ttk.LabelFrame(
            lower_paned, text="Group Chat (read-only - full text, not the masked version bystanders see)"
        )
        board_frame = ttk.LabelFrame(lower_paned, text="Board")
        lower_paned.add(chat_frame, weight=1)
        lower_paned.add(board_frame, weight=1)

        chat_top_bar = ttk.Frame(chat_frame)
        chat_top_bar.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Button(chat_top_bar, text="Refresh", command=self._refresh_group_chat).pack(side="left", padx=2)

        self.group_chat_box = tk.Text(chat_frame, wrap="word", state="disabled")
        self.group_chat_box.pack(fill="both", expand=True, padx=4, pady=4)

        board_top_bar = ttk.Frame(board_frame)
        board_top_bar.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Button(board_top_bar, text="New post", command=self.new_board_post).pack(side="left", padx=2)
        ttk.Button(board_top_bar, text="Save post", command=self.save_board_post).pack(side="left", padx=2)
        ttk.Button(board_top_bar, text="Delete post", command=self.delete_board_post).pack(side="left", padx=2)

        board_tree_frame = ttk.Frame(board_frame)
        board_tree_frame.pack(fill="x", padx=4, pady=4)
        board_scrollbar = ttk.Scrollbar(board_tree_frame, orient="vertical")
        self.board_tree = ttk.Treeview(
            board_tree_frame,
            columns=("id", "subject", "author", "timestamp", "text"),
            show="headings",
            yscrollcommand=board_scrollbar.set,
            height=4,
        )
        for col, width in (("id", 40), ("subject", 140), ("author", 90), ("timestamp", 130), ("text", 260)):
            self.board_tree.heading(col, text=col.capitalize())
            self.board_tree.column(col, width=width, stretch=(col == "text"))
        board_scrollbar.config(command=self.board_tree.yview)
        self.board_tree.pack(side="left", fill="both", expand=True)
        board_scrollbar.pack(side="right", fill="y")
        self.board_tree.bind("<<TreeviewSelect>>", self._on_board_select)

        board_edit_frame = ttk.Frame(board_frame)
        board_edit_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        ttk.Label(board_edit_frame, text="Subject:").grid(row=0, column=0, sticky="w")
        self.board_subject_var = tk.StringVar(value="")
        ttk.Entry(board_edit_frame, textvariable=self.board_subject_var, width=24).grid(row=0, column=1, sticky="w", padx=(2, 12))
        ttk.Label(board_edit_frame, text="Author:").grid(row=0, column=2, sticky="w")
        self.board_author_var = tk.StringVar(value="")
        ttk.Entry(board_edit_frame, textvariable=self.board_author_var, width=14).grid(row=0, column=3, sticky="w", padx=(2, 12))
        ttk.Label(board_edit_frame, text="Timestamp:").grid(row=0, column=4, sticky="w")
        self.board_timestamp_var = tk.StringVar(value="")
        ttk.Entry(board_edit_frame, textvariable=self.board_timestamp_var, width=20).grid(row=0, column=5, sticky="w", padx=(2, 0))
        # ~50% of the window's height (Teddy's ask, 2026-09-10), same as
        # the Voices tab's message editor.
        self.board_text_box = scrolledtext.ScrolledText(board_edit_frame, wrap="word", height=16)
        self.board_text_box.grid(row=1, column=0, columnspan=6, sticky="nsew", pady=(4, 0))
        board_edit_frame.grid_columnconfigure(5, weight=1)
        board_edit_frame.grid_rowconfigure(1, weight=1)

    def _populate_groups_list(self):
        self._current_group_names = list_groups(self.world_name)
        self.groups_listbox.delete(0, "end")
        for name in self._current_group_names:
            self.groups_listbox.insert("end", name)

    def _on_group_select(self, event):
        selection = self.groups_listbox.curselection()
        if not selection:
            return
        self._load_group(self._current_group_names[selection[0]])

    def _load_group(self, name):
        state = load_group_state(self.world_name, name) or default_group_state(name)
        self.displayed_group = name
        self.group_name_var.set(state.get("name", name))
        self.group_members_listbox.delete(0, "end")
        for voice in state.get("members", []):
            self.group_members_listbox.insert("end", voice)
        self._refresh_group_member_candidates()
        self._refresh_group_chat()
        self._current_board = state.get("board", [])
        self._populate_board_tree()
        self._clear_board_edit()

    def _refresh_group_chat(self):
        """Read-only - see group_chat_transcript()'s own docstring for
        why this shows the real full text, not what any one member's
        inbox actually received."""
        if not self.displayed_group:
            return
        entries = group_chat_transcript(self.world_name, self.displayed_group)
        lines = [f"[{m['timestamp']}] {m['speaker']}: {m['text']}" for m in entries]
        self.group_chat_box.config(state="normal")
        self.group_chat_box.delete("1.0", "end")
        self.group_chat_box.insert("end", "\n\n".join(lines) if lines else "(no chat yet)")
        self.group_chat_box.config(state="disabled")

    def _populate_board_tree(self):
        self.board_tree.delete(*self.board_tree.get_children())
        for p in sorted(self._current_board, key=lambda p: p["id"]):
            preview = p["text"].replace("\n", " ")
            if len(preview) > 100:
                preview = preview[:100] + "..."
            self.board_tree.insert(
                "", "end", iid=str(p["id"]),
                values=(p["id"], p["subject"], p["author"], p["timestamp"], preview),
            )

    def _clear_board_edit(self):
        self.selected_post_id = None
        self.board_subject_var.set("")
        self.board_author_var.set("")
        self.board_timestamp_var.set("")
        self.board_text_box.delete("1.0", "end")

    def _on_board_select(self, event):
        selection = self.board_tree.selection()
        if not selection:
            return
        post_id = int(selection[0])
        post = next((p for p in self._current_board if p["id"] == post_id), None)
        if not post:
            return
        self.selected_post_id = post_id
        self.board_subject_var.set(post["subject"])
        self.board_author_var.set(post["author"])
        self.board_timestamp_var.set(post["timestamp"])
        self.board_text_box.delete("1.0", "end")
        self.board_text_box.insert("end", post["text"])

    def new_board_post(self):
        if not self.displayed_group:
            return
        self._clear_board_edit()
        self.board_author_var.set(self.displayed_group)
        self.board_timestamp_var.set(datetime.now().isoformat(timespec="seconds"))

    def save_board_post(self):
        if not self.displayed_group:
            return
        subject = self.board_subject_var.get().strip()
        author = self.board_author_var.get().strip()
        timestamp = self.board_timestamp_var.get().strip()
        text = self.board_text_box.get("1.0", "end-1c")
        if not subject or not author or not timestamp:
            messagebox.showerror("Fenra", "Subject, author, and timestamp can't be empty.")
            return
        if self.selected_post_id is not None:
            for p in self._current_board:
                if p["id"] == self.selected_post_id:
                    p["subject"], p["author"], p["timestamp"], p["text"] = subject, author, timestamp, text
                    break
        else:
            next_id = max((p["id"] for p in self._current_board), default=0) + 1
            self._current_board.append({
                "id": next_id, "subject": subject, "author": author,
                "timestamp": timestamp, "text": text, "seen": {},
            })
            self.selected_post_id = next_id
        state = load_group_state(self.world_name, self.displayed_group) or default_group_state(self.displayed_group)
        state["board"] = self._current_board
        save_group_state(self.world_name, self.displayed_group, state)
        self._populate_board_tree()
        self.status_var.set("Post saved")

    def delete_board_post(self):
        if not self.displayed_group or self.selected_post_id is None:
            return
        if not messagebox.askyesno("Fenra", "Delete this post? This can't be undone."):
            return
        self._current_board = [p for p in self._current_board if p["id"] != self.selected_post_id]
        state = load_group_state(self.world_name, self.displayed_group) or default_group_state(self.displayed_group)
        state["board"] = self._current_board
        save_group_state(self.world_name, self.displayed_group, state)
        self._populate_board_tree()
        self._clear_board_edit()

    def _refresh_group_member_candidates(self):
        """The 'Add voice' combobox - every world voice not already a
        member of the currently displayed group."""
        if not self.displayed_group:
            self.group_add_voice_combo["values"] = []
            return
        state = load_group_state(self.world_name, self.displayed_group) or {}
        members = set(state.get("members", []))
        candidates = [v for v in list_voices(self.world_name) if v not in members]
        self.group_add_voice_combo["values"] = candidates
        if candidates:
            self.group_add_voice_var.set(candidates[0])
        else:
            self.group_add_voice_var.set("")

    def new_group(self):
        name = simpledialog.askstring("New Group", "Group name:", parent=self.root)
        if not name:
            return
        name = sanitize_name(name)
        if not name:
            return
        if name in list_groups(self.world_name):
            messagebox.showerror("Fenra", f"a group named '{name}' already exists.")
            return
        save_group_state(self.world_name, name, default_group_state(name))
        self._populate_groups_list()

    def rename_group(self):
        if not self.displayed_group:
            return
        old_name = self.displayed_group
        new_name = simpledialog.askstring("Rename Group", "New name:", initialvalue=old_name, parent=self.root)
        if not new_name:
            return
        new_name = sanitize_name(new_name)
        if not new_name or new_name == old_name:
            return
        if new_name in list_groups(self.world_name):
            messagebox.showerror("Fenra", f"a group named '{new_name}' already exists.")
            return
        state = load_group_state(self.world_name, old_name) or default_group_state(old_name)
        state["name"] = new_name
        save_group_state(self.world_name, new_name, state)
        delete_group(self.world_name, old_name)
        self.displayed_group = new_name
        self._populate_groups_list()

    def delete_group(self):
        if not self.displayed_group:
            return
        name = self.displayed_group
        if not messagebox.askyesno("Fenra", f"Delete group '{name}'? This can't be undone."):
            return
        delete_group(self.world_name, name)
        self.displayed_group = None
        self._populate_groups_list()
        self.group_name_var.set("")
        self.group_members_listbox.delete(0, "end")
        self._clear_group_chat()
        self._current_board = []
        self.board_tree.delete(*self.board_tree.get_children())
        self._clear_board_edit()

    def _clear_group_chat(self):
        self.group_chat_box.config(state="normal")
        self.group_chat_box.delete("1.0", "end")
        self.group_chat_box.config(state="disabled")

    def add_group_member(self):
        if not self.displayed_group:
            return
        voice = self.group_add_voice_var.get()
        if not voice:
            return
        state = load_group_state(self.world_name, self.displayed_group) or default_group_state(self.displayed_group)
        if voice not in state["members"]:
            state["members"].append(voice)
            save_group_state(self.world_name, self.displayed_group, state)
        self._load_group(self.displayed_group)

    def remove_group_members(self):
        if not self.displayed_group:
            return
        selection = self.group_members_listbox.curselection()
        if not selection:
            return
        to_remove = {self.group_members_listbox.get(i) for i in selection}
        state = load_group_state(self.world_name, self.displayed_group) or default_group_state(self.displayed_group)
        state["members"] = [v for v in state.get("members", []) if v not in to_remove]
        save_group_state(self.world_name, self.displayed_group, state)
        self._load_group(self.displayed_group)

    # --------------------------------------------------------------- worlds --

    def _load_world(self, name):
        if self.running:
            self.toggle_loop()
        if self.displayed_voice:
            self._save_voice_snapshot(self.displayed_voice)

        ensure_world_dir(name)
        state = load_world_state(name)
        self.world_name = name
        self.world_var.set(name)
        self.host_var.set(state.get("host", DEFAULT_HOST))
        self.interval_var.set(str(state.get("interval", DEFAULT_INTERVAL_SEC)))
        self.urge_model_var.set(state.get("urge_model", DEFAULT_URGE_MODEL))
        self.num_predict_var.set(str(state.get("num_predict", 1500)))
        self.urge_num_predict_var.set(str(state.get("urge_num_predict", 250)))
        self.repeat_penalty_var.set(str(state.get("repeat_penalty", 1.3)))
        self.world_voices = state.get("voices", [])
        self.voice_rotation_index = state.get("voice_rotation_index", 0)

        self.displayed_voice = None
        self.displayed_group = None
        self._populate_voices_list()
        self._populate_groups_list()
        self.group_name_var.set("")
        self.group_members_listbox.delete(0, "end")
        self.identity_box.delete("1.0", "end")
        for var in self.currency_vars.values():
            var.set("0")
        self.hud_summary_box.config(state="normal")
        self.hud_summary_box.delete("1.0", "end")
        self.hud_summary_box.config(state="disabled")
        self._current_messages = []
        self.messages_tree.delete(*self.messages_tree.get_children())
        self._clear_message_edit()
        self._clear_group_chat()
        self._current_board = []
        self.board_tree.delete(*self.board_tree.get_children())
        self._clear_board_edit()
        self.status_var.set("Idle")
        self._rebuild_worlds_menu()

    def new_world(self):
        name = simpledialog.askstring("New World", "World name:", parent=self.root)
        if not name:
            return
        name = sanitize_name(name)
        if not name:
            return
        if name in list_worlds():
            if messagebox.askyesno("Fenra", f"World '{name}' already exists. Load it instead?"):
                self._load_world(name)
            return
        ensure_world_dir(name)
        save_world_state(name, default_world_state())
        self._load_world(name)

    def rename_world(self):
        if not self.world_name:
            return
        old_name = self.world_name
        new_name = simpledialog.askstring("Rename World", "New name:", initialvalue=old_name, parent=self.root)
        if not new_name:
            return
        new_name = sanitize_name(new_name)
        if not new_name or new_name == old_name:
            return
        if new_name in list_worlds():
            messagebox.showerror("Fenra", f"a world named '{new_name}' already exists.")
            return
        # Real bug, caught by the live smoke test: _load_world's own
        # save-before-switch step (correct for an ordinary world switch,
        # where the world being left genuinely still exists on disk)
        # writes into self.world_name, which is still the OLD name at
        # that point since _load_world only updates it partway through.
        # For a rename specifically, the old directory has just been
        # physically moved away - writing to it afterward resurrects a
        # stale, empty duplicate under the old name. Clearing the
        # displayed selection first means _load_world has nothing to
        # save before the switch, since there's no "old world" left to
        # save it into - this world didn't go away, it just changed name.
        self.displayed_voice = None
        self.displayed_group = None
        os.rename(world_dir(old_name), world_dir(new_name))
        self._load_world(new_name)

    def _save_world_controls(self):
        if not self.world_name:
            return
        state = {
            "host": self.host_var.get(),
            "interval": self.interval_var.get(),
            "model_default": DEFAULT_MODEL,
            "voices": self.world_voices,
            "voice_rotation_index": self.voice_rotation_index,
            "urge_model": self.urge_model_var.get(),
            "num_predict": self.num_predict_var.get(),
            "urge_num_predict": self.urge_num_predict_var.get(),
            "repeat_penalty": self.repeat_penalty_var.get(),
        }
        save_world_state(self.world_name, state)

    # ----------------------------------------------------------- the loop --

    def toggle_loop(self):
        if self.running:
            self.running = False
            self.start_stop_btn.config(text="Start")
            self.status_var.set("Stopping...")
        else:
            if not self.world_voices:
                messagebox.showinfo("Fenra", "This world has no voices yet - add one on the Voices tab first.")
                return
            self.running = True
            self.start_stop_btn.config(text="Stop")
            self.status_var.set("Running")
            self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)
            self.loop_thread.start()

    def _run_loop(self):
        while self.running:
            try:
                self._tick()
            except Exception as exc:  # keep the loop alive on a transient error
                self.root.after(0, self.status_var.set, f"Error: {exc}")
            try:
                interval = float(self.interval_var.get())
            except ValueError:
                interval = DEFAULT_INTERVAL_SEC
            for _ in range(int(interval * 10)):
                if not self.running:
                    break
                time.sleep(0.1)
        self.root.after(0, self.status_var.set, "Idle")

    def _tick(self):
        if not self.world_voices:
            return
        index = self.voice_rotation_index % len(self.world_voices)
        active_voice = self.world_voices[index]
        self.voice_rotation_index = (index + 1) % len(self.world_voices)
        self.root.after(0, self._save_world_controls)
        # "Thinking..." status (2026-09-11) - shows who's mid-call during
        # the wait on a slow model, not just after it returns. The
        # existing post-response status_var.set below already overwrites
        # this once the call finishes.
        self.root.after(0, self.status_var.set, f"Running ({active_voice} is thinking...)")

        # If the currently-displayed voice is the one about to run, its
        # in-flight widget edits are the authoritative copy - persist
        # them first so a same-voice tick doesn't clobber an unsaved
        # edit. Gated on equality with active_voice (2026-09-10 fix) -
        # it used to fire for the displayed voice on *every* tick
        # regardless of whose turn it was, which meant a stale in-memory
        # snapshot of whatever voice happened to be selected in the GUI
        # silently overwrote real deliveries appended to its state.json
        # by other voices' turns in between (found via Dash only ever
        # showing its own messages). The residual gap that gate alone
        # left open - the displayed voice's widget still going stale
        # from a delivery received *between* its own turns, then this
        # same save clobbering it on the next one - is closed by the
        # live-refresh in the delivery loop below (2026-09-10, found via
        # Milo silently losing a real delivery from Orin).
        if self.displayed_voice == active_voice:
            self.root.after(0, self._save_voice_snapshot, self.displayed_voice)

        state = load_voice_state(self.world_name, active_voice)
        model = state.get("model", DEFAULT_MODEL)

        try:
            repeat_penalty = float(self.repeat_penalty_var.get())
        except ValueError:
            repeat_penalty = 1.3

        # Urge system (2026-09-11) - see xleud()/compute_urge_snapshot()/
        # apply_urge_tick() for the mechanics. Entirely prompt-only, same
        # as build_hud() - urge_block never gets written to
        # state["messages"] or anywhere on disk, computed fresh every tick.
        urge_snapshot = compute_urge_snapshot(state)
        urge_block = ""
        if urge_snapshot["understand_wins"]:
            # A recent real error is the single highest signal right now -
            # skip the roleplay urge agent entirely and inject a fixed,
            # deterministic corrective instead (never LLM-generated, so
            # it's always the exact real functions() search syntax).
            winner_name, _ = urge_snapshot["understand_winner"]
            urge_block = (
                "You have the urge to call functions()."
                if winner_name is None
                else f"You have the urge to call functions({winner_name})."
            )
        elif urge_snapshot["top_perform"]:
            top_perform = urge_snapshot["top_perform"]
            urge_prompt = build_urge_agent_prompt(top_perform)
            try:
                urge_num_predict = int(self.urge_num_predict_var.get())
            except ValueError:
                urge_num_predict = 250
            try:
                urge_para = call_ollama(
                    self.host_var.get(), self.urge_model_var.get(), urge_prompt,
                    options={"num_predict": urge_num_predict, "repeat_penalty": repeat_penalty},
                ).strip()
            except requests.RequestException:
                # Fail open (confirmed with Teddy, 2026-09-11) - the urge
                # block is supplementary flavor text, not essential; a
                # hiccup in the (separate, smaller) urge-agent model
                # shouldn't cost the voice its whole turn the way the main
                # model failing does below.
                urge_para = ""
            if urge_para:
                reminders = build_call_syntax_reminders([name for name, _ in top_perform])
                urge_block = f"{urge_para}\n\n{reminders}"

        hud = build_hud(self.world_name, active_voice)
        prompt = f"{render_messages(state.get('messages', []))}\n\n{hud}"
        if urge_block:
            prompt = f"{prompt}\n\n{urge_block}"

        try:
            num_predict = int(self.num_predict_var.get())
        except ValueError:
            num_predict = 1500
        try:
            response = call_ollama(
                self.host_var.get(), model, prompt,
                options={"num_predict": num_predict, "repeat_penalty": repeat_penalty},
            )
        except requests.RequestException as exc:
            self.root.after(0, self.status_var.set, f"Error calling {model}: {exc}")
            return
        response = response.strip()
        if not response:
            return
        full_response, masked_response, outcomes = run_function_calls(self.world_name, active_voice, response)
        apply_urge_tick(self.world_name, active_voice, outcomes)

        timestamp = datetime.now().isoformat(timespec="seconds")

        # The speaker's own record - the real thing, calls and results
        # both, tagged with every group it just went out to (see
        # append_message's `groups` docstring - this is what the
        # Groups-tab chat view is reconstructed from). Everyone else
        # sees the masked version (see run_function_calls) - that a
        # call happened, never its arguments or result.
        member_groups = groups_containing(self.world_name, active_voice)
        append_message(self.world_name, active_voice, active_voice, full_response, timestamp, groups=member_groups)

        # Every OTHER member of every group the speaker belongs to - one
        # append per listener max, even if they share more than one
        # group with the speaker.
        already_notified = {active_voice}
        for group_name in member_groups:
            gstate = load_group_state(self.world_name, group_name) or {}
            for member in gstate.get("members", []):
                if member in already_notified:
                    continue
                already_notified.add(member)
                append_message(self.world_name, member, active_voice, masked_response, timestamp)
                # Live-refresh (2026-09-10 fix): if this recipient is the
                # voice currently displayed in the GUI, its in-memory
                # _current_messages is now stale relative to what we just
                # wrote to disk. Left alone, that staleness survives until
                # this voice's own next turn, when the _tick pre-save
                # above (gated on displayed_voice == active_voice) would
                # persist the stale copy right back over this delivery -
                # clobbering it. Reloading now keeps the widget (and any
                # snapshot later saved from it) honest. Same reload used
                # for the speaker's own turn below, so this shares its
                # side effects: identity/model/HUD refresh, and any
                # in-progress unsaved manual edit in the message editor
                # is cleared - accepted, matches existing precedent.
                if member == self.displayed_voice:
                    self.root.after(0, self._load_voice, member)

        if active_voice == self.displayed_voice:
            self.root.after(0, self._load_voice, active_voice)
        self.root.after(0, self.status_var.set, f"Running ('{active_voice}' spoke)")


def main():
    root = tk.Tk()
    app = FenraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
