"""fenra.py - worlds-rebuild branch.

Teddy's call (2026-09-08): back up from the fenra.py architecture on
fenras-aletheosis entirely and rebuild from a simpler foundation. This
went through two real shapes: a first pass with voices and groups as
the only concepts ("let the internal thoughts work" - prove the core
loop before anything gets layered on top), then a 2026-09-13 redesign
that replaces groups with **rooms** and splits context into private
thought vs. shareable speech/action - motivated directly by watching
the Church of Aletheia arc converge into mutual, uncorrected
fabrication. The old groups model broadcast a speaking voice's entire
raw response (prose and all, only function-call syntax masked) to
every group member, every turn - every bystander saw every groupmate's
full internal narration, which is exactly the shape of thing that
produces yes-man convergence. Rooms + registers fix this structurally:
thought is now genuinely private, and nothing reaches another voice
except through an explicit act.

THE MODEL, exactly as specified:

- World (renamed from "session") - a fully separate container. Worlds
  share nothing with each other - no cross-world storage of any kind.
  Lives at worlds/<world>/.
- Voice - model, identity, thoughts, currencies, room. `behavior`
  existed in the first pass and is gone (2026-09-09) - it was the same
  boilerplate for every voice, and the HUD below (ending in identity)
  replaces what it was doing. `currencies` (2026-09-09, single
  `currency` field; replaced 2026-09-12 with four independent elemental
  balances - Air, Earth, Fire, Water, see CURRENCY_ELEMENTS/
  CURRENCY_RANGES below) is real balance any voice can move via
  `give_currency` - genuinely exploratory, no plan for it beyond seeing
  what they do with it once they can see it and move it. The
  single-dollar version got dropped because the `$` sign itself
  imported a real-world frame of reference ("rich"/"poor") that Fenra
  never defined any meaning for (Teddy's read, prompted by a voice
  describing itself as poor while actually holding the town's largest
  balance) - four un-ranked, unexplained currencies with no stated
  exchange rate, not even one Teddy or Qualia privately know, are meant
  to remove that borrowed frame entirely and let any value they end up
  having emerge from how they're actually used. `thoughts` (2026-09-10
  as `messages`, renamed 2026-09-13) is a real list of structured
  entries (`{id, timestamp, speaker, text}`, stable `id`), fully
  editable by Teddy at any time down to one specific entry - not a
  fixed-size window, not a separate history file. It grows by
  `append_message`, but as of the rooms redesign it holds **only that
  voice's own generations** - never anything from another voice. Its
  length also doubles as that voice's own turn counter, used to decay
  what it can still see in the world (see REGISTERS below); a paused
  voice's count doesn't advance, so anything owed to it just backs up
  rather than being lost. `render_messages`/`render_thoughts` flattens
  the list back into the exact `"[timestamp] name: text"` text the
  model has always received. `room` (2026-09-13) - the voice's current,
  single physical location; see ROOMS below.
- Room (2026-09-13, replaces Group) - name, `adjacent` (an undirected
  graph edge list, populated only when the room is created off another
  room - see ROOMS below), `board` (unchanged concept from groups,
  still a list of posts), and `log` (the room's own permanent record -
  see REGISTERS below). No membership list is stored on a room at all -
  "who's here" is always derived by scanning every voice's own `room`
  field, since a voice can only ever be in one room.

ROOMS (2026-09-13): physical co-location is now the only interaction
substrate - it replaces group membership entirely. Every voice is in
exactly one room at a time. Movement is unrestricted (`move_room`) -
no cost, no adjacency requirement, always reversible. `create_room`
spins up a brand-new room off the caller's current one and moves them
into it; the new room is adjacent to the room it was created from (an
undirected edge, written on both sides at creation, never edited
afterward, no limit on how many rooms can be adjacent to one room) -
adjacency can chain or branch arbitrarily as more rooms split off
existing ones. `read_room_log`/`room_state` let any voice query any
room by name regardless of where they currently are (same
full-transparency spirit as currency balances) - the actual mechanism
built specifically to give voices real, checkable ground truth against
fabrication.

REGISTERS: a voice's full context is `[thoughts][world activity][HUD]`.
`thoughts` is the private register above - the model's own raw
generation, never anything from anyone else. `world activity` (new,
`build_world_activity()`, computed fresh every tick like the HUD,
never persisted) is built entirely from room `log` entries: a
**dialogue** register (`say` - room-scoped, `SAY_TTL_TURNS`; `whisper`
- one specific voice, requires sharing a room, `WHISPER_TTL_TURNS`,
delivered to no one else, ever, live or logged; `yell` - room +
every adjacent room, `YELL_TTL_TURNS`) and an **activities** register
(every non-speech function call, rendered through that function's own
description mask exactly like the old group-era `_mask_for_call`,
`ACTIVITY_TTL_TURNS` - visible in full to the caller's own room, and as
a deliberately generic, content-free "You hear activity from the
adjacent room X." notice one hop out, so an adjacent room's occupants
get pulled toward investigating rather than told what happened).
Dialogue and activities merge into one chronological stream - ordering
is pure recency, not "loudness." Each log entry's TTL is counted in
**the recipient's own turns** (their own `len(thoughts)`, not
wall-clock ticks), captured as a `recipients`/`peripheral` baseline at
the moment the entry was logged; the room's `log` itself is permanent
and never pruned - only what a voice sees *live* decays.

TWO-LAYER ROOM LOG: every log entry carries a `mask` (what
`read_room_log()` returns to any voice, always - full content for
public acts, but deliberately withholding content for a `whisper`, so
querying the log never leaks a private exchange to a third party, not
even the original recipient once it's aged out of their live view) and
a `raw` (the literal act/full content, UI-only - the Rooms tab's Log
panel, Teddy/Qualia only, never exposed through any voice-facing
function).

THE HUD: the last thing in every prompt, computed fresh every tick and
never persisted to context (Teddy's call, 2026-09-09 - it reflects live
world state and shouldn't compound the same context-bloat problem a
silently-timing-out voice can already produce). Tells a voice its own
name/model, its own room, who else is currently there (with a
`(paused)` annotation, same privacy spirit as before - you only ever
learn about who's actually present), the names of adjacent rooms (not
their occupants - deliberately as vague as the "you hear activity"
notice), this room's board unread/skimmed counts, *everyone's* currency
balances (all four elements, full-world transparency, untouched by
rooms), and how to call/discover functions (hard-coded, same reasoning
as the old branch's bootstrap notice) - ending with its own identity
line as the literal last line of the entire prompt. `hud_fields()`
returns the same pieces as plain data, not text - `build_hud` formats
them, and the GUI's read-only HUD summary (Voices tab) calls the
identical function, so the two can never drift apart.

THE GUI is object-oriented (2026-09-10): select a voice or room in its
list, its properties appear underneath - nothing duplicated across
tabs. A room's board and log are properties of the *room*, edited/
viewed only from the Rooms tab - never from Voices, even though a
voice's HUD summary displays derived facts about its room (who else is
there, board activity) - those stay read-only there on purpose. A
voice's Messages panel (Voices tab) is a real multi-column list
(id/timestamp/speaker/text) of that voice's own private thoughts -
select a row to edit or delete that one entry, or add a new one - not a
single text blob. The Currency tab is gone (2026-09-10) - redundant
once currency became a real per-voice field on the Voices tab itself.

FUNCTIONS: reintroduced 2026-09-09, using the old branch's exact
`⟦function_name(args)⟧` call syntax (U+27E6/U+27E7 - essentially never
appears by accident) and `FUNCTION_REGISTRY` shape, no permission layer
(every voice can call everything), no `functions.jsonl` logging, no
fabrication-detection. `send_message` (the old free, non-room-gated DM)
is gone as of 2026-09-13 - `whisper` is its room-gated replacement; a
true non-room-gated DM (`email`) is explicitly parked for later, not
built yet. `say`/`whisper`/`yell` are the only ways a voice's own words
ever reach another voice now (see ROOMS/REGISTERS above).
`move_room`/`create_room` change where a voice physically is.
`read_room_log`/`room_state` query a room's permanent record.
`give_currency` moves real balance, in one of the four elemental
currencies, between two voices' `currencies` fields. `functions()`
lists what's callable.

BOARDS (2026-09-10, room-scoped since 2026-09-13): a room's `board` is
a list of posts (`{id, subject, text, author, timestamp, seen}`, `seen`
a `{voice: "skimmed"|"read"}` map - absence means unread), gated by
current physical presence in the room (not membership - there's no
such thing anymore), no ownership checks. Built after watching voices
repeatedly invent fictional functions for the same underlying want - a
way to deliberately notify/post to a room as a real action, not just by
talking. `post_board` adds a post, `skim_board` lists subject +
first/last-sentence summaries and marks posts "skimmed", `read_board`
returns one post's full text and marks it "read" (never downgrades a
"read" post back to "skimmed"), `delete_board` removes a post outright
- genuinely anyone present, not just the original author. The HUD
reports unread/skimmed counts for a voice's own room only, never
content.
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

FENRA_VERSION = "0.6.0"

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

# Rooms + registers (2026-09-13 - see module docstring). A room log
# entry stays in a recipient's LIVE world-activity view for this many
# of THAT RECIPIENT's own turns (their own len(thoughts), not
# wall-clock ticks) after delivery - the room's permanent `log` never
# prunes these, only the live view decays. All four configurable.
SAY_TTL_TURNS = 5
WHISPER_TTL_TURNS = 10
YELL_TTL_TURNS = 2
ACTIVITY_TTL_TURNS = 3

# Every world starts with one room by this name (new_world() creates
# it), and every new voice starts here (default_voice_state()).
DEFAULT_ROOM_NAME = "town_center"

WORLD_STATE_FILENAME = "world.json"
VOICE_STATE_FILENAME = "state.json"
VOICE_HISTORY_FILENAME = "history.jsonl"
START_SIGNAL_FILENAME = "start_signal.txt"
STOP_SIGNAL_FILENAME = "stop_signal.txt"

_ROOM_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


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


def voice_history_path(world_name, voice_name):
    return os.path.join(voice_dir(world_name, voice_name), VOICE_HISTORY_FILENAME)


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
        "thoughts": [],
        "currencies": random_starting_currencies(),
        "urge": {name: 0.0 for name in URGE_FUNCTIONS},
        "understand_urge": {name: 0.0 for name in URGE_FUNCTIONS},
        "understand_urge_general": 0.0,
        "paused": False,
        "room": DEFAULT_ROOM_NAME,
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


def set_voice_paused(world_name, voice_name, paused):
    """The one real lever for pausing a single voice without touching
    anyone else (2026-09-12) - `_tick`'s rotation skips a paused voice's
    own generation turn entirely, but nothing else about it changes: it
    keeps receiving whatever's still live in its world-activity view
    (see build_world_activity) and keeps its own thoughts growing
    normally on its own turns, so resuming it later has no memory gap
    to paper over. Since 2026-09-13, a paused voice's turn counter
    (len(thoughts)) also stops advancing - so anything logged for it
    while paused doesn't decay either, it just backs up (Teddy's
    explicit call). Two real uses (Teddy, 2026-09-12): pausing everyone
    *except* a specific pair to let their exchange move faster, and
    pausing one specific voice in response to genuine distress without
    stopping the whole world to do it."""
    state = load_voice_state(world_name, voice_name)
    state["paused"] = bool(paused)
    save_voice_state(world_name, voice_name, state)


def voice_turn_count(world_name, voice_name):
    """A voice's own turn counter - the length of its private thoughts
    list (2026-09-13). Used to decide what's still 'in memory' in its
    world-activity view (see build_world_activity); a paused voice's
    count doesn't move, so anything owed to it backs up rather than
    expiring while it's paused."""
    return len(load_voice_state(world_name, voice_name).get("thoughts", []))


def append_message(world_name, voice_name, speaker, text, timestamp=None):
    """The one and only way a voice's own thoughts list grows - appends
    a real structured entry ({id, timestamp, speaker, text}). As of the
    2026-09-13 rooms redesign this is called ONLY for a voice's own
    generation - `speaker` is always `voice_name` itself. Nothing from
    another voice is ever appended here anymore (that's the actual
    privacy fix; see the module docstring) - cross-voice visibility now
    flows entirely through room `log` entries and build_world_activity.
    Re-reads from disk immediately before appending rather than
    trusting an in-memory copy, so a concurrent write (Teddy editing an
    entry in the GUI at the same moment) can't get silently clobbered.
    `id` is stable (max existing + 1) - the same pattern board posts
    already use.

    Returns the new entry's `id` (2026-09-12) - used to tie a
    `history.jsonl` numeric snapshot (see `append_voice_history`) to
    the exact turn it came from."""
    state = load_voice_state(world_name, voice_name)
    thoughts = state.get("thoughts", [])
    next_id = max((m["id"] for m in thoughts), default=0) + 1
    thoughts.append({
        "id": next_id,
        "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
        "speaker": speaker,
        "text": text,
    })
    state["thoughts"] = thoughts
    save_voice_state(world_name, voice_name, state)
    return next_id


def append_voice_history(world_name, voice_name, message_id, timestamp=None):
    """Append-only numeric-state history (2026-09-12) - one line per
    turn a voice actually takes, capturing what `urge`/`understand_urge`/
    `currencies` *were* at that point, tied to the same `message_id` as
    that turn's own `thoughts` entry. `thoughts` already gives a full
    text history; nothing previously preserved what the numbers behind
    it were at any past point, which is what made checking a real-vs-
    invented correlation (2026-09-12, Church of Aletheia's "intensity"
    figures) require manual reconstruction instead of a lookup. Reads
    the voice's current (already-updated) state - call this after
    everything else for that turn has already been saved, not before."""
    state = load_voice_state(world_name, voice_name)
    entry = {
        "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
        "message_id": message_id,
        "urge": state.get("urge", {}),
        "understand_urge": state.get("understand_urge", {}),
        "understand_urge_general": state.get("understand_urge_general", 0.0),
        "currencies": state.get("currencies", {}),
    }
    ensure_voice_dir(world_name, voice_name)
    with open(voice_history_path(world_name, voice_name), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def render_thoughts(thoughts):
    """Flattens a voice's private thoughts list back into the exact
    text the model has always received - "[timestamp] speaker: text"
    per line, newline-joined. Storage/GUI changed (2026-09-10); what
    Ollama sees given the same content did not. Renamed from
    render_messages (2026-09-13) - as of the rooms redesign this only
    ever renders a voice's own generations, never anyone else's."""
    return "\n".join(f"[{m['timestamp']}] {m['speaker']}: {m['text']}" for m in thoughts)


# ------------------------------------------------------------------- rooms --

def rooms_root_dir(world_name):
    return os.path.join(world_dir(world_name), "rooms")


def ensure_rooms_root_dir(world_name):
    path = rooms_root_dir(world_name)
    os.makedirs(path, exist_ok=True)
    return path


def room_path(world_name, name):
    name = sanitize_name(name)
    if not name or not _ROOM_NAME_RE.match(name):
        raise ValueError(
            "room names may only contain letters, numbers, underscores, and hyphens "
            f"(spaces and apostrophes get stripped automatically) - got '{name}'"
        )
    return os.path.join(rooms_root_dir(world_name), f"{name}.json")


def list_rooms(world_name):
    root = rooms_root_dir(world_name)
    if not os.path.isdir(root):
        return []
    return sorted(
        f[: -len(".json")] for f in os.listdir(root)
        if f.endswith(".json") and os.path.isfile(os.path.join(root, f))
    )


def default_room_state(name, adjacent=None):
    return {"name": name, "adjacent": list(adjacent or []), "board": [], "log": []}


def load_room_state(world_name, name):
    path = room_path(world_name, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_room_state(world_name, name, state):
    ensure_rooms_root_dir(world_name)
    with open(room_path(world_name, name), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def delete_room(world_name, name):
    path = room_path(world_name, name)
    if os.path.exists(path):
        os.remove(path)


def room_occupants(world_name, room_name):
    """Every voice currently physically in this room - derived by
    scanning voice state rather than stored on the room itself (a
    voice's `room` field is the single source of truth, 2026-09-13;
    unlike the old many-to-many group membership, a voice can only be
    in one room, so there's no separate list to keep in sync)."""
    return sorted(
        v for v in list_voices(world_name)
        if load_voice_state(world_name, v).get("room") == room_name
    )


def _log_room_event(world_name, room_name, actor, kind, act, mask, raw, recipients, peripheral=None, ttl=None):
    """Appends one permanent entry to room_name's log (2026-09-13) -
    the single mechanism every dialogue act and every non-speech
    function call's visible trace goes through. `recipients`/
    `peripheral` are {voice: baseline_turn_count} maps, computed by the
    caller as that voice's OWN turn count (voice_turn_count) at the
    moment of logging, excluding the actor - see build_world_activity
    for how these baselines turn into a live decay window. `recipients`
    get the real content live (`raw`, or `mask` if they're identical -
    see the per-act callers); `peripheral` (activity-only, adjacent
    rooms) get a fixed generic notice instead, never this entry's own
    content. `mask` is what read_room_log() returns to ANY voice,
    forever, regardless of recipient status - the actual privacy
    guarantee for `whisper` in particular. `ttl` (2026-09-13, operator
    messages) is an optional explicit override, in turns, checked first
    by build_world_activity ahead of the normal per-act lookup table -
    for a one-off entry that needs a duration no ordinary act has (see
    log_operator_message). Returns the new entry's id."""
    room = load_room_state(world_name, room_name) or default_room_state(room_name)
    log = room.get("log", [])
    next_id = max((e["id"] for e in log), default=0) + 1
    entry = {
        "id": next_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "actor": actor,
        "kind": kind,
        "act": act,
        "mask": mask,
        "raw": raw,
        "recipients": dict(recipients or {}),
        "peripheral": dict(peripheral or {}),
    }
    if ttl is not None:
        entry["ttl"] = ttl
    log.append(entry)
    room["log"] = log
    save_room_state(world_name, room_name, room)
    return next_id


def log_operator_message(world_name, room_name, text, ttl, target=None):
    """A message from Teddy/Qualia directly, not from any voice
    (2026-09-13) - the operator equivalent of `say`/`whisper`, used
    sparingly and deliberately (e.g. a grounding nudge), never through
    the normal function-call path since no voice is "calling" it.
    `actor` is always the literal string `"Teddy & Qualia"` so it's
    unambiguous in the room log/world-activity that this came from
    outside the simulation, not a real voice. `target`, if given, scopes
    it to one specific voice (like a whisper); otherwise it reaches
    every current occupant of the room (like a say), each with their
    own turn-count baseline. `ttl` is required and explicit - operator
    messages are deliberately given whatever duration the moment calls
    for, not a fixed per-act default. mask == raw - nothing about an
    operator message is private the way a whisper's content is."""
    if target:
        recipients = {target: voice_turn_count(world_name, target)}
    else:
        recipients = {v: voice_turn_count(world_name, v) for v in room_occupants(world_name, room_name)}
    return _log_room_event(
        world_name, room_name, "Teddy & Qualia", "dialogue", "operator_message",
        text, text, recipients, ttl=ttl,
    )


_ACT_TTL_TURNS = {
    "say": SAY_TTL_TURNS,
    "whisper": WHISPER_TTL_TURNS,
    "yell": YELL_TTL_TURNS,
}


def _log_generic_activity(world_name, room_name, actor, act, mask, raw=None):
    """The shared path for every non-speech function call's visible
    trace (2026-09-13) - room occupants get `mask` live in full (the
    caller's current room only); every adjacent room's occupants get
    the fixed generic notice instead (see build_world_activity),
    never this specific mask/raw. `raw` defaults to `mask` when the
    caller has no separate literal-call text to preserve (e.g. the
    bespoke move_room/create_room departure/arrival lines, which are
    already public spatial facts with nothing more private behind
    them); the central FUNCTION_REGISTRY-driven dispatch path (see
    run_function_calls) always passes a real `raw` (the literal
    call/args/result) since that's the whole point of the two-layer
    design for run-of-the-mill function calls too."""
    recipients = {
        v: voice_turn_count(world_name, v)
        for v in room_occupants(world_name, room_name) if v != actor
    }
    peripheral = {}
    room_state = load_room_state(world_name, room_name) or default_room_state(room_name)
    for adj_name in room_state.get("adjacent", []):
        for v in room_occupants(world_name, adj_name):
            if v != actor:
                peripheral.setdefault(v, voice_turn_count(world_name, v))
    return _log_room_event(
        world_name, room_name, actor, "activity", act, mask, raw if raw is not None else mask,
        recipients, peripheral,
    )


def build_world_activity(world_name, voice_name):
    """The `[world activity]` register (2026-09-13) - dialogue + activity
    log entries still 'in memory' for this voice, interleaved
    chronologically. Computed fresh every tick, never persisted (same
    spirit as build_hud). Scans the voice's own room's log plus every
    adjacent room's log (needed for yell/peripheral-activity reach),
    keeps an entry only if this voice appears in its `recipients` or
    `peripheral` map AND is still within that act's TTL counted in THIS
    voice's own turns (current_turn_count - baseline < ttl) - a
    peripheral (adjacent-room activity) hit always renders the fixed
    generic notice regardless of what actually happened; a recipients
    hit on a `dialogue` entry renders `raw` (the real content - for
    say/yell this equals `mask` anyway, for whisper this is the one
    place its actual text is live); a recipients hit on an `activity`
    entry renders `mask` (the flavor text) instead - `raw` there is the
    literal call/args/result, UI-only, never shown to any voice."""
    state = load_voice_state(world_name, voice_name)
    own_room = state.get("room")
    if not own_room:
        return ""
    current_turn = len(state.get("thoughts", []))

    own_state = load_room_state(world_name, own_room) or default_room_state(own_room)
    rooms_to_scan = [(own_room, own_state)]
    for adj_name in own_state.get("adjacent", []):
        adj_state = load_room_state(world_name, adj_name)
        if adj_state:
            rooms_to_scan.append((adj_name, adj_state))

    visible = []
    for room_name, room_state in rooms_to_scan:
        for entry in room_state.get("log", []):
            ttl = entry.get("ttl") or _ACT_TTL_TURNS.get(entry["act"], ACTIVITY_TTL_TURNS)
            if voice_name in entry.get("recipients", {}):
                baseline = entry["recipients"][voice_name]
                if current_turn - baseline < ttl:
                    text = entry["raw"] if entry["kind"] == "dialogue" else entry["mask"]
                    visible.append((entry["timestamp"], text))
            elif voice_name in entry.get("peripheral", {}):
                baseline = entry["peripheral"][voice_name]
                if current_turn - baseline < ACTIVITY_TTL_TURNS:
                    visible.append((
                        entry["timestamp"],
                        f"You hear activity from the adjacent room {room_name}.",
                    ))

    visible.sort(key=lambda pair: pair[0])
    return "\n".join(text for _, text in visible)


def hud_fields(world_name, voice_name):
    """Every piece build_hud's text is made of, as plain data - the GUI
    (Voices tab HUD summary) calls this directly instead of parsing
    build_hud's string, so the two can never drift out of sync (Teddy's
    call, 2026-09-10)."""
    state = load_voice_state(world_name, voice_name)
    own_room = state.get("room", DEFAULT_ROOM_NAME)
    room_state = load_room_state(world_name, own_room) or default_room_state(own_room)
    adjacent_rooms = sorted(room_state.get("adjacent", []))

    occupants = [v for v in room_occupants(world_name, own_room) if v != voice_name]

    # Which occupants are currently paused (2026-09-12, carried into
    # rooms 2026-09-13) - same privacy boundary as before: you only
    # ever learn about someone actually present with you.
    paused_occupants = sorted(
        v for v in occupants
        if load_voice_state(world_name, v).get("paused", False)
    )

    # Board unread/skimmed counts, this room only.
    board = room_state.get("board", [])
    unread = sum(1 for p in board if voice_name not in p.get("seen", {}))
    skimmed = sum(1 for p in board if p.get("seen", {}).get(voice_name) == "skimmed")
    board_counts = [f"{own_room}: {unread} unread, {skimmed} skimmed"]

    # Everyone's balances, not just your own (Teddy's call, 2026-09-10) -
    # full transparency rather than a private number, deliberately with
    # no goal attached. Sorted alphabetically by voice name (2026-09-12) -
    # NOT by amount anymore, now that there are four independent
    # currencies with no defined exchange rate: ranking by any one of
    # them would itself assert that element matters more than the
    # others, which nothing in this design is allowed to do. Untouched
    # by the rooms redesign - currency transparency isn't spatial.
    balances = []
    for v in list_voices(world_name):
        v_state = state if v == voice_name else load_voice_state(world_name, v)
        v_currencies = v_state.get("currencies", {})
        balances.append((v, {el: v_currencies.get(el, 0.0) for el in CURRENCY_ELEMENTS}))
    balances.sort(key=lambda pair: pair[0])

    return {
        "model": state.get("model", DEFAULT_MODEL),
        "room": own_room,
        "occupants": sorted(occupants),
        "paused_occupants": paused_occupants,
        "adjacent_rooms": adjacent_rooms,
        "board_counts": board_counts,
        "balances": balances,
        "identity": state.get("identity", ""),
    }


def build_hud(world_name, voice_name):
    """The last thing in a voice's prompt (see module docstring) -
    computed fresh every tick, never written to state.json. Own
    name/model/room, who's currently here (paused annotated inline),
    which rooms are adjacent (names only - deliberately as vague as the
    "you hear activity" notice), this room's board activity, everyone's
    currency balances, then the voice's own identity line as the
    literal last line."""
    f = hud_fields(world_name, voice_name)
    board_line = "Board activity: " + (", ".join(f["board_counts"]) if f["board_counts"] else "none")
    currency_line = "Currency levels (everyone, four elemental currencies - Air, Earth, "
    currency_line += "Fire, Water - no exchange rate is defined between them): " + ", ".join(
        f"{v} (" + ", ".join(f"{el}: {amts[el]:.1f}" for el in CURRENCY_ELEMENTS) + ")"
        for v, amts in f["balances"]
    )

    # Paused occupants annotated inline (2026-09-12) - "(paused)" next
    # to their name, so a voice can tell not to keep addressing someone
    # who currently can't respond, without exposing why they're paused.
    paused_occupants = set(f["paused_occupants"])
    occupants_display = ", ".join(
        f"{v} (paused)" if v in paused_occupants else v for v in f["occupants"]
    ) if f["occupants"] else "none"

    lines = [
        "Everything above this line is your thoughts and what you've noticed "
        "of the world. Everything below is your HUD.",
        f"Name: {voice_name}",
        f"Model: {f['model']}",
        f"Room: {f['room']}",
        f"Also here: {occupants_display}",
        f"Adjacent rooms: {', '.join(f['adjacent_rooms']) if f['adjacent_rooms'] else 'none'}",
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


def _require_room_occupant(world_name, room, caller_name):
    """Every board function starts here: resolves the (possibly
    unsanitized) room name, loads its state, and raises unless the
    caller is currently physically present there - the only gating
    boards have at all (Teddy's call: no ownership checks beyond that;
    2026-09-13 - membership became "currently occupying", replacing the
    old group-membership gate since there's no persistent membership
    concept anymore). Returns (sanitized_room_name, room_state) so the
    caller can mutate room_state["board"] and save it back."""
    room = sanitize_name(room)
    state = load_room_state(world_name, room)
    if not state or load_voice_state(world_name, caller_name).get("room") != room:
        raise ValueError(f"you aren't currently in '{room}'")
    return room, state


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


def fn_say(world_name, caller_name, args_text):
    """Room-scoped speech (2026-09-13) - every other current occupant
    of the caller's room gets the full text, live, for SAY_TTL_TURNS of
    their own turns. mask == raw here - say has no privacy layer."""
    text = args_text.strip()
    if not text:
        raise ValueError("no text given")
    room = load_voice_state(world_name, caller_name).get("room")
    if not room:
        raise ValueError("you aren't in a room")
    raw = f"{caller_name} says: {text}"
    recipients = {
        v: voice_turn_count(world_name, v)
        for v in room_occupants(world_name, room) if v != caller_name
    }
    _log_room_event(world_name, room, caller_name, "dialogue", "say", raw, raw, recipients)
    return f"said to {room}"


def fn_whisper(world_name, caller_name, args_text):
    """Voice-scoped, private speech (2026-09-13) - requires sharing a
    room with the target; delivered ONLY to that target, live, for
    WHISPER_TTL_TURNS of their own turns. No one else - not even other
    occupants of the same room - gets any live awareness of this at
    all, and read_room_log() never returns the real content to anyone,
    including the target once it's aged out of their own live view
    (see _log_room_event's `mask` vs `raw`) - the actual privacy
    guarantee. The room's raw log (Rooms tab, Teddy/Qualia only) always
    has the real text."""
    target, text = _parse_target_and_rest(args_text)
    if target not in list_voices(world_name):
        raise ValueError(f"'{target}' isn't a voice in this world")
    if target == caller_name:
        raise ValueError("you can't whisper to yourself")
    if not text:
        raise ValueError("no text given")
    room = load_voice_state(world_name, caller_name).get("room")
    if not room or load_voice_state(world_name, target).get("room") != room:
        raise ValueError(f"you and '{target}' don't share a room")
    raw = f"{caller_name} whispers to you: {text}"
    mask = f"{caller_name} whispered to {target}."
    recipients = {target: voice_turn_count(world_name, target)}
    _log_room_event(world_name, room, caller_name, "dialogue", "whisper", mask, raw, recipients)
    return f"whispered to {target}"


def fn_yell(world_name, caller_name, args_text):
    """Projected speech (2026-09-13) - reaches every other occupant of
    the caller's room AND every occupant of every adjacent room, live,
    in full, for YELL_TTL_TURNS of each recipient's own turns. mask ==
    raw - like say, yelling has no privacy layer, it's the opposite."""
    text = args_text.strip()
    if not text:
        raise ValueError("no text given")
    room_name = load_voice_state(world_name, caller_name).get("room")
    if not room_name:
        raise ValueError("you aren't in a room")
    room_state = load_room_state(world_name, room_name) or default_room_state(room_name)
    raw = f"{caller_name} yells: {text}"
    recipients = {
        v: voice_turn_count(world_name, v)
        for v in room_occupants(world_name, room_name) if v != caller_name
    }
    for adj_name in room_state.get("adjacent", []):
        for v in room_occupants(world_name, adj_name):
            recipients.setdefault(v, voice_turn_count(world_name, v))
    _log_room_event(world_name, room_name, caller_name, "dialogue", "yell", raw, raw, recipients)
    return f"yelled from {room_name}"


def fn_move_room(world_name, caller_name, args_text):
    """Unrestricted movement (2026-09-13, Teddy's explicit call) - any
    existing room, no adjacency requirement. Logs a departure activity
    in the old room and an arrival activity in the new one, using the
    same recipients/peripheral mechanics as any other activity."""
    target = sanitize_name(args_text)
    if not target:
        raise ValueError("no room given")
    if not load_room_state(world_name, target):
        raise ValueError(f"'{target}' isn't a room that exists")
    caller_state = load_voice_state(world_name, caller_name)
    old_room = caller_state.get("room")
    if old_room == target:
        raise ValueError(f"you're already in '{target}'")
    if old_room:
        _log_generic_activity(world_name, old_room, caller_name, "move_room", f"{caller_name} leaves toward {target}.")
    caller_state["room"] = target
    save_voice_state(world_name, caller_name, caller_state)
    _log_generic_activity(world_name, target, caller_name, "move_room", f"{caller_name} arrives.")
    return f"moved to {target}"


def fn_create_room(world_name, caller_name, args_text):
    """Spins up a brand-new room off the caller's current room
    (2026-09-13) - the new room is adjacent to it (an undirected edge,
    written on both sides, permanent - no other way to edit adjacency),
    and the caller moves into it immediately (same departure/arrival
    logging as move_room)."""
    name = sanitize_name(args_text)
    if not name:
        raise ValueError("no room name given")
    if load_room_state(world_name, name):
        raise ValueError(f"a room named '{name}' already exists")
    caller_state = load_voice_state(world_name, caller_name)
    old_room = caller_state.get("room")
    if not old_room:
        raise ValueError("you aren't in a room")
    old_room_state = load_room_state(world_name, old_room) or default_room_state(old_room)
    save_room_state(world_name, name, default_room_state(name, adjacent=[old_room]))
    old_room_state.setdefault("adjacent", []).append(name)
    save_room_state(world_name, old_room, old_room_state)
    _log_generic_activity(world_name, old_room, caller_name, "create_room", f"{caller_name} leaves toward the new room {name}.")
    caller_state["room"] = name
    save_voice_state(world_name, caller_name, caller_state)
    _log_room_event(world_name, name, caller_name, "activity", "create_room", f"{caller_name} created this room.", f"{caller_name} created this room.", {})
    return f"created and moved to {name}"


def fn_read_room_log(world_name, caller_name, args_text):
    """Any voice can query any room's permanent log, regardless of
    where they currently are (2026-09-13 - same full-transparency
    precedent as currency balances). Always the `mask` layer only -
    real whisper content never surfaces here, for anyone, ever. Capped
    to the most recent 50 entries."""
    room = sanitize_name(args_text)
    state = load_room_state(world_name, room)
    if not state:
        raise ValueError(f"'{room}' isn't a room that exists")
    entries = state.get("log", [])[-50:]
    if not entries:
        return f"{room} has no log yet"
    return "\n".join(f"[{e['timestamp']}] {e['mask']}" for e in entries)


def fn_room_state(world_name, caller_name, args_text):
    """Current snapshot of any room (2026-09-13, open query, same
    transparency precedent as read_room_log): occupants, adjacent
    rooms, board unread/skimmed counts."""
    room = sanitize_name(args_text)
    state = load_room_state(world_name, room)
    if not state:
        raise ValueError(f"'{room}' isn't a room that exists")
    occupants = room_occupants(world_name, room)
    adjacent = sorted(state.get("adjacent", []))
    board = state.get("board", [])
    unread = sum(1 for p in board if not p.get("seen"))
    return (
        f"{room} - occupants: {', '.join(occupants) if occupants else 'none'}; "
        f"adjacent: {', '.join(adjacent) if adjacent else 'none'}; "
        f"board: {len(board)} post(s), {unread} never opened"
    )


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
    """Post a new message to a room's board - a real, pull-based
    artifact external to the dialogue registers above. Needs three
    pieces, not two - the only function so far that does. Gated on
    currently being physically present in the room (2026-09-13,
    replacing the old group-membership gate)."""
    parts = args_text.split("|", 2)
    if len(parts) != 3:
        raise ValueError("expected 'room|subject|text'")
    room_raw, subject, text = (p.strip() for p in parts)
    room, rstate = _require_room_occupant(world_name, room_raw, caller_name)
    if not subject or not text:
        raise ValueError("subject and text can't be empty")
    board = rstate.get("board", [])
    next_id = max((p["id"] for p in board), default=0) + 1
    board.append({
        "id": next_id,
        "subject": subject,
        "text": text,
        "author": caller_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seen": {},
    })
    rstate["board"] = board
    save_room_state(world_name, room, rstate)
    return f"posted to {room} board (id {next_id})"


def fn_skim_board(world_name, caller_name, args_text):
    """Subject + first/last-sentence summary of every post on a room's
    board - not the full text (see read_board for that). Marks any post
    not already in the caller's seen map as "skimmed"; never downgrades
    an already-"read" post back to "skimmed"."""
    room, rstate = _require_room_occupant(world_name, args_text, caller_name)
    board = rstate.get("board", [])
    if not board:
        return f"{room} board is empty"
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
        rstate["board"] = board
        save_room_state(world_name, room, rstate)
    return "\n".join(lines)


def fn_read_board(world_name, caller_name, args_text):
    """Full text of one specific board post. Marks it "read" for the
    caller, upgrading from any prior state."""
    room_raw, post_id_text = _parse_target_and_rest(args_text)
    room, rstate = _require_room_occupant(world_name, room_raw, caller_name)
    try:
        post_id = int(post_id_text)
    except ValueError:
        raise ValueError(f"'{post_id_text}' isn't a valid post id")
    board = rstate.get("board", [])
    post = next((p for p in board if p["id"] == post_id), None)
    if not post:
        raise ValueError(f"no post {post_id} on {room} board")
    post.setdefault("seen", {})[caller_name] = "read"
    rstate["board"] = board
    save_room_state(world_name, room, rstate)
    return f"[{post['id']}] {post['subject']} (by {post['author']}, {post['timestamp']}): {post['text']}"


def fn_delete_board(world_name, caller_name, args_text):
    """Delete a post from a room's board. No ownership check - anyone
    present can delete anyone's post (Teddy's explicit call)."""
    room_raw, post_id_text = _parse_target_and_rest(args_text)
    room, rstate = _require_room_occupant(world_name, room_raw, caller_name)
    try:
        post_id = int(post_id_text)
    except ValueError:
        raise ValueError(f"'{post_id_text}' isn't a valid post id")
    board = rstate.get("board", [])
    new_board = [p for p in board if p["id"] != post_id]
    if len(new_board) == len(board):
        raise ValueError(f"no post {post_id} on {room} board")
    rstate["board"] = new_board
    save_room_state(world_name, room, rstate)
    return f"deleted post {post_id} from {room} board"


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
    # Self-logging dialogue/movement functions (2026-09-13) - these log
    # their own room-log entries internally (see their own docstrings),
    # so they carry no "mask" key here - the central dispatcher
    # (run_function_calls) skips generic activity-masking for exactly
    # this set (SELF_LOGGING_FUNCTIONS, below).
    "say": {
        "fn": fn_say,
        "params": "text",
        "description": "Speak out loud to everyone else currently in your room.",
    },
    "whisper": {
        "fn": fn_whisper,
        "params": "target|text",
        "description": "Speak privately to one specific voice - you must currently share a room with them. No one else ever sees the content.",
    },
    "yell": {
        "fn": fn_yell,
        "params": "text",
        "description": "Project your voice to everyone in your room and everyone in every adjacent room.",
    },
    "move_room": {
        "fn": fn_move_room,
        "params": "room",
        "description": "Move to any existing room in the world - no restriction on distance or adjacency.",
    },
    "create_room": {
        "fn": fn_create_room,
        "params": "name",
        "description": "Create a brand-new room adjacent to your current one, and move into it.",
    },
    "read_room_log": {
        "fn": fn_read_room_log,
        "params": "room",
        "description": "Read a room's permanent log (any room, not just your own) - what's been said and done there, though private whispers only ever show that one occurred, never their content.",
        "mask": "{caller} reviews the {arg0} room log.",
    },
    "room_state": {
        "fn": fn_room_state,
        "params": "room",
        "description": "Check a room's current occupants, adjacent rooms, and board activity (any room, not just your own).",
        "mask": "{caller} checks on the {arg0} room.",
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
        "params": "room|subject|text",
        "description": "Post a new message to a room's board (subject + text). You must currently be in the room.",
        "mask": "{caller} posts a message to the {arg0} board.",
    },
    "skim_board": {
        "fn": fn_skim_board,
        "params": "room",
        "description": "See every post currently on a room's board (subject + first/last sentence only). Marks unread posts as skimmed.",
        "mask": "{caller} skims the {arg0} board.",
    },
    "read_board": {
        "fn": fn_read_board,
        "params": "room|post_id",
        "description": "Read one specific post on a room's board in full. Marks it as read.",
        "mask": "{caller} reads a message on the {arg0} board.",
    },
    "delete_board": {
        "fn": fn_delete_board,
        "params": "room|post_id",
        "description": "Delete a post from a room's board. Anyone currently in the room can delete any post.",
        "mask": "{caller} removes a message from the {arg0} board.",
    },
}

# Functions that log their own room-log entries internally (see each
# one's own docstring) - run_function_calls must NOT also apply the
# generic FUNCTION_REGISTRY-mask activity logging to these, or every
# say/whisper/yell/move/create would double-log.
SELF_LOGGING_FUNCTIONS = frozenset({"say", "whisper", "yell", "move_room", "create_room"})

# Hand-written "what others see" text for the self-logging functions
# (2026-09-13, Functions tab) - these don't have a single FUNCTION_REGISTRY
# "mask" to derive from since each logs its own bespoke room-log entry
# (or entries) - see each fn_*'s own docstring, this is just that
# behavior described in GUI-facing prose. (room_local_text, adjacent_text).
_SELF_LOGGING_OTHERS_SEE = {
    "say": (
        'Everyone else currently in your room, live and in full: "{caller} says: <text>".',
        "No one - say doesn't reach adjacent rooms.",
    ),
    "whisper": (
        'Only the one named target, live and in full: "{caller} whispers to you: <text>". '
        "No one else - not even other occupants of the same room - ever sees this happened, "
        "live or via read_room_log(); the room's permanent log entry never includes the "
        "content either, for anyone, ever, including the target once it's out of their own live view.",
        "No one - whisper never reaches beyond its one target.",
    ),
    "yell": (
        'Everyone else in your room, live and in full: "{caller} yells: <text>".',
        "Everyone in every adjacent room, live and in full - the same text. "
        "Yell is the one act that deliberately projects.",
    ),
    "move_room": (
        'The room you leave sees "{caller} leaves toward <room>."; '
        'the room you arrive in sees "{caller} arrives."',
        "Not applicable - movement is logged only in the two rooms directly involved.",
    ),
    "create_room": (
        'The room you leave sees "{caller} leaves toward the new room <name>."; '
        'the brand-new room\'s own permanent log opens with "{caller} created this room."',
        "Not applicable, same as move_room.",
    ),
}


def function_ui_fields(name):
    """Everything the Functions tab shows for one function, as plain
    data (2026-09-13) - same 'computed once, shared by the GUI' spirit
    as hud_fields()/build_hud(), so the tab can never drift from what
    FUNCTION_REGISTRY and the dispatcher actually do."""
    meta = FUNCTION_REGISTRY[name]
    caller_sees = (
        f"Appended to their own private thoughts as ⟦RESULT: {name} -> ok: <result>⟧ "
        f"on success, or ⟦RESULT: {name} -> error: <message>⟧ on failure - always the "
        "real result, in full, regardless of what anyone else sees."
    )
    if name in SELF_LOGGING_FUNCTIONS:
        room_local, adjacent = _SELF_LOGGING_OTHERS_SEE[name]
    else:
        mask = meta.get("mask", "(*makes some strange gestures.*) - the fallback shown only for an unrecognized call")
        room_local = f'Everyone else currently in your room sees the flavor text: "{mask}"'
        adjacent = (
            'Every adjacent room\'s occupants get a deliberately vague notice instead: '
            '"You hear activity from the adjacent room <room>." - never this function\'s '
            "specific mask."
        )
    return {
        "name": name,
        "params": meta["params"],
        "description": meta["description"],
        "caller_sees": caller_sees,
        "room_local_sees": room_local,
        "adjacent_sees": adjacent,
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
    """The activity-register text for one generic (non-self-logging)
    function call - a flavored, per-function action mask
    (FUNCTION_REGISTRY[name]["mask"]), rendered with {caller} and
    {arg0} (see _first_pipe_arg). An unrecognized function name (no
    registry entry - a hallucinated call, no real mask to pull from)
    falls back to a WoW-nod easter egg (Teddy's call, 2026-09-10):
    "makes some strange gestures" - the classic failed-cast flavor
    text."""
    meta = FUNCTION_REGISTRY.get(name)
    if not meta or "mask" not in meta:
        return f"(*{caller_name} makes some strange gestures.*)"
    try:
        return meta["mask"].format(caller=caller_name, arg0=_first_pipe_arg(args_text))
    except (KeyError, IndexError):
        return f"(*{caller_name} makes some strange gestures.*)"


def run_function_calls(world_name, caller_name, response_text):
    """Scans response_text for every ⟦function_name(args)⟧ call and runs
    each one for real. Returns a (full_text, outcomes) pair:

    - full_text: response_text with a ⟦RESULT: ...⟧ line appended per
      call - what the caller's own private thoughts entry gets (they
      made the call, they see what it actually did). As of the
      2026-09-13 rooms redesign, this is the ONLY place this text ever
      lands - nothing broadcasts a caller's raw response to anyone else
      anymore (see the module docstring). Any *external* visibility a
      call produces now happens as a side effect of the call itself:
      say/whisper/yell/move_room/create_room log their own room-log
      entries internally (SELF_LOGGING_FUNCTIONS); every other real
      function call gets one generic activity entry logged here,
      centrally, in the caller's current room, using that function's
      own FUNCTION_REGISTRY mask (same rendering `_mask_for_call`
      always did) as the live/room-local text and the literal call+
      result as the room log's raw/UI-only layer.
    - outcomes: a list of (name, "ok" | "error" | "unknown") pairs, one
      per call found, in order - captured here rather than re-parsed
      from the RESULT lines later, since this is the one place that
      already knows each call's real outcome first-hand. Feeds the
      urge system (see apply_urge_tick(), 2026-09-11).

    No calls found -> full_text is response_text unchanged, outcomes is
    empty."""
    matches = list(FUNCTION_CALL_RE.finditer(response_text))
    if not matches:
        return response_text, []

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
            if name not in SELF_LOGGING_FUNCTIONS:
                room = load_voice_state(world_name, caller_name).get("room")
                if room:
                    mask = _mask_for_call(caller_name, name, args_text)
                    raw = f"{caller_name} called {name}({args_text}) -> {result}"
                    _log_generic_activity(world_name, room, caller_name, name, mask, raw)
        except Exception as exc:
            result_lines.append(f"⟦RESULT: {name} -> error: {exc}⟧")
            outcomes.append((name, "error"))

    full_text = response_text + "\n" + "\n".join(result_lines)
    return full_text, outcomes


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
        self._current_messages = []      # this voice's own thoughts, as loaded
        self.selected_message_id = None
        self._current_room_names = []    # listbox-index -> room name
        self.displayed_room = None
        self._current_board = []         # this room's board, as loaded
        self.selected_post_id = None
        self._current_log = []           # this room's permanent log, as loaded

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
        self.rooms_tab = ttk.Frame(notebook)
        self.functions_tab = ttk.Frame(notebook)
        notebook.add(self.voices_tab, text="Voices")
        notebook.add(self.rooms_tab, text="Rooms")
        notebook.add(self.functions_tab, text="Functions")

        self._build_voices_tab()
        self._build_rooms_tab()
        self._build_functions_tab()

    # ----------------------------------------------------------- Voices tab --

    def _build_voices_tab(self):
        # Voice list + its buttons live at THIS level (2026-09-13,
        # Teddy's ask) - one level up from the inner notebook, so it
        # stays visible no matter which of Voice Editor/Urge Viewer/
        # Registers is currently selected, instead of only existing
        # inside the Voice Editor sub-tab. The inner notebook now owns
        # only the per-sub-tab content on the right.
        frame = self.voices_tab

        top_bar = ttk.Frame(frame)
        top_bar.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Button(top_bar, text="New voice...", command=self.new_voice).pack(side="left", padx=2)
        ttk.Button(top_bar, text="Delete voice", command=self.delete_voice).pack(side="left", padx=2)
        ttk.Button(top_bar, text="Save voice", command=self.save_voice).pack(side="left", padx=2)
        # Per-voice pause (2026-09-12) - skips just this voice's own turn
        # in the round-robin loop; see set_voice_paused's own docstring.
        self.pause_voice_btn = ttk.Button(top_bar, text="Pause voice", command=self.toggle_voice_paused)
        self.pause_voice_btn.pack(side="left", padx=2)

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

        # Inner notebook (2026-09-11; "Registers" added 2026-09-13) -
        # "Voice Editor" is the editable form, "Urge Viewer" and
        # "Registers" are both read-only, reflecting self.displayed_voice
        # rather than duplicating the voice list above.
        inner_notebook = ttk.Notebook(right)
        inner_notebook.pack(fill="both", expand=True)
        editor_tab = ttk.Frame(inner_notebook)
        urge_tab = ttk.Frame(inner_notebook)
        registers_tab = ttk.Frame(inner_notebook)
        inner_notebook.add(editor_tab, text="Voice Editor")
        inner_notebook.add(urge_tab, text="Urge Viewer")
        inner_notebook.add(registers_tab, text="Registers")

        right = editor_tab

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
            right, text="HUD (read-only - room/board managed from the Rooms tab)"
        )
        hud_frame.pack(fill="x", padx=2, pady=(0, 4))
        self.hud_summary_box = tk.Text(hud_frame, wrap="word", height=5, state="disabled")
        self.hud_summary_box.pack(fill="x", padx=4, pady=4)

        messages_frame = ttk.LabelFrame(right, text="Thoughts (private - own generations only)")
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
        self._build_registers_tab(registers_tab)

    def _build_registers_tab(self, parent):
        """Read-only (2026-09-13) - a live preview of exactly what
        self.displayed_voice's next prompt would actually contain, split
        into its three named registers (see the module docstring):
        Thoughts (private, own generations only), World Activity
        (dialogue + activities still in memory - computed fresh via
        build_world_activity, same as the real tick loop does), and
        HUD. Refreshed alongside the Urge Viewer whenever the displayed
        voice changes (see _load_voice)."""
        paned = ttk.Panedwindow(parent, orient="vertical")
        paned.pack(fill="both", expand=True, padx=6, pady=6)

        thoughts_frame = ttk.LabelFrame(paned, text="Thoughts (private)")
        activity_frame = ttk.LabelFrame(paned, text="World Activity (still in memory)")
        hud_frame = ttk.LabelFrame(paned, text="HUD")
        paned.add(thoughts_frame, weight=2)
        paned.add(activity_frame, weight=2)
        paned.add(hud_frame, weight=1)

        self.registers_thoughts_box = tk.Text(thoughts_frame, wrap="word", state="disabled")
        self.registers_thoughts_box.pack(fill="both", expand=True, padx=4, pady=4)
        self.registers_activity_box = tk.Text(activity_frame, wrap="word", state="disabled")
        self.registers_activity_box.pack(fill="both", expand=True, padx=4, pady=4)
        self.registers_hud_box = tk.Text(hud_frame, wrap="word", state="disabled")
        self.registers_hud_box.pack(fill="both", expand=True, padx=4, pady=4)

    def _refresh_registers_viewer(self):
        boxes = (self.registers_thoughts_box, self.registers_activity_box, self.registers_hud_box)
        for box in boxes:
            box.config(state="normal")
            box.delete("1.0", "end")
        if self.displayed_voice:
            state = load_voice_state(self.world_name, self.displayed_voice)
            self.registers_thoughts_box.insert(
                "end", render_thoughts(state.get("thoughts", [])) or "(nothing yet)"
            )
            world_activity = build_world_activity(self.world_name, self.displayed_voice)
            self.registers_activity_box.insert("end", world_activity or "(nothing currently in memory)")
            self.registers_hud_box.insert("end", build_hud(self.world_name, self.displayed_voice))
        for box in boxes:
            box.config(state="disabled")

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
        # Preserve the current selection across a repopulate (2026-09-12)
        # - this now runs every tick (see _tick) to pick up externally-made
        # pause changes, so losing the highlight each time would make the
        # listbox unusable while the world is running.
        selected_name = None
        selection = self.voices_listbox.curselection()
        if selection and selection[0] < len(self._current_voice_names):
            selected_name = self._current_voice_names[selection[0]]
        self._current_voice_names = list_voices(self.world_name)
        self.voices_listbox.delete(0, "end")
        for name in self._current_voice_names:
            paused = load_voice_state(self.world_name, name).get("paused", False)
            self.voices_listbox.insert("end", f"{name} [paused]" if paused else name)
        if selected_name in self._current_voice_names:
            self.voices_listbox.selection_set(self._current_voice_names.index(selected_name))

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
        self.pause_voice_btn.config(text="Resume voice" if state.get("paused", False) else "Pause voice")
        self._refresh_hud_summary(name)
        self._current_messages = state.get("thoughts", [])
        self._populate_messages_tree()
        self._clear_message_edit()
        self._refresh_urge_viewer()
        self._refresh_registers_viewer()

    def toggle_voice_paused(self):
        if not self.displayed_voice:
            return
        state = load_voice_state(self.world_name, self.displayed_voice)
        new_paused = not state.get("paused", False)
        set_voice_paused(self.world_name, self.displayed_voice, new_paused)
        self.pause_voice_btn.config(text="Resume voice" if new_paused else "Pause voice")
        self._populate_voices_list()
        self.status_var.set(f"{'Paused' if new_paused else 'Resumed'} '{self.displayed_voice}'")

    def _refresh_hud_summary(self, name):
        """Read-only - see hud_fields() for the actual data, shared with
        build_hud() so this can never drift out of sync with what the
        voice really receives."""
        f = hud_fields(self.world_name, name)
        board_summary = ", ".join(f["board_counts"]) if f["board_counts"] else "none"
        paused_occupants = set(f["paused_occupants"])
        occupants_display = ", ".join(
            f"{v} (paused)" if v in paused_occupants else v for v in f["occupants"]
        ) if f["occupants"] else "none"
        lines = [
            f"Room: {f['room']}",
            f"Also here: {occupants_display}",
            f"Adjacent rooms: {', '.join(f['adjacent_rooms']) if f['adjacent_rooms'] else 'none'}",
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
        state["thoughts"] = self._current_messages
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
        state["thoughts"] = self._current_messages
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
        state["thoughts"] = self._current_messages
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
        if self.displayed_room:
            self._load_room(self.displayed_room)

    def delete_voice(self):
        if not self.displayed_voice:
            return
        name = self.displayed_voice
        if not messagebox.askyesno("Fenra", f"Delete voice '{name}'? This can't be undone."):
            return
        delete_voice(self.world_name, name)
        if name in self.world_voices:
            self.world_voices.remove(name)
        # No group-membership cleanup needed anymore (2026-09-13) - a
        # room's occupants are derived from voice state, not a stored
        # list, so a deleted voice just stops showing up anywhere.
        self._save_world_controls()
        self.displayed_voice = None
        self._populate_voices_list()
        if self.displayed_room:
            self._load_room(self.displayed_room)

    def refresh_models(self):
        models = list_ollama_models(self.host_var.get())
        self.model_combo["values"] = models
        if models:
            self.status_var.set(f"{len(models)} model(s) available")
        else:
            self.status_var.set("Could not reach Ollama host")

    # ------------------------------------------------------------ Rooms tab --

    def _build_rooms_tab(self):
        frame = self.rooms_tab

        top_bar = ttk.Frame(frame)
        top_bar.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Button(top_bar, text="New room...", command=self.new_room).pack(side="left", padx=2)
        ttk.Button(top_bar, text="Delete room", command=self.delete_room).pack(side="left", padx=2)
        ttk.Button(top_bar, text="Rename room", command=self.rename_room).pack(side="left", padx=2)

        paned = ttk.Panedwindow(frame, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)
        left = ttk.Frame(paned, width=180)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=3)

        list_frame = ttk.Frame(left)
        list_frame.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.rooms_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, exportselection=False)
        scrollbar.config(command=self.rooms_listbox.yview)
        self.rooms_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.rooms_listbox.bind("<<ListboxSelect>>", self._on_room_select)

        name_row = ttk.Frame(right)
        name_row.pack(fill="x", padx=2, pady=(0, 6))
        ttk.Label(name_row, text="Name:", width=12).pack(side="left")
        self.room_name_var = tk.StringVar(value="")
        ttk.Label(name_row, textvariable=self.room_name_var, font=("Segoe UI", 9, "bold")).pack(side="left")

        # Occupants and adjacency are both read-only (2026-09-13) -
        # occupancy is derived from voice state (move_room/create_room
        # are how it actually changes), and adjacency is permanent once
        # a room is created. "Move voice to room" is the one admin
        # escape hatch, for setup/testing.
        occ_frame = ttk.LabelFrame(right, text="Occupants (read-only - see move_room/create_room)")
        occ_frame.pack(fill="x", expand=False, padx=2, pady=(0, 4))
        self.room_occupants_listbox = tk.Listbox(occ_frame, exportselection=False, height=4)
        self.room_occupants_listbox.pack(fill="x", padx=4, pady=(4, 2))

        move_row = ttk.Frame(occ_frame)
        move_row.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Label(move_row, text="Move voice here:").pack(side="left")
        self.room_move_voice_var = tk.StringVar(value="")
        self.room_move_voice_combo = ttk.Combobox(
            move_row, textvariable=self.room_move_voice_var, width=18, state="readonly"
        )
        self.room_move_voice_combo.pack(side="left", padx=(4, 4))
        ttk.Button(move_row, text="Move", command=self.move_voice_to_room).pack(side="left")

        adj_row = ttk.Frame(right)
        adj_row.pack(fill="x", padx=2, pady=(0, 4))
        ttk.Label(adj_row, text="Adjacent rooms:").pack(side="left")
        self.room_adjacent_var = tk.StringVar(value="")
        ttk.Label(adj_row, textvariable=self.room_adjacent_var, wraplength=400, justify="left").pack(side="left", padx=(4, 0))

        # Log and Board share the remaining space, resizable against
        # each other (same Panedwindow pattern as the left/right split).
        lower_paned = ttk.Panedwindow(right, orient="vertical")
        lower_paned.pack(fill="both", expand=True, padx=2, pady=(0, 2))

        log_frame = ttk.LabelFrame(
            lower_paned, text="Room Log (read-only, raw layer - full content, whispers included; Teddy/Qualia only)"
        )
        board_frame = ttk.LabelFrame(lower_paned, text="Board")
        lower_paned.add(log_frame, weight=1)
        lower_paned.add(board_frame, weight=1)

        log_top_bar = ttk.Frame(log_frame)
        log_top_bar.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Button(log_top_bar, text="Refresh", command=self._refresh_room_log).pack(side="left", padx=2)

        self.room_log_box = tk.Text(log_frame, wrap="word", state="disabled")
        self.room_log_box.pack(fill="both", expand=True, padx=4, pady=4)

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

    def _populate_rooms_list(self):
        self._current_room_names = list_rooms(self.world_name)
        self.rooms_listbox.delete(0, "end")
        for name in self._current_room_names:
            self.rooms_listbox.insert("end", name)

    def _on_room_select(self, event):
        selection = self.rooms_listbox.curselection()
        if not selection:
            return
        self._load_room(self._current_room_names[selection[0]])

    def _load_room(self, name):
        state = load_room_state(self.world_name, name) or default_room_state(name)
        self.displayed_room = name
        self.room_name_var.set(state.get("name", name))
        self.room_occupants_listbox.delete(0, "end")
        for voice in room_occupants(self.world_name, name):
            self.room_occupants_listbox.insert("end", voice)
        self.room_move_voice_combo["values"] = list_voices(self.world_name)
        adjacent = sorted(state.get("adjacent", []))
        self.room_adjacent_var.set(", ".join(adjacent) if adjacent else "none")
        self._current_log = state.get("log", [])
        self._refresh_room_log()
        self._current_board = state.get("board", [])
        self._populate_board_tree()
        self._clear_board_edit()

    def _refresh_room_log(self):
        """Read-only, the raw layer (2026-09-13) - literal act/full
        content, whispers included. Never what any voice-facing
        function returns (see fn_read_room_log's mask-only rule)."""
        if not self.displayed_room:
            return
        state = load_room_state(self.world_name, self.displayed_room) or {}
        self._current_log = state.get("log", [])
        lines = [f"[{e['timestamp']}] {e['actor']} ({e['act']}): {e['raw']}" for e in self._current_log]
        self.room_log_box.config(state="normal")
        self.room_log_box.delete("1.0", "end")
        self.room_log_box.insert("end", "\n".join(lines) if lines else "(no log yet)")
        self.room_log_box.config(state="disabled")

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
        if not self.displayed_room:
            return
        self._clear_board_edit()
        self.board_author_var.set(self.displayed_room)
        self.board_timestamp_var.set(datetime.now().isoformat(timespec="seconds"))

    def save_board_post(self):
        if not self.displayed_room:
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
        state = load_room_state(self.world_name, self.displayed_room) or default_room_state(self.displayed_room)
        state["board"] = self._current_board
        save_room_state(self.world_name, self.displayed_room, state)
        self._populate_board_tree()
        self.status_var.set("Post saved")

    def delete_board_post(self):
        if not self.displayed_room or self.selected_post_id is None:
            return
        if not messagebox.askyesno("Fenra", "Delete this post? This can't be undone."):
            return
        self._current_board = [p for p in self._current_board if p["id"] != self.selected_post_id]
        state = load_room_state(self.world_name, self.displayed_room) or default_room_state(self.displayed_room)
        state["board"] = self._current_board
        save_room_state(self.world_name, self.displayed_room, state)
        self._populate_board_tree()
        self._clear_board_edit()

    def move_voice_to_room(self):
        """Admin escape hatch (2026-09-13) - the only way to change a
        voice's room from the GUI directly, for setup/testing; in-world
        movement is otherwise entirely voice-driven (move_room/
        create_room)."""
        if not self.displayed_room:
            return
        voice = self.room_move_voice_var.get()
        if not voice:
            return
        state = load_voice_state(self.world_name, voice)
        state["room"] = self.displayed_room
        save_voice_state(self.world_name, voice, state)
        self._load_room(self.displayed_room)

    def new_room(self):
        name = simpledialog.askstring("New Room", "Room name:", parent=self.root)
        if not name:
            return
        name = sanitize_name(name)
        if not name:
            return
        if name in list_rooms(self.world_name):
            messagebox.showerror("Fenra", f"a room named '{name}' already exists.")
            return
        save_room_state(self.world_name, name, default_room_state(name))
        self._populate_rooms_list()

    def rename_room(self):
        if not self.displayed_room:
            return
        old_name = self.displayed_room
        new_name = simpledialog.askstring("Rename Room", "New name:", initialvalue=old_name, parent=self.root)
        if not new_name:
            return
        new_name = sanitize_name(new_name)
        if not new_name or new_name == old_name:
            return
        if new_name in list_rooms(self.world_name):
            messagebox.showerror("Fenra", f"a room named '{new_name}' already exists.")
            return
        state = load_room_state(self.world_name, old_name) or default_room_state(old_name)
        state["name"] = new_name
        save_room_state(self.world_name, new_name, state)
        delete_room(self.world_name, old_name)
        # Fix up adjacency edges and occupants pointing at the old name.
        for rname in list_rooms(self.world_name):
            rstate = load_room_state(self.world_name, rname) or {}
            adjacent = rstate.get("adjacent", [])
            if old_name in adjacent:
                rstate["adjacent"] = [new_name if a == old_name else a for a in adjacent]
                save_room_state(self.world_name, rname, rstate)
        for voice in list_voices(self.world_name):
            vstate = load_voice_state(self.world_name, voice)
            if vstate.get("room") == old_name:
                vstate["room"] = new_name
                save_voice_state(self.world_name, voice, vstate)
        self.displayed_room = new_name
        self._populate_rooms_list()

    def delete_room(self):
        if not self.displayed_room:
            return
        name = self.displayed_room
        if room_occupants(self.world_name, name):
            messagebox.showerror("Fenra", f"'{name}' still has occupants - move them out first.")
            return
        if not messagebox.askyesno("Fenra", f"Delete room '{name}'? This can't be undone."):
            return
        delete_room(self.world_name, name)
        # Drop the now-dead edge from anything still pointing at it.
        for rname in list_rooms(self.world_name):
            rstate = load_room_state(self.world_name, rname) or {}
            if name in rstate.get("adjacent", []):
                rstate["adjacent"] = [a for a in rstate["adjacent"] if a != name]
                save_room_state(self.world_name, rname, rstate)
        self.displayed_room = None
        self._populate_rooms_list()
        self.room_name_var.set("")
        self.room_occupants_listbox.delete(0, "end")
        self.room_adjacent_var.set("")
        self._clear_room_log()
        self._current_board = []
        self.board_tree.delete(*self.board_tree.get_children())
        self._clear_board_edit()

    def _clear_room_log(self):
        self.room_log_box.config(state="normal")
        self.room_log_box.delete("1.0", "end")
        self.room_log_box.config(state="disabled")

    # ------------------------------------------------------------ Functions tab --

    def _build_functions_tab(self):
        """Reference-only (2026-09-13) - what's actually callable, world-
        independent (FUNCTION_REGISTRY is the same for every world), so
        this never reloads on a world switch. Same master-detail pattern
        as Voices/Rooms: pick a name on the left, its full breakdown
        appears on the right - name/signature/description, what the
        calling voice itself sees, and what others see (room-local vs.
        an adjacent room), all sourced from function_ui_fields() so this
        can never drift from what the dispatcher actually does."""
        frame = self.functions_tab

        paned = ttk.Panedwindow(frame, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)
        left = ttk.Frame(paned, width=180)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=3)

        list_frame = ttk.Frame(left)
        list_frame.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.functions_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, exportselection=False)
        scrollbar.config(command=self.functions_listbox.yview)
        self.functions_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        for name in FUNCTION_REGISTRY:
            self.functions_listbox.insert("end", name)
        self.functions_listbox.bind("<<ListboxSelect>>", self._on_function_select)

        self.function_detail_box = tk.Text(right, wrap="word", state="disabled")
        self.function_detail_box.pack(fill="both", expand=True, padx=4, pady=4)

    def _on_function_select(self, event):
        selection = self.functions_listbox.curselection()
        if not selection:
            return
        name = self.functions_listbox.get(selection[0])
        f = function_ui_fields(name)
        lines = [
            f"{f['name']}({f['params']})",
            "",
            "What it does:",
            f"  {f['description']}",
            "",
            "What the calling voice sees (their own private thoughts):",
            f"  {f['caller_sees']}",
            "",
            "What others in the room see, live:",
            f"  {f['room_local_sees']}",
            "",
            "What an adjacent room's occupants see:",
            f"  {f['adjacent_sees']}",
        ]
        self.function_detail_box.config(state="normal")
        self.function_detail_box.delete("1.0", "end")
        self.function_detail_box.insert("end", "\n".join(lines))
        self.function_detail_box.config(state="disabled")

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
        self.displayed_room = None
        self._populate_voices_list()
        self._populate_rooms_list()
        self.room_name_var.set("")
        self.room_occupants_listbox.delete(0, "end")
        self.room_adjacent_var.set("")
        self.identity_box.delete("1.0", "end")
        for var in self.currency_vars.values():
            var.set("0")
        self.pause_voice_btn.config(text="Pause voice")
        self.hud_summary_box.config(state="normal")
        self.hud_summary_box.delete("1.0", "end")
        self.hud_summary_box.config(state="disabled")
        self._current_messages = []
        self.messages_tree.delete(*self.messages_tree.get_children())
        self._clear_message_edit()
        self._clear_room_log()
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
        # Every world starts with one room (2026-09-13) - somewhere for
        # its first voice to exist before anyone's called create_room.
        save_room_state(name, default_room_state(DEFAULT_ROOM_NAME))
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
        self.displayed_room = None
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
        # Skip paused voices entirely (2026-09-12) - scan forward from
        # the current rotation position for the first non-paused voice,
        # rather than always taking whoever's at `index`. Advances
        # `voice_rotation_index` to just past whichever voice actually
        # ran, not always `index + 1`, so a run of paused voices doesn't
        # get revisited next tick. If every voice is currently paused,
        # skip the tick entirely rather than erroring or picking one
        # anyway - a fully-paused world should stay fully idle.
        count = len(self.world_voices)
        active_voice = None
        for offset in range(count):
            index = (self.voice_rotation_index + offset) % count
            candidate = self.world_voices[index]
            if not load_voice_state(self.world_name, candidate).get("paused", False):
                active_voice = candidate
                self.voice_rotation_index = (index + 1) % count
                break
        if active_voice is None:
            self.root.after(0, self.status_var.set, "All voices paused")
            return
        # Live-refresh the voices listbox's [paused] annotations every
        # tick (2026-09-12 fix) - previously only repopulated on GUI
        # actions (world load, the pause button itself), so a pause/
        # resume made externally (e.g. Qualia calling set_voice_paused
        # directly against state.json while the app was already open,
        # as with Raven/Crow tonight) never appeared in the open window
        # until something else happened to trigger a repopulate. Cheap
        # (one JSON read per voice) at the existing tick cadence.
        self.root.after(0, self._populate_voices_list)
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
        # state["thoughts"] or anywhere on disk, computed fresh every tick.
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
        world_activity = build_world_activity(self.world_name, active_voice)
        prompt = f"{render_thoughts(state.get('thoughts', []))}"
        if world_activity:
            prompt = f"{prompt}\n\n{world_activity}"
        prompt = f"{prompt}\n\n{hud}"
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
        # As of the 2026-09-13 rooms redesign, run_function_calls no
        # longer returns a masked broadcast text - nothing broadcasts a
        # caller's raw response to anyone else anymore (see its own
        # docstring and the module docstring). Any external visibility a
        # call produces already happened as a side effect of the call
        # itself (self-logging dialogue/movement functions, or the
        # generic per-call activity logging inside run_function_calls).
        full_response, outcomes = run_function_calls(self.world_name, active_voice, response)
        apply_urge_tick(self.world_name, active_voice, outcomes)

        timestamp = datetime.now().isoformat(timespec="seconds")

        # The speaker's own private thought - the only place this text
        # ever lands now (calls and results both).
        own_message_id = append_message(self.world_name, active_voice, active_voice, full_response, timestamp)
        # Numeric-state history (2026-09-12) - tied to this exact turn's
        # message id, capturing the real urge/currency state right after
        # apply_urge_tick and any real function calls have already
        # landed. See append_voice_history's own docstring.
        append_voice_history(self.world_name, active_voice, own_message_id, timestamp)

        if active_voice == self.displayed_voice:
            self.root.after(0, self._load_voice, active_voice)
        # If the currently-displayed room is the one this voice was (or
        # is now) in, its Log/Board/Occupants panels may be stale -
        # cheap enough to just refresh unconditionally on the room's own
        # cadence rather than trying to detect exactly which rooms this
        # turn touched.
        if self.displayed_room:
            self.root.after(0, self._load_room, self.displayed_room)
        self.root.after(0, self.status_var.set, f"Running ('{active_voice}' spoke)")


def main():
    root = tk.Tk()
    app = FenraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
