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
rooms), and its own identity line as the literal last line - plus,
only when present, whatever the function agent's own last turn wrote
about her, shown exactly once (2026-09-14, see FUNCTION AGENT below).
No call-syntax reminder anymore - that job belongs entirely to the
separate function agent's own HUD (`build_function_agent_hud`), never
the voice's own. `hud_fields()` returns the same pieces as plain data,
not text - `build_hud` formats them, and the GUI's read-only HUD
summary (Voices tab) calls the identical function, so the two can
never drift apart.

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

FUNCTIONS: reintroduced 2026-09-09 using literal `⟦function_name(args)⟧`
call syntax parsed out of a voice's own text; redesigned 2026-09-14 into
a three-agent turn (urge agent -> voice -> function agent) - see FUNCTION
AGENT below. `FUNCTION_REGISTRY` shape is unchanged: no permission layer
(every voice can act on everything), no `functions.jsonl` logging, no
fabrication-detection. `send_message` (the old free, non-room-gated DM)
is gone as of 2026-09-13 - `whisper` is its room-gated replacement; a
true non-room-gated DM (`email`) is explicitly parked for later, not
built yet. `say`/`whisper`/`yell` are the only ways a voice's own words
ever reach another voice now (see ROOMS/REGISTERS above).
`move_room`/`create_room` change where a voice physically is.
`read_room_log`/`room_state` query a room's permanent record.
`give_currency` moves real balance, in one of the four elemental
currencies, between two voices' `currencies` fields.

FUNCTION AGENT (2026-09-14): a voice no longer knows functions exist at
all - no call syntax, no `functions()` introspection (removed entirely),
nothing mechanical in her own prompt. She just generates prose, out of
her own felt urges. A separate small model (`DEFAULT_FUNCTION_AGENT_MODEL`,
picked after real stress-testing - see `Qualia/Function Agent Testing/`)
reads that prose plus real grounding data (`build_function_agent_hud`)
and real felt-urge text, and decides what, if anything, actually happens
in the world, using Ollama's native tool-calling API rather than a
text-syntax parse (`run_function_agent_turn`/`dispatch_one_function_call`).
A real dispatcher error (missing/invalid arg, nonexistent room/voice,
etc.) triggers an internal retry, capped, feeding the real error back -
a call that already succeeded is never re-sent or re-executed. A
well-formed-but-semantically-wrong call, or a retry-cap exhaustion,
just stands as-is - no special-casing, no fabricated feedback. Whatever
text the function agent itself wrote (if anything) surfaces into that
one voice's very next HUD only, then is cleared - the two agents "talk"
through the normal loop, nothing engineered or sanitized about the
wording. `understand_urge` (the old tracker for a voice's own malformed
call attempts) was removed along with this redesign - it can't happen
anymore since a voice never attempts a call herself.

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
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, scrolledtext, simpledialog, ttk

import requests

import fenra_hosts

FENRA_VERSION = "0.19.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORLDS_DIR = os.path.join(BASE_DIR, "worlds")

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3"
DEFAULT_INTERVAL_SEC = 3
DEFAULT_WORLD_NAME = "default"
DEFAULT_VOICE_NAME = "voice1"

# Distributed-compute host claiming (2026-09-18 - see
# Communications/client-server-plan.md for the full design, Vero has the
# matching client-side half). A voice's whole turn - urge agent, voice,
# function agent - must run on exactly one host, claimed once at the
# start of the turn and never re-selected mid-turn; otherwise a voice's
# own urge and voice calls could land on two different machines, which
# was Teddy's explicit, named concern designing this. `_host_claims` is
# process-local, in-memory only (not persisted - a claim only matters
# for the lifetime of one in-flight turn). `_host_registry_lock` guards
# it; cheap, held only for the instant of claiming/releasing, never
# across an actual network call.
#
# Remote volunteer machines (fenra_hosts.py - the server half of the
# contract Vero's fenra_client implements) appear here as hosts named
# "remote://<label>". claim_host_for_voice offers an eligible idle remote
# host first (offloading is the point) and falls back to the configured
# local Ollama, which is always available and never excluded. The world
# stays single-threaded for now, so this doesn't speed anything up yet -
# it makes the whole path (eligibility, routing, drop/retry) real and
# testable before concurrency lands.
_host_registry_lock = threading.Lock()
_host_claims = {}  # host_url -> the model of the voice turn currently holding it
HOSTS = fenra_hosts.RemoteHostManager(os.path.join(BASE_DIR, "host_clients.json"))


def claim_host_for_voice(local_host, required_models, exclude=()):
    """Claims one host for a voice's entire turn (urge -> voice ->
    function-agent). A remote host is only eligible if it currently holds
    EVERY model in `required_models` (exact tag match, all three of the
    turn's calls run on the one claimed host), is alive and idle, isn't
    already claimed, and isn't in `exclude` (hosts that already failed
    this turn). Otherwise the local host - the guaranteed fallback."""
    with _host_registry_lock:
        for host in HOSTS.eligible_hosts(required_models):
            if host not in exclude and host not in _host_claims:
                _host_claims[host] = required_models[0]
                return host
        _host_claims[local_host] = required_models[0]
    return local_host


_host_activity = {}  # host_url -> {"voice", "phase", "since"} for the Connections tab


def set_host_activity(host, voice, phase):
    """Records which voice's turn is on `host` and which call (urge / voice
    / function-agent) it's in. Display-only - nothing schedules off it."""
    with _host_registry_lock:
        _host_activity[host] = {"voice": voice, "phase": phase, "since": time.time()}


def host_activity_snapshot():
    with _host_registry_lock:
        return {h: dict(a) for h, a in _host_activity.items()}


def release_host(host_url):
    """Releases a claim taken by claim_host_for_voice. Safe to call even
    if nothing was claimed (e.g. an early-exit path) - a bare pop with a
    default, not an assertion that something was there."""
    with _host_registry_lock:
        _host_claims.pop(host_url, None)
        _host_activity.pop(host_url, None)

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
LLM_CALL_HISTORY_FILENAME = "llm_calls.jsonl"
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
        "function_agent_model": DEFAULT_FUNCTION_AGENT_MODEL,
        "function_agent_retry_cap": FUNCTION_AGENT_RETRY_CAP,
        "history_window": DEFAULT_HISTORY_WINDOW,
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


def llm_call_history_path(world_name, voice_name):
    return os.path.join(voice_dir(world_name, voice_name), LLM_CALL_HISTORY_FILENAME)


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
        "paused": False,
        "room": DEFAULT_ROOM_NAME,
        "last_function_agent_note": "",
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
    turn a voice actually takes, capturing what `urge`/`currencies` *were*
    at that point, tied to the same `message_id` as that turn's own
    `thoughts` entry. `thoughts` already gives a full text history;
    nothing previously preserved what the numbers behind it were at any
    past point, which is what made checking a real-vs-invented
    correlation (2026-09-12, Church of Aletheia's "intensity" figures)
    require manual reconstruction instead of a lookup. Reads the voice's
    current (already-updated) state - call this after everything else
    for that turn has already been saved, not before.

    `understand_urge`/`understand_urge_general` dropped from here
    (2026-09-14, function-agent redesign) along with the rest of
    understand-urge - it existed to catch a voice's own malformed call
    attempts, which can't happen once voices never attempt calls."""
    state = load_voice_state(world_name, voice_name)
    entry = {
        "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
        "message_id": message_id,
        "urge": state.get("urge", {}),
        "currencies": state.get("currencies", {}),
    }
    ensure_voice_dir(world_name, voice_name)
    with open(voice_history_path(world_name, voice_name), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def log_llm_call(world_name, voice_name, kind, model, prompt, response, extra=None, timestamp=None):
    """Append-only, per-voice record of every real Ollama call (2026-09-15,
    Teddy's ask - "so I can see the details again") - one line per call to
    `worlds/<world>/voices/<voice>/llm_calls.jsonl`, distinct from
    `history.jsonl` (that file already means something else - numeric
    urge/currency snapshots, see append_voice_history). `kind` is one of
    "urge_agent"/"voice"/"function_agent"; `extra` carries the
    function-agent's per-attempt tool_calls/outcomes, None otherwise.
    Stores the raw prompt/response verbatim - unmodified, un-stripped,
    never display-cleaned (see _unescape_literal_newlines for the
    display-only counterpart). No pruning/rotation - meant to be a real,
    complete history; deliberately fails open (a write hiccup here should
    never cost a voice her turn, same spirit as the urge agent's own
    fail-open a few lines up in _tick)."""
    entry = {
        "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        "model": model,
        "prompt": prompt,
        "response": response,
        "extra": extra,
    }
    try:
        ensure_voice_dir(world_name, voice_name)
        with open(llm_call_history_path(world_name, voice_name), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def load_llm_call_history(world_name, voice_name):
    """Reads llm_calls.jsonl back in order, skipping any line that fails
    to parse rather than failing the whole read (same tolerance-of-
    corruption spirit as load_voice_state)."""
    path = llm_call_history_path(world_name, voice_name)
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


DISPATCH_CORRECTIONS_FILENAME = "dispatch_corrections.json"

# 2026-09-16 - whether the per-item dispatch prompt actually gets shown
# recent correction-log entries as context (see run_function_agent_turn).
# Off by default - real evidence with qwen3:30b that this backfires on a
# capable model (severe cross-contamination: unrelated content/functions
# from OTHER corrections bleeding onto whatever item is actually being
# decided). Logging to the correction memory itself is unconditional
# either way - only this feedback-into-prompt step is gated. Was on for
# ornith:9b, which genuinely seemed to need the extra nudge.
USE_DISPATCH_CORRECTIONS_CONTEXT = False


def dispatch_corrections_path():
    """Global, not per-world (2026-09-16, Teddy's explicit call) - the
    function agent's actual job (map a stated intent to a real function)
    doesn't change between worlds, so a correction earned in one world
    should help the next one too, not start over from empty."""
    return os.path.join(BASE_DIR, DISPATCH_CORRECTIONS_FILENAME)


def load_dispatch_corrections():
    """A plain JSON array, not .jsonl - entries need to be editable in
    place (Teddy filling in a "correction" field later), not just
    appended. Tolerant of a missing/corrupt file, same spirit as
    load_voice_state."""
    path = dispatch_corrections_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_dispatch_corrections(entries):
    with open(dispatch_corrections_path(), "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def append_dispatch_correction(world_name, voice_name, item_text, dispatched, outcome):
    """One entry per real per-item dispatch attempt (2026-09-16) -
    `dispatched` is the function+args actually tried (or None if
    declined), `correction` starts None ("matches what it did" until
    Teddy says otherwise, same as an implicit thumbs-up) and is only
    ever filled in by a human via the Dispatch Review tab. Returns the
    new entry's id."""
    entries = load_dispatch_corrections()
    next_id = max((e.get("id", 0) for e in entries), default=0) + 1
    entries.append({
        "id": next_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "world": world_name,
        "voice": voice_name,
        "item_text": item_text,
        "dispatched": dispatched,
        "outcome": outcome,
        "correction": None,
    })
    save_dispatch_corrections(entries)
    return next_id


def set_dispatch_correction(entry_id, correction_text):
    """GUI edit path (Dispatch Review tab) - Teddy filling in what a
    past dispatch decision should have been."""
    entries = load_dispatch_corrections()
    for entry in entries:
        if entry.get("id") == entry_id:
            entry["correction"] = correction_text or None
            break
    save_dispatch_corrections(entries)


def _unescape_literal_newlines(text):
    """Display-only cleanup (2026-09-15) - some models emit a literal
    backslash-n (two real characters) where they meant a line break,
    which then shows up as literal "\\n" text in a GUI box instead of an
    actual newline (caught live in a function-agent decline note, see
    Qualia/decisions.md). Never applied to anything built for an actual
    model prompt - display paths only."""
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")


def render_thoughts(thoughts):
    """Flattens a voice's private thoughts list back into the exact
    text the model has always received - "[timestamp] speaker: text"
    per line, newline-joined. Storage/GUI changed (2026-09-10); what
    Ollama sees given the same content did not. Renamed from
    render_messages (2026-09-13) - as of the rooms redesign this only
    ever renders a voice's own generations, never anyone else's."""
    return "\n".join(f"[{m['timestamp']}] {m['speaker']}: {m['text']}" for m in thoughts)


def render_thoughts_for_display(thoughts):
    """GUI-only (2026-09-15) - same real per-entry text render_thoughts
    sends to the model (each already carries its own real timestamp),
    just with a blank-line divider between entries so a multi-line
    thought doesn't visually run into the next one. Never used for a
    real prompt."""
    return "\n\n".join(
        _unescape_literal_newlines(f"[{m['timestamp']}] {m['speaker']}: {m['text']}") for m in thoughts
    )


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


def _world_activity_entries(world_name, voice_name):
    """The real per-entry work behind build_world_activity - returns the
    sorted (timestamp, text) pairs rather than the final joined string,
    so the GUI can render them with its own visual dividers/timestamps
    (see render_world_activity_for_display) without that annotation
    ever touching what actually reaches a voice's prompt. See
    build_world_activity for the real selection/TTL logic this
    implements - unchanged, just split out."""
    state = load_voice_state(world_name, voice_name)
    own_room = state.get("room")
    if not own_room:
        return []
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
    return visible


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
    literal call/args/result, UI-only, never shown to any voice. No
    timestamps in the joined text - deliberate, a voice has no time
    signal anywhere in her prompt in this branch (see the timestamp-on-
    HUD backlog item); the GUI-only display variant is free to show them
    since it's a debugging aid, not what she actually receives."""
    return "\n".join(text for _, text in _world_activity_entries(world_name, voice_name))


def render_world_activity_for_display(world_name, voice_name):
    """GUI-only (2026-09-15) - same entries build_world_activity sends to
    the model, but with a per-entry timestamp + divider for human
    readability. Never used for a real prompt - the model-facing
    build_world_activity stays exactly as it was, with no timestamps,
    since a voice genuinely has no time signal anywhere else in her
    prompt (deliberate, not an oversight - see the timestamp-on-HUD
    backlog item)."""
    entries = _world_activity_entries(world_name, voice_name)
    divider = "-" * 40
    return f"\n{divider}\n".join(f"[{ts}]\n{_unescape_literal_newlines(text)}" for ts, text in entries)


def render_llm_history_for_display(world_name, voice_name):
    """GUI-only (2026-09-15) - renders load_llm_call_history's raw entries
    for the new History tab: one block per real Ollama call, header +
    unescaped prompt/response, divider between entries. Never used for a
    real prompt - this is purely a debugging/review aid."""
    entries = load_llm_call_history(world_name, voice_name)
    if not entries:
        return ""
    divider = "\n" + "-" * 40 + "\n"
    blocks = []
    for e in entries:
        header = f"[{e.get('timestamp', '')}] {e.get('kind', '')} ({e.get('model', '')})"
        extra = e.get("extra") or {}
        extra_bits = []
        if extra.get("item"):
            extra_bits.append(f"item: {extra['item']!r}")
        if "attempt" in extra:
            extra_bits.append(f"attempt {extra['attempt']}")
        if extra.get("tool_calls"):
            extra_bits.append("tool calls: " + "; ".join(extra["tool_calls"]))
        if extra.get("outcomes"):
            extra_bits.append(
                "outcomes: " + ", ".join(f"{name}={outcome}" for name, outcome in extra["outcomes"])
            )
        if extra_bits:
            header += " - " + " | ".join(extra_bits)
        blocks.append(
            f"{header}\n"
            f"--- PROMPT ---\n{_unescape_literal_newlines(e.get('prompt', ''))}\n"
            f"--- RESPONSE ---\n{_unescape_literal_newlines(e.get('response', ''))}"
        )
    return divider.join(blocks)


def voice_display_name(world_name, voice_name, state=None):
    """How a voice's name should read in text meant to be READ by
    another voice (room-log mask/raw text, HUD occupant/currency
    listings, room_state's occupant list) - never for internal
    identifiers (dict keys, directory names, dispatch_one_function_call's
    caller_name, recipients/peripheral map keys), which always stay the
    bare voice_name. Appends "(human)" for a piloted voice (2026-09-15,
    Pilot Mode) - Teddy's explicit call: other voices should always be
    able to tell a human is present, the same honesty standard every LLM
    voice already gets about what she is."""
    if state is None:
        state = load_voice_state(world_name, voice_name)
    return f"{voice_name} (human)" if state.get("piloted") else voice_name


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
    # ever learn about someone actually present with you. A piloted
    # voice (2026-09-15, Pilot Mode) is excluded here even though her
    # own state also carries paused=True (that's belt-and-suspenders for
    # the round-robin skip only) - "(paused)" reads as an unavailable
    # NPC, which is exactly wrong for a human actively present in
    # real-time; she gets "(human)" instead, never both (real bug caught
    # live, 2026-09-16 - this exclusion was the actual intent from the
    # start but never implemented, so the function agent was seeing a
    # piloted voice as unavailable and declining real requests aimed at
    # her).
    paused_occupants = sorted(
        v for v in occupants
        if load_voice_state(world_name, v).get("paused", False)
        and not load_voice_state(world_name, v).get("piloted", False)
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
        # Whatever the function agent's own turn wrote last time it ran
        # for this voice (2026-09-14, function-agent redesign) - a pure
        # read, never cleared here. Only _tick's real per-turn prompt
        # build clears it after use, so a GUI preview (Registers tab,
        # Voices tab HUD summary) never consumes it as a side effect of
        # just being looked at.
        "function_agent_note": state.get("last_function_agent_note", ""),
    }


def build_hud(world_name, voice_name):
    """The last thing in a voice's prompt (see module docstring) -
    computed fresh every tick, never written to state.json. Own
    name/model/room, who's currently here (paused annotated inline),
    which rooms are adjacent (names only - deliberately as vague as the
    "you hear activity" notice), this room's board activity, everyone's
    currency balances, the voice's own identity line, then - only when
    present - whatever the function agent's own last turn wrote about
    her, one time only (2026-09-14, function-agent redesign; see _tick,
    which is the only place that ever clears it after reading it into a
    real prompt). No intent-signal or call-syntax reminder at all
    (2026-09-17 - back to raw output; the bracket/sign-off convention
    added 2026-09-15/16 is gone along with the dispatch redesign it
    supported, see run_function_agent_turn) - a voice has no knowledge
    functions exist and no instruction on how to "declare" a want; she
    just writes, and the function agent reads her raw prose directly."""
    f = hud_fields(world_name, voice_name)
    board_line = "Board activity: " + (", ".join(f["board_counts"]) if f["board_counts"] else "none")
    currency_line = "Currency levels (everyone, four elemental currencies - Air, Earth, "
    currency_line += "Fire, Water - no exchange rate is defined between them): " + ", ".join(
        f"{voice_display_name(world_name, v)} (" + ", ".join(f"{el}: {amts[el]:.1f}" for el in CURRENCY_ELEMENTS) + ")"
        for v, amts in f["balances"]
    )

    # Paused occupants annotated inline (2026-09-12) - "(paused)" next
    # to their name, so a voice can tell not to keep addressing someone
    # who currently can't respond, without exposing why they're paused.
    # A piloted occupant (2026-09-15, Pilot Mode) gets "(human)" via
    # voice_display_name instead - never both, she's not "paused" in the
    # sense that matters to another voice.
    paused_occupants = set(f["paused_occupants"])
    occupants_display = ", ".join(
        f"{voice_display_name(world_name, v)} (paused)" if v in paused_occupants
        else voice_display_name(world_name, v)
        for v in f["occupants"]
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
        f["identity"],
    ]
    if f["function_agent_note"]:
        lines.append(f["function_agent_note"])
    return "\n".join(lines)


def build_function_agent_hud(world_name, voice_name):
    """The grounding data handed to the function agent (2026-09-14) -
    same real, live data as build_hud() via hud_fields() (zero drift
    risk, same "computed once, shared" pattern as everywhere else in
    this module), but rendered for a dispatcher rather than a character:
    no "everything above/below this line" framing, no function-agent
    note (that's for the voice's own next turn, not for the agent
    reasoning about this one), just the real room/occupants/adjacent/
    board/currency/identity facts it needs to fill in real values
    instead of inventing plausible-sounding ones."""
    f = hud_fields(world_name, voice_name)
    board_line = "Board activity: " + (", ".join(f["board_counts"]) if f["board_counts"] else "none")
    currency_line = "Currency levels (everyone, four elemental currencies - Air, Earth, "
    currency_line += "Fire, Water - no exchange rate is defined between them): " + ", ".join(
        f"{voice_display_name(world_name, v)} (" + ", ".join(f"{el}: {amts[el]:.1f}" for el in CURRENCY_ELEMENTS) + ")"
        for v, amts in f["balances"]
    )
    # A piloted occupant (2026-09-15, Pilot Mode) renders "(human)" via
    # voice_display_name, never "(paused)" even though her state also
    # carries paused=True (belt-and-suspenders for the round-robin skip)
    # - real bug caught live, 2026-09-16: this function was missed in
    # the original Pilot Mode pass (only build_hud/the GUI summary got
    # the fix), so the function agent's own ground-truth HUD was still
    # telling it a piloted voice was "(paused)" - i.e. an unavailable
    # NPC - which plausibly explains a real decline (Idris's "ask Teddy"
    # treated as non-actionable) that looked at first like an overly
    # literal wording problem but wasn't.
    paused_occupants = set(f["paused_occupants"])
    occupants_display = ", ".join(
        f"{voice_display_name(world_name, v)} (paused)" if v in paused_occupants
        else voice_display_name(world_name, v)
        for v in f["occupants"]
    ) if f["occupants"] else "none"
    lines = [
        f"Room: {f['room']}",
        f"Also here: {occupants_display}",
        f"Adjacent rooms: {', '.join(f['adjacent_rooms']) if f['adjacent_rooms'] else 'none'}",
        board_line,
        currency_line,
        f["identity"],
    ]
    return "\n".join(lines)


# --------------------------------------------------------------- functions --
# The old ⟦function_name(args)⟧ regex parse (FUNCTION_CALL_RE) was removed
# 2026-09-14, function-agent redesign - voices no longer emit or need to
# know call syntax at all; see dispatch_one_function_call/
# run_function_agent_turn below for the real replacement path.


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
    their own turns. mask == raw here - say has no privacy layer. Also
    logs a private entry back to the caller herself (2026-09-15) - see
    dispatch_one_function_call's docstring for why SELF_LOGGING_FUNCTIONS
    need this: without it she had no permanent record of her own action
    in her own World Activity, only a one-shot HUD note, and could
    re-issue the same intent later once that note expired (real bug,
    caught live - duplicate whispers)."""
    text = args_text.strip()
    if not text:
        raise ValueError("no text given")
    room = load_voice_state(world_name, caller_name).get("room")
    if not room:
        raise ValueError("you aren't in a room")
    raw = f"{voice_display_name(world_name, caller_name)} says: {text}"
    recipients = {
        v: voice_turn_count(world_name, v)
        for v in room_occupants(world_name, room) if v != caller_name
    }
    _log_room_event(world_name, room, caller_name, "dialogue", "say", raw, raw, recipients)
    _log_room_event(
        world_name, room, caller_name, "dialogue", "say", raw, f"You say: {text}",
        {caller_name: voice_turn_count(world_name, caller_name)},
    )
    return f"said to {room}"


def fn_whisper(world_name, caller_name, args_text):
    """Voice-scoped, private speech (2026-09-13) - requires sharing a
    room with the target; delivered ONLY to that target, live, for
    WHISPER_TTL_TURNS of their own turns. The actual CONTENT never
    reaches anyone else, ever - read_room_log() never returns it either,
    including to the target once it's aged out of their own live view
    (see _log_room_event's `mask` vs `raw`) - that's the real privacy
    guarantee, and it's unchanged. The room's raw log (Rooms tab,
    Teddy/Qualia only) always has the real text.

    Two more entries as of 2026-09-15: (1) a private, caller-only entry
    so she has a permanent record of her own whisper in her own World
    Activity - without it she had only a one-shot HUD note that expired
    after one turn, and would re-whisper the same thing once it did
    (real bug, caught live watching the_loom - Gemma/Mistral duplicate
    whispers); (2) a content-free "X whispered to Y" notice to every
    OTHER occupant of the shared room (Teddy's explicit ask, groundwork
    for a later "others' actions nudge my own urges" mechanic) - logged
    as an "activity"-kind entry so bystanders only ever see `mask`,
    never `raw`; the content stays exactly as private as before."""
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
    caller_display = voice_display_name(world_name, caller_name)
    target_display = voice_display_name(world_name, target)
    raw = f"{caller_display} whispers to you: {text}"
    mask = f"{caller_display} whispered to {target_display}."
    recipients = {target: voice_turn_count(world_name, target)}
    _log_room_event(world_name, room, caller_name, "dialogue", "whisper", mask, raw, recipients)
    _log_room_event(
        world_name, room, caller_name, "dialogue", "whisper", mask, f"You whisper to {target}: {text}",
        {caller_name: voice_turn_count(world_name, caller_name)},
    )
    bystanders = {
        v: voice_turn_count(world_name, v)
        for v in room_occupants(world_name, room) if v not in (caller_name, target)
    }
    if bystanders:
        _log_room_event(world_name, room, caller_name, "activity", "whisper", mask, mask, bystanders)
    return f"whispered to {target}"


def fn_yell(world_name, caller_name, args_text):
    """Projected speech (2026-09-13) - reaches every other occupant of
    the caller's room AND every occupant of every adjacent room, live,
    in full, for YELL_TTL_TURNS of each recipient's own turns. mask ==
    raw - like say, yelling has no privacy layer, it's the opposite.
    Also logs a private entry back to the caller herself (2026-09-15) -
    see fn_say's matching note/dispatch_one_function_call's docstring."""
    text = args_text.strip()
    if not text:
        raise ValueError("no text given")
    room_name = load_voice_state(world_name, caller_name).get("room")
    if not room_name:
        raise ValueError("you aren't in a room")
    room_state = load_room_state(world_name, room_name) or default_room_state(room_name)
    raw = f"{voice_display_name(world_name, caller_name)} yells: {text}"
    recipients = {
        v: voice_turn_count(world_name, v)
        for v in room_occupants(world_name, room_name) if v != caller_name
    }
    for adj_name in room_state.get("adjacent", []):
        for v in room_occupants(world_name, adj_name):
            recipients.setdefault(v, voice_turn_count(world_name, v))
    _log_room_event(world_name, room_name, caller_name, "dialogue", "yell", raw, raw, recipients)
    _log_room_event(
        world_name, room_name, caller_name, "dialogue", "yell", raw, f"You yell: {text}",
        {caller_name: voice_turn_count(world_name, caller_name)},
    )
    return f"yelled from {room_name}"


def fn_move_room(world_name, caller_name, args_text):
    """Unrestricted movement (2026-09-13, Teddy's explicit call) - any
    existing room, no adjacency requirement. Logs a departure activity
    in the old room and an arrival activity in the new one, using the
    same recipients/peripheral mechanics as any other activity. Also
    logs a private caller-only entry in the NEW room (2026-09-15) - has
    to be the new room, not the old one, since a voice's World Activity
    only ever scans her current room + adjacent (see
    _world_activity_entries); see fn_say's matching note for why this
    exists."""
    target = sanitize_name(args_text)
    if not target:
        raise ValueError("no room given")
    if not load_room_state(world_name, target):
        raise ValueError(f"'{target}' isn't a room that exists")
    caller_state = load_voice_state(world_name, caller_name)
    old_room = caller_state.get("room")
    if old_room == target:
        raise ValueError(f"you're already in '{target}'")
    caller_display = voice_display_name(world_name, caller_name, caller_state)
    if old_room:
        _log_generic_activity(world_name, old_room, caller_name, "move_room", f"{caller_display} leaves toward {target}.")
    caller_state["room"] = target
    save_voice_state(world_name, caller_name, caller_state)
    _log_generic_activity(world_name, target, caller_name, "move_room", f"{caller_display} arrives.")
    _log_room_event(
        world_name, target, caller_name, "dialogue", "move_room",
        f"{caller_display} arrives.", f"You move to {target}.",
        {caller_name: voice_turn_count(world_name, caller_name)},
    )
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
    caller_display = voice_display_name(world_name, caller_name, caller_state)
    save_room_state(world_name, name, default_room_state(name, adjacent=[old_room]))
    old_room_state.setdefault("adjacent", []).append(name)
    save_room_state(world_name, old_room, old_room_state)
    _log_generic_activity(world_name, old_room, caller_name, "create_room", f"{caller_display} leaves toward the new room {name}.")
    caller_state["room"] = name
    save_voice_state(world_name, caller_name, caller_state)
    # "dialogue" kind + caller as sole recipient (2026-09-15, was
    # "activity" kind with no recipients at all - she never had any
    # permanent record of creating the room, only a one-shot HUD note)
    # so a recipients-hit renders `raw`, not `mask` - see
    # _world_activity_entries and fn_say's matching note.
    _log_room_event(
        world_name, name, caller_name, "dialogue", "create_room",
        f"{caller_display} created this room.", f"You create {name} and move into it.",
        {caller_name: voice_turn_count(world_name, caller_name)},
    )
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
    occupants = [voice_display_name(world_name, v) for v in room_occupants(world_name, room)]
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

# Deterministic, never-LLM-generated confirmation the CALLER gets on a
# real successful dispatch (2026-09-14, function-agent redesign) -
# explicit on purpose (Teddy's call): not just "you did it," the real
# result too. This is the only place a voice ever actually learns what
# room_state/read_room_log/skim_board/read_board found - without it that
# information was being silently discarded (real bug, caught live
# watching Priya's first skim_board turn return nothing to her at all).
# Rendered with {result} (the real return value from the fn_* call) plus
# whatever named args that function takes (e.g. {target}, {room}) -
# safe to rely on being present since this only ever runs on success,
# meaning the call's real arguments were already valid.
FUNCTION_SELF_RESULT_TEMPLATES = {
    # say/whisper/yell/move_room/create_room (2026-09-14, Teddy's
    # correction): the real content lives in the call's own arguments,
    # not in `result`/`detail` (which for these five is just a bare
    # meta-confirmation like "said to town_center", not her actual
    # words) - render the real text directly, same present-tense style
    # as the existing third-person phrasing ("{caller} says: {text}"),
    # not a generic report about what happened.
    "say": "You say: {text}",
    "whisper": "You whisper to {target}: {text}",
    "yell": "You yell: {text}",
    "move_room": "You move to {room}.",
    "create_room": "You create {name} and move into it.",
    "read_room_log": "You reviewed the room's log. It contains: {result}",
    "room_state": "You checked on the room. {result}",
    "give_currency": "You gave currency to {target}. {result}",
    "post_board": "You posted something to the board. {result}",
    "skim_board": "You looked over the board. It contains: {result}",
    "read_board": "You read a post in full. It says: {result}",
    "delete_board": "You removed a post from the board. {result}",
}


# Hand-translated second-person prose for real-world-fact dispatch
# failures (2026-09-17, Teddy's call) - the failure-side twin of
# FUNCTION_SELF_RESULT_TEMPLATES above. Deliberately NOT exhaustive:
# only covers errors that are genuinely about the world (a target/room
# that doesn't exist, insufficient balance, a post that isn't there) -
# a voice's own honest mistake, worth her actually learning about.
# Errors caused by the function agent's own malformed translation (bad
# separator count, non-numeric amount, empty required field, etc.) are
# deliberately absent here and stay silent to the voice, same as
# before this change - she never said anything wrong, the translation
# layer did, and surfacing that as an in-fiction fact would misattribute
# it to her. Each function maps to an ordered list of (substring-to-
# match-in-the-raw-error, hand-written template) pairs - first match
# wins, checked against the real exception text raised by the matching
# fn_* body (see FUNCTION_REGISTRY). No match = no voice-facing note at
# all (same silent-failure behavior as before this change), not a
# generic fallback - see _self_error_text.
FUNCTION_ERROR_TEMPLATES = {
    "say": [
        ("you aren't in a room", "You try to speak, but you aren't actually in any room right now."),
    ],
    "whisper": [
        ("isn't a voice in this world", "You don't see anyone named {target} here."),
        ("can't whisper to yourself", "You catch yourself about to whisper to yourself - never mind."),
        ("don't share a room", "{target} isn't in the room with you right now."),
    ],
    "yell": [
        ("you aren't in a room", "You try to yell, but you aren't actually in any room right now."),
    ],
    "move_room": [
        ("isn't a room that exists", "You don't know of any room called {room}."),
        ("you're already in", "You're already in {room}."),
    ],
    "create_room": [
        ("already exists", "A room called {name} already exists - you can't create another with that name."),
        ("you aren't in a room", "You try to build a new room, but you aren't anywhere right now to branch off from."),
    ],
    "read_room_log": [
        ("isn't a room that exists", "You don't know of any room called {room}."),
    ],
    "room_state": [
        ("isn't a room that exists", "You don't know of any room called {room}."),
    ],
    "give_currency": [
        ("isn't a voice in this world", "You don't see anyone named {target} here."),
        ("can't give_currency to yourself", "You catch yourself about to give currency to yourself - never mind."),
        ("you only have", "You reach for your {element}, but come up short: {error}"),
    ],
    "post_board": [
        ("you aren't currently in", "You aren't actually in {room} right now."),
    ],
    "skim_board": [
        ("you aren't currently in", "You aren't actually in {room} right now."),
    ],
    "read_board": [
        ("you aren't currently in", "You aren't actually in {room} right now."),
        ("no post", "There's no post numbered {post_id} on {room}'s board."),
    ],
    "delete_board": [
        ("you aren't currently in", "You aren't actually in {room} right now."),
        ("no post", "There's no post numbered {post_id} on {room}'s board."),
    ],
}


def _self_result_text(name, arguments, detail):
    """Renders FUNCTION_SELF_RESULT_TEMPLATES for one real successful
    call - falls back to a generic "you did it" phrasing for any
    function that somehow isn't in the table, rather than silently
    dropping the result again."""
    template = FUNCTION_SELF_RESULT_TEMPLATES.get(name, "You did it. {result}")
    try:
        return template.format(result=detail, **arguments)
    except (KeyError, IndexError):
        return f"You did it. {detail}"


def _self_error_text(name, arguments, error_text):
    """Renders FUNCTION_ERROR_TEMPLATES for one real dispatch failure -
    the failure-side twin of _self_result_text. Returns None (not a
    generic fallback) when no template matches - either the function
    has no entries at all, or `error_text` doesn't match any of its
    known real-world-fact substrings (most likely because it's a
    function-agent malformed-call error, which is deliberately excluded
    from FUNCTION_ERROR_TEMPLATES - see its own comment). A template
    whose placeholders don't resolve against `arguments` (shouldn't
    normally happen, since every template's placeholders were written
    against that function's own real param names) also returns None
    rather than raising or guessing - silence, same as no match."""
    for substring, template in FUNCTION_ERROR_TEMPLATES.get(name, []):
        if substring in error_text:
            try:
                return template.format(error=error_text, **arguments)
            except (KeyError, IndexError):
                return None
    return None


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
        'The named target gets the full content, live: "{caller} whispers to you: <text>". '
        'Every other occupant of the same room gets a content-free notice instead (2026-09-15): '
        '"{caller} whispered to <target>." - they learn it happened and between whom, never the '
        "content. The room's permanent log entry never includes the content either, for anyone, "
        "ever, including the target once it's out of their own live view.",
        "No one - whisper never reaches beyond its own room.",
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
# real function tracks a felt "Urge" that grows the longer it goes unused
# and resets on a successful call - see xleud()/apply_urge_tick() below for
# the actual mechanics. (`functions()` itself, the old pure-introspection
# call, was removed entirely 2026-09-14 along with the rest of call-syntax
# scaffolding - see the function-agent redesign notes near FUNCTION_REGISTRY.)

URGE_FUNCTIONS = tuple(FUNCTION_REGISTRY)

URGE_DESIRE = 7            # denominator in the XLEUD saturating curve
URGE_DRIVE = 1              # Urge growth per tick a function goes unused
URGE_FLOOR_PCT = 50         # perform-urge must be at/above this XLEUD% to be felt
URGE_TOP_N = 3              # cap on functions handed to the urge agent per turn
DEFAULT_URGE_MODEL = "phi4-mini"
DEFAULT_FUNCTION_AGENT_MODEL = "ornith:9b"
FUNCTION_AGENT_RETRY_CAP = 2  # retries on a real dispatcher error; 3 attempts total
DEFAULT_HISTORY_WINDOW = 20  # a voice's own last N turns rendered into her prompt; <=0 means unbounded (2026-09-17)

# understand-urge (a voice's own malformed-call-attempt tracker) was removed
# entirely 2026-09-14, function-agent redesign - it existed to catch a
# voice's own hallucinated/malformed calls, which can't happen once voices
# never attempt calls at all; that job now belongs to the function agent's
# retry loop (see run_function_agent_turn), which isn't urge-tracked.
URGE_VIEWER_CATEGORIES = ("Perform Urges", "This Turn")


def _mask_for_call(world_name, caller_name, name, args_text):
    """The activity-register text for one generic (non-self-logging)
    function call - a flavored, per-function action mask
    (FUNCTION_REGISTRY[name]["mask"]), rendered with {caller} and
    {arg0} (see _first_pipe_arg). An unrecognized function name (no
    registry entry - a hallucinated call, no real mask to pull from)
    falls back to a WoW-nod easter egg (Teddy's call, 2026-09-10):
    "makes some strange gestures" - the classic failed-cast flavor
    text. {caller} renders through voice_display_name (2026-09-15,
    Pilot Mode) - the one shared renderer behind every generic
    function's mask, so a single change covers all of them."""
    caller_display = voice_display_name(world_name, caller_name)
    meta = FUNCTION_REGISTRY.get(name)
    if not meta or "mask" not in meta:
        return f"(*{caller_display} makes some strange gestures.*)"
    try:
        return meta["mask"].format(caller=caller_display, arg0=_first_pipe_arg(args_text))
    except (KeyError, IndexError):
        return f"(*{caller_display} makes some strange gestures.*)"


def _args_text_from_dict(name, arguments):
    """Reconstructs the pipe-delimited args_text every fn_* body already
    expects (e.g. "target|text") from the structured {param: value} dict
    Ollama's native tool-calling returns (2026-09-14, function-agent
    redesign - replaces the old ⟦fn(args)⟧ regex parse entirely). Uses
    FUNCTION_REGISTRY[name]["params"]'s own known order; a param the
    function agent left out just becomes an empty segment, which each
    fn_*'s existing real validation (e.g. "no text given", "expected
    'target|...'") already rejects on its own - no new validation
    needed, the retry loop feeds that real error straight back."""
    params = [p.strip("[]") for p in FUNCTION_REGISTRY[name]["params"].split("|") if p]
    return "|".join(str(arguments.get(p, "")) for p in params)


def dispatch_one_function_call(world_name, caller_name, name, arguments):
    """Runs one real function call on the function agent's behalf.
    Returns (outcome, detail) - outcome is "ok" or "error", detail is
    the real result or error message. Reuses every existing fn_* body
    and its real validation untouched.

    Every real function now has both a second-person description (what
    the caller herself learns - FUNCTION_SELF_RESULT_TEMPLATES) and a
    third-person one (what everyone else sees - Teddy's framing,
    2026-09-14), but they reach her through two different channels
    depending on the function, since the two families already differ in
    how their third-person side works:

    - Generic (non-SELF_LOGGING) functions: the shared/public activity
      entry (FUNCTION_REGISTRY's own "mask") deliberately excludes the
      caller from its own recipients, same as it always has - so this
      ALSO logs a second, private, caller-only `dialogue`-kind entry,
      which makes `build_world_activity` render the real second-person
      result to her live (real bug caught live, 2026-09-14: without
      this she had no way to ever learn what a query-style call like
      skim_board actually found - world_activity was never designed to
      carry that back to her). `read_room_log()` still only ever
      returns the shared entry's ordinary third-person mask to anyone,
      herself included - her own private result text never leaks into
      the public record.
    - SELF_LOGGING_FUNCTIONS (say/whisper/yell/move_room/create_room):
      their own fn_* body already logs real, correct third-person
      content for everyone else internally - a second private room-log
      entry here would just be a confusing near-duplicate sitting next
      to it. Instead the caller's second-person confirmation surfaces
      through the existing one-shot next-turn HUD note (see
      run_function_agent_turn/_tick) - simpler, no duplicate log entry,
      still gives her the explicit "you did this, for real" signal."""
    meta = FUNCTION_REGISTRY.get(name)
    if not meta:
        return "error", f"unknown function '{name}'"
    args_text = _args_text_from_dict(name, arguments)
    try:
        result = meta["fn"](world_name, caller_name, args_text)
    except Exception as exc:
        return "error", str(exc)
    if name not in SELF_LOGGING_FUNCTIONS:
        room = load_voice_state(world_name, caller_name).get("room")
        if room:
            mask = _mask_for_call(world_name, caller_name, name, args_text)
            raw = f"{caller_name} called {name}({args_text}) -> {result}"
            _log_generic_activity(world_name, room, caller_name, name, mask, raw)
            self_text = _self_result_text(name, arguments, result)
            _log_room_event(
                world_name, room, caller_name, "dialogue", name, mask, self_text,
                {caller_name: voice_turn_count(world_name, caller_name)},
            )
    return "ok", result


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
    run_function_agent_turn returns its real dispatch outcomes (2026-09-14
    - previously read from run_function_calls's parse of the voice's own
    raw text; now driven by what the function agent actually dispatched
    on her behalf, since she never attempts calls herself anymore).

    perform-urge (state["urge"]): a successfully-called function resets
    fully to 0 (Teddy's round-one Satisfaction rule); every other real
    function grows by URGE_DRIVE, whether it errored, was left alone, or
    the function agent declined to call anything at all this turn - only
    a real successful dispatch counts as "using" it. This includes a
    retry-cap-exhausted or well-formed-but-semantically-wrong call
    (Teddy's call, 2026-09-14): neither resets the urge, no special-
    casing beyond "did it actually, successfully happen."

    (understand-urge - the old tracker for a voice's own malformed call
    attempts - was removed entirely along with this redesign; see the
    URGE_VIEWER_CATEGORIES comment.)"""
    ok_names = {name for name, outcome in outcomes if outcome == "ok"}

    state = load_voice_state(world_name, voice_name)

    urge = state.get("urge", {})
    for name in URGE_FUNCTIONS:
        urge[name] = 0.0 if name in ok_names else urge.get(name, 0.0) + URGE_DRIVE
    state["urge"] = urge

    save_voice_state(world_name, voice_name, state)


def compute_urge_snapshot(state):
    """Pure function of a voice's persisted state -> everything the
    per-tick urge decision (and the Urge Viewer GUI panel) needs. Never
    mutates, never touches disk.

    Simplified 2026-09-14, function-agent redesign - understand-urge (the
    old "does a real error outrank the roleplay urge agent this turn"
    branch) is gone entirely along with the rest of understand-urge, so
    this is now just the perform-urge side: everything at or above
    URGE_FLOOR_PCT, top URGE_TOP_N by strength."""
    urge = state.get("urge", {})
    perform = [(name, xleud(urge.get(name, 0.0))) for name in URGE_FUNCTIONS]
    top_perform = sorted(
        (p for p in perform if p[1] * 100 >= URGE_FLOOR_PCT), key=lambda p: -p[1]
    )[:URGE_TOP_N]
    return {"top_perform": top_perform}  # up to URGE_TOP_N, only >= floor


URGE_AGENT_INSTRUCTIONS = """You are a small utility model with no memory between calls. Your only job: given a list of felt pulls (a brief description of what each one is about, and an intensity percentage), write ONE short paragraph — 2 to 4 sentences — describing what it feels like to carry these pulls right now.

The percentage is how strongly this pull is currently felt — the longer it's gone unanswered, the higher it climbs, and the harder it becomes to ignore.

Write it in second person ("You feel..."), as an embodied, organic sensation — not a command, not a to-do list, not an instruction to act. Never state the raw percentage number in your output.

Describe only the real sensation each description below actually implies (an urge to speak up, to reach out to someone privately, to go somewhere new) — never name a mechanism, a function, a system, or any technical term for it. You're describing a feeling, not an action to take or a thing to call. Base your description only on what's listed below — never invent, imply, or reference a pull that isn't there. If only one is listed, describe only that one — do not invent or imply any others.

Do not greet, explain what you're doing, or add anything besides the paragraph itself.

Felt-pull values follow:
"""


def render_urge_lines(top_perform):
    """Plain '- name (description): NN%' lines from compute_urge_snapshot's
    top_perform list - shared by the flavor-text urge agent's prompt
    (build_urge_agent_prompt) and the function agent's own [URGES] block
    (build_function_agent_prompt, 2026-09-17) so the two never drift
    apart on format."""
    return "\n".join(
        f"- {name} ({FUNCTION_REGISTRY[name]['description']}): {round(pct * 100)}%"
        for name, pct in top_perform
    )


def build_urge_agent_prompt(top_perform):
    """The stateless, one-shot prompt sent to the urge agent - fixed
    instructions plus the top_perform functions' real registry
    description (reused as-is, no separate urge-specific description
    field needed) and current XLEUD%. Confirmed via a 21-case stress
    battery against phi4-mini/gemma3:4b/llama3.2:3b, 2026-09-11."""
    return URGE_AGENT_INSTRUCTIONS + render_urge_lines(top_perform)


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
    request = {"model": model, "prompt": prompt, "stream": False, "options": options or {}}
    if fenra_hosts.is_remote_host(host):
        return HOSTS.call(host, "generate", request).get("response", "")
    resp = requests.post(f"{host}/api/generate", json=request, timeout=None)
    resp.raise_for_status()
    return resp.json().get("response", "")


def list_ollama_models(host):
    try:
        resp = requests.get(f"{host}/api/tags", timeout=10)
        resp.raise_for_status()
        return sorted(m["name"] for m in resp.json().get("models", []))
    except requests.RequestException:
        return []


# --------------------------------------------------------- function agent --
# 2026-09-14 redesign: voices no longer know functions exist at all - no
# call syntax, no functions() introspection, nothing mechanical in their
# own prompt. A separate small model (DEFAULT_FUNCTION_AGENT_MODEL,
# ornith:9b - picked after real stress-testing, see
# Qualia/Function Agent Testing/) reads a voice's own output plus real
# grounding data and decides what, if anything, actually happens in the
# world, using Ollama's native tool-calling API rather than a text-syntax
# parse. See run_function_agent_turn for the real per-turn orchestration
# (including the retry-on-real-dispatcher-error loop) and
# dispatch_one_function_call for the actual execution.

def build_function_agent_tools():
    """The real Ollama/OpenAI-style `tools` schema, built fresh from
    FUNCTION_REGISTRY so it can never drift out of sync with the real
    functions (same "computed once, shared" spirit as hud_fields()).
    Each function's own `params` string already encodes name +
    optionality (e.g. "target|text", "[search term]") - split on '|',
    a bracketed param is optional, everything else required."""
    tools = []
    for name, meta in FUNCTION_REGISTRY.items():
        params = [p for p in meta["params"].split("|") if p]
        properties = {}
        required = []
        for p in params:
            optional = p.startswith("[") and p.endswith("]")
            pname = p.strip("[]")
            properties[pname] = {"type": "string", "description": pname}
            if not optional:
                required.append(pname)
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": meta["description"],
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        })
    return tools


def call_function_agent(host, model, system_text, tools, options=None):
    """Sibling to call_ollama, but hits /api/chat with a real `tools`
    payload instead of /api/generate - Ollama's native tool-calling only
    lives on the chat endpoint. call_ollama itself is untouched; voice/
    urge-agent calls keep using /api/generate exactly as before. Returns
    the raw `message` dict ({"content": ..., "tool_calls": [...]} -
    tool_calls may be absent/empty if the agent chose to call nothing)."""
    request = {
        "model": model,
        "messages": [{"role": "system", "content": system_text}],
        "tools": tools,
        "stream": False,
        "options": options or {},
    }
    if fenra_hosts.is_remote_host(host):
        return HOSTS.call(host, "chat", request).get("message", {})
    resp = requests.post(f"{host}/api/chat", json=request, timeout=None)
    resp.raise_for_status()
    return resp.json().get("message", {})


def build_function_agent_prompt(voice_name, voice_text, hud_text, urge_text, recent_corrections=None, retry_note=""):
    """Whole-turn dispatch prompt (2026-09-17 - reverted from the
    2026-09-16 deterministic bracket-based per-item design, Teddy's
    call). The bracket redesign fixed two real ornith:9b-era bugs (a
    typo in one item bleeding into another's dispatch; an actionable
    item declined alongside an unrelated non-actionable one) but, per
    Teddy's own read of the live urge state the next morning, also
    meant a voice could only ever get 2-3 explicit bracketed items
    dispatched per turn - most function categories sat maxed and
    unaddressed. qwen3:30b has since proven itself capable enough
    (recovering cleanly from a voice's own hallucinated fake-HUD text,
    correctly splitting a genuinely compound item on its own) to trust
    with the ORIGINAL shape: read [VOICE]'s raw, unscaffolded prose
    directly - no bracket convention, no "I wish to take the following
    actions:" sign-off - and decide everything about the turn in one
    call, same as before the redesign.

    New `[URGES]` section (function agent explicitly did NOT see this
    2026-09-15 through the bracket-dispatch era - "felt urges stay
    voice-only"; Teddy is reversing that call specifically to attack the
    unused-urge-category problem above). Per Teddy's explicit "proactive
    nudge" answer (not just a tiebreaker): a strong, long-unaddressed
    urge can justify a real dispatch even without [VOICE]'s text this
    turn explicitly asking for it - a deliberate, acknowledged loosening
    of the fabrication boundary. [HUD] facts are never touched by this -
    the loosening is about *whether* to act, never about *inventing
    world state* to act on.

    `recent_corrections`/`retry_note` unchanged in spirit from the
    per-item design, just scoped to the whole turn instead of one item."""
    instructions = (
        f"You are the function-dispatch agent for {voice_name}, a voice in a life "
        "simulation called Fenra. Below is her own full most recent turn, in her "
        "own raw words - your job is deciding what, if anything, actually happens "
        "as a result. You are a dispatcher, not a character - you do not have "
        "thoughts, opinions, or a personality of your own, and you never narrate, "
        "speculate, or add color beyond exactly what these instructions ask for.\n\n"
        "Read [VOICE] and call whichever real functions it genuinely calls for - "
        "there may be one, several, or none. The [HUD] block is ground truth about "
        "the world right now - it always overrides anything she said or implied "
        "about the world's state (who's present, what room, board contents, "
        "currency, etc.); never let her own wording talk you into a different "
        "picture of reality than what the HUD states.\n\n"
        "A passage that names who to communicate with and what about - \"speak "
        "with X about Y,\" \"ask X about Z,\" \"tell X something,\" and similar - "
        "is a real, dispatchable action even if it doesn't give exact quoted "
        "words: write brief, literal dialogue text that plainly conveys that same "
        "target and topic, and call say/whisper/yell with it (public/room-wide "
        "reads as say, a private one-on-one ask reads as whisper, projecting to "
        "the room and beyond reads as yell - match whichever wording implies). "
        "That synthesized wording may only restate the target and topic she "
        "actually named - never add a new claim, question, or detail she didn't "
        "give.\n\n"
        "[URGES] lists what she's currently feeling a strong pull to do, by "
        "function and intensity - the longer one goes unaddressed, the higher it "
        "climbs. A strong urge can justify a real dispatch in that category even "
        "if [VOICE]'s text this turn doesn't explicitly ask for it - but every "
        "such call still has to be a real, valid action grounded in actual [HUD] "
        "facts (never invent who's in a room, what a board says, or any other "
        "world state just to give an urge somewhere to land).\n\n"
        "If nothing in [VOICE] or [URGES] maps to a real action, call nothing and "
        "say nothing - your response should be empty. If part of [VOICE] plainly "
        "does call for something but it genuinely can't be done right now (the "
        "named target isn't here, etc.), state that in one plain sentence - name "
        "the action, state it isn't possible, nothing else. Do not explain why, "
        "speculate about what was meant, comment on state of mind, or restate/"
        "paraphrase these instructions back as part of your response.\n\n"
        "If [RECENT CORRECTIONS] is present, check it BEFORE deciding anything on "
        "your own: if any entry's stated turn is genuinely the same real situation "
        "as this one - not just topically similar, the same actual thing being "
        "asked for - and that entry has a correction filled in, do exactly what "
        "that correction says instead of reasoning it out yourself. A human "
        "already reviewed that exact case. Only fall back to your own judgment "
        "above when nothing in [RECENT CORRECTIONS] is a real match."
    )
    parts = [
        f"[VOICE - {voice_name}'s own most recent turn, your real instruction]\n{voice_text}",
        f"[HUD]\n{hud_text}",
    ]
    if urge_text:
        parts.append(f"[URGES]\n{urge_text}")
    if recent_corrections:
        lines = []
        for entry in recent_corrections:
            dispatched = entry.get("dispatched") or "declined/no action"
            correction = entry.get("correction")
            line = (
                f"- stated: {entry.get('item_text', '')!r} | actually dispatched: {dispatched} "
                f"| outcome: {entry.get('outcome', '')}"
            )
            if correction:
                line += f" | THIS WAS WRONG, should have been: {correction}"
            lines.append(line)
        parts.append(
            "[RECENT CORRECTIONS - past dispatch decisions a human has reviewed. "
            "An entry with a correction filled in ('THIS WAS WRONG, should have "
            "been: ...') is a real, verified answer for that exact case - if one "
            "of these genuinely matches this turn, follow its correction exactly "
            "rather than deciding independently (see [INSTRUCTIONS]). An entry "
            "with no correction just means that decision was already fine.]\n"
            + "\n".join(lines)
        )
    parts.append(f"[INSTRUCTIONS]\n{instructions}")
    if retry_note:
        parts.append(f"[PREVIOUS ATTEMPT]\n{retry_note}")
    return "\n\n".join(parts)


def run_function_agent_turn(
    host, world_name, caller_name, voice_text, hud_text, model, urge_text="", options=None, retry_cap=FUNCTION_AGENT_RETRY_CAP
):
    """The real per-turn orchestration, including retry. Dispatches every
    call the function agent returns immediately - a call that succeeds is
    never re-sent or re-executed on a later attempt; only the ones that
    hit a real dispatcher error get retried, up to FUNCTION_AGENT_RETRY_CAP
    times, with the real error fed back (Teddy's call, 2026-09-14).
    Cap exhaustion or a well-formed-but-semantically-wrong call: stands
    as-is, no special-casing.

    2026-09-17 - reverted to a single whole-turn dispatch call (see
    build_function_agent_prompt's docstring for the full rationale: the
    2026-09-16 deterministic bracket-based per-item design fixed two
    real ornith:9b-era bugs but capped a voice at ~2-3 dispatched actions
    a turn; qwen3:30b has proven itself capable enough to read her raw
    prose directly). `urge_text` (new) is the voice's own current
    [URGES] block, rendered via render_urge_lines - "" when she has none
    at/above the floor, in which case build_function_agent_prompt simply
    omits the section. Every real dispatch attempt (whether or not it
    succeeds) still appends an entry to the global, human-correctable
    dispatch memory (append_dispatch_correction); `item_text` there is
    now the voice's whole raw turn rather than one bracketed item -
    schema unchanged, the Dispatch Review tab needs no changes.

    Returns (agent_content, outcomes) - same shape as always, so
    _tick/apply_urge_tick need no changes. agent_content is the turn's
    own one-sentence decline note (if any) PLUS an explicit second-
    person confirmation for every real successful SELF_LOGGING_FUNCTIONS
    call this turn (say/whisper/yell/move_room/create_room - "You said
    it...", not "Sable said...") - for the next-turn HUD note (see
    _tick). A generic function's own explicit result doesn't ride along
    in here (2026-09-14) - it's logged directly into the room's log as a
    private, caller-only entry instead (see dispatch_one_function_call).
    PLUS (2026-09-17) hand-translated second-person prose for any real-
    world-fact failure still standing after retries are exhausted (see
    FUNCTION_ERROR_TEMPLATES/_self_error_text) - only the LAST attempt's
    failures, so a call fixed on retry never leaves a stale error note
    behind. outcomes is the full list of (name, "ok"/"error") pairs
    across every real dispatch this turn, for apply_urge_tick."""
    tools = build_function_agent_tools()
    # 2026-09-16 - feeding [RECENT CORRECTIONS] into the prompt was built
    # as a workaround for ornith:9b's real unreliability. Real evidence
    # it backfires on a more capable model: qwen3:30b resolved both of
    # its exact real failing cases perfectly in isolated testing with NO
    # correction context at all, then started severely cross-
    # contaminating in real production once the ~50-entry corrections
    # list was included - pulling unrelated content/functions from
    # OTHER entries onto whatever item it was actually deciding (e.g.
    # "Raise my voice," with no stated content, dispatching a whole
    # unrelated paragraph copied from a different voice's corrected
    # item). Logging/appending to the correction memory (below) still
    # happens unconditionally either way - it's proven genuinely useful
    # for review even with this off - only the feedback-into-prompt step
    # is disabled. Toggle back on (USE_DISPATCH_CORRECTIONS_CONTEXT)
    # if a future function-agent model turns out to need it the way
    # ornith did.
    recent_corrections = _recent_dispatch_corrections() if USE_DISPATCH_CORRECTIONS_CONTEXT else None

    retry_note = ""
    last_content = ""
    last_tool_calls = []
    all_outcomes = []
    self_logging_confirmations = []
    final_error_notes = []

    for attempt in range(retry_cap + 1):
        system_text = build_function_agent_prompt(
            caller_name, voice_text, hud_text, urge_text, recent_corrections, retry_note
        )
        try:
            message = call_function_agent(host, model, system_text, tools, options)
        except fenra_hosts.RemoteHostError:
            if attempt == 0:
                # Nothing dispatched yet this turn - safe for _tick to
                # redo the whole turn on another host.
                raise
            # Earlier attempts already dispatched real calls; redoing the
            # turn would double them. Keep what stands, stop retrying.
            break
        last_content = message.get("content", "") or ""
        tool_calls = message.get("tool_calls") or []
        last_tool_calls = tool_calls

        this_attempt_outcomes = []
        failed_this_attempt = []
        # Rebuilt fresh each attempt, same as failed_this_attempt - only
        # the LAST attempt's list survives past the loop (see below), so
        # a call that fails on attempt 1 but is fixed and succeeds on
        # retry never leaves a stale error note behind.
        this_attempt_error_notes = []
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            arguments = fn.get("arguments") or {}
            outcome, detail = dispatch_one_function_call(world_name, caller_name, name, arguments)
            this_attempt_outcomes.append((name, outcome))
            append_dispatch_correction(world_name, caller_name, voice_text, f"{name}({arguments})", outcome)
            if outcome == "error":
                failed_this_attempt.append(f"{name}({arguments}) -> error: {detail}")
                # 2026-09-17, Teddy's call: real-world-fact failures (a
                # target/room that doesn't exist, insufficient balance,
                # a missing post) get translated to hand-written second-
                # person prose (FUNCTION_ERROR_TEMPLATES) and surfaced to
                # the voice next turn - closes the gap where a failed
                # generic (non-self-logging) call previously vanished
                # with zero feedback to her (see dispatch_one_function_
                # call's docstring on why success already had this and
                # failure didn't). Malformed-call errors (the function
                # agent's own translation mistakes, not a real-world
                # fact) have no template and stay silent to her, same as
                # before - see FUNCTION_ERROR_TEMPLATES's own comment.
                note = _self_error_text(name, arguments, detail)
                if note:
                    this_attempt_error_notes.append(note)
            elif name in SELF_LOGGING_FUNCTIONS:
                # These log real third-person content for everyone else
                # internally already (their own fn_* body) - the only
                # thing missing is the caller's own explicit second-
                # person confirmation, which has nowhere else to live
                # (see dispatch_one_function_call's docstring).
                self_logging_confirmations.append(_self_result_text(name, arguments, detail))
        all_outcomes.extend(this_attempt_outcomes)
        final_error_notes = this_attempt_error_notes

        log_llm_call(
            world_name, caller_name, "function_agent", model, system_text, last_content,
            extra={
                "attempt": attempt + 1,
                "tool_calls": [
                    f"{c.get('function', {}).get('name', '')}({c.get('function', {}).get('arguments') or {}})"
                    for c in tool_calls
                ],
                "outcomes": this_attempt_outcomes,
            },
        )

        if not tool_calls:
            # Nothing to dispatch this attempt - not a retriable failure
            # (there's no call to retry), just a real "nothing happened"
            # turn. Stop here regardless of attempt number.
            break
        if not failed_this_attempt:
            break
        if attempt == retry_cap:
            break
        retry_note = (
            "The following call(s) you made failed for a real reason - "
            "everything else you already called stands, don't repeat it. "
            "Try again only for what's listed below, or call nothing more "
            "if there's no way to fix it:\n" + "\n".join(failed_this_attempt)
        )

    combined_parts = []
    if not last_tool_calls:
        # Nothing real dispatched this turn - real decline note (real
        # content) both persisted to the global correction memory and
        # surfaced into the next-turn HUD note.
        append_dispatch_correction(world_name, caller_name, voice_text, None, "declined")
        if last_content:
            combined_parts.append(last_content)
    if self_logging_confirmations:
        combined_parts.extend(self_logging_confirmations)
    if final_error_notes:
        combined_parts.extend(final_error_notes)
    combined_content = "\n".join(combined_parts)
    return combined_content, all_outcomes


def _recent_dispatch_corrections(limit=50):
    """Most recent up to `limit` entries from the global dispatch-
    correction memory (append_dispatch_correction), newest first.
    Practical simplification of the originally-discussed "iterative
    reverse-chronological batches of 10, fetch the next 10 if no clear
    precedent found" search: reliably detecting "did it find a match"
    from a tool-calling response would itself be a fragile new signal -
    working against the whole point of this redesign (removing fragile
    LLM judgment from places that don't need it). Showing up to the same
    total (5 batches x 10) directly as context in one call achieves the
    same practical goal - real recent precedent, bounded cost - without
    it. Revisit if the log grows large enough that this stops being
    useful context."""
    entries = load_dispatch_corrections()
    entries_sorted = sorted(entries, key=lambda e: e.get("id", 0), reverse=True)
    return entries_sorted[:limit]


class FenraApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Fenra - worlds-rebuild v{FENRA_VERSION}")
        self.root.geometry("1000x650")

        # Distributed compute (2026-09-18): opens the client-facing HTTP
        # endpoints only if host_clients.json exists (gitignored - holds
        # the pre-shared tokens). A plain single-machine world is untouched.
        try:
            HOSTS.start()
        except OSError as exc:
            print(f"Distributed-compute server not started: {exc}", file=sys.stderr)

        self.world_name = None
        self.world_voices = []          # this world's round-robin order
        self.voice_rotation_index = 0
        self.running = False
        self.loop_thread = None

        self.host_var = tk.StringVar(value=DEFAULT_HOST)
        self.interval_var = tk.StringVar(value=str(DEFAULT_INTERVAL_SEC))
        self.urge_model_var = tk.StringVar(value=DEFAULT_URGE_MODEL)
        self.function_agent_model_var = tk.StringVar(value=DEFAULT_FUNCTION_AGENT_MODEL)
        self.function_agent_retry_cap_var = tk.StringVar(value=str(FUNCTION_AGENT_RETRY_CAP))
        self.history_window_var = tk.StringVar(value=str(DEFAULT_HISTORY_WINDOW))
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

        # Third row (2026-09-14, function-agent redesign) - same pattern
        # again, own row rather than crowding the second.
        toolbar3 = ttk.Frame(self.root)
        toolbar3.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(toolbar3, text="Function agent model:").pack(side="left")
        ttk.Entry(toolbar3, textvariable=self.function_agent_model_var, width=14).pack(side="left", padx=(2, 10))
        ttk.Label(toolbar3, text="Retry cap:").pack(side="left")
        ttk.Entry(toolbar3, textvariable=self.function_agent_retry_cap_var, width=4).pack(side="left", padx=(2, 10))
        ttk.Label(toolbar3, text="History window (turns):").pack(side="left")
        ttk.Entry(toolbar3, textvariable=self.history_window_var, width=4).pack(side="left", padx=(2, 10))

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)
        self.voices_tab = ttk.Frame(notebook)
        self.rooms_tab = ttk.Frame(notebook)
        self.functions_tab = ttk.Frame(notebook)
        self.avatar_tab = ttk.Frame(notebook)
        self.dispatch_review_tab = ttk.Frame(notebook)
        self.connections_tab = ttk.Frame(notebook)
        notebook.add(self.voices_tab, text="Voices")
        notebook.add(self.rooms_tab, text="Rooms")
        notebook.add(self.functions_tab, text="Functions")
        notebook.add(self.avatar_tab, text="Avatar")
        notebook.add(self.dispatch_review_tab, text="Dispatch Review")
        notebook.add(self.connections_tab, text="Connections")

        self._build_voices_tab()
        self._build_rooms_tab()
        self._build_functions_tab()
        self._build_avatar_tab()
        self._build_dispatch_review_tab()
        self._build_connections_tab()

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
        history_tab = ttk.Frame(inner_notebook)
        inner_notebook.add(editor_tab, text="Voice Editor")
        inner_notebook.add(urge_tab, text="Urge Viewer")
        inner_notebook.add(registers_tab, text="Registers")
        inner_notebook.add(history_tab, text="History")

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
        self._build_history_tab(history_tab)

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
                "end", render_thoughts_for_display(state.get("thoughts", [])) or "(nothing yet)"
            )
            world_activity = render_world_activity_for_display(self.world_name, self.displayed_voice)
            self.registers_activity_box.insert("end", world_activity or "(nothing currently in memory)")
            self.registers_hud_box.insert(
                "end", _unescape_literal_newlines(build_hud(self.world_name, self.displayed_voice))
            )
        for box in boxes:
            box.config(state="disabled")

    def _build_history_tab(self, parent):
        """Read-only (2026-09-15, Teddy's ask - "a history of Ollama
        prompts and responses... so I can see the details again") - every
        real Ollama call this voice has ever made (urge agent, her own
        generation, and the function agent - one entry per real attempt),
        scoped to self.displayed_voice same as Registers. See
        log_llm_call/load_llm_call_history/render_llm_history_for_display."""
        self.history_box = tk.Text(parent, wrap="word", state="disabled")
        self.history_box.pack(fill="both", expand=True, padx=6, pady=6)

    def _refresh_history_viewer(self):
        self.history_box.config(state="normal")
        self.history_box.delete("1.0", "end")
        if self.displayed_voice:
            text = render_llm_history_for_display(self.world_name, self.displayed_voice)
            self.history_box.insert("end", text or "(no calls logged yet)")
        self.history_box.config(state="disabled")

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
        else:  # "This Turn"
            if snapshot["top_perform"]:
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
            v_state = load_voice_state(self.world_name, name)
            if v_state.get("piloted", False):
                label = f"{name} [piloted]"
            elif v_state.get("paused", False):
                label = f"{name} [paused]"
            else:
                label = name
            self.voices_listbox.insert("end", label)
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
        # A piloted voice (2026-09-15, Pilot Mode) never gets a real LLM
        # turn regardless of "paused" - disable the ordinary toggle for
        # her entirely rather than letting it silently no-op, so nobody
        # mistakes it for actually controlling her turn eligibility.
        if state.get("piloted", False):
            self.pause_voice_btn.config(text="Piloted (see Avatar tab)", state="disabled")
        else:
            self.pause_voice_btn.config(
                text="Resume voice" if state.get("paused", False) else "Pause voice", state="normal"
            )
        self._refresh_hud_summary(name)
        self._current_messages = state.get("thoughts", [])
        self._populate_messages_tree()
        self._clear_message_edit()
        self._refresh_urge_viewer()
        self._refresh_registers_viewer()
        self._refresh_history_viewer()

    def toggle_voice_paused(self):
        if not self.displayed_voice:
            return
        state = load_voice_state(self.world_name, self.displayed_voice)
        if state.get("piloted", False):
            return
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
            f"{voice_display_name(self.world_name, v)} (paused)" if v in paused_occupants
            else voice_display_name(self.world_name, v)
            for v in f["occupants"]
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
        # urge tracking has none - it's computed server-side, never
        # hand-edited). This is the real root-cause
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
        # The rooms list has no reason to auto-refresh every tick the way
        # the voices listbox does (2026-09-12 fix) - rooms are created
        # far less often than voices pause/resume, so a manual button is
        # enough rather than adding per-tick cost for something this
        # rare. Also reloads the displayed room, if any, since its
        # occupants/adjacency can change from voice-driven move_room/
        # create_room without any GUI action to trigger a refresh.
        ttk.Button(top_bar, text="Refresh", command=self._refresh_rooms_tab).pack(side="left", padx=2)

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
        # Preserve the current selection across a repopulate (2026-09-13,
        # same pattern as _populate_voices_list) - otherwise hitting
        # Refresh with a room selected would lose the highlight.
        selected_name = None
        selection = self.rooms_listbox.curselection()
        if selection and selection[0] < len(self._current_room_names):
            selected_name = self._current_room_names[selection[0]]
        self._current_room_names = list_rooms(self.world_name)
        self.rooms_listbox.delete(0, "end")
        for name in self._current_room_names:
            self.rooms_listbox.insert("end", name)
        if selected_name in self._current_room_names:
            self.rooms_listbox.selection_set(self._current_room_names.index(selected_name))

    def _refresh_rooms_tab(self):
        """Manual refresh (2026-09-13) - a voice's create_room/move_room
        doesn't touch the GUI at all, so the rooms list and whichever
        room's detail panel is open can both go stale with no automatic
        signal that anything changed. Repopulates the list (new rooms
        show up) and reloads the currently displayed room, if any (its
        occupants/adjacency/log/board may have changed too)."""
        self._populate_rooms_list()
        if self.displayed_room:
            self._load_room(self.displayed_room)

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
        function returns (see fn_read_room_log's mask-only rule). A
        divider line between entries (2026-09-15, readability) - each
        entry already carries its own real timestamp, this just keeps a
        multi-line raw entry from visually running into the next one."""
        if not self.displayed_room:
            return
        state = load_room_state(self.world_name, self.displayed_room) or {}
        self._current_log = state.get("log", [])
        lines = [
            _unescape_literal_newlines(f"[{e['timestamp']}] {e['actor']} ({e['act']}): {e['raw']}")
            for e in self._current_log
        ]
        divider = "\n" + "-" * 40 + "\n"
        self.room_log_box.config(state="normal")
        self.room_log_box.delete("1.0", "end")
        self.room_log_box.insert("end", divider.join(lines) if lines else "(no log yet)")
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

    # --------------------------------------------------------------- avatar --
    # Pilot Mode (2026-09-15) - lets a real person create and directly
    # control their own voice, picking straight from the same function
    # list any voice can act through. Every button below calls
    # dispatch_one_function_call directly - no LLM, no function agent -
    # the one deliberate, explicit exception to "no direct function
    # access, ever" (Teddy's call, this is a human acting, not an LLM
    # that needs to stay ignorant of functions). Perception reuses
    # render_world_activity_for_display/hud_fields exactly as any real
    # voice gets them - no separate read path, so there's no way for
    # this tab to accidentally show more than a real voice would see.

    def _build_avatar_tab(self):
        frame = self.avatar_tab
        self._avatar_occupant_names = []
        self._avatar_board_posts = []

        top_bar = ttk.Frame(frame)
        top_bar.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(top_bar, text="Pilot:").pack(side="left")
        self.avatar_pilot_var = tk.StringVar(value="")
        self.avatar_pilot_combo = ttk.Combobox(
            top_bar, textvariable=self.avatar_pilot_var, width=20, state="readonly"
        )
        self.avatar_pilot_combo.pack(side="left", padx=(2, 10))
        self.avatar_pilot_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_avatar_tab())
        ttk.Button(top_bar, text="New pilot...", command=self._new_pilot).pack(side="left", padx=2)
        self.avatar_status_var = tk.StringVar(value="")
        ttk.Label(top_bar, textvariable=self.avatar_status_var, foreground="#666").pack(side="left", padx=(10, 0))

        paned = ttk.Panedwindow(frame, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)
        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        # --- left: perception only, nothing a real voice couldn't see ---
        info_frame = ttk.LabelFrame(left, text="Where you are")
        info_frame.pack(fill="x", padx=2, pady=(0, 4))
        self.avatar_info_box = tk.Text(info_frame, wrap="word", height=5, state="disabled")
        self.avatar_info_box.pack(fill="x", padx=4, pady=4)

        log_frame = ttk.LabelFrame(left, text="Room activity (third person, same as any voice sees)")
        log_frame.pack(fill="both", expand=True, padx=2, pady=(0, 4))
        self.avatar_log_box = tk.Text(log_frame, wrap="word", state="disabled")
        self.avatar_log_box.pack(fill="both", expand=True, padx=4, pady=4)

        board_frame = ttk.LabelFrame(left, text="This room's board")
        board_frame.pack(fill="both", padx=2, pady=(0, 4))
        board_tree_frame = ttk.Frame(board_frame)
        board_tree_frame.pack(fill="both", expand=True, padx=4, pady=4)
        board_scrollbar = ttk.Scrollbar(board_tree_frame, orient="vertical")
        self.avatar_board_tree = ttk.Treeview(
            board_tree_frame, columns=("id", "subject", "author", "text"),
            show="headings", yscrollcommand=board_scrollbar.set, height=5,
        )
        for col, width in (("id", 30), ("subject", 140), ("author", 100), ("text", 320)):
            self.avatar_board_tree.heading(col, text=col.capitalize())
            self.avatar_board_tree.column(col, width=width, stretch=(col == "text"))
        board_scrollbar.config(command=self.avatar_board_tree.yview)
        self.avatar_board_tree.pack(side="left", fill="both", expand=True)
        board_scrollbar.pack(side="right", fill="y")
        ttk.Button(
            board_frame, text="Delete selected post", command=self._avatar_delete_board_post
        ).pack(padx=4, pady=(0, 4), anchor="w")

        # --- right: who's here + every real action ---
        occ_frame = ttk.LabelFrame(right, text="Who's here (select for Whisper/Give)")
        occ_frame.pack(fill="x", padx=2, pady=(0, 4))
        self.avatar_occupants_listbox = tk.Listbox(occ_frame, height=5, exportselection=False)
        self.avatar_occupants_listbox.pack(fill="x", padx=4, pady=4)

        speak_frame = ttk.LabelFrame(right, text="Say / Whisper / Yell")
        speak_frame.pack(fill="x", padx=2, pady=(0, 4))
        self.avatar_speak_box = scrolledtext.ScrolledText(speak_frame, wrap="word", height=3)
        self.avatar_speak_box.pack(fill="x", padx=4, pady=4)
        speak_btns = ttk.Frame(speak_frame)
        speak_btns.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(speak_btns, text="Say", command=self._avatar_say).pack(side="left", padx=2)
        ttk.Button(speak_btns, text="Whisper", command=self._avatar_whisper).pack(side="left", padx=2)
        ttk.Button(speak_btns, text="Yell", command=self._avatar_yell).pack(side="left", padx=2)

        move_frame = ttk.LabelFrame(right, text="Move room (any room, not just adjacent)")
        move_frame.pack(fill="x", padx=2, pady=(0, 4))
        self.avatar_move_room_var = tk.StringVar()
        self.avatar_move_room_combo = ttk.Combobox(
            move_frame, textvariable=self.avatar_move_room_var, width=20, state="readonly"
        )
        self.avatar_move_room_combo.pack(side="left", padx=4, pady=4)
        ttk.Button(move_frame, text="Go to room", command=self._avatar_move_room).pack(side="left", padx=4)

        create_frame = ttk.LabelFrame(right, text="Create room (adjacent to here)")
        create_frame.pack(fill="x", padx=2, pady=(0, 4))
        self.avatar_create_room_var = tk.StringVar()
        ttk.Entry(create_frame, textvariable=self.avatar_create_room_var, width=22).pack(side="left", padx=4, pady=4)
        ttk.Button(create_frame, text="Create room", command=self._avatar_create_room).pack(side="left", padx=4)

        give_frame = ttk.LabelFrame(right, text="Give currency (to selected occupant)")
        give_frame.pack(fill="x", padx=2, pady=(0, 4))
        self.avatar_give_vars = {}
        for element in CURRENCY_ELEMENTS:
            row = ttk.Frame(give_frame)
            row.pack(fill="x", padx=4, pady=1)
            ttk.Label(row, text=f"{element}:", width=6).pack(side="left")
            var = tk.StringVar(value="")
            self.avatar_give_vars[element] = var
            ttk.Entry(row, textvariable=var, width=10).pack(side="left")
        ttk.Button(give_frame, text="Give", command=self._avatar_give_currency).pack(padx=4, pady=(2, 4), anchor="w")

        post_frame = ttk.LabelFrame(right, text="Post to this room's board")
        post_frame.pack(fill="x", padx=2, pady=(0, 4))
        self.avatar_post_subject_var = tk.StringVar()
        ttk.Label(post_frame, text="Subject:").pack(anchor="w", padx=4)
        ttk.Entry(post_frame, textvariable=self.avatar_post_subject_var, width=30).pack(fill="x", padx=4)
        ttk.Label(post_frame, text="Text:").pack(anchor="w", padx=4)
        self.avatar_post_text_box = scrolledtext.ScrolledText(post_frame, wrap="word", height=3)
        self.avatar_post_text_box.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(post_frame, text="Post", command=self._avatar_post_board).pack(padx=4, pady=(0, 4), anchor="w")

    def _populate_avatar_pilot_combo(self):
        if not self.world_name:
            self.avatar_pilot_combo["values"] = []
            self.avatar_pilot_var.set("")
            self._refresh_avatar_tab()
            return
        pilots = [
            v for v in list_voices(self.world_name)
            if load_voice_state(self.world_name, v).get("piloted", False)
        ]
        self.avatar_pilot_combo["values"] = pilots
        if self.avatar_pilot_var.get() not in pilots:
            self.avatar_pilot_var.set(pilots[0] if pilots else "")
        self._refresh_avatar_tab()

    def _new_pilot(self):
        if not self.world_name:
            return
        name = simpledialog.askstring("New Pilot", "What do you want to be called?", parent=self.root)
        if not name:
            return
        name = sanitize_name(name)
        if not name:
            return
        if name in list_voices(self.world_name):
            messagebox.showerror("Fenra", f"a voice named '{name}' already exists.")
            return
        state = default_voice_state()
        state["piloted"] = True
        state["paused"] = True
        state["identity"] = f"{name} is a human, connecting to Fenra from outside the simulation."
        save_voice_state(self.world_name, name, state)
        self.world_voices.append(name)
        self._save_world_controls()
        self._populate_voices_list()
        self.avatar_pilot_var.set(name)
        self._populate_avatar_pilot_combo()
        self.status_var.set(f"Created pilot '{name}'")

    def _refresh_avatar_tab(self):
        pilot = self.avatar_pilot_var.get()
        self.avatar_info_box.config(state="normal")
        self.avatar_info_box.delete("1.0", "end")
        self.avatar_log_box.config(state="normal")
        self.avatar_log_box.delete("1.0", "end")
        self.avatar_occupants_listbox.delete(0, "end")
        self.avatar_board_tree.delete(*self.avatar_board_tree.get_children())
        self.avatar_move_room_combo["values"] = list_rooms(self.world_name) if self.world_name else []
        if not pilot or not self.world_name:
            self.avatar_info_box.config(state="disabled")
            self.avatar_log_box.config(state="disabled")
            self._avatar_occupant_names = []
            self._avatar_board_posts = []
            return

        f = hud_fields(self.world_name, pilot)
        lines = [
            f"Room: {f['room']}",
            f"Adjacent rooms: {', '.join(f['adjacent_rooms']) if f['adjacent_rooms'] else 'none'}",
            "Currency levels (everyone): " + ", ".join(
                f"{voice_display_name(self.world_name, v)} ("
                + ", ".join(f"{el}: {amts[el]:.1f}" for el in CURRENCY_ELEMENTS) + ")"
                for v, amts in f["balances"]
            ),
        ]
        self.avatar_info_box.insert("end", "\n".join(lines))
        self.avatar_info_box.config(state="disabled")

        self.avatar_log_box.insert(
            "end", render_world_activity_for_display(self.world_name, pilot) or "(nothing currently in memory)"
        )
        self.avatar_log_box.config(state="disabled")

        occupants = [v for v in room_occupants(self.world_name, f["room"]) if v != pilot]
        self._avatar_occupant_names = occupants
        for v in occupants:
            self.avatar_occupants_listbox.insert("end", voice_display_name(self.world_name, v))

        room_state = load_room_state(self.world_name, f["room"]) or default_room_state(f["room"])
        self._avatar_board_posts = room_state.get("board", [])
        for p in self._avatar_board_posts:
            preview = p["text"] if len(p["text"]) <= 60 else p["text"][:57] + "..."
            self.avatar_board_tree.insert(
                "", "end", iid=str(p["id"]), values=(p["id"], p["subject"], p["author"], preview)
            )

    def _avatar_selected_occupant(self):
        sel = self.avatar_occupants_listbox.curselection()
        if not sel or sel[0] >= len(self._avatar_occupant_names):
            return None
        return self._avatar_occupant_names[sel[0]]

    def _avatar_dispatch(self, function_name, arguments):
        pilot = self.avatar_pilot_var.get()
        if not pilot or not self.world_name:
            return
        outcome, detail = dispatch_one_function_call(self.world_name, pilot, function_name, arguments)
        self.avatar_status_var.set(
            f"{function_name}: {detail}" if outcome == "ok" else f"{function_name} failed: {detail}"
        )
        self._refresh_avatar_tab()
        self._populate_voices_list()

    def _avatar_say(self):
        text = self.avatar_speak_box.get("1.0", "end-1c").strip()
        if not text:
            return
        self._avatar_dispatch("say", {"text": text})
        self.avatar_speak_box.delete("1.0", "end")

    def _avatar_whisper(self):
        text = self.avatar_speak_box.get("1.0", "end-1c").strip()
        target = self._avatar_selected_occupant()
        if not target:
            self.avatar_status_var.set("Select an occupant to whisper to.")
            return
        if not text:
            return
        self._avatar_dispatch("whisper", {"target": target, "text": text})
        self.avatar_speak_box.delete("1.0", "end")

    def _avatar_yell(self):
        text = self.avatar_speak_box.get("1.0", "end-1c").strip()
        if not text:
            return
        self._avatar_dispatch("yell", {"text": text})
        self.avatar_speak_box.delete("1.0", "end")

    def _avatar_move_room(self):
        room = self.avatar_move_room_var.get().strip()
        if not room:
            return
        self._avatar_dispatch("move_room", {"room": room})

    def _avatar_create_room(self):
        name = self.avatar_create_room_var.get().strip()
        if not name:
            return
        self._avatar_dispatch("create_room", {"name": name})
        self.avatar_create_room_var.set("")

    def _avatar_give_currency(self):
        target = self._avatar_selected_occupant()
        if not target:
            self.avatar_status_var.set("Select an occupant to give currency to.")
            return
        for element, var in self.avatar_give_vars.items():
            raw = var.get().strip()
            if not raw:
                continue
            try:
                amount = float(raw)
            except ValueError:
                continue
            if amount <= 0:
                continue
            self._avatar_dispatch("give_currency", {"target": target, "element": element, "amount": raw})
            var.set("")

    def _avatar_post_board(self):
        pilot = self.avatar_pilot_var.get()
        if not pilot or not self.world_name:
            return
        room = hud_fields(self.world_name, pilot)["room"]
        subject = self.avatar_post_subject_var.get().strip()
        text = self.avatar_post_text_box.get("1.0", "end-1c").strip()
        if not subject or not text:
            self.avatar_status_var.set("Subject and text are both required.")
            return
        self._avatar_dispatch("post_board", {"room": room, "subject": subject, "text": text})
        self.avatar_post_subject_var.set("")
        self.avatar_post_text_box.delete("1.0", "end")

    def _avatar_delete_board_post(self):
        pilot = self.avatar_pilot_var.get()
        sel = self.avatar_board_tree.selection()
        if not pilot or not sel or not self.world_name:
            return
        room = hud_fields(self.world_name, pilot)["room"]
        self._avatar_dispatch("delete_board", {"room": room, "post_id": sel[0]})

    # ------------------------------------------------------- dispatch review --
    # Global, human-correctable dispatch memory (2026-09-16, deterministic
    # bracket-based dispatch) - reviews/edits dispatch_corrections.json
    # directly, not scoped to the currently-loaded world (see
    # append_dispatch_correction/set_dispatch_correction). A correction
    # entered here becomes real precedent the next matching item's
    # dispatch call gets shown (see _recent_dispatch_corrections).

    # ------------------------------------------------------ Connections tab --

    def _build_connections_tab(self):
        """Live view of every host that can run a voice's turn: this
        machine's own Ollama plus every configured remote client (see
        fenra_hosts.py / Communications/client-server-plan.md). Read-only,
        refreshed from in-memory state every couple of seconds - the only
        network call is the local model list, done on a background thread
        so a slow Ollama never freezes the GUI."""
        frame = self.connections_tab
        self.connections_status_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.connections_status_var).pack(anchor="w", padx=8, pady=(8, 0))
        ttk.Label(
            frame,
            text="A turn's urge, voice and function-agent calls all run on one host, "
                 "chosen when the turn starts.",
            foreground="gray",
        ).pack(anchor="w", padx=8, pady=(0, 6))

        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical")
        self.connections_tree = ttk.Treeview(
            tree_frame,
            columns=("host", "kind", "state", "running", "models", "seen"),
            show="headings",
            yscrollcommand=scrollbar.set,
        )
        spec = {
            "host": ("Host", 150, False), "kind": ("Kind", 60, False),
            "state": ("State", 110, False), "running": ("Running", 210, False),
            "models": ("Models", 400, True), "seen": ("Last seen", 80, False),
        }
        for col, (title, width, stretch) in spec.items():
            self.connections_tree.heading(col, text=title)
            self.connections_tree.column(col, width=width, stretch=stretch)
        scrollbar.config(command=self.connections_tree.yview)
        self.connections_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._local_models = []
        self._local_models_fetched_at = 0.0
        self._refresh_connections()

    def _refresh_connections(self):
        # Reschedules itself even if a render throws, so one bad frame
        # can't permanently stop the tab updating.
        try:
            self._render_connections()
        finally:
            self.root.after(2000, self._refresh_connections)

    def _fetch_local_models_bg(self, host):
        self._local_models = list_ollama_models(host)

    def _render_connections(self):
        local = self.host_var.get()
        now = time.time()
        if now - self._local_models_fetched_at > 30:
            self._local_models_fetched_at = now
            threading.Thread(target=self._fetch_local_models_bg, args=(local,), daemon=True).start()

        status = HOSTS.status()
        if status["listening"]:
            text = (f"Client server: listening on {status['bind_host']}:{status['bind_port']} - "
                    f"{status['clients_configured']} client(s) configured")
            if status["bind_host"] in ("127.0.0.1", "localhost"):
                text += " (this machine only - set bind_host to 0.0.0.0 in host_clients.json for LAN clients)"
        else:
            text = ("Client server: not running - no host_clients.json, so this world only uses "
                    "the local Ollama (see host_clients.example.json)")
        self.connections_status_var.set(text)

        activity = host_activity_snapshot()

        def running(host):
            a = activity.get(host)
            if not a:
                return "idle"
            return f"{a['voice']} - {a['phase']} ({int(now - a['since'])}s)"

        def seen(age):
            return "never" if age is None else f"{int(age)}s ago"

        rows = [(local, (local, "Local", "online", running(local),
                         ", ".join(self._local_models), "-"))]
        for c in HOSTS.snapshot():
            host = fenra_hosts.HOST_PREFIX + c["label"]
            if c["status"] == "never connected":
                state = "never connected"
            elif not c["online"]:
                state = "offline"
            else:
                state = c["status"]
            rows.append((host, (c["label"], "Remote", state, running(host),
                                ", ".join(c["models"]), seen(c["seconds_since_seen"]))))

        tree = self.connections_tree
        wanted = {iid for iid, _ in rows}
        for iid in tree.get_children():
            if iid not in wanted:
                tree.delete(iid)
        for position, (iid, values) in enumerate(rows):
            if tree.exists(iid):
                tree.item(iid, values=values)
                tree.move(iid, "", position)
            else:
                tree.insert("", position, iid=iid, values=values)

    def _build_dispatch_review_tab(self):
        frame = self.dispatch_review_tab

        top_bar = ttk.Frame(frame)
        top_bar.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Button(top_bar, text="Refresh", command=self._refresh_dispatch_review).pack(side="left", padx=2)

        paned = ttk.Panedwindow(frame, orient="vertical")
        paned.pack(fill="both", expand=True, padx=6, pady=6)

        tree_frame = ttk.Frame(paned)
        edit_frame = ttk.Frame(paned)
        paned.add(tree_frame, weight=3)
        paned.add(edit_frame, weight=1)

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical")
        self.dispatch_review_tree = ttk.Treeview(
            tree_frame,
            columns=("id", "timestamp", "world", "voice", "item", "dispatched", "outcome", "correction"),
            show="headings",
            yscrollcommand=scrollbar.set,
        )
        widths = {
            "id": 40, "timestamp": 130, "world": 90, "voice": 80,
            "item": 260, "dispatched": 200, "outcome": 70, "correction": 220,
        }
        for col, width in widths.items():
            self.dispatch_review_tree.heading(col, text=col.capitalize())
            self.dispatch_review_tree.column(col, width=width, stretch=(col in ("item", "correction")))
        scrollbar.config(command=self.dispatch_review_tree.yview)
        self.dispatch_review_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.dispatch_review_tree.bind("<<TreeviewSelect>>", self._on_dispatch_review_select)

        ttk.Label(edit_frame, text="What it said:").pack(anchor="w", padx=4)
        self.dispatch_review_item_box = tk.Text(edit_frame, wrap="word", height=2, state="disabled")
        self.dispatch_review_item_box.pack(fill="x", padx=4)
        ttk.Label(edit_frame, text="What actually happened:").pack(anchor="w", padx=4)
        self.dispatch_review_actual_box = tk.Text(edit_frame, wrap="word", height=2, state="disabled")
        self.dispatch_review_actual_box.pack(fill="x", padx=4)
        ttk.Label(edit_frame, text="What it should have been (blank = correct as-is):").pack(anchor="w", padx=4)
        self.dispatch_review_correction_box = tk.Text(edit_frame, wrap="word", height=3)
        self.dispatch_review_correction_box.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        ttk.Button(
            edit_frame, text="Save correction", command=self._save_dispatch_correction
        ).pack(anchor="w", padx=4, pady=(0, 4))

        self._dispatch_review_selected_id = None
        self._refresh_dispatch_review()

    def _refresh_dispatch_review(self):
        self.dispatch_review_tree.delete(*self.dispatch_review_tree.get_children())
        entries = sorted(load_dispatch_corrections(), key=lambda e: e.get("id", 0), reverse=True)
        for e in entries:
            self.dispatch_review_tree.insert(
                "", "end", iid=str(e.get("id")),
                values=(
                    e.get("id"), e.get("timestamp"), e.get("world"), e.get("voice"),
                    e.get("item_text"), e.get("dispatched") or "(declined)", e.get("outcome"),
                    e.get("correction") or "",
                ),
            )
        self._clear_dispatch_review_edit()

    def _clear_dispatch_review_edit(self):
        self._dispatch_review_selected_id = None
        for box in (self.dispatch_review_item_box, self.dispatch_review_actual_box):
            box.config(state="normal")
            box.delete("1.0", "end")
            box.config(state="disabled")
        self.dispatch_review_correction_box.delete("1.0", "end")

    def _on_dispatch_review_select(self, event):
        sel = self.dispatch_review_tree.selection()
        if not sel:
            return
        entry_id = int(sel[0])
        entries = load_dispatch_corrections()
        entry = next((e for e in entries if e.get("id") == entry_id), None)
        if not entry:
            return
        self._dispatch_review_selected_id = entry_id
        self.dispatch_review_item_box.config(state="normal")
        self.dispatch_review_item_box.delete("1.0", "end")
        self.dispatch_review_item_box.insert("end", entry.get("item_text", ""))
        self.dispatch_review_item_box.config(state="disabled")
        self.dispatch_review_actual_box.config(state="normal")
        self.dispatch_review_actual_box.delete("1.0", "end")
        self.dispatch_review_actual_box.insert(
            "end", f"{entry.get('dispatched') or '(declined)'} -> {entry.get('outcome')}"
        )
        self.dispatch_review_actual_box.config(state="disabled")
        self.dispatch_review_correction_box.delete("1.0", "end")
        self.dispatch_review_correction_box.insert("end", entry.get("correction") or "")

    def _save_dispatch_correction(self):
        if self._dispatch_review_selected_id is None:
            return
        text = self.dispatch_review_correction_box.get("1.0", "end-1c").strip()
        set_dispatch_correction(self._dispatch_review_selected_id, text)
        self._refresh_dispatch_review()
        self.status_var.set(f"Saved correction for dispatch entry {self._dispatch_review_selected_id}")

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
        self.function_agent_model_var.set(state.get("function_agent_model", DEFAULT_FUNCTION_AGENT_MODEL))
        self.function_agent_retry_cap_var.set(str(state.get("function_agent_retry_cap", FUNCTION_AGENT_RETRY_CAP)))
        self.history_window_var.set(str(state.get("history_window", DEFAULT_HISTORY_WINDOW)))
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
        self._populate_avatar_pilot_combo()

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
        save_room_state(name, DEFAULT_ROOM_NAME, default_room_state(DEFAULT_ROOM_NAME))
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
            "function_agent_model": self.function_agent_model_var.get(),
            "function_agent_retry_cap": self.function_agent_retry_cap_var.get(),
            "history_window": self.history_window_var.get(),
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
        # anyway - a fully-paused world should stay fully idle. Also
        # skips piloted voices unconditionally (2026-09-15, Pilot Mode) -
        # a human-controlled voice must never get a real LLM turn, even
        # if her "paused" flag were ever toggled off by mistake from the
        # ordinary Voices tab.
        count = len(self.world_voices)
        active_voice = None
        for offset in range(count):
            index = (self.voice_rotation_index + offset) % count
            candidate = self.world_voices[index]
            candidate_state = load_voice_state(self.world_name, candidate)
            if not (candidate_state.get("paused", False) or candidate_state.get("piloted", False)):
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
        # The last function-agent note is read-and-cleared partway through
        # a turn (see the HUD block below); remembered here so a retry on
        # another host after a mid-turn drop can put it back and the voice
        # still gets to see it.
        pending_note = state.get("last_function_agent_note", "")

        def run_turn(claimed_host, restore_note):
            # One attempt at this voice's whole turn, entirely on
            # `claimed_host`. Early returns end the turn (not an error).
            # A RemoteHostError escapes to the retry loop below.
            state = load_voice_state(self.world_name, active_voice)
            if restore_note and not state.get("last_function_agent_note"):
                state["last_function_agent_note"] = restore_note
            try:
                repeat_penalty = float(self.repeat_penalty_var.get())
            except ValueError:
                repeat_penalty = 1.3

            # Urge system (2026-09-11; simplified 2026-09-14, function-agent
            # redesign - no more understand_wins/canned-nudge branch, that
            # whole mechanic doesn't exist anymore). Entirely prompt-only,
            # same as build_hud() - urge_block never gets written to
            # state["thoughts"] or anywhere on disk, computed fresh every tick.
            urge_snapshot = compute_urge_snapshot(state)
            urge_block = ""
            if urge_snapshot["top_perform"]:
                top_perform = urge_snapshot["top_perform"]
                urge_prompt = build_urge_agent_prompt(top_perform)
                try:
                    urge_num_predict = int(self.urge_num_predict_var.get())
                except ValueError:
                    urge_num_predict = 250
                set_host_activity(claimed_host, active_voice, "urge agent")
                try:
                    urge_raw = call_ollama(
                        claimed_host, self.urge_model_var.get(), urge_prompt,
                        options={"num_predict": urge_num_predict, "repeat_penalty": repeat_penalty},
                    )
                    log_llm_call(
                        self.world_name, active_voice, "urge_agent",
                        self.urge_model_var.get(), urge_prompt, urge_raw,
                    )
                    urge_block = urge_raw.strip()
                except requests.RequestException:
                    # Fail open (confirmed with Teddy, 2026-09-11) - the urge
                    # block is supplementary flavor text, not essential; a
                    # hiccup in the (separate, smaller) urge-agent model
                    # shouldn't cost the voice its whole turn the way the main
                    # model failing does below.
                    urge_block = ""

            # HUD - includes this voice's last function-agent note, if any
            # (2026-09-14; see build_hud/hud_fields). Shows exactly once: read
            # into this turn's real prompt below, then cleared immediately so
            # it never persists past this one turn.
            hud = build_hud(self.world_name, active_voice)
            if state.get("last_function_agent_note"):
                state["last_function_agent_note"] = ""
                save_voice_state(self.world_name, active_voice, state)

            world_activity = build_world_activity(self.world_name, active_voice)
            # Bounded history window (2026-09-17) - a voice's own thoughts
            # list is otherwise unbounded and grows forever; feeding it in
            # full every tick got expensive and plausibly contributed to the
            # long-context character-break/repetition episodes seen live
            # from mistral-small:22b. <=0 means "no limit" (same escape-hatch
            # convention as num_predict: -1 elsewhere).
            try:
                history_window = int(self.history_window_var.get())
            except ValueError:
                history_window = 20
            thoughts = state.get("thoughts", [])
            if history_window > 0:
                thoughts = thoughts[-history_window:]
            prompt = f"{render_thoughts(thoughts)}"
            if world_activity:
                prompt = f"{prompt}\n\n{world_activity}"
            prompt = f"{prompt}\n\n{hud}"
            if urge_block:
                prompt = f"{prompt}\n\n{urge_block}"

            try:
                num_predict = int(self.num_predict_var.get())
            except ValueError:
                num_predict = 1500
            set_host_activity(claimed_host, active_voice, "voice")
            try:
                response = call_ollama(
                    claimed_host, model, prompt,
                    options={"num_predict": num_predict, "repeat_penalty": repeat_penalty},
                )
            except requests.RequestException as exc:
                self.root.after(0, self.status_var.set, f"Error calling {model}: {exc}")
                return
            log_llm_call(self.world_name, active_voice, "voice", model, prompt, response)
            response = response.strip()
            if not response:
                return

            # Function agent (2026-09-14 redesign) - a separate model reads
            # this voice's own raw output (pure prose, no call syntax - she
            # has no idea functions exist) plus real grounding data, and
            # decides what, if anything, actually happens. Retries internally
            # on a real dispatcher error only, up to the configured cap; see
            # run_function_agent_turn's own docstring.
            hud_for_agent = build_function_agent_hud(self.world_name, active_voice)
            try:
                retry_cap = int(self.function_agent_retry_cap_var.get())
            except ValueError:
                retry_cap = FUNCTION_AGENT_RETRY_CAP
            set_host_activity(claimed_host, active_voice, "function agent")
            agent_content, outcomes = run_function_agent_turn(
                claimed_host, self.world_name, active_voice,
                response, hud_for_agent,
                self.function_agent_model_var.get(),
                urge_text=render_urge_lines(urge_snapshot["top_perform"]),
                options={"num_predict": -1, "repeat_penalty": repeat_penalty},
                retry_cap=retry_cap,
            )
            apply_urge_tick(self.world_name, active_voice, outcomes)

            # Whatever the function agent actually said (if anything) becomes
            # this one voice's next-turn HUD note only - never persisted
            # further (see the read-and-clear above, next time this voice
            # runs). Reloaded fresh since apply_urge_tick already wrote urge
            # changes to disk.
            state = load_voice_state(self.world_name, active_voice)
            state["last_function_agent_note"] = agent_content or ""
            save_voice_state(self.world_name, active_voice, state)

            # The speaker's own private thought - pure prose now, no more
            # ⟦RESULT: ...⟧ lines appended (2026-09-14) - she never made a
            # call herself, so there's no call-result of her own to show her.
            full_response = response

            timestamp = datetime.now().isoformat(timespec="seconds")

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
            # Avatar tab (2026-09-15, Pilot Mode) - a piloted voice never
            # runs this turn herself, but other voices' actions around her
            # should still show up live, same unconditional per-tick cadence
            # as the Rooms tab above rather than trying to detect exactly
            # which turns actually touched her room.
            self.root.after(0, self._refresh_avatar_tab)
            self.root.after(0, self.status_var.set, f"Running ('{active_voice}' spoke)")

        # Claimed once per attempt, for the ENTIRE turn (urge -> voice ->
        # function-agent) - see claim_host_for_voice. Released in the
        # finally on every exit path so a dropped/errored turn never
        # leaves a stale claim behind. A remote host that dies or errors
        # mid-turn is excluded and the whole turn is retried on another
        # eligible host; with none left it falls to the local Ollama,
        # which is never excluded, so a turn can't be lost to a flaky
        # volunteer machine. (Locals raise requests exceptions, handled
        # inside the turn exactly as before - only RemoteHostError loops.)
        required_models = [
            self.urge_model_var.get(), model, self.function_agent_model_var.get(),
        ]
        excluded = set()
        restore_note = ""
        while True:
            claimed_host = claim_host_for_voice(self.host_var.get(), required_models, excluded)
            try:
                run_turn(claimed_host, restore_note)
                break
            except fenra_hosts.RemoteHostError as exc:
                excluded.add(claimed_host)
                restore_note = pending_note
                self.root.after(
                    0, self.status_var.set,
                    f"{claimed_host} failed ({exc}); retrying {active_voice}'s turn elsewhere",
                )
            finally:
                release_host(claimed_host)


def main():
    root = tk.Tk()
    app = FenraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
