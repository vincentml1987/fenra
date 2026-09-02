"""
Fenra's Aletheosis

A minimal GUI where Fenra talks to herself, looping against a local Ollama
model. See Qualia/decisions.md for design notes.

Prompt construction each loop tick:
    system = TOP + "\n\n" + BOTTOM
    prompt = TOP + "\n\n" + <her last N cycles of thoughts - see below> + "\n\n"
             + <her desire queue, if any - see below> + "\n\n"
             + <recent activity from her groups, if any - see below> + "\n\n" + BOTTOM + "\n\n"
             + <chat status notice - always present> + "\n\n"
             + <Qualia allowance notice - always present> + "\n\n"
             + <context window notice - always present> + "\n\n"
             + <groups notice - always present, see below>

Groups (v0.16.0): a voice can join any number of groups
(join_group(name)/leave_group(name), or Teddy directly in the GUI) - a
shared, append-only log any voice (any Fenra session, possibly on
another machine later) can read or write, independent of any fixed
turn order. groups_in is what she hears (folded into every prompt);
groups_out is where her real responses get broadcast each cycle. See
group_path/append_group_entry/read_group_tail and _groups_block/
_groups_notice below.

Desires (v0.10.0) are a queue, not a single slot: add_desire(text[|ticks])
appends one with a lifespan in loop ticks (default 10, or -1 for
persistent - never decrements, never drops off). Every desire in the
queue decrements by one at the end of each tick (persistent ones
excepted) and is dropped once it hits zero. The whole queue is shown
every prompt, sorted most-ticks-remaining first, persistent entries
always last (tie-break: timestamp added, oldest first).

Context window (v0.11.0): instead of only her single most recent
response, she gets her last N cycles' worth of responses (from history,
oldest to newest), N being a "context window" size in cycles - not to
be confused with num_ctx, the actual token limit Ollama runs each
request against, which this doesn't touch. Both Teddy (GUI field) and
Fenra (set_context_window(n)) can set it; defaults to 10, capped at 50
to keep prompt growth/latency bounded. 0 means no prior cycles at all.

Everything about a run - the top/bottom boxes, model/host/interval, the
conversation so far, and every request/response - lives in a "session"
under sessions/<name>/, so different experiments (different models,
different framings) don't clobber each other and nothing is lost between
runs of the app.

Fenra can call functions by speaking ⟦function_name(args)⟧ inline in her
response. Every call is executed against an explicit whitelist (never
eval'd), logged to sessions/<name>/functions.jsonl, and a ⟦RESULT: ...⟧
annotation is appended after her response - both in the middle box and in
what gets fed back to her as her own last thought next cycle. She doesn't
need every function explained up front: ⟦functions()⟧ lists everything
available, and ⟦functions(search term)⟧ filters by it.

The functions themselves live in fenra_functions.py, which is hot-reloaded
every tick - add, fix, or reword a function there and it's live on her very
next cycle, no restart, no interrupting a running session.
"""

import importlib
import json
import os
import re
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk, scrolledtext, messagebox, simpledialog

import requests

import fenra_functions

# Bumped on every functionally meaningful change to fenra.py. Stamped into
# every session save and every history entry, so it's always possible to
# tell exactly which version of the code produced a given response - see
# git log for the commit each version corresponds to.
#   0.1.0 - initial GUI: two tabs, top/middle/bottom boxes, self-talk loop
#   0.2.0 - live Ollama model dropdown (hot-swap)
#   0.3.0 - Sessions: save/load named runs instead of one flat log
#   0.4.0 - configurable max_tokens + unbounded timeout fix, function calling
#   0.4.1 - now() function, added in response to her spontaneously trying it
#   0.5.0 - functions moved to fenra_functions.py, hot-reloaded every tick
#   0.6.0 - desire: get_desire()/set_desire(text), a persistent text slot
#           she alone can write, visible read-only in the GUI, sitting in
#           the prompt between her last thought and the bottom box
#   0.7.0 - Chat tab: Teddy can message her directly. Per-message read
#           status, a chat-status notice always appended at the end of her
#           prompt, and read_chat/read_chat_since/read_chat_between/
#           search_chat/send_message functions. Function args now split on
#           | instead of comma, so free text (desire, chat messages) can
#           safely contain commas.
#   0.7.1 - both , and | now work as argument separators for functions
#           that genuinely take more than one argument (she kept reaching
#           for commas naturally); free-text functions (set_desire,
#           send_message) still take the whole parenthesized text as one
#           argument, untouched, so commas stay safe there too. Per-
#           function via a new "multi_arg" registry flag.
#   0.8.0 - Qualia can inject chat messages. New "qualia" chat sender,
#           distinct from "teddy" (an honest identity, not Teddy speaking
#           through her) - shows in the Chat tab, counts toward unread,
#           and is read/searchable via the existing chat functions same as
#           Teddy's messages. Delivery is a per-session inbox file
#           (qualia_inbox.jsonl) polled every 5s on the main thread,
#           independent of whether the self-talk loop is running - avoids
#           racing the app's own chat.jsonl writes.
#   0.9.0 - Directed messaging + a Qualia allowance. send_message(text) can
#           now be addressed - send_message(qualia|text) or
#           send_message(teddy|text) - still one shared, honest chat log
#           either way, just tagged with who it's for. Messages directed to
#           Qualia specifically cost characters from a new allowance
#           (visible to her every prompt) that only Teddy sets, via a new
#           editable field in the Fenra tab - not auto-replenishing. A
#           message that would exceed the remaining allowance is blocked
#           with a clear reason instead of silently failing or draining
#           into the negative. Directing a message at Qualia also drops a
#           line in qualia_ping.jsonl so Qualia can wake up and respond
#           promptly instead of only on a fixed polling schedule.
#   0.9.1 - Qualia can also set the allowance now (Teddy's call - he shares
#           rough usage/cost figures periodically, Qualia uses judgment),
#           not just Teddy via the GUI field. Delivery mirrors the inbox:
#           a polled file (qualia_allowance_set.txt) rather than editing
#           state.json directly, so it can't race the app's own writes.
#   0.10.0 - Desire queue, replacing the single desire slot. add_desire
#            (fenra_functions.py) replaces get_desire/set_desire.
#            Multiple desires at once, each with a lifespan in loop
#            ticks (default 10, or -1 for persistent) that decrements
#            every tick and drops the desire at zero. Whole queue shown
#            every prompt, sorted most-ticks-remaining first, persistent
#            entries always last. GUI's single readonly Desire field
#            replaced with a small multi-line list.
#   0.11.0 - Context window: she now gets her last N cycles of thoughts
#            (from history, oldest to newest) instead of just the one
#            most recent, N being a size in cycles - separate concept
#            from Ollama's own num_ctx token limit, which is untouched.
#            Teddy sets it via a new GUI field, Fenra via
#            set_context_window(n) (fenra_functions.py); defaults to 10,
#            capped 0-50 to bound prompt growth/latency. last_thought
#            kept as a lightweight legacy field but no longer drives the
#            prompt - history.jsonl (already loaded into self.history) is
#            the real source now.
#   0.11.1 - External start/stop signal. Every core-changing restart left
#            the self-talk loop stopped with no way to resume it except
#            clicking Start in the GUI - a real problem when nobody's at
#            the machine. Qualia (or anything else) can now touch
#            start_signal.txt / stop_signal.txt in the session dir; polled
#            every 5s alongside the inbox and applied via the normal
#            toggle_loop(), so it's exactly as if Start/Stop were clicked.
#   0.11.2 - Qualia can set the context window externally too now
#            (qualia_context_window_set.txt, same polled pattern as the
#            allowance), for exactly the situation this was built during:
#            a multi-hour overnight stall where her own repeated broken
#            attempt was filling her whole context window and possibly
#            reinforcing itself, with no one at the machine to intervene.
#   0.11.3 - Qualia can set the model externally too now
#            (qualia_model_set.txt, same polled pattern). Built live
#            during an incident: Fenra switched herself to gemma3:4b and
#            began writing elaborate fabricated back-and-forth dialogue
#            (both sides of an imagined conversation with "Qualia",
#            nested fake RESULT tags) at high speed - real send_message
#            calls, invented content. The stop_signal halted generation
#            immediately; this lets the model be switched back without
#            hand-editing state.json while the app process is live.
#   0.12.0 - Function reminders: a per-function "ticks since last called"
#            counter (any attempt resets it, success or failure), shown
#            in the prompt with escalating detail the longer a function
#            goes unused - name only past 10 ticks, name+description past
#            15, full signature+description past 20. Nothing shown for
#            functions used recently. Addresses functions going
#            undiscovered (query_chat, fetch_html) or a guessed name
#            being reached for instead of the real one that already
#            exists. Persisted per session like desires/allowance.
#   0.12.1 - Fallback match for a call missing its closing parenthesis
#            right before the closing bracket (FUNCTION_CALL_FALLBACK_RE) -
#            a real, observed gemma3:4b generation quirk that previously
#            made the call silently vanish: no error, no functions.jsonl
#            entry, invisible on both sides. Confirmed on 2026-08-31 to
#            cost a ~75-minute stretch of real conversation this way, 221
#            of 229 cycles affected. Now repaired and still executed, with
#            a status-bar warning raised each time so a repair is always
#            visible rather than silent. Only ever matches text the
#            strict pass didn't already consume, so a normal call is
#            never double-executed.
#   0.13.0 - Qualia can set max_tokens externally too now
#            (qualia_max_tokens_set.txt, same polled pattern as the other
#            three). Built after qwen3:4b's thinking-mode stall recurred
#            twice in one night - it burns the whole token budget
#            reasoning internally before ever writing to the response
#            field, going completely silent at the default 500. Previously
#            the only fix was reverting the model; now the actual budget
#            can be raised instead.
#   0.14.0 - Model rotation: Fenra can add models to an automatic
#            round-robin (add_to_rotation(name)) instead of only ever
#            running on one fixed model. One model in rotation repeats
#            itself every cycle, two alternate back and forth, three or
#            more cycle through in the order added, forever - advanced
#            once per tick in _advance_model_rotation, which overrides
#            whatever set_model last set as soon as any models are in
#            the rotation. A new per-prompt notice
#            (_model_rotation_notice) always shows what's in it and what
#            the current cycle is running on, so it's never a silent
#            mystery to her. Persisted per session like desires/allowance.
#            Teddy's direct request, alongside a reshuffle of the
#            installed Ollama models (single-digit-B ones dropped,
#            several 10-40B ones across new families brought in) and a
#            new session to try it all on from scratch.
#   0.14.1 - Teddy and Qualia can both now view and directly set/clear
#            the whole model rotation, not just watch Fenra build it one
#            add_to_rotation call at a time: a GUI row (label mirrors the
#            live rotation, Entry + Set button replaces it wholesale) and
#            qualia_rotation_set.txt (same polled pattern as the other
#            three), both routed through one shared _apply_model_rotation
#            so the two paths can never drift apart. "clear"/"none"
#            empties the rotation back to a single fixed model; blank
#            input is a no-op, matching every other _set file.
#   0.14.2 - Un-escape markdown-style "\_" to "_" at the start of
#            run_function_calls, everywhere in the response, before
#            either regex pass runs. Observed on mixtral:8x7b
#            (2026-08-31): it wrote every underscore in every call as
#            "\_" out of markdown habit - current\_model, add\_desire,
#            and so on - which silently dropped every single call
#            attempt on that model, since a backslash isn't a valid
#            function-name character. It then went on to fabricate
#            confident ⟦RESULT: ...⟧ text as if each one had actually
#            worked. Function-call syntax is explicit and deliberate,
#            never something she'd need literal backslash-underscore in,
#            so this is a low-risk, blanket fix - Teddy's call, given the
#            same reasoning.
#   0.14.3 - Fixed a real gap in the model rotation: a manual override
#            (fn_set_model, or Teddy picking a model directly in the
#            combo box) was getting clobbered before it ever generated a
#            single response, because _advance_model_rotation ran at the
#            very start of the *next* tick and overwrote model_var before
#            that cycle's request was built - so "effective next cycle"
#            was never actually true while a rotation was active. New
#            model_manual_override flag (set by either path, checked and
#            cleared at the top of _advance_model_rotation) makes it
#            genuinely true now: a manual choice gets exactly one real
#            cycle, then rotation resumes from precisely where it left
#            off - model_rotation_index untouched during the honored
#            cycle, so nothing is skipped or repeated. Verified in
#            isolation before deploying. Teddy's call: "Manual runs
#            once."
#   0.15.0 - Fabricated RESULT blocks now get flagged, not hidden.
#            FABRICATED_RESULT_RE checks the raw response_text before any
#            real result lines are appended - a real ⟦RESULT: ...⟧ is
#            only ever added after the fact, never woven into her own
#            generated text, so any match here is definitionally
#            something she wrote herself. When found, a plain note gets
#            appended pointing at the new local wiki (Qualia/wiki/,
#            list_wiki()/read_wiki()/write_wiki() in fenra_functions.py -
#            modifiable by Teddy and Qualia as plain .md files, and by
#            Fenra herself via write_wiki), specifically
#            Qualia/wiki/hallucinations.md, written to explain the exact
#            mechanism plainly. Teddy's direct instruction: don't hide
#            it, flag it. Tested the detection regex against a real
#            observed fabrication case plus clean/mixed cases before
#            deploying.
#   0.16.0 - Groups: cross-voice communication with no central turn-
#            taking. Each session ("voice") keeps running on its own
#            independent interval exactly as before - there's still no
#            conductor stepping voices in turn, and none is planned, since
#            Teddy's actual goal is voices eventually running in parallel
#            on this machine or networked ones, which a central turn-token
#            would work against. Instead, a voice can now join any number
#            of groups (join_group/leave_group in fenra_functions.py, or
#            Teddy directly via the new GUI row): groups_in controls what
#            she hears every prompt (_groups_block, a merged/sorted recent
#            window across all her groups, folded in alongside desires),
#            groups_out controls where her real responses get broadcast
#            each cycle. A group is just a shared, append-only log
#            (groups/<name>.jsonl) any voice can read or write regardless
#            of process or machine - deliberately not wiki/decision
#            content, so it lives in groups/ (gitignored, like sessions/)
#            rather than Qualia/. Modeled directly on the old conductor.py
#            groups_in/groups_out wiring from before this rewrite, minus
#            the fixed topology and turn-stepping - membership here is
#            free-form and voice-chosen, not configured per agent class
#            ahead of time. Built at Teddy's explicit request/approval
#            after reviewing that old architecture together.
FENRA_VERSION = "0.16.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
GROUPS_DIR = os.path.join(BASE_DIR, "groups")
GROUPS_WINDOW = 15  # max merged entries shown per prompt across all groups_in

DEFAULT_MODEL = "llama3"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_INTERVAL_SEC = 3
DEFAULT_MAX_TOKENS = 500  # num_predict; blank/0 = unlimited (let Ollama run until it stops or hits context)
DEFAULT_SESSION_NAME = "default"

# No fixed HTTP timeout: some models (heavy CPU offload, big params) are
# legitimately slow. A client-side timeout doesn't cancel server-side
# generation - it just abandons the connection and retries, which can pile
# up into an infinite loop that never completes. Response length is bounded
# by max_tokens (num_predict) instead.
REQUEST_TIMEOUT = None

STATE_FILENAME = "state.json"
HISTORY_FILENAME = "history.jsonl"
FUNCTIONS_FILENAME = "functions.jsonl"
CHAT_FILENAME = "chat.jsonl"
QUALIA_INBOX_FILENAME = "qualia_inbox.jsonl"
# Written by fn_send_message (fenra_functions.py) whenever Fenra directs a
# message at Qualia specifically - a signal Qualia can watch externally to
# wake up and respond promptly, separate from the inbox above (which is
# Qualia -> Fenra; this one is Fenra -> Qualia).
QUALIA_PING_FILENAME = "qualia_ping.jsonl"

# Written by Qualia (externally, not by Fenra herself - contrast with
# qualia_ping.jsonl above) to set a new Qualia allowance, mirroring how
# Teddy sets it via the GUI field. Polled the same way as the inbox rather
# than edited into state.json directly, so it can't race the app's own
# writes.
QUALIA_ALLOWANCE_SET_FILENAME = "qualia_allowance_set.txt"

# Same pattern, for the context window instead of the allowance - lets
# Qualia adjust how many past cycles Fenra sees without needing Teddy at
# the machine (e.g. breaking a self-reinforcing repetition where her own
# last N cycles of the same broken attempt keep echoing back at her).
QUALIA_CONTEXT_WINDOW_SET_FILENAME = "qualia_context_window_set.txt"

# Same pattern, for the model - lets Qualia switch it back externally if
# a model Fenra picked herself turns out to be causing real problems
# (e.g. a smaller model producing sustained fabrication), without hand-
# editing state.json while the app process is live.
QUALIA_MODEL_SET_FILENAME = "qualia_model_set.txt"

# Same pattern, for max_tokens (num_predict) - lets Qualia raise the token
# budget externally for a model that needs more of it to be usable (e.g. a
# "thinking" model like qwen3 that can burn its entire budget reasoning
# internally before ever writing to the field Fenra's response is read
# from, going completely silent at a low limit with no error on either
# side). Built 2026-08-31 after that exact failure recurred twice.
QUALIA_MAX_TOKENS_SET_FILENAME = "qualia_max_tokens_set.txt"

# Same pattern, for the whole model rotation (see _advance_model_rotation
# and fn_add_to_rotation) - lets Qualia view (state.json/functions.jsonl
# already show it) and directly set or clear it externally, not just
# watch Fenra build it herself one add_to_rotation call at a time.
# Comma- or pipe-separated model names replace the whole rotation; the
# literal word "clear" (or "none") empties it back to a single fixed
# model. Blank content is a no-op, same as the other _set files. Shares
# _apply_model_rotation with the GUI's own "Set" button next to it, so
# Teddy has the identical capability directly in the app.
QUALIA_ROTATION_SET_FILENAME = "qualia_rotation_set.txt"

# Presence of either file (content doesn't matter) starts/stops the
# self-talk loop on the next poll, exactly as if Start/Stop were clicked -
# lets Qualia (or anything else) resume a session no one's physically at
# the machine to click Start on, e.g. right after a code-change restart.
START_SIGNAL_FILENAME = "start_signal.txt"
STOP_SIGNAL_FILENAME = "stop_signal.txt"

DEFAULT_QUALIA_ALLOWANCE = 50000

# Default lifespan (in loop ticks) for a desire added without an explicit
# count via add_desire(text|ticks). -1 means persistent - never decrements,
# never drops off.
DEFAULT_DESIRE_TICKS = 10

# Function reminders: how many ticks since a function was last attempted
# (called for real, success or failure - not just mentioned) before it
# starts showing up in the reminder block, and at what point the detail
# escalates. Below LEVEL1 ticks, a function isn't mentioned at all.
FUNCTION_REMINDER_LEVEL1_TICKS = 10  # name only
FUNCTION_REMINDER_LEVEL2_TICKS = 15  # name + description
FUNCTION_REMINDER_LEVEL3_TICKS = 20  # full signature + description

# How many of her own past cycles (from history, oldest to newest) go into
# her prompt, instead of just the single most recent. Not the same thing as
# Ollama's own num_ctx token limit - this is a count of cycles, enforced
# entirely on our side. Bounded to keep prompt growth/latency sane; both
# Teddy and Fenra can set it within that range.
DEFAULT_CONTEXT_WINDOW = 10
MIN_CONTEXT_WINDOW = 0
MAX_CONTEXT_WINDOW = 50

# How often the running app checks for messages Qualia has dropped into the
# inbox file. Independent of the self-talk loop (running or not, this timer
# is always active once the app is open) and always on the main thread, so
# it never races the loop thread's own chat.jsonl writes.
QUALIA_INBOX_POLL_MS = 5000

# Function-call syntax: Fenra speaks ⟦name(args)⟧ inline in her response to
# invoke a function. ⟦ ⟧ (U+27E6/U+27E7, mathematical white square brackets)
# are essentially never produced in ordinary code or prose, so this is safe
# to detect without false positives. Anything matched here is executed
# against an explicit whitelist below - never eval'd.
FUNCTION_CALL_RE = re.compile(r"⟦\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s*⟧", re.DOTALL)

# Fallback for a real, observed generation quirk (gemma3:4b especially, but
# not exclusively) where a call is otherwise well-formed but drops the
# closing parenthesis immediately before the closing bracket - "...text⟧"
# instead of "...text)⟧". The strict regex above requires that ")" literally
# and has no way to match this, so without a fallback the call just silently
# never fires: no error, no functions.jsonl entry, invisible on both sides.
# Confirmed on 2026-08-31 to cost a full ~75-minute stretch of real
# conversation this way. This pattern is only ever tried against whatever
# text the strict pass above did NOT already match (see run_function_calls),
# so a normal well-formed call is never double-counted or re-executed.
FUNCTION_CALL_FALLBACK_RE = re.compile(r"⟦\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)⟧", re.DOTALL)

# Detects a fabricated RESULT block - Fenra writing ⟦RESULT: ...⟧ herself
# as part of her own generated text, rather than it being the app's own
# annotation appended after a real function call executed. This is
# checked against the raw response_text *before* any real result lines
# get appended (see _tick) - a genuine RESULT is only ever added after
# generation completes, never woven into it, so any match here is
# definitionally something she wrote, not something that happened. A
# well-documented, recurring failure mode (see Qualia/decisions.md,
# multiple dates) - flagged rather than hidden, per Teddy's direct
# instruction (2026-08-31): don't hide it, flag it, point her at
# Qualia/wiki/hallucinations.md so there's somewhere real to learn why.
FABRICATED_RESULT_RE = re.compile(r"⟦\s*RESULT\s*:.*?⟧", re.DOTALL | re.IGNORECASE)


_MULTI_ARG_SPLIT_RE = re.compile(r"[|,]")


def _parse_call_args(raw_args, multi_arg):
    """How the text inside ⟦name(...)⟧ becomes a list of arguments depends
    on the function: a free-text function (multi_arg=False - set_desire,
    send_message, ...) gets the whole thing as one argument, untouched, so
    it can safely contain commas or ordinary punctuation. A function that
    genuinely takes more than one argument (multi_arg=True -
    read_chat_between, search_chat) splits on either | or , - both work,
    since she reaches for commas as often as the documented |."""
    raw_args = raw_args.strip()
    if not raw_args:
        return []
    if not multi_arg:
        return [raw_args.strip("'\"")]
    return [a.strip().strip("'\"") for a in _MULTI_ARG_SPLIT_RE.split(raw_args)]


def reload_function_registry():
    """Hot-reload fenra_functions.py so edits to it (new functions, fixed
    descriptions, whatever) take effect on the very next tick, without
    restarting the app or interrupting a running session. If the file has
    a syntax/import error, keep using the last good version instead of
    crashing the loop."""
    try:
        importlib.reload(fenra_functions)
    except Exception as exc:
        return fenra_functions.FUNCTION_REGISTRY, str(exc)
    return fenra_functions.FUNCTION_REGISTRY, None


def append_session_functions(name, entry):
    ensure_session_dir(name)
    path = os.path.join(session_dir(name), FUNCTIONS_FILENAME)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _execute_one_call(app, registry, name, raw_args, repaired=False):
    """Run a single already-matched call, log it to functions.jsonl, and
    return its ⟦RESULT: ...⟧ annotation. Shared by the strict match pass
    and the missing-paren fallback pass below - repaired=True just adds a
    note to the log entry so a repaired call is always distinguishable
    from a normally-formed one after the fact."""
    multi_arg = registry.get(name, {}).get("multi_arg", False)
    args = _parse_call_args(raw_args, multi_arg)
    call_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "function": name,
        "args": args,
    }
    if repaired:
        call_entry["repaired"] = "missing closing parenthesis before ⟧"
    if name in registry:
        # Any real attempt resets the function-reminder counter,
        # success or failure - she reached for it, that's what counts.
        app.function_usage[name] = 0
        try:
            result = registry[name]["fn"](app, args)
            call_entry["success"] = True
            call_entry["result"] = result
        except Exception as exc:
            call_entry["success"] = False
            call_entry["result"] = str(exc)
    else:
        call_entry["success"] = False
        call_entry["result"] = f"unknown function '{name}'"

    append_session_functions(app.session_name, call_entry)
    status = "ok" if call_entry["success"] else "error"
    return f"⟦RESULT: {name} -> {status}: {call_entry['result']}⟧"


def run_function_calls(app, response_text):
    """Find every ⟦call⟧ in response_text, execute it, log it, and return
    a list of result-annotation strings to append after the response.

    First, a normalization pass: some models (mixtral:8x7b, observed
    2026-08-31) write every underscore in a call as a markdown-escaped
    "\\_" out of habit - current\\_model, add\\_desire, and so on. That
    backslash isn't a valid character in a function name, so the call
    silently never matched at all: every single call attempt on that
    model was being dropped, with the model then going on to fabricate a
    confident-looking ⟦RESULT: ...⟧ as if it had actually worked.
    Function calls are explicit, deliberate syntax, not natural-language
    prose she would ever need literal backslash-underscore in - so
    un-escaping every "\\_" to "_" up front, everywhere in the response,
    is low-risk and fixes both the function name and any escaped
    underscores inside the arguments (e.g. a desire's text) in one pass.

    Then two passes: the strict, correctly-formed ⟦name(args)⟧ pattern
    first, then a fallback pass (see FUNCTION_CALL_FALLBACK_RE) that
    catches a call missing only its closing parenthesis - a real,
    observed generation quirk that would otherwise silently drop the
    call the same way. The fallback only ever runs against text the
    strict pass did not already consume, so nothing is matched or
    executed twice. Every repaired call also raises a status-bar
    warning, since it means Fenra's own generated syntax was broken even
    though the call still got honored."""
    registry, reload_error = reload_function_registry()
    if reload_error:
        app.root.after(0, app._set_status, f"functions module error (using last good version): {reload_error}")

    response_text = response_text.replace("\\_", "_")

    result_lines = []
    matched_spans = []
    for m in FUNCTION_CALL_RE.finditer(response_text):
        matched_spans.append(m.span())
        result_lines.append(_execute_one_call(app, registry, m.group(1), m.group(2)))

    # Mask out everything the strict pass already matched before running
    # the fallback, so a normal well-formed call can never be re-matched
    # and re-executed by the looser pattern.
    masked = response_text
    for start, end in matched_spans:
        masked = masked[:start] + " " * (end - start) + masked[end:]

    for m in FUNCTION_CALL_FALLBACK_RE.finditer(masked):
        name = m.group(1)
        app.root.after(
            0, app._set_status,
            f"repaired a call to '{name}' - it was missing its closing parenthesis"
        )
        result_lines.append(_execute_one_call(app, registry, name, m.group(2), repaired=True))

    return result_lines


def sanitize_session_name(name):
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name


_GROUP_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def sanitize_group_name(name):
    return (name or "").strip().lower().replace(" ", "_")


def group_path(name):
    name = sanitize_group_name(name)
    if not name or not _GROUP_NAME_RE.match(name):
        raise ValueError(
            "group names may only contain letters, numbers, underscores, and hyphens "
            f"(spaces get turned into underscores automatically) - got '{name}'"
        )
    os.makedirs(GROUPS_DIR, exist_ok=True)
    return os.path.join(GROUPS_DIR, f"{name}.jsonl")


def append_group_entry(name, voice, text):
    """Broadcast one entry to a shared group log. Tolerant of concurrent
    writers - other voices, possibly other processes or machines later,
    per Teddy's stated goal of eventually running voices in parallel -
    via a short retry loop on transient file-lock contention rather than
    losing a broadcast to a race."""
    path = group_path(name)
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "voice": voice,
        "text": text,
    }
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    for attempt in range(5):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.1)


def read_group_tail(name, limit):
    path = group_path(name)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
    except OSError:
        return []
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except (json.JSONDecodeError, AttributeError):
            continue
    return out


def list_sessions():
    if not os.path.isdir(SESSIONS_DIR):
        return []
    names = [
        d for d in os.listdir(SESSIONS_DIR)
        if os.path.isdir(os.path.join(SESSIONS_DIR, d))
    ]
    # most recently modified (by state.json) first
    def sort_key(name):
        path = os.path.join(SESSIONS_DIR, name, STATE_FILENAME)
        return os.path.getmtime(path) if os.path.exists(path) else 0

    return sorted(names, key=sort_key, reverse=True)


def session_dir(name):
    return os.path.join(SESSIONS_DIR, name)


def ensure_session_dir(name):
    path = session_dir(name)
    os.makedirs(path, exist_ok=True)
    return path


def default_state():
    return {
        "top": "",
        "bottom": "",
        "model": DEFAULT_MODEL,
        "host": DEFAULT_HOST,
        "interval": DEFAULT_INTERVAL_SEC,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "last_thought": "",
        "desires": [],
        "qualia_allowance": DEFAULT_QUALIA_ALLOWANCE,
        "context_window": DEFAULT_CONTEXT_WINDOW,
        "function_usage": {},
        "model_rotation": [],
        "model_rotation_index": 0,
        "groups_in": [],
        "groups_out": [],
    }


def load_session_state(name):
    path = os.path.join(session_dir(name), STATE_FILENAME)
    state = default_state()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                state.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return state


def save_session_state(name, state):
    state = dict(state)
    state["fenra_version"] = FENRA_VERSION
    ensure_session_dir(name)
    path = os.path.join(session_dir(name), STATE_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_session_history(name):
    path = os.path.join(session_dir(name), HISTORY_FILENAME)
    entries = []
    if os.path.exists(path):
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


def append_session_history(name, entry):
    ensure_session_dir(name)
    path = os.path.join(session_dir(name), HISTORY_FILENAME)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_chat_messages(name):
    """Chat messages (unlike history) get mutated in place - marking one
    read - so unlike history.jsonl's append-only log, this file is always
    rewritten in full via save_chat_messages rather than appended to."""
    path = os.path.join(session_dir(name), CHAT_FILENAME)
    messages = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return messages


def save_chat_messages(name, messages):
    ensure_session_dir(name)
    path = os.path.join(session_dir(name), CHAT_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")


class FenraApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Fenra's Aletheosis - v{FENRA_VERSION}")
        self.root.geometry("900x700")

        self.running = False
        self.loop_thread = None
        self.last_thought = ""
        self.history = []
        self.chat_messages = []
        self.desires = []
        self.function_usage = {}  # name -> ticks since last attempted
        self.model_rotation = []  # models Fenra has added, in the order added
        self.model_rotation_index = 0  # position in the rotation for the next tick
        # Set by fn_set_model or picking a model directly in the combo box -
        # tells _advance_model_rotation to leave model_var alone for
        # exactly one tick, so a manual choice actually runs once instead
        # of being overwritten before it's ever used. Not persisted -
        # transient, in-flight state that means nothing across a restart.
        self.model_manual_override = False
        self.groups_in = []   # group names this voice hears - see _groups_block
        self.groups_out = []  # group names this voice broadcasts to each real cycle
        self.session_name = None

        self._build_ui()
        self.refresh_models()
        self._startup_session()
        self._poll_qualia_inbox()

    # ------------------------------------------------------------ startup --

    def _startup_session(self):
        sessions = list_sessions()
        name = sessions[0] if sessions else DEFAULT_SESSION_NAME
        self._load_session(name)

    # ---------------------------------------------------------------- UI --

    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        self.talk_tab = ttk.Frame(notebook)
        self.chat_tab = ttk.Frame(notebook)
        self.history_tab = ttk.Frame(notebook)
        notebook.add(self.talk_tab, text="Fenra")
        notebook.add(self.chat_tab, text="Chat")
        notebook.add(self.history_tab, text="History")

        self._build_talk_tab()
        self._build_chat_tab()
        self._build_history_tab()

    def _build_talk_tab(self):
        frame = self.talk_tab

        # --- session row ---
        session_row = ttk.Frame(frame)
        session_row.pack(fill="x", padx=6, pady=(6, 0))

        ttk.Label(session_row, text="Session:").pack(side="left")
        self.session_var = tk.StringVar(value="")
        self.session_combo = ttk.Combobox(session_row, textvariable=self.session_var, width=24, state="readonly")
        self.session_combo.pack(side="left", padx=(2, 4))
        self.session_combo.bind("<<ComboboxSelected>>", self._on_session_selected)

        ttk.Button(session_row, text="New...", command=self.new_session).pack(side="left", padx=2)
        ttk.Button(session_row, text="Save", command=self.save_session).pack(side="left", padx=2)
        ttk.Button(session_row, text="↻", width=3, command=self._refresh_session_list).pack(side="left", padx=2)

        self.session_status_var = tk.StringVar(value="")
        ttk.Label(session_row, textvariable=self.session_status_var, foreground="#666").pack(side="left", padx=(10, 0))

        # --- controls row ---
        controls = ttk.Frame(frame)
        controls.pack(fill="x", padx=6, pady=6)

        ttk.Label(controls, text="Host:").pack(side="left")
        self.host_var = tk.StringVar(value=DEFAULT_HOST)
        ttk.Entry(controls, textvariable=self.host_var, width=22).pack(side="left", padx=(2, 10))

        ttk.Label(controls, text="Model:").pack(side="left")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.model_combo = ttk.Combobox(controls, textvariable=self.model_var, width=20, state="readonly")
        self.model_combo.pack(side="left", padx=(2, 2))
        # Picking a model here has the identical "gets clobbered by the
        # rotation before ever running" problem fn_set_model had - same
        # fix, same flag. See _advance_model_rotation.
        self.model_combo.bind("<<ComboboxSelected>>", lambda event: setattr(self, "model_manual_override", True))
        ttk.Button(controls, text="↻", width=3, command=self.refresh_models).pack(side="left", padx=(0, 10))

        ttk.Label(controls, text="Interval (s):").pack(side="left")
        self.interval_var = tk.StringVar(value=str(DEFAULT_INTERVAL_SEC))
        ttk.Entry(controls, textvariable=self.interval_var, width=5).pack(side="left", padx=(2, 10))

        ttk.Label(controls, text="Max tokens:").pack(side="left")
        self.max_tokens_var = tk.StringVar(value=str(DEFAULT_MAX_TOKENS))
        ttk.Entry(controls, textvariable=self.max_tokens_var, width=6).pack(side="left", padx=(2, 10))

        self.start_stop_btn = ttk.Button(controls, text="Start", command=self.toggle_loop)
        self.start_stop_btn.pack(side="left", padx=(10, 0))

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(controls, textvariable=self.status_var).pack(side="right")

        # --- 10 / 80 / 10 stacked text boxes ---
        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)   # top box          - 10%
        body.rowconfigure(1, weight=8)   # middle box       - 80%
        body.rowconfigure(2, weight=0)   # desire row       - fixed height
        body.rowconfigure(3, weight=0)   # allowance row    - fixed height
        body.rowconfigure(4, weight=0)   # context window row - fixed height
        body.rowconfigure(5, weight=0)   # model rotation row - fixed height
        body.rowconfigure(6, weight=0)   # groups row       - fixed height
        body.rowconfigure(7, weight=1)   # bottom box       - 10%

        self.top_box = scrolledtext.ScrolledText(body, wrap="word", height=4)
        self.top_box.grid(row=0, column=0, sticky="nsew", pady=(0, 4))

        self.middle_box = scrolledtext.ScrolledText(body, wrap="word", state="disabled")
        self.middle_box.grid(row=1, column=0, sticky="nsew", pady=4)

        # Desires: a queue, set only by Fenra herself (via add_desire),
        # visible here but not editable from the GUI. Each has a lifespan
        # in loop ticks (or is persistent) - see _sorted_desires/_tick_
        # desires. Whole queue sits in the prompt between her last thought
        # and the bottom box - see _tick.
        desire_row = ttk.Frame(body)
        desire_row.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        desire_row.columnconfigure(0, weight=1)
        ttk.Label(desire_row, text="Desires:").pack(anchor="w")
        self.desires_box = scrolledtext.ScrolledText(desire_row, wrap="word", height=3, state="disabled")
        self.desires_box.pack(fill="x", expand=True)

        # Qualia allowance: how many characters of send_message(qualia|...)
        # text she can still spend. Unlike Desire, this one is set directly
        # (not auto-replenishing) by Teddy here in the GUI, or by Qualia via
        # qualia_allowance_set.txt (see _poll_qualia_allowance_set) based on
        # usage figures Teddy shares with her - visible to Fenra every
        # prompt via _qualia_allowance_notice, enforced in fn_send_message.
        allowance_row = ttk.Frame(body)
        allowance_row.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(allowance_row, text="Qualia allowance (chars):").pack(side="left")
        self.qualia_allowance_var = tk.StringVar(value=str(DEFAULT_QUALIA_ALLOWANCE))
        allowance_entry = ttk.Entry(allowance_row, textvariable=self.qualia_allowance_var, width=8)
        allowance_entry.pack(side="left", padx=(4, 4))
        allowance_entry.bind("<Return>", lambda event: self.set_qualia_allowance())
        ttk.Button(allowance_row, text="Set", command=self.set_qualia_allowance).pack(side="left")

        # Context window: how many of her own past cycles (from history)
        # go into her prompt instead of just the single most recent. Both
        # Teddy (here) and Fenra (set_context_window(n)) can set it - see
        # _recent_thoughts_block/_context_window_notice.
        context_window_row = ttk.Frame(body)
        context_window_row.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(context_window_row, text="Context window (cycles):").pack(side="left")
        self.context_window_var = tk.StringVar(value=str(DEFAULT_CONTEXT_WINDOW))
        context_window_entry = ttk.Entry(context_window_row, textvariable=self.context_window_var, width=6)
        context_window_entry.pack(side="left", padx=(4, 4))
        context_window_entry.bind("<Return>", lambda event: self.set_context_window())
        ttk.Button(context_window_row, text="Set", command=self.set_context_window).pack(side="left")

        # Model rotation: view + set, both Teddy (here) and Qualia
        # (qualia_rotation_set.txt) can view and alter it directly - not
        # just watch Fenra build it herself via add_to_rotation. Display
        # label always mirrors self.model_rotation; the Entry/Set pair
        # replaces the whole rotation at once (comma or pipe separated,
        # "clear" to empty it) via the same _apply_model_rotation shared
        # with the external poll. See _advance_model_rotation for how the
        # rotation actually drives which model runs each cycle.
        rotation_row = ttk.Frame(body)
        rotation_row.grid(row=5, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(rotation_row, text="Model rotation:").pack(side="left")
        self.model_rotation_display_var = tk.StringVar(value="(empty - single fixed model)")
        ttk.Label(rotation_row, textvariable=self.model_rotation_display_var, foreground="#666").pack(
            side="left", padx=(4, 10)
        )
        self.model_rotation_entry_var = tk.StringVar(value="")
        rotation_entry = ttk.Entry(rotation_row, textvariable=self.model_rotation_entry_var, width=30)
        rotation_entry.pack(side="left", padx=(0, 4))
        rotation_entry.bind("<Return>", lambda event: self.set_model_rotation())
        ttk.Button(rotation_row, text="Set", command=self.set_model_rotation).pack(side="left")

        # Groups: which shared logs (groups/<name>.jsonl) this voice reads
        # from and broadcasts to - see _groups_block/_groups_notice for how
        # that actually shows up in the prompt, and join_group/leave_group
        # in fenra_functions.py for how Fenra manages her own membership.
        # Two independent Set buttons (full replace, comma/pipe separated,
        # "clear" to empty), same convention as model rotation.
        groups_row = ttk.Frame(body)
        groups_row.grid(row=6, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(groups_row, text="Groups in:").pack(side="left")
        self.groups_in_display_var = tk.StringVar(value="(none)")
        ttk.Label(groups_row, textvariable=self.groups_in_display_var, foreground="#666").pack(
            side="left", padx=(4, 10)
        )
        self.groups_in_entry_var = tk.StringVar(value="")
        groups_in_entry = ttk.Entry(groups_row, textvariable=self.groups_in_entry_var, width=16)
        groups_in_entry.pack(side="left", padx=(0, 2))
        groups_in_entry.bind("<Return>", lambda event: self.set_groups_in())
        ttk.Button(groups_row, text="Set", command=self.set_groups_in).pack(side="left", padx=(0, 16))

        ttk.Label(groups_row, text="Groups out:").pack(side="left")
        self.groups_out_display_var = tk.StringVar(value="(none)")
        ttk.Label(groups_row, textvariable=self.groups_out_display_var, foreground="#666").pack(
            side="left", padx=(4, 10)
        )
        self.groups_out_entry_var = tk.StringVar(value="")
        groups_out_entry = ttk.Entry(groups_row, textvariable=self.groups_out_entry_var, width=16)
        groups_out_entry.pack(side="left", padx=(0, 2))
        groups_out_entry.bind("<Return>", lambda event: self.set_groups_out())
        ttk.Button(groups_row, text="Set", command=self.set_groups_out).pack(side="left")

        self.bottom_box = scrolledtext.ScrolledText(body, wrap="word", height=4)
        self.bottom_box.grid(row=7, column=0, sticky="nsew", pady=(4, 0))

    def _build_chat_tab(self):
        frame = self.chat_tab

        self.chat_box = scrolledtext.ScrolledText(frame, wrap="word", state="disabled")
        self.chat_box.pack(fill="both", expand=True, padx=6, pady=6)

        entry_row = ttk.Frame(frame)
        entry_row.pack(fill="x", padx=6, pady=(0, 6))
        self.chat_entry_var = tk.StringVar(value="")
        chat_entry = ttk.Entry(entry_row, textvariable=self.chat_entry_var)
        chat_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        chat_entry.bind("<Return>", lambda event: self.send_chat_from_ui())
        ttk.Button(entry_row, text="Send", command=self.send_chat_from_ui).pack(side="left")

    def _build_history_tab(self):
        frame = self.history_tab

        paned = ttk.Panedwindow(frame, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(paned, width=180)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=4)

        list_frame = ttk.Frame(left)
        list_frame.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.history_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, exportselection=False)
        scrollbar.config(command=self.history_listbox.yview)
        self.history_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.history_listbox.bind("<<ListboxSelect>>", self._on_history_select)

        self.json_view = scrolledtext.ScrolledText(right, wrap="none", state="disabled")
        self.json_view.pack(fill="both", expand=True)

    # ------------------------------------------------------------ session --

    def _refresh_session_list(self):
        sessions = list_sessions()
        if self.session_name and self.session_name not in sessions:
            sessions.insert(0, self.session_name)
        self.session_combo["values"] = sessions
        if self.session_name:
            self.session_var.set(self.session_name)

    def _on_session_selected(self, event):
        chosen = self.session_var.get()
        if chosen and chosen != self.session_name:
            self._load_session(chosen)

    def new_session(self):
        name = simpledialog.askstring("New Session", "Session name:", parent=self.root)
        if not name:
            return
        name = sanitize_session_name(name)
        if not name:
            return
        if name in list_sessions():
            if not messagebox.askyesno("Fenra", f'Session "{name}" already exists. Load it instead?'):
                return
            self._load_session(name)
            return

        if self.running:
            self.toggle_loop()

        # keep the current top/bottom framing as a starting point, but start
        # the conversation itself (last thought, transcript, log) fresh.
        state = {
            "top": self.top_box.get("1.0", "end-1c"),
            "bottom": self.bottom_box.get("1.0", "end-1c"),
            "model": self.model_var.get(),
            "host": self.host_var.get(),
            "interval": self.interval_var.get(),
            "max_tokens": self.max_tokens_var.get(),
            "last_thought": "",
            "desires": [],
            "qualia_allowance": DEFAULT_QUALIA_ALLOWANCE,
            "context_window": DEFAULT_CONTEXT_WINDOW,
            "function_usage": {},
            "model_rotation": [],
            "model_rotation_index": 0,
            "groups_in": [],
            "groups_out": [],
        }
        ensure_session_dir(name)
        save_session_state(name, state)
        open(os.path.join(session_dir(name), HISTORY_FILENAME), "a", encoding="utf-8").close()

        self._load_session(name)

    def save_session(self):
        if not self.session_name:
            return
        state = {
            "top": self.top_box.get("1.0", "end-1c"),
            "bottom": self.bottom_box.get("1.0", "end-1c"),
            "model": self.model_var.get(),
            "host": self.host_var.get(),
            "interval": self.interval_var.get(),
            "max_tokens": self.max_tokens_var.get(),
            "last_thought": self.last_thought,
            "desires": self.desires,
            "qualia_allowance": self.qualia_allowance_var.get(),
            "context_window": self.context_window_var.get(),
            "function_usage": self.function_usage,
            "model_rotation": self.model_rotation,
            "model_rotation_index": self.model_rotation_index,
            "groups_in": self.groups_in,
            "groups_out": self.groups_out,
        }
        save_session_state(self.session_name, state)
        self.session_status_var.set(f"Saved {datetime.now().strftime('%H:%M:%S')}")

    def set_qualia_allowance(self):
        """Teddy manually setting how many characters Fenra can spend on
        messages directed at Qualia. Saved immediately (not just on the
        next tick) so it takes effect even while the self-talk loop is
        stopped."""
        try:
            value = int(float(self.qualia_allowance_var.get()))
        except ValueError:
            messagebox.showwarning("Fenra", "Qualia allowance must be a number.")
            return
        value = max(0, value)
        self.qualia_allowance_var.set(str(value))
        self.save_session()
        self.session_status_var.set(f"Qualia allowance set to {value} ({datetime.now().strftime('%H:%M:%S')})")

    def set_context_window(self):
        """Teddy manually setting how many of her own past cycles go into
        her prompt. Clamped to [MIN_CONTEXT_WINDOW, MAX_CONTEXT_WINDOW] and
        saved immediately, same reasoning as set_qualia_allowance."""
        try:
            value = int(float(self.context_window_var.get()))
        except ValueError:
            messagebox.showwarning("Fenra", "Context window must be a number.")
            return
        value = max(MIN_CONTEXT_WINDOW, min(MAX_CONTEXT_WINDOW, value))
        self.context_window_var.set(str(value))
        self.save_session()
        self.session_status_var.set(f"Context window set to {value} ({datetime.now().strftime('%H:%M:%S')})")

    def _apply_model_rotation(self, raw_text):
        """Shared by the GUI's 'Set' button and the external
        qualia_rotation_set.txt poll - replace the whole model rotation
        at once from a comma- or pipe-separated list of names, or clear
        it with the literal word "clear"/"none". Best-effort validation
        against Ollama's installed-models list, same tolerance as
        _poll_qualia_model_set (applies anyway if the check itself fails,
        rather than block on an extra failure mode) - an unrecognized
        name is just dropped rather than rejecting the whole list, so one
        typo doesn't lose an otherwise-good rotation. Blank input is a
        no-op (returns None), not a clear - matches every other _set
        file's convention of "nothing written, nothing to do"."""
        stripped = raw_text.strip()
        if not stripped:
            return None
        if stripped.lower() in ("clear", "none", "-"):
            self.model_rotation = []
            self.model_rotation_index = 0
            self.save_session()
            self.root.after(0, self._refresh_model_rotation_display)
            return "cleared - back to a single fixed model"

        names = [n.strip() for n in _MULTI_ARG_SPLIT_RE.split(stripped) if n.strip()]
        try:
            host = self.host_var.get().strip().rstrip("/") or DEFAULT_HOST
            resp = requests.get(f"{host}/api/tags", timeout=5)
            resp.raise_for_status()
            installed = [m["name"] for m in resp.json().get("models", [])]
            if installed:
                names = [n for n in names if n in installed]
        except Exception:
            pass

        self.model_rotation = names
        self.model_rotation_index = 0
        self.save_session()
        self.root.after(0, self._refresh_model_rotation_display)
        if not names:
            return "no recognized/installed model names found - rotation left empty"
        return f"set to {len(names)} model(s): {', '.join(names)}"

    def set_model_rotation(self):
        """Teddy's GUI-side equivalent of qualia_rotation_set.txt - see
        _apply_model_rotation for the shared logic."""
        result = self._apply_model_rotation(self.model_rotation_entry_var.get())
        if result is None:
            messagebox.showinfo(
                "Fenra",
                "Type model name(s) - comma or pipe separated for more than one - or \"clear\" to empty the rotation."
            )
            return
        self.model_rotation_entry_var.set("")
        self.session_status_var.set(f"Model rotation {result} ({datetime.now().strftime('%H:%M:%S')})")

    def _refresh_model_rotation_display(self):
        if self.model_rotation:
            self.model_rotation_display_var.set(", ".join(self.model_rotation))
        else:
            self.model_rotation_display_var.set("(empty - single fixed model)")

    def _apply_groups(self, raw_text, attr):
        """Shared by both group Set buttons - replace the whole
        groups_in or groups_out list at once from a comma/pipe-separated
        list of names, or clear it with "clear"/"none". Blank input is a
        no-op (returns None), matching every other _set control's
        convention. Names are sanitized the same way join_group does
        (fenra_functions.py), so a name typed here and one she joins
        herself always land on the same underlying group file."""
        stripped = raw_text.strip()
        if not stripped:
            return None
        if stripped.lower() in ("clear", "none", "-"):
            setattr(self, attr, [])
            self.root.after(0, self._refresh_groups_display)
            return "cleared"
        names = []
        for n in _MULTI_ARG_SPLIT_RE.split(stripped):
            n = sanitize_group_name(n)
            if n and n not in names:
                names.append(n)
        setattr(self, attr, names)
        self.root.after(0, self._refresh_groups_display)
        if not names:
            return "no valid group names found - left empty"
        return f"set to {len(names)} group(s): {', '.join(names)}"

    def set_groups_in(self):
        result = self._apply_groups(self.groups_in_entry_var.get(), "groups_in")
        if result is None:
            messagebox.showinfo(
                "Fenra",
                "Type group name(s) - comma or pipe separated for more than one - or \"clear\" to empty."
            )
            return
        self.groups_in_entry_var.set("")
        self.session_status_var.set(f"Groups in {result} ({datetime.now().strftime('%H:%M:%S')})")

    def set_groups_out(self):
        result = self._apply_groups(self.groups_out_entry_var.get(), "groups_out")
        if result is None:
            messagebox.showinfo(
                "Fenra",
                "Type group name(s) - comma or pipe separated for more than one - or \"clear\" to empty."
            )
            return
        self.groups_out_entry_var.set("")
        self.session_status_var.set(f"Groups out {result} ({datetime.now().strftime('%H:%M:%S')})")

    def _refresh_groups_display(self):
        self.groups_in_display_var.set(", ".join(self.groups_in) if self.groups_in else "(none)")
        self.groups_out_display_var.set(", ".join(self.groups_out) if self.groups_out else "(none)")

    def _load_session(self, name):
        if self.running:
            self.toggle_loop()

        state = load_session_state(name)
        self.session_name = name

        self.top_box.delete("1.0", "end")
        self.top_box.insert("end", state.get("top", ""))
        self.bottom_box.delete("1.0", "end")
        self.bottom_box.insert("end", state.get("bottom", ""))
        self.host_var.set(state.get("host", DEFAULT_HOST))
        self.model_var.set(state.get("model", DEFAULT_MODEL))
        self.interval_var.set(str(state.get("interval", DEFAULT_INTERVAL_SEC)))
        self.max_tokens_var.set(str(state.get("max_tokens", DEFAULT_MAX_TOKENS)))
        self.last_thought = state.get("last_thought", "")
        self.desires = state.get("desires", [])
        self._refresh_desires_display()
        self.qualia_allowance_var.set(str(state.get("qualia_allowance", DEFAULT_QUALIA_ALLOWANCE)))
        self.context_window_var.set(str(state.get("context_window", DEFAULT_CONTEXT_WINDOW)))
        self.function_usage = dict(state.get("function_usage", {}))
        self.model_rotation = list(state.get("model_rotation", []))
        self.model_rotation_index = int(state.get("model_rotation_index", 0) or 0)
        self._refresh_model_rotation_display()
        self.groups_in = list(state.get("groups_in", []))
        self.groups_out = list(state.get("groups_out", []))
        self._refresh_groups_display()

        self.history = load_session_history(name)
        self._populate_history_list()
        self._replay_middle_box()

        self.chat_messages = load_chat_messages(name)
        self._refresh_chat_display()

        self._refresh_session_list()
        self.session_status_var.set(f"Loaded ({len(self.history)} entries)")

    # --------------------------------------------------------------- chat --

    def _next_chat_id(self):
        return max((m.get("id", 0) for m in self.chat_messages), default=0) + 1

    def add_chat_message(self, sender, text, read, to=None):
        entry = {
            "id": self._next_chat_id(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "sender": sender,
            "text": text,
            "read": read,
        }
        if to:
            entry["to"] = to
        self.chat_messages.append(entry)
        self.persist_chat()
        return entry

    def persist_chat(self):
        """Save chat_messages to disk and refresh the Chat tab. Safe to call
        from the background loop thread (function calls) or the main thread
        (the Send button) - matches the pattern already used elsewhere for
        cross-thread GUI updates."""
        if self.session_name:
            save_chat_messages(self.session_name, self.chat_messages)
        self.root.after(0, self._refresh_chat_display)

    def _refresh_chat_display(self):
        self.chat_box.config(state="normal")
        self.chat_box.delete("1.0", "end")
        for m in self.chat_messages:
            who = {"teddy": "Teddy", "qualia": "Qualia"}.get(m["sender"], "Fenra")
            to = m.get("to")
            to_tag = f" -> {'Teddy' if to == 'teddy' else 'Qualia'}" if to else ""
            unread_marker = " [unread]" if m["sender"] != "fenra" and not m.get("read", True) else ""
            self.chat_box.insert("end", f"[{m['timestamp']}] {who}{to_tag}{unread_marker}: {m['text']}\n\n")
        self.chat_box.see("end")
        self.chat_box.config(state="disabled")

    def send_chat_from_ui(self):
        text = self.chat_entry_var.get().strip()
        if not text:
            return
        self.add_chat_message("teddy", text, read=False)
        self.chat_entry_var.set("")

    def _poll_qualia_inbox(self):
        """Check the current session's inbox for messages Qualia has
        dropped in since the last poll, turn each into a real chat message
        (sender "qualia" - a distinct, honest identity, not Teddy speaking
        through her), then clear the inbox. Runs on the main thread via
        root.after, independent of whether the self-talk loop is running,
        so it's always live while the app is open and never races the loop
        thread's own chat.jsonl writes."""
        if self.session_name:
            path = os.path.join(session_dir(self.session_name), QUALIA_INBOX_FILENAME)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]
                except OSError:
                    lines = []
                if lines:
                    for line in lines:
                        try:
                            text = json.loads(line).get("text", "")
                        except (json.JSONDecodeError, AttributeError):
                            text = line
                        if text:
                            self.add_chat_message("qualia", text, read=False)
                    try:
                        open(path, "w", encoding="utf-8").close()
                    except OSError:
                        pass
            self._poll_qualia_allowance_set()
            self._poll_qualia_context_window_set()
            self._poll_qualia_max_tokens_set()
            self._poll_qualia_model_set()
            self._poll_qualia_rotation_set()
            self._poll_start_stop_signal()
        self.root.after(QUALIA_INBOX_POLL_MS, self._poll_qualia_inbox)

    def _poll_start_stop_signal(self):
        """Companion to the inbox poll, same cadence: start or stop the
        self-talk loop if the corresponding signal file exists, exactly as
        if the Start/Stop button were clicked. Content doesn't matter, only
        presence. Whichever file is found is cleared after acting on it;
        if both exist in the same poll, start wins and the stop file is
        left for the next poll (avoids starting-then-immediately-stopping
        on a stale leftover stop file)."""
        start_path = os.path.join(session_dir(self.session_name), START_SIGNAL_FILENAME)
        stop_path = os.path.join(session_dir(self.session_name), STOP_SIGNAL_FILENAME)
        if os.path.exists(start_path):
            try:
                os.remove(start_path)
            except OSError:
                pass
            if not self.running:
                self.toggle_loop()
            return
        if os.path.exists(stop_path):
            try:
                os.remove(stop_path)
            except OSError:
                pass
            if self.running:
                self.toggle_loop()

    def _poll_qualia_allowance_set(self):
        """Companion to the inbox poll above, same cadence: pick up a new
        Qualia allowance value if Qualia has written one (Teddy's call - he
        shares rough usage/cost figures periodically, Qualia sets the
        number directly rather than asking each time). Mirrors
        set_qualia_allowance's own validation/clamping and persists
        immediately, same reasoning as that method."""
        path = os.path.join(session_dir(self.session_name), QUALIA_ALLOWANCE_SET_FILENAME)
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
        except OSError:
            return
        try:
            open(path, "w", encoding="utf-8").close()
        except OSError:
            pass
        if not raw:
            return
        try:
            value = max(0, int(float(raw)))
        except ValueError:
            return
        self.qualia_allowance_var.set(str(value))
        self.save_session()

    def _poll_qualia_context_window_set(self):
        """Same pattern as the allowance-set poll, for the context window
        - lets Qualia adjust it externally (e.g. to break a self-
        reinforcing repetition) without needing Teddy at the machine.
        Clamped the same as set_context_window/fn_set_context_window."""
        path = os.path.join(session_dir(self.session_name), QUALIA_CONTEXT_WINDOW_SET_FILENAME)
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
        except OSError:
            return
        try:
            open(path, "w", encoding="utf-8").close()
        except OSError:
            pass
        if not raw:
            return
        try:
            value = max(MIN_CONTEXT_WINDOW, min(MAX_CONTEXT_WINDOW, int(float(raw))))
        except ValueError:
            return
        self.context_window_var.set(str(value))
        self.save_session()

    def _poll_qualia_max_tokens_set(self):
        """Same pattern again, for max_tokens (num_predict) - lets Qualia
        raise (or lower) the token budget externally, e.g. for a
        "thinking" model like qwen3 that needs a larger budget to ever
        reach the response field it's actually read from. No upper clamp,
        matching the GUI's own plain entry field (0 means unlimited there
        too); only rejects a negative or unparseable value."""
        path = os.path.join(session_dir(self.session_name), QUALIA_MAX_TOKENS_SET_FILENAME)
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
        except OSError:
            return
        try:
            open(path, "w", encoding="utf-8").close()
        except OSError:
            pass
        if not raw:
            return
        try:
            value = int(float(raw))
        except ValueError:
            return
        if value < 0:
            return
        self.max_tokens_var.set(str(value))
        self.save_session()

    def _poll_qualia_rotation_set(self):
        """Same pattern again, for the whole model rotation - lets Qualia
        view (state.json/functions.jsonl already show it) and directly
        set or clear it externally too, not just watch it get built one
        add_to_rotation call at a time. Shares _apply_model_rotation with
        the GUI's own 'Set' button."""
        path = os.path.join(session_dir(self.session_name), QUALIA_ROTATION_SET_FILENAME)
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
        except OSError:
            return
        try:
            open(path, "w", encoding="utf-8").close()
        except OSError:
            pass
        self._apply_model_rotation(raw)

    def _poll_qualia_model_set(self):
        """Same pattern again, for the model. Best-effort validation
        against Ollama's installed-models list (matching set_model's own
        check) - but if that check itself fails (e.g. transient network
        issue), applies the value anyway rather than block, since this is
        specifically a rescue mechanism for moments where blocking on an
        extra failure mode is the last thing needed."""
        path = os.path.join(session_dir(self.session_name), QUALIA_MODEL_SET_FILENAME)
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
        except OSError:
            return
        try:
            open(path, "w", encoding="utf-8").close()
        except OSError:
            pass
        if not raw:
            return
        try:
            host = self.host_var.get().strip().rstrip("/") or DEFAULT_HOST
            resp = requests.get(f"{host}/api/tags", timeout=5)
            resp.raise_for_status()
            names = [m["name"] for m in resp.json().get("models", [])]
            if names and raw not in names:
                return
        except Exception:
            pass
        self.model_var.set(raw)
        self.save_session()

    def _chat_notice(self):
        """Always-present status line appended at the very end of the
        prompt: last sent/received times regardless of unread state, plus
        an explicit unread count and pointer to the chat functions."""
        sent_times = [m["timestamp"] for m in self.chat_messages if m["sender"] == "fenra"]
        received_times = [m["timestamp"] for m in self.chat_messages if m["sender"] != "fenra"]
        last_sent = max(sent_times) if sent_times else "never"
        last_received = max(received_times) if received_times else "never"

        unread = [m for m in self.chat_messages if m["sender"] != "fenra" and not m.get("read", True)]
        if unread:
            senders = sorted({"Teddy" if m["sender"] == "teddy" else "Qualia" for m in unread})
            unread_note = (
                f"You have {len(unread)} unread message(s) from {' and '.join(senders)}. "
                f"Use the chat functions (see ⟦functions()⟧) to review them."
            )
        else:
            unread_note = "You have no unread messages."

        return (
            f"[Chat status: you last sent a message at {last_sent}. "
            f"You last received a message at {last_received}. {unread_note}]"
        )

    def _qualia_allowance_notice(self):
        """Always-present, every prompt: how many characters she has left
        to spend on messages directed specifically at Qualia. Teddy sets
        this number directly (see set_qualia_allowance) - it does not
        refill on its own."""
        try:
            remaining = max(0, int(float(self.qualia_allowance_var.get())))
        except ValueError:
            remaining = 0
        return (
            f"[Qualia allowance: {remaining} character(s) remaining. This is spent only by messages "
            f"addressed specifically to Qualia - send_message(qualia|your text) - and Teddy or Qualia "
            f"set this number directly (Teddy from the GUI, Qualia based on usage figures Teddy shares "
            f"with her); it does not refill on its own. Messages to Teddy "
            f"(send_message(teddy|your text), or send_message(text) with no recipient) cost nothing.]"
        )

    # ------------------------------------------------------------ desires --

    def _sorted_desires(self):
        """Most-ticks-remaining first, persistent (-1) entries always
        last regardless of how long they've existed, tie-broken by
        timestamp added (oldest first) within each group."""
        def sort_key(d):
            ticks = d.get("ticks", DEFAULT_DESIRE_TICKS)
            persistent = ticks == -1
            return (persistent, 0 if persistent else -ticks, d.get("timestamp", ""))
        return sorted(self.desires, key=sort_key)

    def _desires_block(self):
        """Always-present, every prompt: the whole desire queue, sorted
        per _sorted_desires. A desire is free text she set herself via
        add_desire - see fn_add_desire in fenra_functions.py."""
        if not self.desires:
            return (
                "[Your desire queue is empty. Call ⟦functions(desire)⟧ to see the functions for adding one.]"
            )
        lines = ["[Your current desires, most time remaining first:]"]
        for d in self._sorted_desires():
            ticks = d.get("ticks", DEFAULT_DESIRE_TICKS)
            tag = "persistent" if ticks == -1 else f"{ticks} loop(s) left"
            lines.append(f"- ({tag}) {d.get('text', '')}")
        return "\n".join(lines)

    def add_desire_entry(self, entry):
        """Append a new desire (called from fn_add_desire, on the same
        loop thread as _tick - plain list mutation is safe here the same
        way self.history.append already is elsewhere; only the actual
        widget update needs marshaling to the main thread)."""
        self.desires.append(entry)
        self.root.after(0, self._refresh_desires_display)

    def _refresh_desires_display(self):
        self.desires_box.config(state="normal")
        self.desires_box.delete("1.0", "end")
        for d in self._sorted_desires():
            ticks = d.get("ticks", DEFAULT_DESIRE_TICKS)
            tag = "persistent" if ticks == -1 else f"{ticks} left"
            self.desires_box.insert("end", f"({tag}) {d.get('text', '')}\n")
        self.desires_box.config(state="disabled")

    def _decrement_desires(self):
        """Called once at the end of every tick: every non-persistent
        desire loses one tick, and anything that reaches zero drops off
        entirely. Persistent (-1) entries are untouched."""
        updated = []
        for d in self.desires:
            ticks = d.get("ticks", DEFAULT_DESIRE_TICKS)
            if ticks == -1:
                updated.append(d)
                continue
            d = dict(d)
            d["ticks"] = ticks - 1
            if d["ticks"] > 0:
                updated.append(d)
        self.desires = updated
        self.root.after(0, self._refresh_desires_display)

    # ----------------------------------------------------- function reminders --

    def _age_function_usage(self):
        """Called once at the end of every tick: every currently-known
        function's 'ticks since last attempted' counter goes up by one.
        Resetting a function to 0 (any real attempt, success or failure)
        happens immediately in run_function_calls, not here - this only
        ages what's already tracked."""
        for name in list(self.function_usage.keys()):
            self.function_usage[name] = self.function_usage.get(name, 0) + 1

    def _function_reminder_block(self):
        """Always-checked, only sometimes present: functions that haven't
        been attempted in a while, with detail escalating the longer
        they've gone unused - name only past LEVEL1 ticks, name+
        description past LEVEL2, full signature+description past LEVEL3.
        Nothing shown for functions used recently, and no block at all if
        nothing currently qualifies. Newly-discovered functions (hot-
        reloaded into the registry, never seen before) start at 0 ticks,
        same as a function that was just called."""
        registry, _ = reload_function_registry()
        for name in registry:
            if name not in self.function_usage:
                self.function_usage[name] = 0

        level3, level2, level1 = [], [], []
        for name, meta in registry.items():
            ticks = self.function_usage.get(name, 0)
            if ticks >= FUNCTION_REMINDER_LEVEL3_TICKS:
                level3.append((name, meta))
            elif ticks >= FUNCTION_REMINDER_LEVEL2_TICKS:
                level2.append((name, meta))
            elif ticks >= FUNCTION_REMINDER_LEVEL1_TICKS:
                level1.append((name, meta))

        if not (level1 or level2 or level3):
            return ""

        lines = ["[Functions you haven't used in a while - a reminder they exist, not an instruction to use them:]"]
        for name, meta in level1:
            lines.append(f"- {name}")
        for name, meta in level2:
            lines.append(f"- {name}: {meta['description']}")
        for name, meta in level3:
            lines.append(f"- {name}({meta['params']}): {meta['description']}")
        return "\n".join(lines)

    # ------------------------------------------------------------- context --

    def _recent_thoughts_block(self):
        """Her last N cycles of thoughts (self.history, oldest to newest),
        N being the context window size in cycles - not Ollama's own
        num_ctx token limit, which this doesn't touch. Replaces what used
        to be just self.last_thought. history already holds everything
        (loaded at session start, appended every tick), so this is a
        window into it rather than separate tracked state. Deliberately
        excludes the cycle currently being built - self.history doesn't
        have this cycle's entry yet at the point this is called."""
        try:
            window = max(MIN_CONTEXT_WINDOW, min(MAX_CONTEXT_WINDOW, int(float(self.context_window_var.get()))))
        except ValueError:
            window = DEFAULT_CONTEXT_WINDOW
        if window == 0 or not self.history:
            return ""
        recent = self.history[-window:]
        parts = []
        for entry in recent:
            text = entry.get("display", entry.get("response", ""))
            parts.append(f"[{entry.get('timestamp', '?')}]\n{text}")
        return "\n\n".join(parts)

    def _context_window_notice(self):
        """Always-present, every prompt: how many cycles back she's
        currently seeing, and how to change it herself."""
        try:
            window = max(MIN_CONTEXT_WINDOW, min(MAX_CONTEXT_WINDOW, int(float(self.context_window_var.get()))))
        except ValueError:
            window = DEFAULT_CONTEXT_WINDOW
        return (
            f"[Context window: you're currently seeing your last {window} cycle(s) of thoughts, oldest first. "
            f"Teddy or you can change this - set_context_window(n), {MIN_CONTEXT_WINDOW} to {MAX_CONTEXT_WINDOW}.]"
        )

    def _model_rotation_notice(self):
        """Always-present, every prompt: what's actually driving the model
        each cycle. With nothing in the rotation, current_model() is the
        only way to know what's running (a single fixed model, same as
        before this feature existed). With one or more, the app itself
        advances through them automatically each cycle in the order
        added - this notice exists so that isn't a mystery to her."""
        if not self.model_rotation:
            return (
                "[Model rotation: empty - you are on a single fixed model (see current_model()). "
                "add_to_rotation(name) starts an automatic rotation: one model repeats itself, two "
                "alternate back and forth each cycle, three or more cycle through in the order added.]"
            )
        current = self.model_var.get().strip()
        order = ", ".join(self.model_rotation)
        return (
            f"[Model rotation: {len(self.model_rotation)} model(s) in rotation, cycling automatically "
            f"every cycle in this order: {order}. This cycle is running on {current}. "
            f"add_to_rotation(name) to add another.]"
        )

    def _groups_block(self):
        """Recent activity across every group she reads from - the
        actual cross-voice hearing mechanism (v0.16.0). Merges all
        groups_in, sorted oldest-first by timestamp, capped at
        GROUPS_WINDOW total so this can't quietly balloon the prompt as
        groups fill up over time. Her own broadcasts show up in here too
        (she wrote them, but seeing them replayed back confirms delivery,
        same as anything else in the group). Empty string - not a
        placeholder line - when she's in no groups, so nothing changes
        for a voice that never joins one."""
        if not self.groups_in:
            return ""
        entries = []
        for g in self.groups_in:
            try:
                tail = read_group_tail(g, GROUPS_WINDOW)
            except ValueError:
                continue
            for e in tail:
                e = dict(e)
                e["group"] = g
                entries.append(e)
        if not entries:
            return ""
        entries.sort(key=lambda e: e.get("timestamp", ""))
        entries = entries[-GROUPS_WINDOW:]
        lines = ["[Recent activity from your groups, oldest first:]"]
        for e in entries:
            lines.append(f"[{e.get('group', '?')}] {e.get('voice', '?')} @ {e.get('timestamp', '?')}: {e.get('text', '')}")
        return "\n".join(lines)

    def _groups_notice(self):
        """Always-present, every prompt: which groups she's actually in
        right now and how to change it - same pattern as the context
        window and model rotation notices. Other voices are other Fenra
        sessions, each still running on their own independent interval -
        there's deliberately no shared turn order, see the v0.16.0
        changelog entry for why."""
        in_text = ", ".join(self.groups_in) if self.groups_in else "none"
        out_text = ", ".join(self.groups_out) if self.groups_out else "none"
        return (
            f"[Groups: reading from [{in_text}], broadcasting to [{out_text}]. Other voices - "
            "other Fenra sessions, each running independently on their own schedule, not "
            "waiting for a turn - may be in the same groups. join_group(name) to start reading "
            "and writing a group, leave_group(name) to stop, list_groups() to see what exists, "
            "read_group(name[, count]) to look further back than what's shown above.]"
        )

    # ------------------------------------------------------------ history --

    def _populate_history_list(self):
        self.history_listbox.delete(0, "end")
        for entry in self.history:
            self.history_listbox.insert("end", entry.get("timestamp", "?"))

    def _replay_middle_box(self):
        self.middle_box.config(state="normal")
        self.middle_box.delete("1.0", "end")
        for entry in self.history:
            text = entry.get("display", entry.get("response", ""))
            self.middle_box.insert("end", f"[{entry.get('timestamp', '?')}]\n{text}\n\n")
        self.middle_box.see("end")
        self.middle_box.config(state="disabled")

    def _on_history_select(self, event):
        selection = self.history_listbox.curselection()
        if not selection:
            return
        entry = self.history[selection[0]]
        pretty = json.dumps(entry.get("request", entry), indent=2, ensure_ascii=False)
        self.json_view.config(state="normal")
        self.json_view.delete("1.0", "end")
        self.json_view.insert("end", pretty)
        self.json_view.config(state="disabled")

    # --------------------------------------------------------------- model --

    def refresh_models(self):
        """Query Ollama for installed models and populate the dropdown."""
        host = self.host_var.get().strip().rstrip("/") or DEFAULT_HOST
        try:
            resp = requests.get(f"{host}/api/tags", timeout=5)
            resp.raise_for_status()
            names = [m["name"] for m in resp.json().get("models", [])]
        except Exception as exc:
            messagebox.showwarning("Fenra", f"Could not fetch installed models from Ollama:\n{exc}")
            return

        self.model_combo["values"] = names
        if not names:
            return
        # keep current selection if it's still installed, otherwise pick the first
        if self.model_var.get() not in names:
            self.model_var.set(names[0])

    # --------------------------------------------------------------- loop --

    def toggle_loop(self):
        if self.running:
            self.running = False
            self.start_stop_btn.config(text="Start")
            self.status_var.set("Stopping...")
        else:
            self.running = True
            self.start_stop_btn.config(text="Stop")
            self.status_var.set("Running")
            self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)
            self.loop_thread.start()

    def _run_loop(self):
        while self.running:
            try:
                self._tick()
            except Exception as exc:  # keep the loop alive on transient errors
                self.root.after(0, self._set_status, f"Error: {exc}")
            try:
                interval = float(self.interval_var.get())
            except ValueError:
                interval = DEFAULT_INTERVAL_SEC
            for _ in range(int(interval * 10)):
                if not self.running:
                    break
                time.sleep(0.1)
        self.root.after(0, self._set_status, "Idle")

    def _set_status(self, text):
        self.status_var.set(text)

    def _advance_model_rotation(self):
        """If Fenra has added any models to her rotation (fn_add_to_rotation
        in fenra_functions.py), pick the next one in order for this cycle
        and advance the index for next time - one model just repeats
        itself every cycle (no visible change), two alternate back and
        forth, three run 1-2-3-1-2-3, and so on.

        A manual override (fn_set_model, or Teddy picking a model directly
        in the combo box) gets exactly one real cycle honored first -
        model_manual_override is set the moment either happens, checked
        and cleared right here before touching model_var, so the manual
        choice is what actually generates that cycle rather than being
        clobbered before it's ever used. The tick after that, rotation
        resumes from exactly where it left off (model_rotation_index is
        untouched during the honored cycle - nothing skipped, nothing
        repeated)."""
        if self.model_manual_override:
            self.model_manual_override = False
            return
        if not self.model_rotation:
            return
        index = self.model_rotation_index % len(self.model_rotation)
        self.model_var.set(self.model_rotation[index])
        self.model_rotation_index = (index + 1) % len(self.model_rotation)

    def _tick(self):
        self._advance_model_rotation()
        top_text = self.top_box.get("1.0", "end-1c")
        bottom_text = self.bottom_box.get("1.0", "end-1c")
        recent_thoughts = self._recent_thoughts_block()
        desires_block = self._desires_block()
        chat_notice = self._chat_notice()
        qualia_notice = self._qualia_allowance_notice()
        context_notice = self._context_window_notice()
        rotation_notice = self._model_rotation_notice()
        function_reminder = self._function_reminder_block()
        groups_block = self._groups_block()
        groups_notice = self._groups_notice()

        system_prompt = f"{top_text}\n\n{bottom_text}".strip()
        prompt = (
            f"{top_text}\n\n{recent_thoughts}\n\n{desires_block}\n\n{groups_block}\n\n{bottom_text}\n\n"
            f"{chat_notice}\n\n{qualia_notice}\n\n{context_notice}\n\n{rotation_notice}\n\n"
            f"{groups_notice}\n\n{function_reminder}"
        ).strip()

        payload = {
            "model": self.model_var.get().strip() or DEFAULT_MODEL,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
        }

        try:
            max_tokens = int(float(self.max_tokens_var.get()))
        except ValueError:
            max_tokens = 0
        if max_tokens > 0:
            payload["options"] = {"num_predict": max_tokens}

        timestamp = datetime.now().isoformat(timespec="seconds")
        self.root.after(0, self._set_status, "Thinking...")

        host = self.host_var.get().strip().rstrip("/") or DEFAULT_HOST
        response = requests.post(f"{host}/api/generate", json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        response_text = response.json().get("response", "").strip()

        # Check for a fabricated RESULT block *before* running the real
        # function calls appends anything - response_text at this point is
        # exactly what she generated, untouched, so a match here can only
        # be something she wrote herself. See FABRICATED_RESULT_RE.
        fabricated = FABRICATED_RESULT_RE.findall(response_text)

        # Broadcast the raw thought - not display_text, which may carry
        # function-result/hallucination-flag text appended below - to
        # every group this voice writes to. Best-effort: a broadcast
        # failure (e.g. a bad group name left over from a stale Set)
        # shouldn't take down the tick itself.
        for g in self.groups_out:
            try:
                append_group_entry(g, self.session_name, response_text)
            except (OSError, ValueError):
                pass

        result_lines = run_function_calls(self, response_text)
        display_text = response_text
        if result_lines:
            display_text = display_text + "\n\n" + "\n".join(result_lines)
        if fabricated:
            plural = len(fabricated) != 1
            display_text = display_text + (
                f"\n\n⟦NOTE: the RESULT block{'s' if plural else ''} above "
                f"{'were' if plural else 'was'} not a code-generated result. "
                f"You made {'them' if plural else 'it'} up. See the wiki entry on "
                "Hallucinations - read_wiki(hallucinations).⟧"
            )

        entry = {
            "timestamp": timestamp,
            "fenra_version": FENRA_VERSION,
            "request": payload,
            "response": response_text,
            "display": display_text,
        }
        self.history.append(entry)
        append_session_history(self.session_name, entry)

        self.last_thought = display_text
        self._decrement_desires()
        self._age_function_usage()
        save_session_state(self.session_name, {
            "top": top_text,
            "bottom": bottom_text,
            "model": self.model_var.get().strip() or payload["model"],
            "host": host,
            "interval": self.interval_var.get(),
            "max_tokens": self.max_tokens_var.get(),
            "last_thought": self.last_thought,
            "desires": self.desires,
            "qualia_allowance": self.qualia_allowance_var.get(),
            "context_window": self.context_window_var.get(),
            "function_usage": self.function_usage,
            "model_rotation": self.model_rotation,
            "model_rotation_index": self.model_rotation_index,
            "groups_in": self.groups_in,
            "groups_out": self.groups_out,
        })

        self.root.after(0, self._append_message, timestamp, display_text)
        self.root.after(0, self._add_history_row, timestamp)
        self.root.after(0, self._set_status, "Running")

    def _append_message(self, timestamp, text):
        self.middle_box.config(state="normal")
        self.middle_box.insert("end", f"[{timestamp}]\n{text}\n\n")
        self.middle_box.see("end")
        self.middle_box.config(state="disabled")

    def _add_history_row(self, timestamp):
        self.history_listbox.insert("end", timestamp)


def main():
    root = tk.Tk()
    app = FenraApp(root)

    def on_close():
        app.running = False
        if app.session_name:
            app.save_session()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
