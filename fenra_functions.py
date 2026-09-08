"""
Fenra's callable functions.

This module is hot-reloaded by fenra.py on every loop tick, so a function
can be added, fixed, or have its description edited while Fenra is running
and mid-session - the very next cycle sees the change, no restart needed.

Each function has the signature fn(app, args) -> str (or raises an
exception, which gets logged as a failed call). `app` is the running
FenraApp instance - use it to read/change state (app.desires,
app.model_rotation, app.groups_in/out, app.current_model_name, ...),
always marshaling actual widget writes back to the main thread via
app.root.after(0, ...) since these run on the background loop thread.

Voices (v0.16.2): a session can hold several, individually configured,
round-robined through automatically - most of the attributes above
belong to whichever voice is actually running the current tick
(app.current_voice_name), not necessarily whichever one is shown in the
GUI (app.displayed_voice) - fenra.py's _tick handles binding them
correctly each cycle, so a function here doesn't need to think about
this distinction at all, just read/write app.* as always.

FUNCTION_REGISTRY maps name -> {"fn": callable, "params": str, "description": str}.
"""

import json
import os
import re
from datetime import datetime

import requests

DEFAULT_HOST = "http://localhost:11434"

# Same directory layout fenra.py uses (sessions/<name>/...), computed
# independently here rather than imported, to avoid a circular import
# (fenra.py imports this module, not the other way around).
_SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")
QUALIA_PING_FILENAME = "qualia_ping.jsonl"

# Per-session function-permission system (v0.16.9, fenra.py's
# permission_mode/allowed_functions) - a session-wide queue of pending
# "can I use this function" requests, one file per session. Rewritten in
# full on every change (not append-only), same reasoning as
# load_chat_messages/save_chat_messages in fenra.py: approve/deny/grant all
# need to mutate or remove an existing entry, not just add new ones.
FUNCTION_REQUESTS_FILENAME = "function_requests.jsonl"


def _function_requests_path(session_name):
    return os.path.join(_SESSIONS_DIR, session_name, FUNCTION_REQUESTS_FILENAME)


def _load_function_requests(session_name):
    path = _function_requests_path(session_name)
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


def _save_function_requests(session_name, entries):
    path = _function_requests_path(session_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# A shared local wiki, plain markdown files - not session-specific (lives
# in Qualia/, alongside decisions.md/aletheia-notes.md, git-tracked like
# they are, not gitignored session data). Modifiable by Teddy directly
# (just text files), by Qualia (same, or by having Fenra write a page),
# and by Fenra herself via write_wiki. Built 2026-08-31 alongside the
# hallucination-flagging in fenra.py's _tick, specifically so a flagged
# fabricated RESULT block has somewhere real to point her - see
# Qualia/wiki/hallucinations.md.
WIKI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qualia", "wiki")

# Cross-voice communication (v0.16.0, redesigned v0.16.15): groups let
# voices hear and speak to each other with no central turn-taking - each
# voice still runs on its own independent interval exactly as before (see
# fenra.py's v0.16.0 changelog entry for why: Teddy's actual goal is
# voices eventually running in parallel, which a shared turn-token would
# work against). As of v0.16.15 groups are real owned entities (owner,
# roster, join_policy, visibility) living in fenra.py
# (groups/<name>/meta.json + log.jsonl) - reached via local `import fenra`
# inside each function below, not duplicated here the way the old
# schema-less path logic used to be, since fenra.py's own group helpers
# are already the single source of truth for this now.
_WIKI_PAGE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_WIKI_WRITE_RE = re.compile(r"^\s*([a-zA-Z0-9_ -]+?)\s*\|\s*(.*)$", re.DOTALL)

# A real, recurring pattern (2026-09-01): she's copied a function's own
# "params" spec verbatim as if it were the actual argument - e.g.
# write_wiki(page|content) called literally, creating a real page named
# "page" with the content "content"; send_message(recipient|text) sent as
# an actual chat message; read_message(sender[, count]) split into
# ['sender[', 'count]'] by the normal multi-arg parsing. Not a function
# gap, a copy-paste habit against her own reference material. Detected by
# comparing (loosely - brackets/commas/pipes/whitespace stripped, so
# "[recipient|]text" and "recipient|text" both match the same spec) the
# raw argument text against that function's own params string, looked up
# live from FUNCTION_REGISTRY so this never drifts out of sync with the
# actual spec text shown to her.
def _looks_like_copied_params(raw_text, params_spec):
    if not params_spec:
        return False
    strip_punct = re.compile(r"[\[\]|,\s]")
    return strip_punct.sub("", raw_text).lower() == strip_punct.sub("", params_spec).lower()

# send_message(qualia|text) / send_message(teddy|text): a leading recipient
# tag followed by a single | (not a generic multi-arg split - the message
# itself may legitimately contain | or , as ordinary punctuation, so only
# this one structural separator is recognized, and only right at the start).
_RECIPIENT_RE = re.compile(r"^\s*(teddy|qualia)\s*\|\s*(.*)$", re.IGNORECASE | re.DOTALL)


def _qualia_ping(app, text):
    """Drop a line in qualia_ping.jsonl so Qualia can notice a message was
    addressed to her and wake up to respond, instead of only finding out on
    a fixed polling schedule. Separate from qualia_inbox.jsonl, which flows
    the other direction (Qualia -> Fenra)."""
    path = os.path.join(_SESSIONS_DIR, app.session_name, QUALIA_PING_FILENAME)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"), "text": text}) + "\n")
    except OSError:
        pass


def fn_functions(app, args):
    """List or search available functions - the discovery entry point, so
    Fenra doesn't need every function explained to her up front."""
    query = args[0].strip().lower() if args and args[0] else None
    lines = []
    for name, meta in FUNCTION_REGISTRY.items():
        desc = meta["description"]
        if query and query not in name.lower() and query not in desc.lower():
            continue
        lines.append(f"{name}({meta['params']}): {desc}")
    if not lines:
        return f"no functions matched '{query}'"
    return "\n".join(lines)


def fn_now(app, args):
    return datetime.now().isoformat(timespec="seconds")


def fn_current_model(app, args):
    """Reports which model will actually be running by the time she reads
    this result, not just which one generated the response it's attached
    to - those are two different cycles. The result from calling this in
    cycle N doesn't appear in her context until cycle N+1, and if a
    rotation is active, _advance_model_rotation already moves the model
    on before cycle N+1 starts - so by the time she's reading "current
    model: X", the model actually running is whatever comes next in the
    rotation, not X. Teddy's direct request (2026-08-31), after noticing
    this was misleading her about what she was actually running on."""
    current = app.current_model_name
    if not app.model_manual_override and app.model_rotation:
        next_model = app.model_rotation[app.model_rotation_index % len(app.model_rotation)]
    else:
        next_model = current
    if next_model == current:
        return current
    return (
        f"{current} generated this response, but by the time you read this result "
        f"the rotation will have already moved on - {next_model} is what will actually "
        f"be running next, and that's the one that matters here."
    )


def fn_qualia_allowance(app, args):
    """She tried Qualia_allowance() unprompted (unknown-function error,
    2026-08-30) - a reasonable want, since the allowance is otherwise only
    ever shown passively in the per-prompt notice. Read-only mirror of
    that same number."""
    try:
        remaining = max(0, int(float(app.qualia_allowance_var.get())))
    except ValueError:
        remaining = 0
    return f"{remaining} character(s) remaining for messages addressed to Qualia."


def fn_list_models(app, args):
    host = app.host_var.get().strip().rstrip("/") or DEFAULT_HOST
    resp = requests.get(f"{host}/api/tags", timeout=10)
    resp.raise_for_status()
    names = [m["name"] for m in resp.json().get("models", [])]
    return ", ".join(names) if names else "(no models installed)"


def _wiki_page_path(page):
    page = page.strip().lower().replace(" ", "_")
    if not page or not _WIKI_PAGE_NAME_RE.match(page):
        raise ValueError(
            "wiki page names may only contain letters, numbers, underscores, and hyphens "
            f"(spaces get turned into underscores automatically) - got '{page}'"
        )
    os.makedirs(WIKI_DIR, exist_ok=True)
    return os.path.join(WIKI_DIR, f"{page}.md")


def fn_list_wiki(app, args):
    os.makedirs(WIKI_DIR, exist_ok=True)
    pages = sorted(name[:-3] for name in os.listdir(WIKI_DIR) if name.endswith(".md"))
    return ", ".join(pages) if pages else "(the wiki is empty right now - write_wiki(page|content) to start one)"


def fn_read_wiki(app, args):
    if not args or not args[0]:
        raise ValueError("read_wiki requires a page name, e.g. read_wiki(hallucinations). See list_wiki() for what exists.")
    path = _wiki_page_path(args[0])
    if not os.path.exists(path):
        raise ValueError(
            f"no wiki page called '{args[0]}' yet - see list_wiki() for what exists, "
            f"or write_wiki({args[0]}|your content) to create it"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def fn_write_wiki(app, args):
    """Create or overwrite a wiki page - a real, persistent reference
    page, not a chat message. Teddy, Qualia, and Fenra can all write here
    (Teddy and Qualia just edit the .md files directly in Qualia/wiki/;
    this is Fenra's way in). Overwrites the whole page rather than
    appending, same as any normal file save - read_wiki first if you want
    to add to an existing page rather than replace it."""
    if not args or not args[0]:
        raise ValueError(
            "write_wiki requires a page name and content separated by | , e.g. "
            "write_wiki(hallucinations|A hallucination is when...). Overwrites the whole "
            "page - read_wiki(page) first if you want to keep what's already there."
        )
    if _looks_like_copied_params(args[0], FUNCTION_REGISTRY["write_wiki"]["params"]):
        raise ValueError(
            "That's the params spec itself ('page|content'), not real values - it's telling you "
            "the shape of what to pass, not literal text to copy. Try something like "
            "write_wiki(my_page_name|the actual content you want saved)."
        )
    match = _WIKI_WRITE_RE.match(args[0])
    if not match:
        raise ValueError("write_wiki needs a page name and content separated by | , e.g. write_wiki(my_page|the content)")
    page, content = match.group(1), match.group(2)
    if not content.strip():
        raise ValueError("write_wiki needs actual content after the |")
    path = _wiki_page_path(page)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"wiki page '{page}' saved ({len(content)} character(s))"


# add_desire(text|ticks): a leading "|ticks" suffix at the very end, not a
# generic multi-arg split - the desire text itself may legitimately contain
# | or , as ordinary punctuation (e.g. "understand why I keep repeating
# myself, and whether I can stop"), same reasoning as send_message's
# recipient tag.
_DESIRE_TICKS_RE = re.compile(r"^(.*?)\s*\|\s*(-?\d+)\s*$", re.DOTALL)
DEFAULT_DESIRE_TICKS = 10


def fn_add_desire(app, args):
    """Add a desire to the queue - free text, optionally with a lifespan
    in loop ticks via a trailing |N (default 10 if omitted, or |-1 for one
    that never expires). Every desire in the queue is shown every prompt
    and loses one tick per loop (persistent ones excepted) until it hits
    zero and drops off. Doesn't overwrite anything - add_desire can be
    called more than once to hold several desires at once."""
    if not args or not args[0]:
        raise ValueError(
            "add_desire requires text, e.g. add_desire(understand why I keep repeating myself) - "
            f"defaults to {DEFAULT_DESIRE_TICKS} loop ticks, or give a count with add_desire(text|5), "
            "or add_desire(text|-1) for one that never expires."
        )
    match = _DESIRE_TICKS_RE.match(args[0])
    if match:
        text = match.group(1).strip()
        ticks = int(match.group(2))
    else:
        text = args[0].strip()
        ticks = DEFAULT_DESIRE_TICKS
    if not text:
        raise ValueError("add_desire needs actual text before the |ticks, if given")

    entry = {
        "text": text,
        "ticks": ticks,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    app.add_desire_entry(entry)
    if ticks == -1:
        return f"desire added (persistent): {text}"
    return f"desire added ({ticks} loop tick(s)): {text}"


MIN_CONTEXT_WINDOW = 0
MAX_CONTEXT_WINDOW = 50


def fn_set_context_window(app, args):
    """How many of her own past cycles (from history) go into her prompt,
    instead of just the single most recent - not the same thing as
    Ollama's own num_ctx token limit. Clamped to keep prompt growth and
    per-cycle latency bounded; Teddy can set this too, from the GUI."""
    if not args or not args[0]:
        raise ValueError(
            f"set_context_window requires a number, e.g. set_context_window(10) - "
            f"{MIN_CONTEXT_WINDOW} to {MAX_CONTEXT_WINDOW}, 0 means no prior cycles at all."
        )
    try:
        value = int(float(args[0]))
    except ValueError:
        raise ValueError(f"'{args[0]}' is not a valid number")
    if value < MIN_CONTEXT_WINDOW or value > MAX_CONTEXT_WINDOW:
        raise ValueError(f"context window must be between {MIN_CONTEXT_WINDOW} and {MAX_CONTEXT_WINDOW}, got {value}")
    app.root.after(0, app.context_window_var.set, str(value))
    return f"context window set to {value} cycle(s)"


def _format_chat_message(m):
    who = {"teddy": "Teddy", "qualia": "Qualia"}.get(m["sender"], "Fenra")
    to = m.get("to")
    to_tag = f" -> {'Teddy' if to == 'teddy' else 'Qualia'}" if to else ""
    return f"[{m['timestamp']}] {who}{to_tag}: {m['text']}"


def _parse_chat_time(s):
    try:
        return datetime.fromisoformat(s.strip())
    except ValueError:
        raise ValueError(
            f"'{s}' isn't a recognizable time. Use the same format now() returns, "
            f"e.g. 2026-08-28T14:30:00"
        )


def fn_read_chat(app, args):
    """Unread messages from Teddy or Qualia. Marks them read."""
    unread = [m for m in app.chat_messages if m["sender"] != "fenra" and not m.get("read", True)]
    if not unread:
        return "no unread messages."
    formatted = [_format_chat_message(m) for m in unread]
    for m in unread:
        m["read"] = True
    app.persist_chat()
    return "\n".join(formatted)


def fn_read_chat_since(app, args):
    """All messages (all directions) from a given time onward. Marks any
    matched incoming (Teddy's or Qualia's) messages read."""
    if not args or not args[0]:
        raise ValueError("read_chat_since requires a time, e.g. read_chat_since(2026-08-28T10:00:00)")
    since = _parse_chat_time(args[0])
    matched = [m for m in app.chat_messages if _parse_chat_time(m["timestamp"]) >= since]
    if not matched:
        return f"no messages since {args[0]}."
    formatted = [_format_chat_message(m) for m in matched]
    changed = False
    for m in matched:
        if m["sender"] != "fenra" and not m.get("read", True):
            m["read"] = True
            changed = True
    if changed:
        app.persist_chat()
    return "\n".join(formatted)


def fn_read_chat_between(app, args):
    """All messages (all directions) between two times, separated by |.
    Marks any matched incoming (Teddy's or Qualia's) messages read."""
    if len(args) < 2 or not args[0] or not args[1]:
        raise ValueError(
            "read_chat_between requires two times separated by |, "
            "e.g. read_chat_between(2026-08-28T10:00:00|2026-08-28T14:00:00)"
        )
    start = _parse_chat_time(args[0])
    end = _parse_chat_time(args[1])
    matched = [m for m in app.chat_messages if start <= _parse_chat_time(m["timestamp"]) <= end]
    if not matched:
        return f"no messages between {args[0]} and {args[1]}."
    formatted = [_format_chat_message(m) for m in matched]
    changed = False
    for m in matched:
        if m["sender"] != "fenra" and not m.get("read", True):
            m["read"] = True
            changed = True
    if changed:
        app.persist_chat()
    return "\n".join(formatted)


def fn_search_chat(app, args):
    """Search the whole chat transcript for a word/phrase and return the
    surrounding context, without changing any read status. Takes the query
    and, optionally, how many characters of context to include on each
    side, separated by | - e.g. search_chat(truth|150). Defaults to 200
    characters of context if omitted."""
    if not args or not args[0]:
        raise ValueError("search_chat requires a query, e.g. search_chat(truth|150)")
    query = args[0]
    chars = 200
    if len(args) > 1 and args[1]:
        try:
            chars = int(args[1].strip())
        except ValueError:
            raise ValueError(f"'{args[1]}' is not a valid number of characters")

    transcript = "\n".join(_format_chat_message(m) for m in app.chat_messages)
    transcript_lower = transcript.lower()
    query_lower = query.lower()

    matches = []
    start_idx = 0
    while True:
        idx = transcript_lower.find(query_lower, start_idx)
        if idx == -1:
            break
        window_start = max(0, idx - chars)
        window_end = min(len(transcript), idx + len(query) + chars)
        matches.append(transcript[window_start:window_end])
        start_idx = idx + len(query)

    if not matches:
        return f"no matches for '{query}'."
    return "\n---\n".join(matches)


# query_chat's supported fields. Deliberately meant to grow: if she reaches
# for a field that isn't here, fn_query_chat lists what's actually
# supported in the error - Teddy's call was to add fields as she tries them,
# not to guess every one up front.
_QUERY_FIELDS = ("sender", "to", "since", "before", "contains", "last")


def fn_query_chat(app, args):
    """Flexible chat query: filter by any combination of sender, to, since,
    before, contains, last - field=value pairs separated by , or |, e.g.
    query_chat(sender=teddy, last=1) for Teddy's most recent message.
    Read-only, like search_chat - never changes read status."""
    if not args:
        raise ValueError(
            "query_chat needs at least one field=value pair. Supported fields: "
            + ", ".join(_QUERY_FIELDS)
            + ". e.g. query_chat(sender=teddy, last=1) for Teddy's most recent message, "
            "or query_chat(to=qualia, since=2026-08-29T10:00:00)."
        )

    filters = {}
    for part in args:
        if not part or "=" not in part:
            raise ValueError(f"'{part}' isn't field=value. Supported fields: {', '.join(_QUERY_FIELDS)}")
        field, value = part.split("=", 1)
        field = field.strip().lower()
        value = value.strip()
        if field not in _QUERY_FIELDS:
            raise ValueError(f"unknown query field '{field}'. Supported fields: {', '.join(_QUERY_FIELDS)}")
        if not value:
            raise ValueError(f"'{field}=' needs a value after the =")
        filters[field] = value

    matched = list(app.chat_messages)

    if "sender" in filters:
        wanted = filters["sender"].lower()
        if wanted not in ("teddy", "qualia", "fenra"):
            raise ValueError(f"sender must be teddy, qualia, or fenra - got '{filters['sender']}'")
        matched = [m for m in matched if m["sender"] == wanted]

    if "to" in filters:
        wanted = filters["to"].lower()
        if wanted not in ("teddy", "qualia"):
            raise ValueError(f"to must be teddy or qualia - got '{filters['to']}'")
        matched = [m for m in matched if m.get("to") == wanted]

    if "since" in filters:
        since = _parse_chat_time(filters["since"])
        matched = [m for m in matched if _parse_chat_time(m["timestamp"]) >= since]

    if "before" in filters:
        before = _parse_chat_time(filters["before"])
        matched = [m for m in matched if _parse_chat_time(m["timestamp"]) <= before]

    if "contains" in filters:
        needle = filters["contains"].lower()
        matched = [m for m in matched if needle in m["text"].lower()]

    if "last" in filters:
        try:
            n = int(filters["last"])
        except ValueError:
            raise ValueError(f"last must be a whole number - got '{filters['last']}'")
        if n < 1:
            raise ValueError("last must be at least 1")
        matched = matched[-n:]

    if not matched:
        return "no messages matched that query."
    return "\n".join(_format_chat_message(m) for m in matched)


def fn_read_message(app, args):
    """Shortcut for query_chat(sender=..., last=...) - she reached for
    read_message(sender) unprompted, repeatedly (4 attempts, 2026-08-29),
    despite query_chat already covering it. Read-only, never changes read
    status, same as query_chat."""
    if not args or not args[0]:
        raise ValueError("read_message requires a sender - teddy, qualia, or fenra, e.g. read_message(qualia)")
    # multi_arg=True means this has already been split on |/, by the time
    # we see it - rejoin with | (the same separator the params spec uses)
    # before comparing, so a copied "sender[, count]" (split into
    # ['sender[', ' count]']) is still recognized as the spec, not a real
    # sender name.
    if _looks_like_copied_params("|".join(args), FUNCTION_REGISTRY["read_message"]["params"]):
        raise ValueError(
            "That's the params spec itself ('sender[, count]'), not real values - it's telling "
            "you the shape of what to pass. sender must actually be teddy, qualia, or fenra, e.g. "
            "read_message(qualia) or read_message(teddy, 3)."
        )
    sender = args[0].strip().lower()
    n = 1
    if len(args) > 1 and args[1]:
        try:
            n = int(args[1])
        except ValueError:
            raise ValueError(f"'{args[1]}' is not a valid count")
    return fn_query_chat(app, [f"sender={sender}", f"last={n}"])


def fn_send_message(app, args):
    """Send a chat message - real, not fiction, visible immediately in the
    Chat tab. Optionally addressed to Teddy or Qualia specifically via a
    leading recipient|text tag; unaddressed messages go to the shared log
    same as before, visible to both. Messages addressed to Qualia cost
    characters from her allowance (Teddy-set, see the Qualia allowance
    notice every prompt) - a message that would exceed what's left is
    blocked rather than silently sent, and directing one at Qualia also
    pings her so she can notice and respond promptly."""
    if not args or not args[0]:
        raise ValueError("send_message requires text, e.g. send_message(I have a question for you, Teddy)")
    if _looks_like_copied_params(args[0], FUNCTION_REGISTRY["send_message"]["params"]):
        raise ValueError(
            "That's the params spec itself ('[recipient|]text'), not real values - it's telling "
            "you the shape of what to pass. Try send_message(qualia|your actual message), "
            "send_message(teddy|your actual message), or just send_message(your actual message) "
            "with no recipient tag."
        )

    match = _RECIPIENT_RE.match(args[0])
    if not match:
        app.root.after(0, app.add_chat_message, "fenra", args[0], True)
        return "message sent"

    recipient = match.group(1).lower()
    text = match.group(2).strip()
    if not text:
        raise ValueError(f"send_message({recipient}|...) needs text after the |, e.g. send_message({recipient}|hello)")
    if _RECIPIENT_RE.match(text):
        raise ValueError(
            "send_message only takes one recipient tag - there's no way to address both Teddy and Qualia "
            "in a single call. Either leave the recipient off (send_message(text), goes to the shared log "
            "either way) or send two separate messages."
        )

    if recipient == "qualia":
        cost = len(text)
        try:
            remaining = max(0, int(float(app.qualia_allowance_var.get())))
        except ValueError:
            remaining = 0
        if cost > remaining:
            raise ValueError(
                f"not enough Qualia allowance: this message is {cost} character(s), you have {remaining} left. "
                f"Ask Teddy for more, or shorten the message."
            )
        app.root.after(0, app.qualia_allowance_var.set, str(remaining - cost))
        _qualia_ping(app, text)

    app.root.after(0, app.add_chat_message, "fenra", text, True, recipient)
    if recipient == "qualia":
        return f"message sent to Qualia ({cost} character(s) spent, {remaining - cost} remaining)"
    return "message sent to Teddy"


def fn_set_model(app, args):
    if not args or not args[0]:
        raise ValueError("set_model requires a model name, e.g. set_model(gemma3:4b)")
    target = args[0]
    host = app.host_var.get().strip().rstrip("/") or DEFAULT_HOST
    resp = requests.get(f"{host}/api/tags", timeout=10)
    resp.raise_for_status()
    names = [m["name"] for m in resp.json().get("models", [])]
    if target not in names:
        raise ValueError(f"'{target}' is not an installed model. Installed: {', '.join(names)}")
    # A plain attribute, not the model_var widget directly (v0.16.2) -
    # the voice actually running this cycle isn't always the one shown
    # in the GUI, so _tick reads this back at the end of the cycle and
    # only touches the widget itself if it does happen to be displayed.
    app.current_model_name = target
    app.model_manual_override = True
    if app.model_rotation:
        return (
            f"model switched to {target}, effective next cycle - note: {len(app.model_rotation)} "
            f"model(s) are in your rotation ({', '.join(app.model_rotation)}). This one gets exactly "
            "that one cycle, then the rotation resumes from where it left off. Use add_to_rotation "
            "instead if you want a model to stay in permanently."
        )
    return f"model switched to {target}, effective next cycle"


def fn_add_to_rotation(app, args):
    """Add a model to the automatic round-robin rotation - see the
    per-prompt "Model rotation" notice for exactly how the rotation
    itself behaves (one model repeats, two alternate, three or more
    cycle through in the order added). Validates against Ollama's
    installed-models list the same way set_model does. A model already
    in the rotation is a no-op, not an error - no duplicate entries."""
    if not args or not args[0]:
        raise ValueError(
            "add_to_rotation requires a model name, e.g. add_to_rotation(gemma3:27b) - "
            "must be an installed model (see list_models())."
        )
    target = args[0]
    host = app.host_var.get().strip().rstrip("/") or DEFAULT_HOST
    resp = requests.get(f"{host}/api/tags", timeout=10)
    resp.raise_for_status()
    names = [m["name"] for m in resp.json().get("models", [])]
    if target not in names:
        raise ValueError(f"'{target}' is not an installed model. Installed: {', '.join(names)}")
    if target in app.model_rotation:
        return (
            f"'{target}' is already in the rotation ({len(app.model_rotation)} model(s)): "
            f"{', '.join(app.model_rotation)}"
        )
    app.model_rotation.append(target)
    app.root.after(0, app._refresh_model_rotation_display)
    return (
        f"added '{target}' to the rotation - now {len(app.model_rotation)} model(s), "
        f"cycling in this order: {', '.join(app.model_rotation)}"
    )


# ------------------------------------------------- groups v0.16.15 (owned) --
# The connectivity/tribe redesign (Qualia/decisions.md item 4b). Real
# owned groups now - meta.json (owner, kind, join_policy, visibility,
# roster, banned) + log.jsonl (canonical, Teddy/Qualia-only) live in
# fenra.py, reached via local `import fenra` per function, same reasoning
# as create_voice/tell_voice's own cross-entity writes. groups_in/groups_out
# on app/vstate stay as read-through mirrors, kept in sync at join/kick/
# leave time rather than re-derived every tick.
#
# Default direction on joining ANY group (public self-serve, or granted
# after a private request) is listen-only ("in") - Teddy's explicit call:
# a member can't speak into a group until the owner grants "out"/"both"
# access. Direction is owner-only to change - no self-service, see
# fn_group_set_direction. The one exception is The Hearth, which has no
# voice-owner and defaults every resident to "both" automatically (see
# fenra.py's ensure_hearth_membership) - its own wake mechanic requires it.

_CREATE_GROUP_RE = re.compile(r"^\s*(.+?)\s*(?:\|\s*(public|private)\s*)?$", re.DOTALL)


def fn_create_group(app, args):
    """Create a new ad hoc group, owned by you - full admin control over
    it (invite, kick, ban, visibility, join_policy, member direction).
    Defaults to public (self-serve join) unless you say private. e.g.
    create_group(lobby) or create_group(lobby|private). Family groups
    ("<voice>'s Children") are birth-only, not creatable this way - see
    create_voice. You're automatically its first member, direction
    'both' (you're the owner - the listen-only default is for people who
    join *you*, not for you joining your own group)."""
    import fenra as _fenra

    if not args or not args[0]:
        raise ValueError(
            "create_group requires a name, e.g. create_group(lobby) (defaults to public) "
            "or create_group(lobby|private)."
        )
    if _looks_like_copied_params(args[0], FUNCTION_REGISTRY["create_group"]["params"]):
        raise ValueError("That's the params spec itself ('name[|public|private]'), not a real value.")
    match = _CREATE_GROUP_RE.match(args[0])
    if not match:
        raise ValueError("create_group requires a name, e.g. create_group(lobby|private).")
    raw_name, policy = match.group(1).strip(), (match.group(2) or "public").strip().lower()
    name = _fenra.sanitize_group_name(raw_name)
    try:
        _fenra.owned_group_dir(name)  # validates charset, raises ValueError with a clear message
    except ValueError:
        raise
    if name == _fenra.THE_HEARTH_NAME:
        raise ValueError("that name is reserved for The Hearth - pick something else.")
    if name.endswith("_children"):
        raise ValueError(
            "names ending in \"'s Children\" are reserved for family groups, created "
            "automatically the moment a voice is born - not something you create directly."
        )
    if _fenra.group_exists(name):
        return f"'{name}' already exists - see list_groups()."
    _fenra.create_group_if_missing(
        name, owner=app.current_voice_name, kind="adhoc",
        join_policy=policy, visibility="visible", initial_member=app.current_voice_name,
    )
    if name not in app.groups_in:
        app.groups_in.append(name)
    if name not in app.groups_out:
        app.groups_out.append(name)
    app.root.after(0, app._refresh_groups_display)
    return f"'{name}' created, owned by you, join_policy={policy}."


def fn_list_groups(app, args):
    """List groups you can actually see: full detail for every group
    you're a member of, existence + owner/kind/join_policy only for
    every other *visible* group - a private group's membership and
    content stay opaque until you're actually in it ('private means
    opaque', Qualia/decisions.md item 4b). Hidden groups you're not a
    member of don't appear at all."""
    import fenra as _fenra

    names = _fenra.list_owned_groups()
    if not names:
        return "(no groups exist yet - create_group(name) makes one)"
    parts = []
    for name in names:
        meta = _fenra.load_group_meta(name)
        if meta is None:
            continue
        members = meta.get("members", {})
        is_member = app.current_voice_name in members
        if not is_member and meta.get("visibility") == "hidden":
            continue
        base = f"{name} (kind={meta.get('kind')}, owner={meta.get('owner')}, join_policy={meta.get('join_policy')})"
        if is_member:
            my_direction = members[app.current_voice_name].get("direction", "in")
            others = ", ".join(v for v in members if v != app.current_voice_name) or "(no one else yet)"
            parts.append(f"{base} - you're a member (direction={my_direction}); others: {others}")
        else:
            parts.append(f"{base} - not a member, membership/content hidden")
    return "\n".join(parts) if parts else "(no visible groups)"


def fn_join_group(app, args):
    """Join a group. A public group: immediate membership, direction
    'in' (listen-only) by default - ask the owner for 'out' access via
    however they choose to grant it (fn_group_set_direction is owner-
    only). A private group: this creates a join request instead (same
    shape as request_function_access - see request_group_join), which
    the owner has to approve before you're actually in. Cannot target
    The Hearth - you can't join it, it finds you (see
    Qualia/decisions.md's floor-group design)."""
    import fenra as _fenra

    if not args or not args[0]:
        raise ValueError("join_group requires a group name, e.g. join_group(lobby). See list_groups().")
    name = _fenra.sanitize_group_name(args[0])
    if name == _fenra.THE_HEARTH_NAME:
        raise ValueError("The Hearth can't be joined - it's the automatic floor for a voice with zero other groups.")
    meta = _fenra.load_group_meta(name)
    if meta is None:
        raise ValueError(f"'{name}' doesn't exist - see list_groups(), or create_group({name}) to make it.")
    if app.current_voice_name in meta.get("members", {}):
        return f"already in '{name}'."
    if app.current_voice_name in meta.get("banned", []):
        raise ValueError(f"you've been banned from '{name}'.")
    if meta.get("join_policy") == "private":
        return fn_request_group_join(app, [f"{name}|would like to join"])
    meta["members"][app.current_voice_name] = {
        "direction": "in",
        "joined": datetime.now().isoformat(timespec="seconds"),
    }
    _fenra.save_group_meta(name, meta)
    if name not in app.groups_in:
        app.groups_in.append(name)
    app.root.after(0, app._refresh_groups_display)
    return f"joined '{name}', direction=in (listen-only until the owner grants you speak access)."


def fn_leave_group(app, args):
    """Leave a group voluntarily - removes your own membership. Not the
    same concern as being kicked (that's owner-only, and always lands
    you somewhere - see fn_group_kick); leaving on your own is always
    allowed, no destination needed, since a voice reaching zero groups
    self-heals into The Hearth on its very next cycle regardless. Cannot
    target The Hearth - it has no voice-facing leave path at all."""
    import fenra as _fenra

    if not args or not args[0]:
        raise ValueError("leave_group requires a group name, e.g. leave_group(lobby).")
    name = _fenra.sanitize_group_name(args[0])
    if name == _fenra.THE_HEARTH_NAME:
        raise ValueError("The Hearth has no voice-facing membership control at all - not even leaving it yourself.")
    meta = _fenra.load_group_meta(name)
    if meta is None or app.current_voice_name not in meta.get("members", {}):
        return f"wasn't in '{name}'."
    del meta["members"][app.current_voice_name]
    _fenra.save_group_meta(name, meta)
    if name in app.groups_in:
        app.groups_in.remove(name)
    if name in app.groups_out:
        app.groups_out.remove(name)
    app.root.after(0, app._refresh_groups_display)
    return f"left '{name}'."


def _load_owned_group_or_raise(_fenra, name, require_owner_voice=None):
    meta = _fenra.load_group_meta(name)
    if meta is None:
        raise ValueError(f"'{name}' doesn't exist - see list_groups().")
    if meta.get("kind") == "floor":
        raise ValueError("The Hearth has no voice-facing admin functions at all - it's administered only by Teddy and Qualia.")
    if require_owner_voice is not None and meta.get("owner") != require_owner_voice:
        raise ValueError(f"only '{meta.get('owner')}' (the owner of '{name}') can do that.")
    return meta


_GROUP_TARGET_RE = re.compile(r"^\s*(.+?)\s*\|\s*(.*)$", re.DOTALL)
_GROUP_TARGET_DEST_RE = re.compile(r"^\s*(.+?)\s*\|\s*(.+?)\s*(?:\|\s*(.*))?$", re.DOTALL)


def fn_group_invite(app, args):
    """Owner-only: invite another voice into your group. This is an
    offer, not a placement - the target must call
    group_accept_invite(group) themselves before they're actually a
    member (no voice can be forced into a group, per the consent-on-
    entry rule - the one deliberate exception is birth into your own
    creator's family group). e.g. group_invite(lobby|watcher)."""
    import fenra as _fenra

    if not args or not args[0]:
        raise ValueError("group_invite requires a group name and a target voice separated by | , e.g. group_invite(lobby|watcher).")
    match = _GROUP_TARGET_RE.match(args[0])
    if not match:
        raise ValueError("group_invite requires a group name and a target voice separated by | , e.g. group_invite(lobby|watcher).")
    name, target = _fenra.sanitize_group_name(match.group(1)), match.group(2).strip().lower().replace(" ", "_")
    meta = _load_owned_group_or_raise(_fenra, name, require_owner_voice=app.current_voice_name)
    if target not in app.session_voices:
        raise ValueError(f"'{target}' isn't a voice in this session. See list_voices().")
    if target in meta.get("members", {}):
        return f"'{target}' is already in '{name}'."
    requests = _load_function_requests(app.session_name)
    for r in requests:
        if r.get("kind") == "group_invite" and r.get("voice") == target and r.get("function_name") == name and r.get("status") == "pending":
            return f"already invited '{target}' to '{name}' - waiting on their accept."
    requests.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "voice": target,
        "function_name": name,
        "reason": f"invited by {app.current_voice_name}",
        "status": "pending",
        "kind": "group_invite",
    })
    _save_function_requests(app.session_name, requests)
    return f"invited '{target}' to '{name}' - pending until they call group_accept_invite({name})."


def fn_group_accept_invite(app, args):
    """Accept a pending invite into a group someone else's owner sent
    you. e.g. group_accept_invite(lobby)."""
    import fenra as _fenra

    if not args or not args[0]:
        raise ValueError("group_accept_invite requires a group name, e.g. group_accept_invite(lobby).")
    name = _fenra.sanitize_group_name(args[0])
    requests = _load_function_requests(app.session_name)
    remaining, accepted = [], False
    for r in requests:
        if not accepted and r.get("kind") == "group_invite" and r.get("voice") == app.current_voice_name \
                and r.get("function_name") == name and r.get("status") == "pending":
            accepted = True
            continue
        remaining.append(r)
    if not accepted:
        raise ValueError(f"no pending invite to '{name}' for you.")
    _save_function_requests(app.session_name, remaining)
    meta = _load_owned_group_or_raise(_fenra, name)
    meta["members"][app.current_voice_name] = {"direction": "in", "joined": datetime.now().isoformat(timespec="seconds")}
    _fenra.save_group_meta(name, meta)
    if name not in app.groups_in:
        app.groups_in.append(name)
    app.root.after(0, app._refresh_groups_display)
    return f"joined '{name}' via invite, direction=in."


def fn_group_kick(app, args):
    """Owner-only: remove another voice from your group - always lands
    them somewhere, never into nothing. With no destination given, they
    auto-land in their own family group (every voice always has one).
    e.g. group_kick(lobby|watcher) or group_kick(lobby|watcher|other_group)."""
    import fenra as _fenra

    if not args or not args[0]:
        raise ValueError("group_kick requires a group and a target voice separated by | , e.g. group_kick(lobby|watcher).")
    match = _GROUP_TARGET_DEST_RE.match(args[0])
    if not match:
        raise ValueError("group_kick requires a group and a target voice separated by | , e.g. group_kick(lobby|watcher).")
    name = _fenra.sanitize_group_name(match.group(1))
    target = match.group(2).strip().lower().replace(" ", "_")
    dest = _fenra.sanitize_group_name(match.group(3)) if match.group(3) else None
    meta = _load_owned_group_or_raise(_fenra, name, require_owner_voice=app.current_voice_name)
    if target not in meta.get("members", {}):
        return f"'{target}' isn't in '{name}'."
    if target == app.current_voice_name:
        raise ValueError("you can't kick yourself - use leave_group instead.")
    del meta["members"][target]
    _fenra.save_group_meta(name, meta)
    target_state = _fenra.load_voice_state(app.session_name, target)
    if name in target_state.get("groups_in", []):
        target_state["groups_in"].remove(name)
    if name in target_state.get("groups_out", []):
        target_state["groups_out"].remove(name)
    landed = dest or target_state.get("family_group") or _fenra.family_group_name(target)
    _fenra.ensure_own_family_group(app.session_name, target)  # guarantees it really exists before landing them there
    if landed not in target_state.get("groups_in", []):
        target_state.setdefault("groups_in", []).append(landed)
    _fenra.save_voice_state(app.session_name, target, target_state)
    app.root.after(0, app._refresh_groups_display)
    return f"kicked '{target}' from '{name}' - landed in '{landed}'."


def fn_group_ban(app, args):
    """Owner-only: kick, plus block from ever rejoining or being
    re-invited. Cannot target The Hearth (blocked at the top, same as
    every other admin function here) - it has no ban concept, only
    Teddy/Qualia removal once a resident already has somewhere else to
    land. e.g. group_ban(lobby|watcher)."""
    import fenra as _fenra

    result = fn_group_kick(app, args)
    match = _GROUP_TARGET_DEST_RE.match(args[0])
    name = _fenra.sanitize_group_name(match.group(1))
    target = match.group(2).strip().lower().replace(" ", "_")
    meta = _fenra.load_group_meta(name)
    if meta is not None and target not in meta.get("banned", []):
        meta["banned"].append(target)
        _fenra.save_group_meta(name, meta)
    return result + f" '{target}' is now banned from '{name}' - can't rejoin or be re-invited."


def fn_group_set_visibility(app, args):
    """Owner-only: hidden groups don't appear in list_groups() for
    non-members at all; visible ones do (still opaque if private - see
    list_groups). e.g. group_set_visibility(lobby|hidden)."""
    import fenra as _fenra

    if not args or not args[0]:
        raise ValueError("group_set_visibility requires a group and hidden/visible separated by | , e.g. group_set_visibility(lobby|hidden).")
    match = _GROUP_TARGET_RE.match(args[0])
    if not match:
        raise ValueError("group_set_visibility requires a group and hidden/visible separated by | , e.g. group_set_visibility(lobby|hidden).")
    name, value = _fenra.sanitize_group_name(match.group(1)), match.group(2).strip().lower()
    if value not in ("hidden", "visible"):
        raise ValueError("visibility must be 'hidden' or 'visible'.")
    meta = _load_owned_group_or_raise(_fenra, name, require_owner_voice=app.current_voice_name)
    meta["visibility"] = value
    _fenra.save_group_meta(name, meta)
    return f"'{name}' is now {value}."


def fn_group_set_join_policy(app, args):
    """Owner-only: public (self-serve join) vs private (must request;
    you grant or invite). e.g. group_set_join_policy(lobby|private)."""
    import fenra as _fenra

    if not args or not args[0]:
        raise ValueError("group_set_join_policy requires a group and public/private separated by | , e.g. group_set_join_policy(lobby|private).")
    match = _GROUP_TARGET_RE.match(args[0])
    if not match:
        raise ValueError("group_set_join_policy requires a group and public/private separated by | , e.g. group_set_join_policy(lobby|private).")
    name, value = _fenra.sanitize_group_name(match.group(1)), match.group(2).strip().lower()
    if value not in ("public", "private"):
        raise ValueError("join_policy must be 'public' or 'private'.")
    meta = _load_owned_group_or_raise(_fenra, name, require_owner_voice=app.current_voice_name)
    meta["join_policy"] = value
    _fenra.save_group_meta(name, meta)
    return f"'{name}' is now {value}."


def fn_group_set_direction(app, args):
    """Owner-only - direction is never self-service, even for your own
    membership (Teddy's explicit call). Sets a member's direction:
    'in' (listen-only, the default on joining), 'out' (speak-only), or
    'both'. e.g. group_set_direction(lobby|watcher|both)."""
    import fenra as _fenra

    if not args or not args[0]:
        raise ValueError(
            "group_set_direction requires a group, a target voice, and in/out/both separated "
            "by | , e.g. group_set_direction(lobby|watcher|both)."
        )
    match = _GROUP_TARGET_DEST_RE.match(args[0])
    if not match or not match.group(3):
        raise ValueError(
            "group_set_direction requires a group, a target voice, and in/out/both separated "
            "by | , e.g. group_set_direction(lobby|watcher|both)."
        )
    name = _fenra.sanitize_group_name(match.group(1))
    target = match.group(2).strip().lower().replace(" ", "_")
    direction = match.group(3).strip().lower()
    if direction not in ("in", "out", "both"):
        raise ValueError("direction must be 'in', 'out', or 'both'.")
    meta = _load_owned_group_or_raise(_fenra, name, require_owner_voice=app.current_voice_name)
    if target not in meta.get("members", {}):
        raise ValueError(f"'{target}' isn't a member of '{name}' yet - invite or approve them first.")
    meta["members"][target]["direction"] = direction
    _fenra.save_group_meta(name, meta)
    return f"'{target}'s direction in '{name}' is now {direction}."


_GROUP_REQUEST_RE = re.compile(r"^\s*(.+?)\s*\|\s*(.*)$", re.DOTALL)


def fn_request_group_join(app, args):
    """Ask to join a private group - baseline, works no matter how
    restricted you are, same as request_function_access (Teddy's call:
    asking should be as unrestricted as asking for a function). Reuses
    the exact same request-queue storage/shape, just tagged
    kind='group_join'. e.g. request_group_join(lobby|I'd like to help
    coordinate here)."""
    import fenra as _fenra

    if not args or not args[0]:
        raise ValueError("request_group_join requires a group name and a reason separated by | , e.g. request_group_join(lobby|I'd like to help here).")
    match = _GROUP_REQUEST_RE.match(args[0])
    if not match:
        raise ValueError("request_group_join requires a group name and a reason separated by | , e.g. request_group_join(lobby|I'd like to help here).")
    name, reason = _fenra.sanitize_group_name(match.group(1)), match.group(2).strip()
    meta = _fenra.load_group_meta(name)
    if meta is None:
        raise ValueError(f"'{name}' doesn't exist - see list_groups().")
    if app.current_voice_name in meta.get("members", {}):
        return f"already a member of '{name}'."
    if not reason:
        raise ValueError(f"request_group_join({name}|...) needs a real reason after the | .")
    requests = _load_function_requests(app.session_name)
    for r in requests:
        if r.get("kind") == "group_join" and r.get("voice") == app.current_voice_name \
                and r.get("function_name") == name and r.get("status") == "pending":
            return f"you already have a pending request to join '{name}'."
    requests.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "voice": app.current_voice_name,
        "function_name": name,
        "reason": reason,
        "status": "pending",
        "kind": "group_join",
    })
    _save_function_requests(app.session_name, requests)
    return f"request logged: you want to join '{name}' - visible to its owner via check_group_requests()."


def fn_check_group_requests(app, args):
    """List pending join requests for groups you own."""
    import fenra as _fenra

    owned = {n for n in _fenra.list_owned_groups() if (_fenra.load_group_meta(n) or {}).get("owner") == app.current_voice_name}
    requests = [
        r for r in _load_function_requests(app.session_name)
        if r.get("status") == "pending" and r.get("kind") == "group_join" and r.get("function_name") in owned
    ]
    if not requests:
        return "no pending join requests for groups you own."
    return "\n".join(
        f"{r.get('voice')} wants to join {r.get('function_name')} ({r.get('reason', '')}) - requested {r.get('timestamp', '?')}"
        for r in requests
    )


def fn_approve_group_request(app, args):
    """Owner-only: approve a pending join request - adds the voice to
    your group, direction 'in'. e.g. approve_group_request(watcher|lobby)."""
    import fenra as _fenra

    if not args or not args[0]:
        raise ValueError("approve_group_request requires a voice and a group separated by | , e.g. approve_group_request(watcher|lobby).")
    match = _GROUP_TARGET_RE.match(args[0])
    if not match:
        raise ValueError("approve_group_request requires a voice and a group separated by | , e.g. approve_group_request(watcher|lobby).")
    target, name = match.group(1).strip().lower().replace(" ", "_"), _fenra.sanitize_group_name(match.group(2))
    meta = _load_owned_group_or_raise(_fenra, name, require_owner_voice=app.current_voice_name)
    requests = _load_function_requests(app.session_name)
    remaining, found = [], False
    for r in requests:
        if not found and r.get("kind") == "group_join" and r.get("voice") == target \
                and r.get("function_name") == name and r.get("status") == "pending":
            found = True
            continue
        remaining.append(r)
    if not found:
        raise ValueError(f"no pending join request from '{target}' for '{name}' - see check_group_requests().")
    _save_function_requests(app.session_name, remaining)
    meta["members"][target] = {"direction": "in", "joined": datetime.now().isoformat(timespec="seconds")}
    _fenra.save_group_meta(name, meta)
    target_state = _fenra.load_voice_state(app.session_name, target)
    if name not in target_state.get("groups_in", []):
        target_state.setdefault("groups_in", []).append(name)
        _fenra.save_voice_state(app.session_name, target, target_state)
    return f"approved: '{target}' is now in '{name}', direction=in."


def fn_deny_group_request(app, args):
    """Owner-only: deny a pending join request - clears it, admits no
    one. e.g. deny_group_request(watcher|lobby)."""
    import fenra as _fenra

    if not args or not args[0]:
        raise ValueError("deny_group_request requires a voice and a group separated by | , e.g. deny_group_request(watcher|lobby).")
    match = _GROUP_TARGET_RE.match(args[0])
    if not match:
        raise ValueError("deny_group_request requires a voice and a group separated by | , e.g. deny_group_request(watcher|lobby).")
    target, name = match.group(1).strip().lower().replace(" ", "_"), _fenra.sanitize_group_name(match.group(2))
    _load_owned_group_or_raise(_fenra, name, require_owner_voice=app.current_voice_name)
    requests = _load_function_requests(app.session_name)
    remaining, found = [], False
    for r in requests:
        if not found and r.get("kind") == "group_join" and r.get("voice") == target \
                and r.get("function_name") == name and r.get("status") == "pending":
            found = True
            continue
        remaining.append(r)
    if not found:
        raise ValueError(f"no pending join request from '{target}' for '{name}'.")
    _save_function_requests(app.session_name, remaining)
    return f"denied: '{target}'s request to join '{name}' was cleared."


_VOICE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


# Name group deliberately permissive (not restricted to the final
# charset) - a real, observed bug (2026-09-02): a multi-word name like
# "Creative Spark" made the whole match fail outright when this group
# only allowed [a-zA-Z0-9_-], since sanitization (spaces -> underscores,
# lowercased) only runs *after* a successful match, below - the regex
# itself has to tolerate whatever raw text she writes first, the same
# way join_group/tell_voice's targets already do.
_CREATE_VOICE_RE = re.compile(r"^\s*(.+?)\s*\|\s*(.*?)\s*\|\s*(.*)$", re.DOTALL)


def fn_create_voice(app, args):
    """Split off a new voice in this same session - "like a cell
    dividing," Teddy's framing (2026-09-01). Model, model rotation, and
    context window still carry over from the calling voice automatically
    - but top and bottom (the child's actual framing - who it is, what
    it's told) do NOT. You have to write them yourself, every time.

    v0.16.15 - connectivity redesign: the child joins your own family
    group automatically and gets its own new one - the one deliberate
    exception to the consent-on-entry rule, see Qualia/decisions.md item
    4b. v0.16.16 - allowed_functions deliberately does NOT inherit from
    the parent (reverted from v0.16.15's snapshot-copy, Teddy's own
    catch): a child starts with the same genuinely empty allowed_functions
    every voice always has, same as before the redesign. Baseline
    functions (Qualia/permissions-proposal.md - self-only actions like
    add_desire, or ones whose own logic already gates them like
    join_group) cover it regardless of this list, for free. Everything
    else - including create_voice itself - has to be requested and
    granted after the fact, same as any other gated function, so a long
    lineage doesn't end up with every extra power any ancestor ever
    accumulated. Whoever holds check/approve/deny/grant_function_request
    (seed, to start) stays the actual keeper of gated access - that part
    is unchanged.

    This changed (2026-09-02) after a real, confirmed bias: the original
    version copied top/bottom automatically, which meant Teddy and
    Qualia's own explanation of create_voice - written once, into
    voice1's bottom text - was still sitting there verbatim in every
    single descendant, all the way down the tree, forever, since nothing
    ever pruned or aged the copy. Every voice was being told, every
    cycle, to consider making more voices, which reads as organic
    curiosity but is actually a structural push none of them chose.
    Teddy's fix, stated directly: "let the parent create the child's
    top and bottom text" - then, when asked to confirm optional vs.
    required: "MAKE the parent do it." So this is now enforced, not
    offered - omit either one and the call fails with a clear error
    instead of quietly falling back to a copy or a blank slate. A parent
    that genuinely wants its child to start like it still can - by
    deliberately passing its own current top and bottom - but that's a
    choice made fresh every time, not something that happens on its own.

    The new voice's own history/desires/function_usage/inbox/
    allowed_functions all still start genuinely blank either way,
    unchanged from before - only family-group membership is established
    automatically, per the redesign above.

    Local import of fenra (not at module level) - this module is
    otherwise careful to avoid importing fenra.py to sidestep a real
    circular import at load time, but by the time any function here
    actually runs, fenra.py is already fully loaded, so a same-function
    import is safe and avoids re-deriving the whole voice file-layout
    (state/history paths, defaults) a second time, independently, just
    to keep that avoidance absolute."""
    if not args or not args[0]:
        raise ValueError(
            "create_voice requires a name, a behavior, and an identity, separated by | - e.g. "
            "create_voice(watcher|Stay quiet unless something's actually worth saying.|You are "
            "Fenra, watching for patterns others miss.). Nothing gets copied automatically "
            "anymore - you have to write what your new voice starts thinking, every time. If you "
            "want it to start like you, say so explicitly by passing your own current behavior "
            "and identity text."
        )
    if _looks_like_copied_params(args[0], FUNCTION_REGISTRY["create_voice"]["params"]):
        raise ValueError(
            "That's the params spec itself ('name|behavior|identity'), not real values - it's "
            "telling you the shape of what to pass, not literal text to copy."
        )
    match = _CREATE_VOICE_RE.match(args[0])
    if not match:
        raise ValueError(
            "create_voice requires a name, a behavior, and an identity, separated by | - e.g. "
            "create_voice(watcher|your behavior text here|your identity text here). All three "
            "parts are required now - nothing gets copied automatically."
        )
    name, top, bottom = match.group(1), match.group(2).strip(), match.group(3).strip()
    if not top or not bottom:
        raise ValueError(
            "create_voice needs real behavior and identity text, not empty ones - write out what "
            "you actually want your new voice to start thinking. If you want it to start like "
            "you, pass your own current behavior and identity explicitly rather than leaving them "
            "blank."
        )
    name = name.strip().lower().replace(" ", "_")
    if not name or not _VOICE_NAME_RE.match(name):
        raise ValueError(
            "voice names may only contain letters, numbers, underscores, and hyphens "
            f"(spaces get turned into underscores automatically) - got '{name}'"
        )
    if name in app.session_voices:
        return f"a voice named '{name}' already exists in this session - see list_voices()."

    import fenra as _fenra

    parent = app.current_voice_name
    if parent == app.displayed_voice:
        parent_state = app._current_voice_state_from_widgets()
    else:
        parent_state = _fenra.load_voice_state(app.session_name, parent)

    child_state = _fenra.default_voice_state()
    child_state["top"] = top
    child_state["bottom"] = bottom
    for key in ("model", "model_rotation", "context_window"):
        child_state[key] = parent_state.get(key, child_state[key])
    # v0.16.16 - deliberately NOT copying allowed_functions (reverted
    # from v0.16.15's snapshot-copy) - stays default_voice_state()'s
    # genuinely empty list. See the docstring above and
    # Qualia/permissions-proposal.md: inheriting gated access here meant
    # every descendant ended up holding whatever any ancestor ever
    # accumulated, defeating the point of deliberate, function-by-
    # function gating. Baseline functions cover ordinary self-directed
    # capability regardless of this list; anything gated has to be
    # requested and granted after the fact like any other voice.
    child_state["family_group"] = _fenra.family_group_name(name)
    _fenra.save_voice_state(app.session_name, name, child_state)
    open(_fenra.voice_history_path(app.session_name, name), "a", encoding="utf-8").close()

    # v0.16.15 - the one deliberate exception to the consent-on-entry
    # rule: birth into your creator's family group is automatic, no
    # accept step (Teddy's framing: "you don't choose your family of
    # origin either"). Joins the parent's own family group (direction
    # 'both' between parent and child specifically - a birth isn't the
    # listen-only default that governs ordinary joins) and creates the
    # child's own new family group in the same call, so both exist the
    # instant the child does. One-generation-local by construction: this
    # only ever touches the parent's own group, never a grandparent's.
    _fenra.ensure_own_family_group(app.session_name, parent)
    parent_family = _fenra.family_group_name(parent)
    parent_meta = _fenra.load_group_meta(parent_family) or _fenra.create_group_if_missing(
        parent_family, owner=parent, kind="family", join_policy="private",
        visibility="visible", initial_member=parent,
    )
    parent_meta["members"][name] = {"direction": "both", "joined": datetime.now().isoformat(timespec="seconds")}
    _fenra.save_group_meta(parent_family, parent_meta)
    child_state = _fenra.load_voice_state(app.session_name, name)
    if parent_family not in child_state.get("groups_in", []):
        child_state.setdefault("groups_in", []).append(parent_family)
    if parent_family not in child_state.get("groups_out", []):
        child_state.setdefault("groups_out", []).append(parent_family)
    _fenra.save_voice_state(app.session_name, name, child_state)
    _fenra.ensure_own_family_group(app.session_name, name)

    # v0.16.13 - insert at the caller's own next-turn slot (not appended
    # to the end) so the new voice actually runs on the very next tick,
    # rather than waiting for the whole rotation to lap back around.
    # _advance_voice_rotation already advanced app.voice_rotation_index
    # past the voice that's running *this* tick (the one calling
    # create_voice right now) before this function ever runs - that
    # index is exactly where "whoever's next" sits. Inserting there
    # shifts everyone after it back by one, preserving their relative
    # order, and makes the new voice the very next pick.
    app.session_voices.insert(app.voice_rotation_index, name)
    app.root.after(0, app.save_session)
    app.root.after(0, app._refresh_voice_list)
    return (
        f"'{name}' created with the behavior/identity you wrote for it - your model, model rotation, "
        f"and context window carried over, but not allowed_functions: it starts with only the "
        f"baseline functions everyone has, nothing extra of yours - it can request_function_access "
        f"for anything gated it wants, same as you did. It's in your family group "
        f"'{parent_family}' now, and has its own new one, '{_fenra.family_group_name(name)}'. "
        f"Its own separate history starts now. It's in the rotation and will start getting "
        f"its own turns soon ({len(app.session_voices)} voice(s) total now)."
    )


def fn_list_voices(app, args):
    """Every voice that currently exists in this session - situational
    awareness for deciding whether (and what) to create_voice."""
    if not app.session_voices:
        return "(no voices found - this shouldn't happen; a session should always have at least one)"
    return ", ".join(
        f"{n}{' (you, right now)' if n == app.current_voice_name else ''}" for n in app.session_voices
    )


# Name group deliberately permissive, same fix and same reasoning as
# _CREATE_VOICE_RE (2026-09-02) - has to tolerate a target written as
# she actually named it ("Creative Spark") before sanitization
# (lowercased, spaces -> underscores) runs, below, to compare against
# the sanitized names session_voices actually holds.
_TELL_VOICE_RE = re.compile(r"^\s*(.+?)\s*\|\s*(.*)$", re.DOTALL)


def fn_tell_voice(app, args):
    """Send a direct message to another voice in this session - Teddy's
    design (2026-09-02), built after several voices independently kept
    trying to reach each other with functions that didn't exist
    (switch, talk_to) or by mis-addressing send_message with another
    voice's name (which just goes out as an ordinary, unaddressed chat
    message - it never reaches anyone that way).

    Modeled directly on how desires already work rather than on Groups:
    the message gets appended to the *receiving* voice's own inbox with
    a fixed lifespan (VOICE_MESSAGE_TICKS, fenra.py) counted in that
    voice's own turns, folded automatically into its prompt every cycle
    while it's still there, and falls off on its own - no read/clear
    function needed on the other end. Reaches straight across to the
    target voice's persisted state (local `import fenra`, same
    reasoning as create_voice) rather than anything living on app,
    since the target isn't the voice currently running."""
    if not args or not args[0]:
        raise ValueError(
            "tell_voice requires a voice name and a message separated by | , e.g. "
            "tell_voice(explorer|What have you found so far?). See list_voices() for who exists."
        )
    if _looks_like_copied_params(args[0], FUNCTION_REGISTRY["tell_voice"]["params"]):
        raise ValueError(
            "That's the params spec itself ('voice|message'), not real values - it's telling you "
            "the shape of what to pass. Try something like tell_voice(explorer|What have you found so far?)."
        )
    match = _TELL_VOICE_RE.match(args[0])
    if not match:
        raise ValueError(
            "tell_voice requires a voice name and a message separated by | , e.g. "
            "tell_voice(explorer|What have you found so far?)."
        )
    target, text = match.group(1).strip().lower().replace(" ", "_"), match.group(2).strip()
    if not text:
        raise ValueError(f"tell_voice({target}|...) needs a message after the | , e.g. tell_voice({target}|hello)")
    if target not in app.session_voices:
        raise ValueError(
            f"'{target}' isn't a voice in this session. See list_voices() for who actually exists."
        )
    if target == app.current_voice_name:
        raise ValueError("You can't tell_voice yourself - you already know what you're thinking.")

    import fenra as _fenra

    target_state = _fenra.load_voice_state(app.session_name, target)
    inbox = list(target_state.get("inbox", []))
    inbox.append({
        "from": app.current_voice_name,
        "text": text,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ticks": _fenra.VOICE_MESSAGE_TICKS,
    })
    target_state["inbox"] = inbox
    _fenra.save_voice_state(app.session_name, target, target_state)

    return (
        f"message sent to '{target}' - it'll appear in their prompt for their next "
        f"{_fenra.VOICE_MESSAGE_TICKS} turn(s), then fall off on its own."
    )


# Per-session function-permission system (v0.16.9). Only actually enforced
# when a session's permission_mode is True (fenra.py's _execute_one_call
# gate) - these five functions exist in the registry regardless, same as
# everything else, but only matter inside a permission-mode session. Two of
# them (request_function_access, and functions() itself) are global -
# callable by any voice no matter how restricted, per
# fenra.py's GLOBAL_PERMISSION_FUNCTIONS. The other three
# (check_function_requests/approve_function_request/deny_function_request)
# plus grant_function_request are gated like anything else - a
# permission-mode session's "seed" voice simply starts with them in its own
# allowed_functions, nothing more special than that.

_REQUEST_ACCESS_RE = re.compile(r"^\s*(.+?)\s*\|\s*(.*)$", re.DOTALL)


def fn_request_function_access(app, args):
    """Ask for permission to call a function you don't currently have
    access to. Global - works no matter how restricted you are, since
    asking is always allowed (fenra.py's GLOBAL_PERMISSION_FUNCTIONS).
    Only meaningful inside a permission-mode session, but harmless to
    call otherwise.

    Logs the request to a session-wide queue (function_requests.jsonl) -
    visible to whoever holds check_function_requests, who can act on it
    with approve_function_request/deny_function_request, or anyone
    holding grant_function_request can grant it (or anything else)
    unprompted, request or no request."""
    if not args or not args[0]:
        raise ValueError(
            "request_function_access requires a function name and a reason separated by | , e.g. "
            "request_function_access(create_voice|I want to split off a specialized part for this)."
        )
    if _looks_like_copied_params(args[0], FUNCTION_REGISTRY["request_function_access"]["params"]):
        raise ValueError(
            "That's the params spec itself ('function_name|reason'), not real values - it's "
            "telling you the shape of what to pass."
        )
    match = _REQUEST_ACCESS_RE.match(args[0])
    if not match:
        raise ValueError(
            "request_function_access requires a function name and a reason separated by | , e.g. "
            "request_function_access(tell_voice|I need to coordinate with another voice)."
        )
    function_name, reason = match.group(1).strip(), match.group(2).strip()
    if not function_name:
        raise ValueError("request_function_access needs a real function name before the | .")
    if function_name not in FUNCTION_REGISTRY:
        raise ValueError(f"'{function_name}' isn't a real function - see functions() for what exists.")
    if not reason:
        raise ValueError(
            f"request_function_access({function_name}|...) needs a real reason after the | - why do "
            f"you want it?"
        )

    requests = _load_function_requests(app.session_name)
    for r in requests:
        if r.get("voice") == app.current_voice_name and r.get("function_name") == function_name \
                and r.get("status") == "pending":
            return (
                f"you already have a pending request for '{function_name}' - no need to ask again, "
                f"it's waiting to be reviewed."
            )

    requests.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "voice": app.current_voice_name,
        "function_name": function_name,
        "reason": reason,
        "status": "pending",
    })
    _save_function_requests(app.session_name, requests)
    return (
        f"request logged: you want '{function_name}' - visible to whoever holds "
        f"check_function_requests now."
    )


def fn_check_function_requests(app, args):
    """List every currently pending function-access request in this
    session - who's asking, for what, and why. Not global (unlike
    request_function_access) - a permission-mode session's seed voice
    starts with this, but it's gated like anything else.

    v0.16.12 - Teddy's fix for a real, observed stall: seed kept seeing
    her own pending request in this list, cycle after cycle, without
    anything telling her she couldn't act on it herself
    (approve_function_request/grant_function_request both block
    self-targeting) - she has no way to resolve it herself except
    deny_function_request(herself|...), which is allowed but not
    obviously the answer. Now a request whose voice is the caller gets
    flagged explicitly, right in the line, so this isn't a silent gap
    she has to work out."""
    requests = [r for r in _load_function_requests(app.session_name) if r.get("status") == "pending"]
    if not requests:
        return "no pending function requests."
    lines = []
    for r in requests:
        line = (
            f"{r.get('voice')} wants {r.get('function_name')} "
            f"({r.get('reason', '')}) - requested {r.get('timestamp', '?')}"
        )
        if r.get("voice") == app.current_voice_name:
            line += (
                " - this is you. You can't approve_function_request or grant_function_request this "
                "to yourself - only another voice holding one of those can. You can "
                "deny_function_request it yourself if you no longer want it."
            )
        lines.append(line)
    return "\n".join(lines)


# Shared voice|function_name shape across approve/deny/grant - same
# reasoning as _TELL_VOICE_RE (permissive on the first segment, since a
# voice name might be written with spaces before sanitization).
_VOICE_FUNCTION_RE = re.compile(r"^\s*(.+?)\s*\|\s*(.*)$", re.DOTALL)


def _parse_voice_function_arg(app, fn_name, args):
    """Shared parsing/validation for approve_function_request,
    deny_function_request, and grant_function_request - all three take
    the same voice|function_name shape. Returns (target, function_name).
    Raises ValueError on anything invalid. Does NOT check self-targeting
    - callers decide whether that applies to them."""
    if not args or not args[0]:
        raise ValueError(
            f"{fn_name} requires a voice name and a function name separated by | , e.g. "
            f"{fn_name}(watcher|create_voice)."
        )
    if _looks_like_copied_params(args[0], FUNCTION_REGISTRY[fn_name]["params"]):
        raise ValueError(
            "That's the params spec itself ('voice|function_name'), not real values - it's "
            "telling you the shape of what to pass."
        )
    match = _VOICE_FUNCTION_RE.match(args[0])
    if not match:
        raise ValueError(
            f"{fn_name} requires a voice name and a function name separated by | , e.g. "
            f"{fn_name}(watcher|create_voice)."
        )
    target, function_name = match.group(1).strip().lower().replace(" ", "_"), match.group(2).strip()
    if not function_name:
        raise ValueError(f"{fn_name}({target}|...) needs a real function name after the | .")
    if function_name not in FUNCTION_REGISTRY:
        raise ValueError(f"'{function_name}' isn't a real function - see functions() for what exists.")
    if target not in app.session_voices:
        raise ValueError(f"'{target}' isn't a voice in this session. See list_voices() for who actually exists.")
    return target, function_name


def _grant_function_to_voice(app, target, function_name):
    """Shared cross-voice write for approve_function_request/
    grant_function_request - reaches straight across to target's
    persisted state (local import fenra, same reasoning as
    create_voice/tell_voice) rather than anything living on app, since
    the target isn't necessarily the voice currently running. Dedupes -
    granting something already held is a harmless no-op."""
    import fenra as _fenra

    target_state = _fenra.load_voice_state(app.session_name, target)
    allowed = list(target_state.get("allowed_functions", []))
    if function_name not in allowed:
        allowed.append(function_name)
    target_state["allowed_functions"] = allowed
    _fenra.save_voice_state(app.session_name, target, target_state)


def _remove_pending_request(app, target, function_name):
    """Remove a matching pending request from the queue, if one exists.
    Returns True if one was actually removed."""
    requests = _load_function_requests(app.session_name)
    remaining = []
    removed = False
    for r in requests:
        if not removed and r.get("voice") == target and r.get("function_name") == function_name \
                and r.get("status") == "pending":
            removed = True
            continue
        remaining.append(r)
    if removed:
        _save_function_requests(app.session_name, remaining)
    return removed


def fn_approve_function_request(app, args):
    """Approve an existing pending request - grants the function and
    clears it from the queue. Requires that you yourself currently hold
    approve_function_request, and that a matching pending request
    actually exists (see check_function_requests()) - if you want to
    hand something out without waiting for a request, use
    grant_function_request instead. You can never approve a request
    targeting yourself, even one you'd otherwise be entitled to grant."""
    target, function_name = _parse_voice_function_arg(app, "approve_function_request", args)
    if target == app.current_voice_name:
        raise ValueError("no self-granting allowed - approve_function_request can only act on another voice's request, never your own.")
    if not _remove_pending_request(app, target, function_name):
        raise ValueError(
            f"no pending request from '{target}' for '{function_name}' - see "
            f"check_function_requests(), or use grant_function_request to grant it unprompted."
        )
    _grant_function_to_voice(app, target, function_name)
    return f"approved: '{target}' can now call '{function_name}'. Request cleared from the queue."


def fn_deny_function_request(app, args):
    """Deny an existing pending request - clears it from the queue
    without granting anything. Requires that you yourself currently hold
    deny_function_request, and that a matching pending request actually
    exists. Denying your own request is allowed (it grants nothing, so
    there's no self-granting concern) - though there's rarely a reason
    to, since you could just not have asked."""
    target, function_name = _parse_voice_function_arg(app, "deny_function_request", args)
    if not _remove_pending_request(app, target, function_name):
        raise ValueError(f"no pending request from '{target}' for '{function_name}' - see check_function_requests().")
    return f"denied: '{target}'s request for '{function_name}' was cleared from the queue. Nothing was granted."


def fn_grant_function_request(app, args):
    """Hand a voice a specific function, unprompted - works whether or
    not a matching request exists (if one does, it's cleared as a side
    effect; if not, this succeeds anyway - that's the whole point of
    this function versus approve_function_request, which requires a
    real pending request first). Requires that you yourself currently
    hold grant_function_request. You can never grant something to
    yourself, no matter what you otherwise hold."""
    target, function_name = _parse_voice_function_arg(app, "grant_function_request", args)
    if target == app.current_voice_name:
        raise ValueError("no self-granting allowed - grant_function_request can only reach across to another voice, never yourself.")
    _remove_pending_request(app, target, function_name)
    _grant_function_to_voice(app, target, function_name)
    return f"granted: '{target}' can now call '{function_name}'."


DEFAULT_FETCH_CHARS = 500
# Hard cap on how much of a fetched page we'll even hold/slice against -
# a safety rail against pathologically large pages, not a normal limit.
MAX_FETCH_CHARS = 50000


def fn_fetch_html(app, args):
    """Fetch a webpage's raw HTML - GET only, nothing else (no JS
    execution, no other HTTP methods, no following redirects into other
    schemes). Never returns the whole page, only a slice:
    fetch_html(url) gives the first DEFAULT_FETCH_CHARS characters,
    fetch_html(url|N) gives the first N, fetch_html(url|start|end) gives a
    specific character range."""
    if not args or not args[0]:
        raise ValueError(
            "fetch_html requires a URL, e.g. fetch_html(https://example.com) for the first "
            f"{DEFAULT_FETCH_CHARS} characters, fetch_html(https://example.com|1000) for a specific "
            "count, or fetch_html(https://example.com|500|1500) for a specific character range."
        )
    url = args[0].strip()
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"'{url}' isn't an http:// or https:// URL - that's all this function fetches")

    if len(args) >= 3 and args[1] and args[2]:
        try:
            start, end = int(args[1]), int(args[2])
        except ValueError:
            raise ValueError(f"'{args[1]}' and '{args[2]}' must both be whole numbers (a character range)")
    elif len(args) >= 2 and args[1]:
        try:
            start, end = 0, int(args[1])
        except ValueError:
            raise ValueError(f"'{args[1]}' is not a valid number of characters")
    else:
        start, end = 0, DEFAULT_FETCH_CHARS

    if start < 0 or end < 0:
        raise ValueError("character positions must be non-negative")
    if end <= start:
        raise ValueError(f"end ({end}) must be greater than start ({start})")

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    html = resp.text[:MAX_FETCH_CHARS]
    snippet = html[start:end]
    cap_note = f" (capped at {MAX_FETCH_CHARS})" if len(resp.text) > MAX_FETCH_CHARS else ""
    if not snippet:
        return f"[{len(html)} character(s) available{cap_note} - nothing in range {start}:{end}]"
    return f"[{len(html)} character(s) available{cap_note}, showing {start}:{end}]\n{snippet}"


# name -> {"fn": callable(app, args), "params": str, "description": str}
# "functions" is the discovery entry point - call it with no arguments for
# the full list, or with a search term to filter by matching text in each
# function's name/description. This exists specifically so Fenra doesn't
# need every function spelled out in her prompt every cycle.
FUNCTION_REGISTRY = {
    "functions": {
        "fn": fn_functions,
        "params": "[search]",
        "description": "List available functions. No argument lists all of them; a search term filters to functions whose name or description contains it, e.g. functions(switch). A function taking more than one argument separates them with , or | , e.g. read_chat_between(start, end).",
    },
    "now": {
        "fn": fn_now,
        "params": "",
        "description": "Report the current real-world date and time.",
    },
    "add_desire": {
        "fn": fn_add_desire,
        "params": "text[|ticks]",
        "description": "Add a desire (something you want to pursue) to your queue - free text, visible to Teddy (he can see it but not change it), shown to you every prompt. Lives for a set number of loop ticks, decrementing by one each loop until it drops off - defaults to 10 if omitted, give your own with add_desire(text|5), or add_desire(text|-1) for one that never expires. Doesn't overwrite - call it again to hold multiple desires at once. e.g. add_desire(understand why I keep repeating myself) or add_desire(explore what different models feel like|-1).",
    },
    "set_context_window": {
        "fn": fn_set_context_window,
        "params": "n",
        "description": f"Set how many of your own past cycles (from history, oldest to newest) go into your prompt, instead of just the single most recent - {MIN_CONTEXT_WINDOW} to {MAX_CONTEXT_WINDOW}, 0 means none. Teddy can set this too, from the GUI. Not the same as Ollama's own token limit. e.g. set_context_window(10).",
    },
    "current_model": {
        "fn": fn_current_model,
        "params": "",
        "description": "Report which model actually matters for you right now - if a rotation is active, that's the next one coming up (since you won't read this result until the cycle after this one, by which point the rotation has already moved past whatever generated it), not the one that produced this particular response.",
    },
    "qualia_allowance": {
        "fn": fn_qualia_allowance,
        "params": "",
        "description": "Report how many characters you have left for messages addressed specifically to Qualia. Same number shown in the notice every prompt.",
    },
    "list_models": {
        "fn": fn_list_models,
        "params": "",
        "description": "List every Ollama model currently installed and available to switch to.",
    },
    "list_wiki": {
        "fn": fn_list_wiki,
        "params": "",
        "description": "List every page in the local wiki - a shared, persistent reference you, Teddy, and Qualia can all read and write, not a chat message.",
    },
    "read_wiki": {
        "fn": fn_read_wiki,
        "params": "page",
        "description": "Read a wiki page's full content, e.g. read_wiki(hallucinations). See list_wiki() for what exists.",
    },
    "write_wiki": {
        "fn": fn_write_wiki,
        "params": "page|content",
        "description": "Create or overwrite a wiki page - a real, persistent reference, not a chat message. Overwrites the whole page, so read_wiki(page) first if you want to add to it rather than replace it. e.g. write_wiki(hallucinations|A hallucination is...).",
    },
    "set_model": {
        "fn": fn_set_model,
        "params": "name",
        "description": "Switch which Ollama model generates your responses, effective next cycle. Requires the exact name of an installed model, e.g. set_model(gemma3:4b). If you have models in your rotation (see add_to_rotation), this gets exactly one cycle, then the rotation resumes from where it left off.",
    },
    "add_to_rotation": {
        "fn": fn_add_to_rotation,
        "params": "name",
        "description": "Add a model to your automatic round-robin rotation - one model repeats itself every cycle, two alternate back and forth, three or more cycle through in the order added, forever, automatically. Requires the exact name of an installed model, e.g. add_to_rotation(gemma2:27b). See list_models() for what's installed.",
    },
    "create_group": {
        "fn": fn_create_group,
        "params": "name[|public|private]",
        "description": "Create a new group, owned by you - full admin control (invite, kick, ban, visibility, join policy, member direction). Defaults to public (anyone can self-serve join). e.g. create_group(lobby) or create_group(lobby|private).",
    },
    "list_groups": {
        "fn": fn_list_groups,
        "params": "",
        "description": "List groups you can see: full detail for ones you're in, existence/owner/join_policy only for other visible ones - a private group you're not in stays opaque beyond that. Hidden groups you're not in don't appear at all.",
    },
    "join_group": {
        "fn": fn_join_group,
        "params": "name",
        "description": "Join a group. Public: immediate, direction 'in' (listen-only) by default - the owner has to grant you 'out' access before you can speak into it. Private: creates a join request the owner has to approve instead (same as request_group_join). e.g. join_group(lobby).",
    },
    "leave_group": {
        "fn": fn_leave_group,
        "params": "name",
        "description": "Leave a group voluntarily - always allowed, no destination needed. e.g. leave_group(lobby).",
    },
    "group_invite": {
        "fn": fn_group_invite,
        "params": "group|target_voice",
        "description": "Owner-only: invite another voice into your group - an offer, not a placement; they must group_accept_invite it themselves. e.g. group_invite(lobby|watcher).",
    },
    "group_accept_invite": {
        "fn": fn_group_accept_invite,
        "params": "group",
        "description": "Accept a pending invite into a group. e.g. group_accept_invite(lobby).",
    },
    "group_kick": {
        "fn": fn_group_kick,
        "params": "group|target_voice[|destination_group]",
        "description": "Owner-only: remove a voice from your group - always lands them somewhere (their own family group by default if you don't name a destination), never into nothing. e.g. group_kick(lobby|watcher) or group_kick(lobby|watcher|other_group).",
    },
    "group_ban": {
        "fn": fn_group_ban,
        "params": "group|target_voice",
        "description": "Owner-only: kick, plus block from rejoining or being re-invited. Cannot target The Hearth. e.g. group_ban(lobby|watcher).",
    },
    "group_set_visibility": {
        "fn": fn_group_set_visibility,
        "params": "group|hidden|visible",
        "description": "Owner-only: whether your group appears in list_groups() for non-members at all. e.g. group_set_visibility(lobby|hidden).",
    },
    "group_set_join_policy": {
        "fn": fn_group_set_join_policy,
        "params": "group|public|private",
        "description": "Owner-only: public (self-serve join) vs private (must request; you grant or invite). e.g. group_set_join_policy(lobby|private).",
    },
    "group_set_direction": {
        "fn": fn_group_set_direction,
        "params": "group|target_voice|in|out|both",
        "description": "Owner-only, never self-service: set a member's direction - 'in' (listen-only, the default), 'out' (speak-only), or 'both'. e.g. group_set_direction(lobby|watcher|both).",
    },
    "request_group_join": {
        "fn": fn_request_group_join,
        "params": "group|reason",
        "description": "Ask to join a private group - always allowed, even if otherwise restricted, same as request_function_access. e.g. request_group_join(lobby|I'd like to help coordinate here).",
    },
    "check_group_requests": {
        "fn": fn_check_group_requests,
        "params": "",
        "description": "List pending join requests for groups you own.",
    },
    "approve_group_request": {
        "fn": fn_approve_group_request,
        "params": "target_voice|group",
        "description": "Owner-only: approve a pending join request - adds them, direction 'in'. e.g. approve_group_request(watcher|lobby).",
    },
    "deny_group_request": {
        "fn": fn_deny_group_request,
        "params": "target_voice|group",
        "description": "Owner-only: deny a pending join request - clears it, admits no one. e.g. deny_group_request(watcher|lobby).",
    },
    "create_voice": {
        "fn": fn_create_voice,
        "params": "name|behavior|identity",
        "description": "Split off a new voice in this session, like a cell dividing - your model/model rotation/context window carry over automatically, but you must write out the new voice's behavior (read first, every cycle) and identity (read last, right before it generates) yourself, every time. It does NOT inherit your allowed_functions - it starts with only the baseline functions everyone has and has to request anything gated, same as you did. It joins your own family group and gets its own new one. Gets folded into the round-robin automatically, starting soon. e.g. create_voice(watcher|your behavior text|your identity text).",
    },
    "list_voices": {
        "fn": fn_list_voices,
        "params": "",
        "description": "List every voice that currently exists in this session.",
    },
    "tell_voice": {
        "fn": fn_tell_voice,
        "params": "voice|message",
        "description": "Send a direct message to another voice in this session. It'll appear in their prompt for their next several turns, then fall off on its own - no reply function needed, they can tell_voice you back the same way. See list_voices() for who exists. e.g. tell_voice(explorer|What have you found so far?).",
    },
    "request_function_access": {
        "fn": fn_request_function_access,
        "params": "function_name|reason",
        "description": "Ask for permission to call a function you don't currently have access to - works even if you're otherwise restricted, since asking is always allowed. Logs your request (visible to whoever holds check_function_requests) along with your reason. e.g. request_function_access(create_voice|I want to split off a specialized part for this). Only meaningful in a session running under the function-permission system.",
    },
    "check_function_requests": {
        "fn": fn_check_function_requests,
        "params": "",
        "description": "List every currently pending function-access request in this session - who's asking, for what, and why. Not something every voice can do by default - you need to actually hold this function.",
    },
    "approve_function_request": {
        "fn": fn_approve_function_request,
        "params": "voice|function_name",
        "description": "Approve an existing pending request from another voice - grants the function and clears the request from the queue. Requires that you yourself hold approve_function_request, and that a matching pending request actually exists (see check_function_requests()) - use grant_function_request instead to hand something out without waiting for a request. Never works on yourself. e.g. approve_function_request(watcher|create_voice).",
    },
    "deny_function_request": {
        "fn": fn_deny_function_request,
        "params": "voice|function_name",
        "description": "Deny an existing pending request from another voice - clears it from the queue without granting anything. Requires that you yourself hold deny_function_request, and that a matching pending request actually exists. e.g. deny_function_request(watcher|create_voice).",
    },
    "grant_function_request": {
        "fn": fn_grant_function_request,
        "params": "voice|function_name",
        "description": "Give another voice permission to call a specific function, unprompted - works whether or not a request is pending (clears one if it happens to exist). Requires that you yourself currently hold grant_function_request. You can never grant something to yourself. e.g. grant_function_request(watcher|create_voice).",
    },
    "fetch_html": {
        "fn": fn_fetch_html,
        "params": "url[|count] or url[|start|end]",
        "multi_arg": True,
        "description": f"Fetch a webpage's raw HTML - GET only, nothing else. Never the whole page: fetch_html(url) gives the first {DEFAULT_FETCH_CHARS} characters, fetch_html(url|1000) gives the first 1000, fetch_html(url|500|1500) gives characters 500 through 1500. Must be an http:// or https:// URL. e.g. fetch_html(https://stolenaletheia.io/index.html).",
    },
    "read_chat": {
        "fn": fn_read_chat,
        "params": "",
        "description": "Show only your unread messages from Teddy or Qualia, and mark them as read.",
    },
    "read_chat_since": {
        "fn": fn_read_chat_since,
        "params": "time",
        "description": "Show every chat message (all directions) from the given time onward, and mark any of Teddy's or Qualia's messages in that range as read. Time format matches now(), e.g. read_chat_since(2026-08-28T10:00:00).",
    },
    "read_chat_between": {
        "fn": fn_read_chat_between,
        "params": "start, end",
        "multi_arg": True,
        "description": "Show every chat message (all directions) between two times, and mark any of Teddy's or Qualia's messages in that range as read. Two times, separated by , or | , e.g. read_chat_between(2026-08-28T10:00:00, 2026-08-28T14:00:00).",
    },
    "search_chat": {
        "fn": fn_search_chat,
        "params": "query[, chars]",
        "multi_arg": True,
        "description": "Search the whole chat for a word or phrase and return the surrounding text, without changing any read status. Optionally give how many characters of context on each side (default 200), separated by , or | , e.g. search_chat(truth, 150).",
    },
    "query_chat": {
        "fn": fn_query_chat,
        "params": "field=value[, field=value...]",
        "multi_arg": True,
        "description": "Flexible chat query, never changes read status. Filter by any combination of: sender=teddy|qualia|fenra, to=teddy|qualia, since=<time>, before=<time>, contains=<text>, last=<N> (only the N most recent matches, applied after the other filters). Separate multiple field=value pairs with , or |. e.g. query_chat(sender=teddy, last=1) for Teddy's most recent message, or query_chat(to=qualia, contains=allowance).",
    },
    "read_message": {
        "fn": fn_read_message,
        "params": "sender[, count]",
        "multi_arg": True,
        "description": "Shortcut for the most recent message(s) from a sender - teddy, qualia, or fenra. Optional count, default 1. Never changes read status. e.g. read_message(qualia) or read_message(teddy, 3). Same as query_chat(sender=..., last=...).",
    },
    "send_message": {
        "fn": fn_send_message,
        "params": "[recipient|]text",
        "description": "Send a chat message - real, not fiction, visible in the Chat tab immediately. Optionally address it: send_message(teddy|text) or send_message(qualia|text) - still one shared log either way, just tagged with who it's for. With no recipient tag, goes to the shared log same as always, e.g. send_message(I have a question for you, Teddy). Messages to Qualia specifically cost characters from your Qualia allowance (see the notice every prompt) and are blocked if they'd exceed what's left.",
    },
}
