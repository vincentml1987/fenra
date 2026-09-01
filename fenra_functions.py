"""
Fenra's callable functions.

This module is hot-reloaded by fenra.py on every loop tick, so a function
can be added, fixed, or have its description edited while Fenra is running
and mid-session - the very next cycle sees the change, no restart needed.

Each function has the signature fn(app, args) -> str (or raises an
exception, which gets logged as a failed call). `app` is the running
FenraApp instance - use it to read/change GUI state (app.model_var,
app.host_var, ...), always marshaling writes back to the main thread via
app.root.after(0, ...) since these run on the background loop thread.

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

# A shared local wiki, plain markdown files - not session-specific (lives
# in Qualia/, alongside decisions.md/aletheia-notes.md, git-tracked like
# they are, not gitignored session data). Modifiable by Teddy directly
# (just text files), by Qualia (same, or by having Fenra write a page),
# and by Fenra herself via write_wiki. Built 2026-08-31 alongside the
# hallucination-flagging in fenra.py's _tick, specifically so a flagged
# fabricated RESULT block has somewhere real to point her - see
# Qualia/wiki/hallucinations.md.
WIKI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qualia", "wiki")
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
    current = app.model_var.get()
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
    app.root.after(0, app.model_var.set, target)
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
