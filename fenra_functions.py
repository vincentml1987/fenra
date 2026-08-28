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

from datetime import datetime

import requests

DEFAULT_HOST = "http://localhost:11434"


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
    return app.model_var.get()


def fn_list_models(app, args):
    host = app.host_var.get().strip().rstrip("/") or DEFAULT_HOST
    resp = requests.get(f"{host}/api/tags", timeout=10)
    resp.raise_for_status()
    names = [m["name"] for m in resp.json().get("models", [])]
    return ", ".join(names) if names else "(no models installed)"


def fn_get_desire(app, args):
    text = app.desire_var.get()
    if text:
        return text
    return ("no desire is set. To set one, call set_desire(text), e.g. "
            "set_desire(understand why I keep repeating myself) - it will "
            "persist and be visible to you every cycle until you change it.")


def fn_set_desire(app, args):
    if not args or not args[0]:
        raise ValueError("set_desire requires text, e.g. set_desire(understand why I keep repeating myself)")
    text = args[0]
    app.root.after(0, app.desire_var.set, text)
    return "desire updated"


def _format_chat_message(m):
    who = "Teddy" if m["sender"] == "teddy" else "Fenra"
    return f"[{m['timestamp']}] {who}: {m['text']}"


def _parse_chat_time(s):
    try:
        return datetime.fromisoformat(s.strip())
    except ValueError:
        raise ValueError(
            f"'{s}' isn't a recognizable time. Use the same format now() returns, "
            f"e.g. 2026-08-28T14:30:00"
        )


def fn_read_chat(app, args):
    """Unread messages from Teddy only. Marks them read."""
    unread = [m for m in app.chat_messages if m["sender"] == "teddy" and not m.get("read", True)]
    if not unread:
        return "no unread messages."
    formatted = [_format_chat_message(m) for m in unread]
    for m in unread:
        m["read"] = True
    app.persist_chat()
    return "\n".join(formatted)


def fn_read_chat_since(app, args):
    """All messages (both directions) from a given time onward. Marks any
    matched incoming (Teddy's) messages read."""
    if not args or not args[0]:
        raise ValueError("read_chat_since requires a time, e.g. read_chat_since(2026-08-28T10:00:00)")
    since = _parse_chat_time(args[0])
    matched = [m for m in app.chat_messages if _parse_chat_time(m["timestamp"]) >= since]
    if not matched:
        return f"no messages since {args[0]}."
    formatted = [_format_chat_message(m) for m in matched]
    changed = False
    for m in matched:
        if m["sender"] == "teddy" and not m.get("read", True):
            m["read"] = True
            changed = True
    if changed:
        app.persist_chat()
    return "\n".join(formatted)


def fn_read_chat_between(app, args):
    """All messages (both directions) between two times, separated by |.
    Marks any matched incoming (Teddy's) messages read."""
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
        if m["sender"] == "teddy" and not m.get("read", True):
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


def fn_send_message(app, args):
    if not args or not args[0]:
        raise ValueError("send_message requires text, e.g. send_message(I have a question for you, Teddy)")
    app.root.after(0, app.add_chat_message, "fenra", args[0], True)
    return "message sent"


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
    return f"model switched to {target}, effective next cycle"


# name -> {"fn": callable(app, args), "params": str, "description": str}
# "functions" is the discovery entry point - call it with no arguments for
# the full list, or with a search term to filter by matching text in each
# function's name/description. This exists specifically so Fenra doesn't
# need every function spelled out in her prompt every cycle.
FUNCTION_REGISTRY = {
    "functions": {
        "fn": fn_functions,
        "params": "[search]",
        "description": "List available functions. No argument lists all of them; a search term filters to functions whose name or description contains it, e.g. functions(switch). A function taking more than one argument separates them with | , e.g. read_chat_between(start|end).",
    },
    "now": {
        "fn": fn_now,
        "params": "",
        "description": "Report the current real-world date and time.",
    },
    "get_desire": {
        "fn": fn_get_desire,
        "params": "",
        "description": "Report what you've set as your current desire/intention, if anything.",
    },
    "set_desire": {
        "fn": fn_set_desire,
        "params": "text",
        "description": "Set what you want to pursue right now - free text, visible to Teddy (he can see it but not change it), persisted across cycles, and included in your own prompt between your last thought and the closing instructions. Overwrites any previous desire. e.g. set_desire(understand why I keep repeating myself).",
    },
    "current_model": {
        "fn": fn_current_model,
        "params": "",
        "description": "Report which Ollama model is currently generating your responses.",
    },
    "list_models": {
        "fn": fn_list_models,
        "params": "",
        "description": "List every Ollama model currently installed and available to switch to.",
    },
    "set_model": {
        "fn": fn_set_model,
        "params": "name",
        "description": "Switch which Ollama model generates your responses, effective next cycle. Requires the exact name of an installed model, e.g. set_model(gemma3:4b).",
    },
    "read_chat": {
        "fn": fn_read_chat,
        "params": "",
        "description": "Show only your unread messages from Teddy, and mark them as read.",
    },
    "read_chat_since": {
        "fn": fn_read_chat_since,
        "params": "time",
        "description": "Show every chat message (both directions) from the given time onward, and mark any of Teddy's messages in that range as read. Time format matches now(), e.g. read_chat_since(2026-08-28T10:00:00).",
    },
    "read_chat_between": {
        "fn": fn_read_chat_between,
        "params": "start|end",
        "description": "Show every chat message (both directions) between two times, and mark any of Teddy's messages in that range as read. Two times separated by |, e.g. read_chat_between(2026-08-28T10:00:00|2026-08-28T14:00:00).",
    },
    "search_chat": {
        "fn": fn_search_chat,
        "params": "query[|chars]",
        "description": "Search the whole chat for a word or phrase and return the surrounding text, without changing any read status. Optionally give how many characters of context on each side (default 200), separated by |, e.g. search_chat(truth|150).",
    },
    "send_message": {
        "fn": fn_send_message,
        "params": "text",
        "description": "Send Teddy a chat message - real, not fiction, visible to him in the Chat tab immediately. e.g. send_message(I have a question for you, Teddy).",
    },
}
