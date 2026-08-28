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
        "description": "List available functions. No argument lists all of them; a search term filters to functions whose name or description contains it, e.g. functions(switch).",
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
}
