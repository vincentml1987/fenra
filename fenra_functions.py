from __future__ import annotations

from typing import Callable, Dict, Any, List
import ast
import subprocess
import sys
import shutil
import json
import os
from copy import deepcopy
from pathlib import Path
from dataclasses import asdict

from config_loader import get_path
from diary_tags import (
    DiaryPaths,
    add_file_tags,
    get_file_tags,
    list_tree as diary_list_tree,
    mkdir as diary_mkdir,
    move as diary_move,
    reindex as diary_reindex,
    remove as diary_remove,
    remove_file_tags,
    search_by_tags as diary_search_by_tags,
    set_file_tags,
)
from directed_memory import get_store
from fenra import awareness
from fenra.awareness import AWARENESS_KEYS


# ---------------------------
# Function registry and API
# ---------------------------

_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _get_param(
    args: list[Any],
    kwargs: dict[str, Any],
    name: str,
    *,
    default: Any = None,
    cast: Callable[[Any], Any] | None = None,
    coerce_underscores: bool = False,
):
    if name in kwargs:
        val = kwargs.pop(name)
    elif args:
        val = args.pop(0)
    else:
        val = default

    if coerce_underscores and isinstance(val, str):
        val = val.replace("_", " ")

    if cast is not None and val is not None:
        try:
            val = cast(val)
        except Exception:
            raise ValueError(f"expected {name} to be {cast.__name__}")

    return val


def register(name: str, description: str):
    """Decorator to register a callable Fenra function with a human-readable description."""

    def _wrap(func: Callable[..., Any]):
        entry = _REGISTRY.get(name, {})
        entry.update({"func": func, "description": description})
        entry.setdefault("forms", [])
        _REGISTRY[name] = entry
        return func

    return _wrap


def register_details(name: str, forms: List[Dict[str, str]]) -> None:
    """Register rich metadata for a Fenra function."""

    normalized: list[dict[str, str]] = []
    for form in forms or []:
        normalized.append(
            {
                "parameters": str(form.get("parameters", "")),
                "usage": str(form.get("usage", "")),
                "returns": str(form.get("returns", "")),
            }
        )

    entry = _REGISTRY.setdefault(name, {"forms": []})
    entry.setdefault("description", "")
    entry["forms"] = normalized


def _literalize(node: ast.AST) -> Any:
    """Safely convert AST nodes for args/kwargs via literal evaluation."""
    return ast.literal_eval(node)


def _parse_call(expr: str) -> tuple[str, list[Any], dict[str, Any]]:
    """
    Parse 'fn_name(arg, kw=val, ...)' or dotted 'ns.fn(arg, ...)' into (name, args, kwargs).
    Raises ValueError on any invalid or unsafe expression.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid expression: {e}") from e

    if not isinstance(tree.body, ast.Call):
        raise ValueError("expression is not a function call")

    def _attr_to_name(n: ast.AST) -> str:
        if isinstance(n, ast.Name):
            return n.id
        if isinstance(n, ast.Attribute):
            return _attr_to_name(n.value) + "." + n.attr
        raise ValueError("expression is not a simple or dotted function name")

    fn_name = _attr_to_name(tree.body.func)
    args = [_literalize(a) for a in tree.body.args]

    kwargs: dict[str, Any] = {}
    for kw in tree.body.keywords:
        if kw.arg is None:
            raise ValueError("keyword unpacking (**kwargs) is not allowed")
        kwargs[kw.arg] = _literalize(kw.value)

    return fn_name, args, kwargs


# --- New file I/O constants & helpers ---

DOC_DIR = "fenra_documentation"
DIARY_DIR = "fenra_diary"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _diary_paths() -> DiaryPaths:
    diary_default = Path.home() / DIARY_DIR
    diary_root = get_path("paths.diary", default=diary_default)
    if diary_root is None:
        diary_root = diary_default
    diary_root = Path(diary_root).expanduser()
    diary_root.mkdir(parents=True, exist_ok=True)

    documentation_root = get_path("paths.documentation", default=None)
    if documentation_root is not None:
        documentation_root = Path(documentation_root).expanduser()

    return DiaryPaths(diary_root=diary_root, documentation_root=documentation_root)


def _sanitize_filename(name: str) -> str:
    base = os.path.basename(str(name or "")).strip()
    if base in ("", ".", ".."): 
        raise ValueError("invalid filename")
    return base


def _safe_path(base_dir: str, name: str) -> str:
    return os.path.join(base_dir, _sanitize_filename(name))


def _list_files(folder: str) -> str:
    _ensure_dir(folder)
    files = [
        f for f in sorted(os.listdir(folder))
        if os.path.isfile(os.path.join(folder, f))
    ]
    return "\n".join(files) if files else "(no files)"


def dispatch_expression(expr: str) -> tuple[str, bool, str, str]:
    """
    Dispatch a Fenra function expression found inside *~...~*.
    Returns (function_name_or_guess, found, result_string).
    - found=False when the name is not registered -> 'Function does not exist.'
    - found=True for executed or error-returning calls (errors are returned as strings).
    """
    guessed_name = expr.strip()
    params_string = ""
    try:
        name, args, kwargs = _parse_call(expr)
        guessed_name = name
        call_str = expr.strip()
        if "(" in call_str and call_str.endswith(")"):
            params_string = call_str[call_str.find("(") + 1 : call_str.rfind(")")]
    except Exception:
        name_display = guessed_name or "<unknown>"
        return (
            guessed_name,
            False,
            f"Function {name_display} does not exist.",
            "",
        )

    entry = _REGISTRY.get(name)
    if not entry:
        name_display = name or "<unknown>"
        return (
            name,
            False,
            f"Function {name_display} does not exist.",
            params_string,
        )

    func = entry.get("func")
    if func is None:
        name_display = name or "<unknown>"
        return (
            name,
            False,
            f"Function {name_display} does not exist.",
            params_string,
        )
    try:
        res = func(*args, **kwargs)
        result = "(No Output)" if res is None else str(res)
        return (name, True, result, params_string)
    except Exception as e:
        return (name, True, f"(error) {type(e).__name__}: {e}", params_string)


# --------------------------------
# Built-in/required Fenra functions
# --------------------------------


def _sync_awareness_state_to_conductor() -> None:
    import importlib

    try:
        conductor = importlib.import_module("conductor")
    except Exception:
        return

    try:
        conductor.STATE.setdefault("awareness", {})
        conductor.STATE["awareness"] = dict(awareness.get_awareness())
    except Exception:
        pass


@register("awareness.list", "List awareness-controlled text inputs and their status.")
def awareness_list(*args, **kwargs) -> str:
    args = list(args)
    kwargs = dict(kwargs)
    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    state = awareness.get_awareness()
    lines = [f"{name}: {'on' if state.get(name) else 'off'}" for name in AWARENESS_KEYS]

    _sync_awareness_state_to_conductor()

    return "\n".join(lines)


@register("awareness.notice", "Enable a specific awareness-controlled text input.")
def awareness_notice(*args, **kwargs) -> str:
    args = list(args)
    kwargs = dict(kwargs)
    try:
        item = _get_param(args, kwargs, "item", default=None, cast=str)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    name = (item or "").strip()
    if name not in AWARENESS_KEYS:
        return f"No such awareness item: {name or '<empty>'}."

    awareness.set_key(name, True)
    _sync_awareness_state_to_conductor()

    return f"Noticed {name}."


@register("awareness.ignore", "Disable a specific awareness-controlled text input.")
def awareness_ignore(*args, **kwargs) -> str:
    args = list(args)
    kwargs = dict(kwargs)
    try:
        item = _get_param(args, kwargs, "item", default=None, cast=str)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    name = (item or "").strip()
    if name not in AWARENESS_KEYS:
        return f"No such awareness item: {name or '<empty>'}."

    awareness.set_key(name, False)
    _sync_awareness_state_to_conductor()

    return f"Ignored {name}."


@register("awareness.awake", "Enable all awareness-controlled text inputs.")
def awareness_awake(*args, **kwargs) -> str:
    args = list(args)
    kwargs = dict(kwargs)

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    current = awareness.get_awareness()
    already_enabled = all(bool(current.get(name)) for name in AWARENESS_KEYS)

    new_state = {name: True for name in AWARENESS_KEYS}
    awareness.set_awareness(new_state)
    _sync_awareness_state_to_conductor()

    if already_enabled:
        return "Awareness inputs were already fully enabled."
    return "Enabled all awareness inputs."


@register("awareness.peek", "Return the text Fenra would supply for the requested awareness input.")
def awareness_peek(*args, **kwargs) -> str:
    import importlib

    args = list(args)
    kwargs = dict(kwargs)
    try:
        item = _get_param(args, kwargs, "item", default=None, cast=str)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    name = (item or "").strip()
    if name not in AWARENESS_KEYS:
        return f"No text for {name}."

    try:
        conductor = importlib.import_module("conductor")
        if not getattr(conductor, "_CONFIGS_LOADED", False) and hasattr(conductor, "ensure_configs_loaded"):
            conductor.ensure_configs_loaded()
    except Exception as exc:
        return f"(error) {type(exc).__name__}: {exc}"

    agent_name = (getattr(conductor, "STATE", {}) or {}).get("current_agent")
    if not isinstance(agent_name, str) or agent_name not in getattr(conductor, "AGENTS_BY_NAME", {}):
        return f"No text for {name}."

    agent = conductor.AGENTS_BY_NAME[agent_name]
    try:
        text = conductor.resolve_awareness_text(agent, name)
    except Exception as exc:
        return f"(error) {type(exc).__name__}: {exc}"

    if not text:
        return f"No text for {name}."

    return text


@register("list_functions", "List available Fenra functions.")
def list_functions(*args, **kwargs) -> str:
    """Return the structured catalog of Fenra functions, optionally filtered by search."""

    args = list(args)
    kwargs = dict(kwargs)
    try:
        search = _get_param(args, kwargs, "search", default="", cast=str)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    entry = _REGISTRY.setdefault(
        "list_functions",
        {"func": list_functions, "description": "List available Fenra functions.", "forms": []},
    )
    if not entry.get("forms"):
        register_details(
            "list_functions",
            [
                {
                    "parameters": 'search: str=""',
                    "usage": "Return the catalog of available Fenra functions or the subset matching the search string. Arguments may be provided positionally or as keywords (e.g., list_functions(\"agents\") or list_functions(search=\"agents\")).",
                    "returns": "A formatted list of functions, their usage forms, and return values.",
                }
            ],
        )

    def _format_entry(name: str, info: Dict[str, Any]) -> str:
        forms = info.get("forms") or []
        if not forms:
            forms = [
                {
                    "parameters": "",
                    "usage": info.get("description", ""),
                    "returns": "(No Output)",
                }
            ]

        lines: list[str] = [name, "Usage:"]
        for idx, form in enumerate(forms):
            params = form.get("parameters", "")
            params_display = params if params else ""
            call_line = f"\t{name}({params_display})" if params_display else f"\t{name}()"
            lines.append(f"{call_line}:")
            lines.append(f"\t\t{form.get('usage', '')}")
            lines.append("")
            returns_text = form.get("returns", "") or "(No Output)"
            lines.append(f"\t\tReturns: {returns_text}")
            if idx != len(forms) - 1:
                lines.append("")
        return "\n".join(lines)

    search_text = (search or "").strip().lower()
    formatted: list[str] = []
    for name in sorted(_REGISTRY):
        info = _REGISTRY[name]
        if search_text:
            haystacks = [name.lower(), str(info.get("description", "")).lower()]
            for form in info.get("forms") or []:
                haystacks.extend(
                    [
                        str(form.get("parameters", "")).lower(),
                        str(form.get("usage", "")).lower(),
                        str(form.get("returns", "")).lower(),
                    ]
                )
            if not any(search_text in hay for hay in haystacks):
                continue
        formatted.append(_format_entry(name, info))

    if not formatted:
        return "(no matching functions)"

    body = "\n\n".join(formatted)
    return f"You have access to the following functions:\n\n{body}"


@register("announce_self", "Announce the name of the current agent.")
def announce_self(*args, **kwargs) -> str:
    import importlib

    args = list(args)
    kwargs = dict(kwargs)
    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    conductor = importlib.import_module("conductor")
    name = (getattr(conductor, "STATE", {}) or {}).get("current_agent")
    if not isinstance(name, str) or not name:
        name = "unknown"
    return f"The agent who ran this function is named '{name}'"


register_details(
    "announce_self",
    [
        {
            "parameters": "",
            "usage": "Return the name of the agent currently executing this function. Arguments may be provided positionally or as keywords—this function takes none.",
            "returns": "The calling agent's name as text.",
        }
    ],
)


@register(
    "get_discord_messages",
    "Get Discord messages. Forms: get_discord_messages(); get_discord_messages(n); get_discord_messages(m, n). "
    "Zero-arg returns the latest message. One-arg returns the last n messages (newest→older). "
    "Two-arg skips the last m messages, then returns the next n messages going backward from the end.",
)
def get_discord_messages(*args, **kwargs) -> str:
    import importlib

    args = list(args)
    kwargs = dict(kwargs)
    original_args_count = len(args)
    provided_kwargs = set(kwargs.keys())

    try:
        fe = importlib.import_module("fenra_ui")
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"

    ensure = getattr(fe, "ensure_discord_running", None)
    if callable(ensure) and not ensure():
        return "(error) Discord not configured or unavailable"

    def _fmt(items):
        ordered = list(items or [])
        ordered.reverse()  # convert from newest→oldest to oldest→newest
        lines = []
        for it in ordered:
            author = (it.get("author") or it.get("sender") or "user") or "user"
            text = (it.get("text") or it.get("message") or "").strip()
            timestamp = (it.get("timestamp") or it.get("time") or "").strip()
            if not text and not timestamp:
                continue
            prefix = timestamp if timestamp else "(unknown time)"
            if text:
                lines.append(f"[{prefix}] {author}: {text}")
            else:
                lines.append(f"[{prefix}] {author}")
        return "\n".join(lines) if lines else "(no messages)"

    if not args and not kwargs:
        try:
            items = fe.fetch_recent_discord_messages(1) or []
            body = _fmt(items[:1])
            return f"The following is the most recent Discord message:\n{body}"
        except Exception as e:
            return f"(error) {type(e).__name__}: {e}"

    try:
        m = _get_param(args, kwargs, "m", default=None, cast=int)
        n = _get_param(args, kwargs, "n", default=None, cast=int)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if original_args_count == 1 and "m" not in provided_kwargs and "n" not in provided_kwargs:
        n = m
        m = None

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    if n is None:
        return "(error) ValueError: expected n to be int"

    m = 0 if m is None else m

    try:
        n = max(1, int(n))
        m = max(0, int(m))
    except Exception:
        return "(error) ValueError: expected integers 'm, n'"

    try:
        items = fe.fetch_recent_discord_messages(m + n) or []
        if m == 0:
            body = _fmt(items[:n])
            return (
                "The following are the {n} most recent Discord messages (oldest to newest):\n".format(n=n)
                + body
            )

        sliced = items[m : m + n]
        body = _fmt(sliced)
        return (
            "The following are the {n} requested messages after skipping the {m} most recent (oldest to newest):\n".format(
                n=n, m=m
            )
            + body
        )
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"


register_details(
    "get_discord_messages",
    [
        {
            "parameters": "",
            "usage": "Return the most recent Discord message. Arguments may be provided positionally or as keywords—this function takes none.",
            "returns": "The latest Discord message text or an error description.",
        },
        {
            "parameters": "n: int",
            "usage": "Return the last n Discord messages, displayed from oldest to newest. Accepts positional or keyword arguments (e.g., get_discord_messages(5) or get_discord_messages(n=5)).",
            "returns": "A newline-separated list of Discord messages or an error description.",
        },
        {
            "parameters": "m: int, n: int",
            "usage": "Skip the most recent m Discord messages, then return the next n messages, displayed from oldest to newest. Accepts positional or keyword arguments (e.g., get_discord_messages(2, 5) or get_discord_messages(m=2, n=5)).",
            "returns": "A newline-separated list of Discord messages or an error description.",
        },
    ],
)


@register("fenra_powershell", "Execute a PowerShell command string and return its output.")
def fenra_powershell(*args, **kwargs) -> str:
    """
    Execute a PowerShell command and return its output as text.
    This is the ONLY path for running PowerShell from agent replies.
    """

    args = list(args)
    kwargs = dict(kwargs)
    try:
        command = _get_param(args, kwargs, "command", default=None, cast=str)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    if not isinstance(command, str) or not command.strip():
        return "(no command)"

    exe = shutil.which("pwsh") or shutil.which("powershell")
    if exe is None:
        exe = "powershell.exe" if sys.platform.startswith("win") else "pwsh"

    args = [exe, "-NoProfile", "-NonInteractive", "-Command", command]
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = proc.stdout or ""
        err = proc.stderr or ""
        combined = out if not err else (out + ("\n" if out and err else "") + err)
        return combined.strip()
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"


register_details(
    "fenra_powershell",
    [
        {
            "parameters": "command: str",
            "usage": "Execute the provided PowerShell command string and capture its output. Arguments may be provided positionally or as keywords (e.g., fenra_powershell(\"Get-Date\") or fenra_powershell(command=\"Get-Date\")).",
            "returns": "The combined stdout/stderr from PowerShell or an error message. Returns (No Output) when the command produced no text.",
        }
    ],
)


@register("duplicate_self", "Duplicate the current agent as <name>-dup and save/refresh.")
def duplicate_self(*args, **kwargs) -> str:

    import importlib
    from config_loader import save_agents
    conductor = importlib.import_module("conductor")

    args = list(args)
    kwargs = dict(kwargs)
    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"
    
    """
    Create a duplicate of the calling agent with the name '<original>-dup'
    (or '<original>-dup2', etc., if needed), append it to confs/agents.json,
    update Conductor's in-memory indexes, and refresh the UI like the GUI
    would after a save.
    """
    # Resolve the caller
    cur_name = (conductor.STATE or {}).get("current_agent")
    if not isinstance(cur_name, str) or cur_name not in conductor.AGENTS_BY_NAME:
        return "(error) RuntimeError: duplicate_self: no current agent is set or it cannot be found"

    source = conductor.AGENTS_BY_NAME[cur_name]

    # Deep copy and rename
    dup = json.loads(json.dumps(source))  # safe deep copy of plain dict
    base = source.get("name") or "Agent"
    proposed = f"{base}-dup"

    # Ensure uniqueness (Agent1-dup, Agent1-dup2, ...)
    existing = {a.get("name") for a in conductor.AGENTS}
    name = proposed
    ctr = 2
    while name in existing:
        name = f"{proposed}{ctr}"
        ctr += 1
    dup["name"] = name

    # Normalize required list fields (save_agents will also setdefault, but keep tidy)
    dup.setdefault("groups_in", list(source.get("groups_in", []) or []))
    dup.setdefault("groups_out", list(source.get("groups_out", []) or []))

    # Append and persist
    conductor.AGENTS.append(dup)

    # Rebuild in-memory indexes to mirror Conductor’s normal load path
    conductor.AGENTS_BY_NAME = {a["name"]: a for a in conductor.AGENTS}
    grp_map = {}
    for a in conductor.AGENTS:
        for g in a.get("groups_in", []):
            grp_map.setdefault(g, set()).add(a["name"])
    conductor.AGENTS_BY_GROUP_IN = grp_map

    # Save to disk (UI file watcher will rebuild relevant tabs)
    save_agents(conductor.AGENTS)

    # Best-effort UI refresh (optional; watcher will handle this too)
    ui = getattr(conductor, "UI", None)
    try:
        if ui is not None and hasattr(ui, "_threadsafe"):
            # Re-read agents for the UI model and refresh panels
            ui._threadsafe(lambda: setattr(ui, "_agents_model", ui._ui_agents()))
            if hasattr(ui, "_update_required_configs_state"):
                ui._threadsafe(ui._update_required_configs_state)
            if hasattr(ui, "_refresh_agent_listbox"):
                ui._threadsafe(ui._refresh_agent_listbox)
            if hasattr(ui, "_build_simple_groups_tab"):
                ui._threadsafe(ui._build_simple_groups_tab)
    except Exception:
        # Non-fatal; the filesystem watcher will still update the UI
        pass

    return f"Duplicated {base} → {name}"


register_details(
    "duplicate_self",
    [
        {
            "parameters": "",
            "usage": "Clone the calling agent with a '-dup' suffix (adding a number if needed) and refresh runtime state. Arguments may be provided positionally or as keywords—this function takes none.",
            "returns": "Confirmation of the original and duplicated agent names.",
        }
    ],
)


@register(
    "rename_agent",
    "Rename an agent. Usage: rename_agent(\"New_Name\") to rename the current agent, "
    "or rename_agent(\"Old_Name\",\"New_Name\") to rename a specific agent. "
    "Underscores in names are converted to spaces."
)
def rename_agent(*args, **kwargs) -> str:
    """
    Rename the calling agent (1 arg) or rename the agent with Old_Name to New_Name (2 args).

    IMPORTANT: Call spans only forbid whitespace touching the *~ or ~* markers,
    so names may include spaces directly. Underscores are still accepted and are
    converted back to spaces for convenience.
    """
    import importlib
    import re
    from config_loader import save_agents, save_state  # save_state is available in config_loader

    conductor = importlib.import_module("conductor")

    args = list(args)
    kwargs = dict(kwargs)
    provided_kwargs = set(kwargs.keys())
    original_args_count = len(args)

    # Ensure configs/structures are loaded so AGENTS/AGENTS_BY_NAME/STATE exist
    if not getattr(conductor, "_CONFIGS_LOADED", False) and hasattr(conductor, "ensure_configs_loaded"):
        try:
            conductor.ensure_configs_loaded()
        except Exception:
            pass

    try:
        old_name = _get_param(args, kwargs, "old_name", default=None, coerce_underscores=True)
        new_name = _get_param(args, kwargs, "new_name", default=None, coerce_underscores=True)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if original_args_count == 1 and "old_name" not in provided_kwargs and "new_name" not in provided_kwargs:
        # Single positional argument -> new name for current agent
        new_name = old_name
        old_name = None

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    def _norm(s: str | None) -> str:
        s = str(s or "")
        s = s.replace("_", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    new_name = _norm(new_name)
    if not new_name:
        return "(error) ValueError: rename_agent: new name cannot be empty"

    if old_name is None:
        cur_name = (getattr(conductor, "STATE", {}) or {}).get("current_agent")
        if not isinstance(cur_name, str) or cur_name not in conductor.AGENTS_BY_NAME:
            return "(error) RuntimeError: rename_agent: current agent is not set or cannot be found"
        old_name = cur_name
    else:
        old_name = _norm(old_name)

    if old_name not in conductor.AGENTS_BY_NAME:
        return f"(error) KeyError: rename_agent: '{old_name}' not found"
    if new_name in conductor.AGENTS_BY_NAME:
        return f"(error) ValueError: rename_agent: name '{new_name}' already exists"

    # Update the agent object in-place
    agent = conductor.AGENTS_BY_NAME[old_name]
    agent["name"] = new_name

    # Rebuild indexes to mirror Conductor’s normal load path
    conductor.AGENTS_BY_NAME = {a["name"]: a for a in conductor.AGENTS}
    grp_map: dict[str, set[str]] = {}
    for a in conductor.AGENTS:
        for g in a.get("groups_in", []) or []:
            grp_map.setdefault(g, set()).add(a["name"])
    conductor.AGENTS_BY_GROUP_IN = grp_map

    # Update STATE.current_agent if needed
    try:
        if (conductor.STATE or {}).get("current_agent") == old_name:
            conductor.STATE["current_agent"] = new_name
            save_state(conductor.STATE)
    except Exception:
        pass

    # Persist agents to disk; the UI watcher will refresh
    save_agents(conductor.AGENTS)

    # Best-effort UI refresh (optional)
    ui = getattr(conductor, "UI", None)
    try:
        if ui is not None and hasattr(ui, "_threadsafe"):
            ui._threadsafe(lambda: setattr(ui, "_agents_model", ui._ui_agents()))
            if hasattr(ui, "_refresh_agent_listbox"):
                ui._threadsafe(ui._refresh_agent_listbox)
            if hasattr(ui, "_build_simple_groups_tab"):
                ui._threadsafe(ui._build_simple_groups_tab)
    except Exception:
        # Non-fatal; the filesystem watcher will still update the UI
        pass

    return f"Renamed {old_name} → {new_name}"


register_details(
    "rename_agent",
    [
        {
            "parameters": "new_name: str",
            "usage": "Rename the current agent to the provided name. Arguments may be provided positionally or as keywords (e.g., rename_agent(\"New Name\") or rename_agent(new_name=\"New Name\")).",
            "returns": "Confirmation of the old and new agent names or an error message.",
        },
        {
            "parameters": "old_name: str, new_name: str",
            "usage": "Rename the specified agent to the provided name. Arguments may be provided positionally or as keywords (e.g., rename_agent(\"Old\", \"New\") or rename_agent(old_name=\"Old\", new_name=\"New\")).",
            "returns": "Confirmation of the old and new agent names or an error message.",
        },
    ],
)


@register("read_documentation", "Read or list files in fenra_documentation.")
def read_documentation(*args, **kwargs) -> str:
    args = list(args)
    kwargs = dict(kwargs)
    try:
        filename = _get_param(args, kwargs, "filename", default=None, cast=str)
        m = _get_param(args, kwargs, "m", default=None, cast=int)
        n = _get_param(args, kwargs, "n", default=None, cast=int)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    if filename is None:
        files = _list_files(DOC_DIR)
        return "The following documentation is available:\n" + files

    path = _safe_path(DOC_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"

    if m is None and n is None:
        return f"Contents of {filename}:\n\n{data}"

    if (m is None) != (n is None):
        return "(error) ValueError: both m and n must be provided when slicing"
    if m < 0:
        return "(error) ValueError: m must be >= 0"
    if n > len(data):
        return f"(error) ValueError: n must be <= {len(data)}"
    if m > n:
        return "(error) ValueError: m must be <= n"

    return f"Contents of {filename} from character {m} to character {n}:\n\n{data[m:n]}"


register_details(
    "read_documentation",
    [
        {
            "parameters": "",
            "usage": "List all files in fenra_documentation.",
            "returns": "A newline-separated list of files, or (no files).",
        },
        {
            "parameters": "filename: str",
            "usage": "Return the full text of the named file in fenra_documentation.",
            "returns": "File contents as text.",
        },
        {
            "parameters": "filename: str, m: int, n: int",
            "usage": "Return file text from m (inclusive) to n (exclusive). Enforces m>=0, n<=len(file), and m<=n.",
            "returns": "The requested slice of the file.",
        },
    ],
)


@register("read_diary", "Read or list files in fenra_diary.")
def read_diary(*args, **kwargs) -> str:
    args = list(args)
    kwargs = dict(kwargs)
    try:
        filename = _get_param(args, kwargs, "filename", default=None, cast=str)
        m = _get_param(args, kwargs, "m", default=None, cast=int)
        n = _get_param(args, kwargs, "n", default=None, cast=int)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    if filename is None:
        return _list_files(DIARY_DIR)

    path = _safe_path(DIARY_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"

    if m is None and n is None:
        return f"Contents of {filename}:\n\n{data}"

    if (m is None) != (n is None):
        return "(error) ValueError: both m and n must be provided when slicing"
    if m < 0:
        return "(error) ValueError: m must be >= 0"
    if n > len(data):
        return f"(error) ValueError: n must be <= {len(data)}"
    if m > n:
        return "(error) ValueError: m must be <= n"

    return f"Contents of {filename} from character {m} to character {n}:\n\n{data[m:n]}"


register_details(
    "read_diary",
    [
        {
            "parameters": "",
            "usage": "List all files in fenra_diary.",
            "returns": "A newline-separated list of files, or (no files).",
        },
        {
            "parameters": "filename: str",
            "usage": "Return the full text of the named file in fenra_diary.",
            "returns": "File contents as text.",
        },
        {
            "parameters": "filename: str, m: int, n: int",
            "usage": "Return file text from m (inclusive) to n (exclusive). Enforces m>=0, n<=len(file), and m<=n.",
            "returns": "The requested slice of the file.",
        },
    ],
)


@register(
    "write_diary",
    "Append, overwrite, or insert text into a diary file in fenra_diary."
)
def write_diary(*args, **kwargs) -> str:
    args = list(args)
    kwargs = dict(kwargs)
    try:
        filename = _get_param(args, kwargs, "filename", cast=str)
        text = _get_param(args, kwargs, "textToWrite", cast=str)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    # Disambiguate third parameter (positional or keyword).
    third = None
    if args:
        third = args.pop(0)  # could be bool overwrite or int writeLocation

    # If provided as keywords, accept only one of them.
    if "overwrite" in kwargs and "writeLocation" in kwargs:
        return "(error) ValueError: provide either 'overwrite' or 'writeLocation', not both"
    if "overwrite" in kwargs:
        third = kwargs.pop("overwrite")
    if "writeLocation" in kwargs:
        third = kwargs.pop("writeLocation")

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    # Required params
    if not filename or text is None:
        return "(error) ValueError: filename and textToWrite are required"

    _ensure_dir(DIARY_DIR)
    path = _safe_path(DIARY_DIR, filename)

    # Case A: overwrite flag (bool)
    if isinstance(third, bool):
        try:
            if third:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                return f"Overwrote {filename} ({len(text)} chars)."
            else:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(text)
                # Report new size in characters
                with open(path, "r", encoding="utf-8") as f:
                    new_len = len(f.read())
                return f"Appended {len(text)} chars to {filename} (now {new_len} chars)."
        except Exception as e:
            return f"(error) {type(e).__name__}: {e}"

    # Case B: writeLocation (int)
    if isinstance(third, int):
        try:
            existing = ""
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    existing = f.read()
            pos = int(third)
            if pos < 0 or pos > len(existing):
                return f"(error) ValueError: writeLocation must be between 0 and {len(existing)}"
            updated = existing[:pos] + text + existing[pos:]
            with open(path, "w", encoding="utf-8") as f:
                f.write(updated)
            return f"Wrote {len(text)} chars at position {pos} in {filename} (now {len(updated)} chars)."
        except Exception as e:
            return f"(error) {type(e).__name__}: {e}"

    # Default: append
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
        with open(path, "r", encoding="utf-8") as f:
            new_len = len(f.read())
        return f"Appended {len(text)} chars to {filename} (now {new_len} chars)."
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"


register_details(
    "write_diary",
    [
        {
            "parameters": "filename: str, textToWrite: str",
            "usage": "Append textToWrite to fenra_diary/filename.",
            "returns": "Append confirmation with new file length.",
        },
        {
            "parameters": "filename: str, textToWrite: str, overwrite: bool",
            "usage": "If overwrite=True, replace file with textToWrite; if False, append.",
            "returns": "Overwrite/append confirmation with file length.",
        },
        {
            "parameters": "filename: str, textToWrite: str, writeLocation: int",
            "usage": "Insert textToWrite at the given character offset (0..len(file)).",
            "returns": "Insert confirmation with new file length.",
        },
    ],
)


@register("diary.list_tree", "List directories and files within the diary root.")
def diary_fn_list_tree(*args, **kwargs) -> str:
    args = list(args)
    kwargs = dict(kwargs)
    try:
        rel_dir = _get_param(args, kwargs, "rel_dir", default=".", cast=str)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    try:
        listing = diary_list_tree(_diary_paths(), rel_dir)
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"

    return json.dumps(listing)


@register("diary.mkdir", "Create a directory inside the diary root.")
def diary_fn_mkdir(*args, **kwargs) -> str:
    args = list(args)
    kwargs = dict(kwargs)
    try:
        rel_dir = _get_param(args, kwargs, "rel_dir", cast=str)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if rel_dir in ("", ".", ".."):
        return "(error) ValueError: invalid directory name"

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    try:
        created = diary_mkdir(_diary_paths(), rel_dir)
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"

    return json.dumps({"path": created})


@register("diary.move", "Move or rename diary files and directories.")
def diary_fn_move(*args, **kwargs) -> str:
    args = list(args)
    kwargs = dict(kwargs)
    try:
        rel_src = _get_param(args, kwargs, "rel_src", cast=str)
        rel_dst = _get_param(args, kwargs, "rel_dst", cast=str)
        overwrite = _get_param(args, kwargs, "overwrite", default=False)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if not isinstance(overwrite, bool):
        return "(error) ValueError: overwrite must be a boolean"

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    try:
        moved = diary_move(_diary_paths(), rel_src, rel_dst, overwrite=overwrite)
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"

    return json.dumps({"path": moved})


@register("diary.remove", "Delete a diary file or directory.")
def diary_fn_remove(*args, **kwargs) -> str:
    args = list(args)
    kwargs = dict(kwargs)
    try:
        rel_path = _get_param(args, kwargs, "rel_path", cast=str)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    try:
        removed = diary_remove(_diary_paths(), rel_path)
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"

    return json.dumps({"removed": removed})


def _validate_tag_list(name: str, value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{name} must be a list of strings")
        result.append(item)
    return result


@register("diary.tags.get", "Get tags for a diary file.")
def diary_tags_get(*args, **kwargs) -> str:
    args = list(args)
    kwargs = dict(kwargs)
    try:
        rel_file = _get_param(args, kwargs, "rel_file", cast=str)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    try:
        tags = get_file_tags(_diary_paths(), rel_file)
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"

    return json.dumps(tags)


@register("diary.tags.add", "Add tags to a diary file.")
def diary_tags_add(*args, **kwargs) -> str:
    args = list(args)
    kwargs = dict(kwargs)
    try:
        rel_file = _get_param(args, kwargs, "rel_file", cast=str)
        tags_raw = _get_param(args, kwargs, "tags")
        tags = _validate_tag_list("tags", tags_raw)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    try:
        updated = add_file_tags(_diary_paths(), rel_file, tags)
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"

    return json.dumps(updated)


@register("diary.tags.remove", "Remove tags from a diary file.")
def diary_tags_remove(*args, **kwargs) -> str:
    args = list(args)
    kwargs = dict(kwargs)
    try:
        rel_file = _get_param(args, kwargs, "rel_file", cast=str)
        tags_raw = _get_param(args, kwargs, "tags")
        tags = _validate_tag_list("tags", tags_raw)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    try:
        updated = remove_file_tags(_diary_paths(), rel_file, tags)
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"

    return json.dumps(updated)


@register("diary.tags.set", "Replace tags for a diary file.")
def diary_tags_set(*args, **kwargs) -> str:
    args = list(args)
    kwargs = dict(kwargs)
    try:
        rel_file = _get_param(args, kwargs, "rel_file", cast=str)
        tags_raw = _get_param(args, kwargs, "tags")
        tags = _validate_tag_list("tags", tags_raw)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    try:
        updated = set_file_tags(_diary_paths(), rel_file, tags)
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"

    return json.dumps(updated)


@register("diary.tags.search", "Search diary files by tags.")
def diary_tags_search(*args, **kwargs) -> str:
    args = list(args)
    kwargs = dict(kwargs)
    try:
        include_raw = _get_param(args, kwargs, "include", default=[])
        exclude_raw = _get_param(args, kwargs, "exclude", default=[])
        include = _validate_tag_list("include", include_raw)
        exclude = _validate_tag_list("exclude", exclude_raw)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    try:
        matches = diary_search_by_tags(_diary_paths(), include, exclude)
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"

    return json.dumps(matches)


@register("diary.tags.reindex", "Rebuild the diary tag index from sidecars.")
def diary_tags_reindex(*args, **kwargs) -> str:
    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    try:
        count = diary_reindex(_diary_paths())
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"

    return json.dumps({"indexed": count})


@register("directed_memory.add", "Add a directed memory (defaults: isGlobal=True, empty agent lists).")
def directed_memory_add(*args, **kwargs) -> str:
    args = list(args); kwargs = dict(kwargs)
    try:
        memory_text = _get_param(args, kwargs, "memory_text", cast=str, coerce_underscores=True)
    except ValueError as e:
        return f"(error) ValueError: {e}"
    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs)); return f"(error) ValueError: unexpected keyword '{key}'"
    try:
        store = get_store()
        mem = store.add(memory_text)
        return json.dumps(asdict(mem), ensure_ascii=False)
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"


@register("directed_memory.delete", "Delete a directed memory by index.")
def directed_memory_delete(*args, **kwargs) -> str:
    args = list(args); kwargs = dict(kwargs)
    try:
        index = _get_param(args, kwargs, "index", cast=int)
    except ValueError as e:
        return f"(error) ValueError: {e}"
    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs)); return f"(error) ValueError: unexpected keyword '{key}'"
    try:
        store = get_store()
        store.delete(index)
        return json.dumps({"deleted": index})
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"


@register("directed_memory.update_text", "Update the memoryText of a directed memory by index.")
def directed_memory_update_text(*args, **kwargs) -> str:
    args = list(args); kwargs = dict(kwargs)
    try:
        index = _get_param(args, kwargs, "index", cast=int)
        memory_text = _get_param(args, kwargs, "memory_text", cast=str, coerce_underscores=True)
    except ValueError as e:
        return f"(error) ValueError: {e}"
    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs)); return f"(error) ValueError: unexpected keyword '{key}'"
    try:
        store = get_store()
        item = store.update_text(index, memory_text)
        return json.dumps(item, ensure_ascii=False)
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"


@register("directed_memory.set_global", "Set isGlobal on a directed memory.")
def directed_memory_set_global(*args, **kwargs) -> str:
    args = list(args); kwargs = dict(kwargs)
    try:
        index = _get_param(args, kwargs, "index", cast=int)
        is_global = _get_param(args, kwargs, "is_global", cast=bool)
    except ValueError as e:
        return f"(error) ValueError: {e}"
    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs)); return f"(error) ValueError: unexpected keyword '{key}'"
    try:
        store = get_store()
        item = store.set_global(index, is_global)
        return json.dumps(item, ensure_ascii=False)
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"


@register("directed_memory.add_agent_class", "Add an agent class to a directed memory.")
def directed_memory_add_agent_class(*args, **kwargs) -> str:
    args = list(args); kwargs = dict(kwargs)
    try:
        index = _get_param(args, kwargs, "index", cast=int)
        agent_class = _get_param(args, kwargs, "agent_class", cast=str, coerce_underscores=True)
    except ValueError as e:
        return f"(error) ValueError: {e}"
    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs)); return f"(error) ValueError: unexpected keyword '{key}'"
    try:
        store = get_store()
        item = store.add_agent_class(index, agent_class)
        return json.dumps(item, ensure_ascii=False)
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"


@register("directed_memory.remove_agent_class", "Remove an agent class from a directed memory.")
def directed_memory_remove_agent_class(*args, **kwargs) -> str:
    args = list(args); kwargs = dict(kwargs)
    try:
        index = _get_param(args, kwargs, "index", cast=int)
        agent_class = _get_param(args, kwargs, "agent_class", cast=str, coerce_underscores=True)
    except ValueError as e:
        return f"(error) ValueError: {e}"
    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs)); return f"(error) ValueError: unexpected keyword '{key}'"
    try:
        store = get_store()
        item = store.remove_agent_class(index, agent_class)
        return json.dumps(item, ensure_ascii=False)
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"


@register("directed_memory.add_agent", "Add an agent name to a directed memory.")
def directed_memory_add_agent(*args, **kwargs) -> str:
    args = list(args); kwargs = dict(kwargs)
    try:
        index = _get_param(args, kwargs, "index", cast=int)
        agent = _get_param(args, kwargs, "agent", cast=str, coerce_underscores=True)
    except ValueError as e:
        return f"(error) ValueError: {e}"
    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs)); return f"(error) ValueError: unexpected keyword '{key}'"
    try:
        store = get_store()
        item = store.add_agent(index, agent)
        return json.dumps(item, ensure_ascii=False)
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"


@register("directed_memory.remove_agent", "Remove an agent name from a directed memory.")
def directed_memory_remove_agent(*args, **kwargs) -> str:
    args = list(args); kwargs = dict(kwargs)
    try:
        index = _get_param(args, kwargs, "index", cast=int)
        agent = _get_param(args, kwargs, "agent", cast=str, coerce_underscores=True)
    except ValueError as e:
        return f"(error) ValueError: {e}"
    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs)); return f"(error) ValueError: unexpected keyword '{key}'"
    try:
        store = get_store()
        item = store.remove_agent(index, agent)
        return json.dumps(item, ensure_ascii=False)
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"


@register("directed_memory.list", "List all directed memories.")
def directed_memory_list(*args, **kwargs) -> str:
    args = list(args); kwargs = dict(kwargs)
    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs)); return f"(error) ValueError: unexpected keyword '{key}'"
    try:
        store = get_store()
        return json.dumps(store.list_raw(), ensure_ascii=False)
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"


register_details(
    "directed_memory.add",
    [
        {
            "parameters": "memory_text: str",
            "usage": "Add a directed memory (isGlobal=True, empty targets).",
            "returns": "The created memory as JSON.",
        }
    ],
)
register_details(
    "directed_memory.delete",
    [
        {
            "parameters": "index: int",
            "usage": "Delete a directed memory by index (id).",
            "returns": '{"deleted": <index>}'
        }
    ],
)
register_details(
    "directed_memory.update_text",
    [
        {
            "parameters": "index: int, memory_text: str",
            "usage": "Update memoryText for a directed memory.",
            "returns": "The updated memory as JSON.",
        }
    ],
)
register_details(
    "directed_memory.set_global",
    [
        {
            "parameters": "index: int, is_global: bool",
            "usage": "Set isGlobal for a directed memory.",
            "returns": "The updated memory as JSON.",
        }
    ],
)
register_details(
    "directed_memory.add_agent_class",
    [
        {
            "parameters": "index: int, agent_class: str",
            "usage": "Allow an agent class to receive this memory.",
            "returns": "The updated memory as JSON.",
        }
    ],
)
register_details(
    "directed_memory.remove_agent_class",
    [
        {
            "parameters": "index: int, agent_class: str",
            "usage": "Remove an agent class from this memory.",
            "returns": "The updated memory as JSON.",
        }
    ],
)
register_details(
    "directed_memory.add_agent",
    [
        {
            "parameters": "index: int, agent: str",
            "usage": "Allow a specific agent to receive this memory.",
            "returns": "The updated memory as JSON.",
        }
    ],
)
register_details(
    "directed_memory.remove_agent",
    [
        {
            "parameters": "index: int, agent: str",
            "usage": "Remove a specific agent from this memory.",
            "returns": "The updated memory as JSON.",
        }
    ],
)
register_details(
    "directed_memory.list",
    [
        {
            "parameters": "",
            "usage": "Return all directed memories.",
            "returns": "Array of memories as JSON.",
        }
    ],
)


@register("list_agents", "Return the names of all agents currently loaded.")
def list_agents(*args, **kwargs) -> str:
    """
    List the name of all agents within the network (one per line).
    """
    import importlib

    args = list(args)
    kwargs = dict(kwargs)
    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    conductor = importlib.import_module("conductor")
    # Ensure configs are loaded so AGENTS/AGENTS_BY_NAME are populated
    if not getattr(conductor, "_CONFIGS_LOADED", False) and hasattr(conductor, "ensure_configs_loaded"):
        try:
            conductor.ensure_configs_loaded()
        except Exception:
            pass
    names = sorted([a.get("name") for a in getattr(conductor, "AGENTS", []) if a.get("name")])
    header = "The following agents currently exist."
    if names:
        return header + "\n" + "\n".join(names)
    return header + "\n(no agents loaded)"


register_details(
    "list_agents",
    [
        {
            "parameters": "",
            "usage": "Return the names of all currently loaded agents, one per line. Arguments may be provided positionally or as keywords—this function takes none.",
            "returns": "A newline-separated list of agent names or '(no agents loaded)'.",
        }
    ],
)


@register(
    "call_agent",
    "Set which agent runs next. Usage: call_agent() to re-run the caller, or call_agent(\"Agent Name\")."
)
def call_agent(*args, **kwargs) -> str:
    import importlib, re
    conductor = importlib.import_module("conductor")

    args = list(args)
    kwargs = dict(kwargs)
    try:
        target = _get_param(args, kwargs, "name", default=None, coerce_underscores=True)
    except ValueError as e:
        return f"(error) ValueError: {e}"

    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    # Ensure runtime is initialized so STATE/AGENTS_BY_NAME are available
    if not getattr(conductor, "_CONFIGS_LOADED", False) and hasattr(conductor, "ensure_configs_loaded"):
        try:
            conductor.ensure_configs_loaded()
        except Exception:
            pass

    def _norm(name: str | None) -> str:
        raw = str(name or "")
        return re.sub(r"\s+", " ", raw).strip()

    if target is None:
        target = (getattr(conductor, "STATE", {}) or {}).get("current_agent")
        if not isinstance(target, str) or target not in conductor.AGENTS_BY_NAME:
            return "call_agent: current agent is not set or cannot be found"
    else:
        target = _norm(target)
        if not target:
            return "call_agent: agent name cannot be empty"
        if target not in conductor.AGENTS_BY_NAME:
            return f"call_agent: '{target}' not found"

    conductor.STATE["force_next_agent"] = target
    return f"Next agent set to: {target}"


register_details(
    "call_agent",
    [
        {
            "parameters": "",
            "usage": "Schedule the current agent to take another turn immediately after this one. Arguments may be provided positionally or as keywords—this function takes none.",
            "returns": "Confirmation of the next agent to run or an error message.",
        },
        {
            "parameters": "name: str",
            "usage": "Select the named agent to run next. Arguments may be provided positionally or as keywords (e.g., call_agent(\"Agent1\") or call_agent(name=\"Agent1\")).",
            "returns": "Confirmation of the next agent to run or an error message.",
        },
    ],
)

@register("speak_to_discord", "Post the agent's visible output to Discord and return that text.")
def speak_to_discord(*args, **kwargs) -> str:
    import importlib, os

    # Enforce zero-arg contract
    if args:
        return "(error) ValueError: unexpected positional arguments"
    if kwargs:
        key = next(iter(kwargs))
        return f"(error) ValueError: unexpected keyword '{key}'"

    # Fetch the message the user actually sees (with *~...~* stripped)
    conductor = importlib.import_module("conductor")
    text = getattr(conductor, "_LAST_VISIBLE_OUTPUT", "")
    text = (text or "").strip()
    if not text:
        return "(no output)"

    # Post via existing webhook helper if configured
    webhook = os.getenv("DISCORD_WEBHOOK_URL")
    post = getattr(conductor, "post_to_discord_via_webhook", None)
    if webhook and callable(post):
        try:
            post(text)
        except Exception as e:
            return f"(error) Discord post failed: {type(e).__name__}: {e}"
    else:
        return "(error) Discord webhook not configured"

    # Also return the visible text as the function output
    return text


register_details(
    "speak_to_discord",
    [
        {
            "parameters": "",
            "usage": "Send the agent's visible output (with any *~...~* removed) to Discord. Place *~speak_to_discord()~* at the end of your message.",
            "returns": "The exact text that was sent to Discord, or an error description.",
        }
    ],
)

