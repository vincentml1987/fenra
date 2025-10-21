from __future__ import annotations
from typing import Callable, Dict, Tuple, Any
import ast
import subprocess
import sys
import shutil
import re
import json
from copy import deepcopy



# ---------------------------
# Function registry and API
# ---------------------------

_REGISTRY: Dict[str, tuple[Callable[..., str], str]] = {}


def register(name: str, description: str):
    """Decorator to register a callable Fenra function with a human-readable description."""

    def _wrap(func: Callable[..., str]):
        _REGISTRY[name] = (func, description)
        return func

    return _wrap


def _like_to_regex(pattern: str) -> re.Pattern:
    """Translate a % wildcard pattern to a case-insensitive regex."""
    escaped = re.escape(pattern)
    rx = ".*" if not escaped else escaped.replace(r"\%", ".*")
    return re.compile(rx, re.IGNORECASE)


def _literalize(node: ast.AST) -> Any:
    """Safely convert AST nodes for args/kwargs via literal evaluation."""
    return ast.literal_eval(node)


def _parse_call(expr: str) -> tuple[str, list[Any], dict[str, Any]]:
    """
    Parse 'fn_name(arg, kw=val, ...)' into (name, args, kwargs).
    Raises ValueError on any invalid or unsafe expression.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid expression: {e}") from e

    if not isinstance(tree.body, ast.Call) or not isinstance(tree.body.func, ast.Name):
        raise ValueError("expression is not a simple function call")

    fn_name = tree.body.func.id
    args = [_literalize(a) for a in tree.body.args]
    kwargs = {kw.arg: _literalize(kw.value) for kw in tree.body.keywords if kw.arg is not None}
    return fn_name, args, kwargs


def dispatch_expression(expr: str) -> tuple[str, bool, str]:
    """
    Dispatch a Fenra function expression found inside *~...~*.
    Returns (function_name_or_guess, found, result_string).
    - found=False when the name is not registered -> 'Function does not exist.'
    - found=True for executed or error-returning calls (errors are returned as strings).
    """
    guessed_name = expr.strip()
    try:
        name, args, kwargs = _parse_call(expr)
        guessed_name = name
    except Exception:
        return (guessed_name, False, "Function does not exist.")

    entry = _REGISTRY.get(name)
    if not entry:
        return (name, False, "Function does not exist.")

    func, _desc = entry
    try:
        res = func(*args, **kwargs)
        return (name, True, "" if res is None else str(res))
    except Exception as e:
        return (name, True, f"(error) {type(e).__name__}: {e}")


# --------------------------------
# Built-in/required Fenra functions
# --------------------------------


@register("list_functions", "List available Fenra functions; supports % wildcard on name/description.")
def list_functions(search: str = "") -> str:
    """
    Return a newline-separated list of 'name: description'.
    If search is provided, use % as wildcard and match name/description (case-insensitive).
    Always includes this function in the registry.
    """
    _REGISTRY.setdefault(
        "list_functions",
        (list_functions, "List available Fenra functions; supports % wildcard on name/description."),
    )

    items = []
    if search:
        rx = _like_to_regex(search)
        for name, (_fn, desc) in _REGISTRY.items():
            if rx.search(name) or rx.search(desc or ""):
                items.append(f"{name}: {desc}")
    else:
        for name in sorted(_REGISTRY):
            items.append(f"{name}: {_REGISTRY[name][1]}")
    return "\n".join(items) if items else "(no matching functions)"


@register("fenra_powershell", "Execute a PowerShell command string and return its output.")
def fenra_powershell(command: str) -> str:
    """
    Execute a PowerShell command and return its output as text.
    This is the ONLY path for running PowerShell from agent replies.
    """
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


@register("duplicate_self", "Duplicate the current agent as <name>-dup and save/refresh.")
def duplicate_self() -> str:

    import importlib
    from config_loader import save_agents
    conductor = importlib.import_module("conductor")
    
    """
    Create a duplicate of the calling agent with the name '<original>-dup'
    (or '<original>-dup2', etc., if needed), append it to confs/agents.json,
    update Conductor's in-memory indexes, and refresh the UI like the GUI
    would after a save.
    """
    # Resolve the caller
    cur_name = (conductor.STATE or {}).get("current_agent")
    if not isinstance(cur_name, str) or cur_name not in conductor.AGENTS_BY_NAME:
        raise RuntimeError("duplicate_self: no current agent is set or it cannot be found")

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


@register("list_agents", "Return the names of all agents currently loaded.")
def list_agents() -> str:
    """
    List the name of all agents within the network (one per line).
    """
    import importlib

    conductor = importlib.import_module("conductor")
    # Ensure configs are loaded so AGENTS/AGENTS_BY_NAME are populated
    if not getattr(conductor, "_CONFIGS_LOADED", False) and hasattr(conductor, "ensure_configs_loaded"):
        try:
            conductor.ensure_configs_loaded()
        except Exception:
            pass
    names = sorted([a.get("name") for a in getattr(conductor, "AGENTS", []) if a.get("name")])
    return "\n".join(names) if names else "(no agents loaded)"

