from __future__ import annotations

from typing import Callable, Dict, Any, List
import ast
import subprocess
import sys
import shutil
import json
from copy import deepcopy



# ---------------------------
# Function registry and API
# ---------------------------

_REGISTRY: Dict[str, Dict[str, Any]] = {}


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
        return (guessed_name, False, "Function does not exist.", "")

    entry = _REGISTRY.get(name)
    if not entry:
        return (name, False, "Function does not exist.", params_string)

    func = entry.get("func")
    if func is None:
        return (name, False, "Function does not exist.", params_string)
    try:
        res = func(*args, **kwargs)
        result = "(No Output)" if res is None else str(res)
        return (name, True, result, params_string)
    except Exception as e:
        return (name, True, f"(error) {type(e).__name__}: {e}", params_string)


# --------------------------------
# Built-in/required Fenra functions
# --------------------------------


@register("list_functions", "List available Fenra functions.")
def list_functions(search: str = "") -> str:
    """Return the structured catalog of Fenra functions, optionally filtered by search."""

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
                    "usage": "Return the catalog of available Fenra functions or the subset matching the search string.",
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
def announce_self() -> str:
    import importlib

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
            "usage": "Return the name of the agent currently executing this function.",
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
def get_discord_messages(*args) -> str:
    import importlib

    try:
        fe = importlib.import_module("fenra_ui")
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"

    ensure = getattr(fe, "ensure_discord_running", None)
    if callable(ensure) and not ensure():
        return "(error) Discord not configured or unavailable"

    def _fmt(items):
        lines = []
        for it in items or []:
            author = (it.get("author") or it.get("sender") or "user") or "user"
            text = (it.get("text") or it.get("message") or "").strip()
            if text:
                lines.append(f"{author}: {text}")
        return "\n".join(lines) if lines else "(no messages)"

    # 0 args → latest only
    if len(args) == 0:
        try:
            items = fe.fetch_recent_discord_messages(1) or []
            return _fmt(items[:1])
        except Exception as e:
            return f"(error) {type(e).__name__}: {e}"

    # 1 arg (N) → last N messages (newest→older)
    if len(args) == 1:
        try:
            n = max(1, int(args[0]))
        except Exception:
            return "(error) ValueError: expected integer 'n'"
        try:
            items = fe.fetch_recent_discord_messages(n) or []
            return _fmt(items)
        except Exception as e:
            return f"(error) {type(e).__name__}: {e}"

    # 2 args (M, N) → skip last M, then take next N (newest→older)
    if len(args) == 2:
        try:
            m = max(0, int(args[0]))
            n = max(1, int(args[1]))
        except Exception:
            return "(error) ValueError: expected integers 'm, n'"
        try:
            items = fe.fetch_recent_discord_messages(m + n) or []
            # items are newest→older; drop the newest M, keep next N
            sliced = items[m:m + n]
            return _fmt(sliced)
        except Exception as e:
            return f"(error) {type(e).__name__}: {e}"

    return "(error) ValueError: get_discord_messages accepts 0, 1, or 2 arguments"


register_details(
    "get_discord_messages",
    [
        {
            "parameters": "",
            "usage": "Return the most recent Discord message.",
            "returns": "The latest Discord message text or an error description.",
        },
        {
            "parameters": "n: int",
            "usage": "Return the last n Discord messages from newest to oldest.",
            "returns": "A newline-separated list of Discord messages or an error description.",
        },
        {
            "parameters": "m: int, n: int",
            "usage": "Skip the most recent m Discord messages, then return the next n messages (newest to older).",
            "returns": "A newline-separated list of Discord messages or an error description.",
        },
    ],
)


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


register_details(
    "fenra_powershell",
    [
        {
            "parameters": "command: str",
            "usage": "Execute the provided PowerShell command string and capture its output.",
            "returns": "The combined stdout/stderr from PowerShell or an error message. Returns (No Output) when the command produced no text.",
        }
    ],
)


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


register_details(
    "duplicate_self",
    [
        {
            "parameters": "",
            "usage": "Clone the calling agent with a '-dup' suffix (adding a number if needed) and refresh runtime state.",
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
def rename_agent(*args) -> str:
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

    # Ensure configs/structures are loaded so AGENTS/AGENTS_BY_NAME/STATE exist
    if not getattr(conductor, "_CONFIGS_LOADED", False) and hasattr(conductor, "ensure_configs_loaded"):
        try:
            conductor.ensure_configs_loaded()
        except Exception:
            pass

    def _norm(s: str) -> str:
        s = str(s or "")
        s = s.replace("_", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    if len(args) == 1:
        # Rename current agent -> New_Name
        cur_name = (getattr(conductor, "STATE", {}) or {}).get("current_agent")
        if not isinstance(cur_name, str) or cur_name not in conductor.AGENTS_BY_NAME:
            raise RuntimeError("rename_agent: current agent is not set or cannot be found")
        old_name = cur_name
        new_name = _norm(args[0])
    elif len(args) == 2:
        # Rename Old_Name -> New_Name
        old_name = _norm(args[0])
        new_name = _norm(args[1])
    else:
        return 'Usage: rename_agent("New_Name") or rename_agent("Old_Name","New_Name")'

    if not new_name:
        raise ValueError("rename_agent: new name cannot be empty")
    if old_name not in conductor.AGENTS_BY_NAME:
        raise KeyError(f"rename_agent: '{old_name}' not found")
    if new_name in conductor.AGENTS_BY_NAME:
        raise ValueError(f"rename_agent: name '{new_name}' already exists")

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


register_details(
    "list_agents",
    [
        {
            "parameters": "",
            "usage": "Return the names of all currently loaded agents, one per line.",
            "returns": "A newline-separated list of agent names or '(no agents loaded)'.",
        }
    ],
)


@register(
    "call_agent",
    "Set which agent runs next. Usage: call_agent() to re-run the caller, or call_agent(\"Agent Name\")."
)
def call_agent(*args) -> str:
    import importlib, re
    conductor = importlib.import_module("conductor")

    # Ensure runtime is initialized so STATE/AGENTS_BY_NAME are available
    if not getattr(conductor, "_CONFIGS_LOADED", False) and hasattr(conductor, "ensure_configs_loaded"):
        try:
            conductor.ensure_configs_loaded()
        except Exception:
            pass

    # Resolve target name
    if len(args) == 0:
        target = (getattr(conductor, "STATE", {}) or {}).get("current_agent")
        if not isinstance(target, str) or target not in conductor.AGENTS_BY_NAME:
            return "call_agent: current agent is not set or cannot be found"
    elif len(args) == 1:
        # Back-compat: allow underscores to mean spaces
        raw = str(args[0] or "")
        target = re.sub(r"\s+", " ", raw.replace("_", " ")).strip()
        if not target:
            return "call_agent: agent name cannot be empty"
        if target not in conductor.AGENTS_BY_NAME:
            return f"call_agent: '{target}' not found"
    else:
        return 'Usage: call_agent() or call_agent("Agent Name")'

    conductor.STATE["force_next_agent"] = target
    return f"Next agent set to: {target}"


register_details(
    "call_agent",
    [
        {
            "parameters": "",
            "usage": "Schedule the current agent to take another turn immediately after this one.",
            "returns": "Confirmation of the next agent to run or an error message.",
        },
        {
            "parameters": "name: str",
            "usage": "Select the named agent to run next.",
            "returns": "Confirmation of the next agent to run or an error message.",
        },
    ],
)

