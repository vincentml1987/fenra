"""Fenra function registry and utilities."""
from __future__ import annotations

import ast
import json
from copy import deepcopy
from typing import Any, Callable, Dict

import conductor
from config_loader import save_agents

REGISTRY: Dict[str, Callable[..., Any]] = {}
DESCRIPTIONS: Dict[str, str | None] = {}


def register(name: str, description: str | None = None):
    """Register a callable Fenra function under ``name``."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        REGISTRY[name] = func
        DESCRIPTIONS[name] = description
        return func

    return decorator


def dispatch_expression(expression: str) -> Any:
    """Parse an expression like ``func(1, key='value')`` and dispatch it."""
    if not isinstance(expression, str):
        raise TypeError("Expression must be a string")
    expr = expression.strip()
    if not expr:
        raise ValueError("Empty expression")
    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid expression '{expression}': {exc}") from exc
    if not isinstance(parsed.body, ast.Call):
        raise ValueError(f"Expression '{expression}' is not a function call")
    func_node = parsed.body.func
    if not isinstance(func_node, ast.Name):
        raise ValueError("Only simple function calls are supported")
    name = func_node.id
    if name not in REGISTRY:
        raise ValueError(f"Unknown Fenra function '{name}'")
    args = [ast.literal_eval(arg) for arg in parsed.body.args]
    kwargs = {}
    for kw in parsed.body.keywords:
        if kw.arg is None:
            raise ValueError("Fenra functions do not support **kwargs")
        kwargs[kw.arg] = ast.literal_eval(kw.value)
    return REGISTRY[name](*args, **kwargs)


@register("duplicate_self", "Duplicate the current agent as <name>-dup and save/refresh.")
def duplicate_self() -> str:
    """Duplicate the currently running agent with a '-dup' suffix."""
    cur_name = (conductor.STATE or {}).get("current_agent")
    if not isinstance(cur_name, str) or cur_name not in conductor.AGENTS_BY_NAME:
        raise RuntimeError(
            "duplicate_self: no current agent is set or it cannot be found"
        )

    source = conductor.AGENTS_BY_NAME[cur_name]
    dup = json.loads(json.dumps(source))
    base = source.get("name") or "Agent"
    proposed = f"{base}-dup"

    existing = {
        a.get("name")
        for a in conductor.AGENTS
        if isinstance(a.get("name"), str)
    }
    name = proposed
    ctr = 2
    while name in existing:
        name = f"{proposed}{ctr}"
        ctr += 1
    dup["name"] = name

    dup.setdefault("groups_in", deepcopy(source.get("groups_in", []) or []))
    dup.setdefault("groups_out", deepcopy(source.get("groups_out", []) or []))

    conductor.AGENTS.append(dup)

    conductor.AGENTS_BY_NAME = {
        a["name"]: a
        for a in conductor.AGENTS
        if isinstance(a.get("name"), str)
    }
    grp_map: Dict[str, set[str]] = {}
    for agent in conductor.AGENTS:
        name_field = agent.get("name")
        if not isinstance(name_field, str):
            continue
        for group in agent.get("groups_in", []) or []:
            grp_map.setdefault(group, set()).add(name_field)
    conductor.AGENTS_BY_GROUP_IN = grp_map

    save_agents(conductor.AGENTS)

    ui = getattr(conductor, "UI", None)
    try:
        if ui is not None and hasattr(ui, "_threadsafe"):
            ui._threadsafe(lambda: setattr(ui, "_agents_model", ui._ui_agents()))
            if hasattr(ui, "_update_required_configs_state"):
                ui._threadsafe(ui._update_required_configs_state)
            if hasattr(ui, "_refresh_agent_listbox"):
                ui._threadsafe(ui._refresh_agent_listbox)
            if hasattr(ui, "_build_simple_groups_tab"):
                ui._threadsafe(ui._build_simple_groups_tab)
    except Exception:  # pragma: no cover - UI refresh is best effort
        pass

    return f"Duplicated {base} → {name}"
