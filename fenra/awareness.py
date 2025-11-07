"""Awareness state helpers shared across Fenra runtime and UI."""

from __future__ import annotations

from typing import Dict

from config_loader import load_state, save_state

AWARENESS_KEYS = [
    "system.global",
    "system.class",
    "system.agent",
    "pre.global",
    "pre.class",
    "pre.agent",
    "post.global",
    "post.class",
    "post.agent",
    "directed_memory",
    "one_time_inject",
    "context.transcript",
    "context.discord_queue",
]

AWARENESS_SLEEP_MESSAGE = (
    "You are currently asleep. Use ~awareness.list()~ to get a list of the inputs you can enable."
)


def _default_awareness() -> Dict[str, bool]:
    return {name: False for name in AWARENESS_KEYS}


def _safe_load_state() -> dict:
    try:
        return load_state()
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _normalize_awareness(data: dict | None) -> tuple[dict[str, bool], bool]:
    source = data if isinstance(data, dict) else {}
    normalized: dict[str, bool] = {}
    changed = not isinstance(data, dict)
    for key in AWARENESS_KEYS:
        val = bool(source.get(key, False))
        if source.get(key) is not val:
            changed = True
        normalized[key] = val
    if set(source.keys()) - set(AWARENESS_KEYS):
        changed = True
    return normalized, changed


def get_awareness() -> Dict[str, bool]:
    """Load the persisted awareness mapping, ensuring all keys are present."""

    state = _safe_load_state()
    normalized, changed = _normalize_awareness(state.get("awareness"))
    if "awareness" not in state or changed:
        state = dict(state)
        state["awareness"] = dict(normalized) if normalized else _default_awareness()
        save_state(state)
    return dict(normalized)


def set_awareness(aw: dict) -> None:
    """Persist a complete awareness mapping."""

    state = _safe_load_state()
    normalized, _ = _normalize_awareness(aw)
    state = dict(state)
    state["awareness"] = dict(normalized) if normalized else _default_awareness()
    save_state(state)


def set_key(name: str, value: bool) -> None:
    """Update a single awareness flag and persist it."""

    aw = get_awareness()
    if name not in aw:
        return
    aw[name] = bool(value)
    set_awareness(aw)


def all_off() -> bool:
    """Return True when every awareness flag is disabled."""

    aw = get_awareness()
    return all(not enabled for enabled in aw.values())

