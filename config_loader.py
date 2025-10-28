# JSON configuration helpers for Fenra.
from __future__ import annotations

import json
import os

from pathlib import Path
from typing import Any, Dict
from config import CONF_DIR


LEGACY_MSG_PDV_KEYS = [
    "incoming_message_pdvms",
    "incoming_message_dpvms",
    "incoming_messages_pdvms",
]


def _strip_legacy_message_pdvms(data: dict) -> dict:
    if not isinstance(data, dict):
        return data
    removed = False
    for key in LEGACY_MSG_PDV_KEYS:
        if key in data:
            data.pop(key, None)
            removed = True
    if removed:
        try:
            print("[Globals] Removed legacy incoming_message_*pdvms config keys")
        except Exception:
            pass
    return data


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _conf_path(name: str) -> str:
    return os.path.join(CONF_DIR, name)

def _atomic_write(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def load_globals(path: str = _conf_path('globals.json')) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing globals config at {path}") from e
    if not isinstance(data, dict) or 'model' not in data:
        raise ValueError("globals.json missing required fields")
    return _strip_legacy_message_pdvms(data)

def save_globals(globals_cfg: dict, path: str = _conf_path('globals.json')) -> None:
    _ensure_parent(path)
    data = _strip_legacy_message_pdvms(dict(globals_cfg or {}))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_pdvs(path: str = _conf_path('pdvs.json')) -> Dict[str, dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing PDVs config at {path}") from e
    if isinstance(data, dict) and 'pdvs' in data:
        pdvs = data.get('pdvs')
        if not isinstance(pdvs, list):
            raise ValueError("pdvs.json: 'pdvs' must be a list")
        out = {}
        for idx, p in enumerate(pdvs):
            if not isinstance(p, dict) or 'name' not in p or 'value' not in p:
                raise ValueError(f"pdvs[{idx}] missing name or value")
            out[p['name']] = p
        return out
    if isinstance(data, dict):
        out: Dict[str, dict] = {}
        for name, cfg in data.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"pdvs[{name!r}] must be an object")
            entry = dict(cfg)
            entry.setdefault('name', name)
            entry.setdefault('description', '')
            entry.setdefault('value', 0.0)
            out[name] = entry
        return out
    raise ValueError("pdvs.json invalid format")

def load_classes(path: str = _conf_path('agent_classes.json')) -> Dict[str, dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing agent classes config at {path}") from e
    if isinstance(data, dict) and 'classes' in data:
        classes = data.get('classes')
        if not isinstance(classes, list):
            raise ValueError("agent_classes.json: 'classes' must be a list")
        out = {}
        for idx, c in enumerate(classes):
            if not isinstance(c, dict) or 'name' not in c or 'triggering_pdv' not in c:
                raise ValueError(f"classes[{idx}] missing name or triggering_pdv")
            out[c['name']] = c
        return out
    if isinstance(data, dict):
        out: Dict[str, dict] = {}
        for name, cfg in data.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"classes[{name!r}] must be an object")
            if 'triggering_pdv' not in cfg:
                raise ValueError(f"classes[{name!r}] missing triggering_pdv")
            entry = dict(cfg)
            entry.setdefault('name', name)
            out[name] = entry
        return out
    raise ValueError("agent_classes.json invalid format")

def save_classes(classes: Dict[str, dict], path: str = _conf_path('agent_classes.json')) -> None:
    _ensure_parent(path)
    data = classes or {}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_agents(path: str = _conf_path('agents.json')) -> list[dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing agents config at {path}") from e
    agents = data.get('agents')
    if not isinstance(agents, list):
        raise ValueError("agents.json: 'agents' must be a list")
    for idx, a in enumerate(agents):
        if not isinstance(a, dict) or 'name' not in a or 'agent_class' not in a:
            raise ValueError(f"agents[{idx}] missing name or agent_class")
        if not isinstance(a.get('groups_in'), list):
            a['groups_in'] = []
        if not isinstance(a.get('groups_out'), list):
            a['groups_out'] = []
    return list(agents)


def try_load_globals(path: str = _conf_path('globals.json')) -> dict:
    try:
        return load_globals(path)
    except Exception:
        return {}


def try_load_pdvs(path: str = _conf_path('pdvs.json')) -> Dict[str, dict]:
    try:
        return load_pdvs(path)
    except Exception:
        return {}


def try_load_classes(path: str = _conf_path('agent_classes.json')) -> Dict[str, dict]:
    try:
        return load_classes(path)
    except Exception:
        return {}


def try_load_agents(path: str = _conf_path('agents.json')) -> list[dict]:
    try:
        agents = load_agents(path)
    except Exception:
        return []
    for agent in agents:
        agent.setdefault('groups_in', [])
        agent.setdefault('groups_out', [])
    return agents

def load_state(path: str = _conf_path('state.json')) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing state config at {path}") from e
    if not isinstance(data, dict):
        raise ValueError("state.json invalid")
    return data

def save_pdvs(pdvs: Dict[str, dict], path: str = _conf_path('pdvs.json')) -> None:
    _ensure_parent(path)
    data = pdvs or {}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def save_agents(agents: list[dict], path: str = _conf_path('agents.json')) -> None:
    _ensure_parent(path)
    agents_normalized = []
    for agent in (agents or []):
        entry = dict(agent)
        entry.setdefault('groups_in', [])
        entry.setdefault('groups_out', [])
        agents_normalized.append(entry)
    data = {'agents': agents_normalized}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def save_state(state: dict, path: str = _conf_path('state.json')) -> None:
    _atomic_write(path, state)


def get_path(key: str, default: str | Path | None = None) -> Path | None:
    """Resolve a filesystem path from the globals configuration."""

    if not key:
        raise ValueError("key must be provided")

    data: Any = try_load_globals()
    for part in key.split('.'):
        if isinstance(data, dict) and part in data:
            data = data[part]
        else:
            data = None
            break

    if data is None:
        data = default

    if data is None:
        return None

    if isinstance(data, Path):
        path = data
    elif isinstance(data, str):
        path = Path(data)
    else:
        raise TypeError(f"Configuration value for {key!r} must be a string path.")

    return path.expanduser()
