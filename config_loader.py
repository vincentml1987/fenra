# JSON configuration helpers for Fenra.
from __future__ import annotations

import json
import os
from typing import Dict

from config import CONF_DIR


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
    return data

def save_globals(globals_cfg: dict, path: str = _conf_path('globals.json')) -> None:
    _atomic_write(path, globals_cfg)

def load_pdvs(path: str = _conf_path('pdvs.json')) -> Dict[str, dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing PDVs config at {path}") from e
    pdvs = data.get('pdvs')
    if not isinstance(pdvs, list):
        raise ValueError("pdvs.json: 'pdvs' must be a list")
    out = {}
    for idx, p in enumerate(pdvs):
        if not isinstance(p, dict) or 'name' not in p or 'value' not in p:
            raise ValueError(f"pdvs[{idx}] missing name or value")
        out[p['name']] = p
    return out

def load_classes(path: str = _conf_path('agent_classes.json')) -> Dict[str, dict]:
    """Load agent classes supporting legacy and modern schemas.

    Accepted schemas:
      1) {"classes": [ {...}, {...} ]}
      2) {"classes": {"Name": {...}, ...}}
      3) {"Name": {...}, "Other": {...}}  # legacy top-level map
    Returns a dict keyed by class name.
    """

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing agent classes config at {path}") from e

    items: list[dict] = []

    def _items_from_mapping(mapping: dict) -> list[dict]:
        mapped: list[dict] = []
        for key, cfg in mapping.items():
            if not isinstance(cfg, dict):
                # Skip non-dict entries (e.g., metadata) when present in legacy files.
                continue
            merged = {**cfg}
            merged.setdefault("name", key)
            mapped.append(merged)
        return mapped

    classes_val = data.get("classes") if isinstance(data, dict) else None

    if isinstance(classes_val, list):
        items = list(classes_val)
    elif isinstance(classes_val, dict):
        items = _items_from_mapping(classes_val)
    elif isinstance(data, dict):
        # Legacy schema: entire file is a mapping of class name -> config
        dict_values = [v for v in data.values() if v is not None]
        if dict_values and all(isinstance(v, dict) for v in dict_values):
            items = _items_from_mapping({k: v for k, v in data.items() if isinstance(v, dict)})

    if not items:
        raise ValueError(
            "agent_classes.json invalid. Expected 'classes' list or a mapping of class objects."
        )

    out: Dict[str, dict] = {}
    for idx, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise ValueError(f"classes[{idx}] must be an object")
        raw_name = raw.get("name")
        if raw_name is None:
            raw_name = f"class_{idx}"
        if not isinstance(raw_name, str):
            raise ValueError(f"classes[{idx}] name must be a string")
        name = raw_name.strip()
        if not name:
            raise ValueError(f"classes[{idx}] missing name")
        if "triggering_pdv" not in raw:
            raise ValueError(f"classes[{idx}] ('{name}') missing triggering_pdv")
        norm = {**raw, "name": name}
        out[name] = norm
    return out

def save_classes(classes: Dict[str, dict], path: str = _conf_path('agent_classes.json')) -> None:
    data = {'classes': list(classes.values())}
    _atomic_write(path, data)

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
    data = {'pdvs': list(pdvs.values())}
    _atomic_write(path, data)

def save_agents(agents: list[dict], path: str = _conf_path('agents.json')) -> None:
    data = {'agents': agents}
    _atomic_write(path, data)

def save_state(state: dict, path: str = _conf_path('state.json')) -> None:
    _atomic_write(path, state)
