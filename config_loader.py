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
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing agent classes config at {path}") from e
    classes = data.get('classes')
    if not isinstance(classes, list):
        raise ValueError("agent_classes.json: 'classes' must be a list")
    out = {}
    for idx, c in enumerate(classes):
        if not isinstance(c, dict) or 'name' not in c or 'triggering_pdv' not in c:
            raise ValueError(f"classes[{idx}] missing name or triggering_pdv")
        out[c['name']] = c
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
