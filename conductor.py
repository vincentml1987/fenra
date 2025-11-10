import os
import json
import time
import random
import shutil
import argparse
import threading
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Iterable
import sys as _sys
import subprocess
import re
import uuid

import requests
from fenra_ui import FenraUI
import importlib
import fenra_functions
from directed_memory import directed_memory_block_for_agent

# Ensure that when this file is executed as a script (__main__), importing
# "conductor" yields the running module instead of creating a duplicate.
if __name__ == "__main__":
    _sys.modules.setdefault("conductor", _sys.modules[__name__])

from ai_model import AIModel
from config_loader import (
    load_globals,
    load_pdvs,
    load_classes,
    load_agents,
    load_state,
    save_pdvs,
    save_agents,
    save_state,
)
from runtime_utils import (
    init_global_logging,
    parse_log_level,
    create_object_logger,
    tokenize_text,
    add_json_watcher,
    OllamaServerError,
)

logger = create_object_logger("Conductor")

TAGS_URL = "http://localhost:11434/api/tags"
PULL_URL = "http://localhost:11434/api/pull"

DEFAULT_OLLAMA_SERVER = {
    "id": "localhost",
    "name": "Localhost",
    "host": "http://localhost",
    "port": 11434,
    "removable": False,
}

STATE_LOCK = threading.RLock()
SERVER_THREAD_LOCK = threading.Lock()
SERVER_CONTEXT_LOCK = threading.Lock()

SERVER_THREADS: dict[str, threading.Thread] = {}
SERVER_STOP_EVENTS: dict[str, threading.Event] = {}
SERVER_STEPS: dict[str, Optional[int]] = {}
SERVER_CONTEXTS: dict[str, str] = {}

_INCOMING_QUEUE: deque[dict] = deque()

_PWSH_BIN_CANDIDATES = ["pwsh", "powershell", "powershell.exe"]
_PWSH_PROC: subprocess.Popen | None = None
_PWSH_LOCK = threading.Lock()


def _which_pwsh() -> str:
    for cand in _PWSH_BIN_CANDIDATES:
        path = shutil.which(cand)
        if path:
            return path
    return "powershell"


def _clip(text: str, limit: int = 8000) -> str:
    t = (text or "").replace("\x00", "")
    if len(t) <= limit:
        return t
    head, tail = t[:4000], t[-2000:]
    return f"{head}\n…[truncated {len(t)-6000} chars]…\n{tail}"


def _is_valid_call_span(span: str) -> bool:
    """Return True iff the span is non-empty and has no leading/trailing whitespace.
    Internal whitespace is allowed."""
    if not isinstance(span, str):
        return False
    trimmed = span.strip()
    return bool(trimmed) and (trimmed == span)


def _extract_pwsh_commands(text: str) -> list[str]:
    """Extract ~...~ call spans that have no leading/trailing whitespace."""

    spans = re.findall(r"~(.*?)~", text or "", flags=re.DOTALL)
    return [s for s in spans if _is_valid_call_span(s)]


def _ps_timeout_seconds() -> int:
    try:
        to = GLOBALS.get("powershell_timeout", 20)
        return int(to) if to is not None else 20
    except Exception:
        return 20


def _start_powershell() -> subprocess.Popen:
    exe = _which_pwsh()
    proc = subprocess.Popen(
        [exe, "-NoLogo", "-NoProfile", "-NonInteractive", "-NoExit", "-Command", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
    )
    init = str(GLOBALS.get("powershell_init_script", "") or "").strip()
    if init:
        try:
            _ps_invoke(init, timeout=_ps_timeout_seconds(), proc=proc)
        except Exception:
            pass
    return proc


def _ensure_powershell_running() -> subprocess.Popen:
    global _PWSH_PROC
    if _PWSH_PROC is None or _PWSH_PROC.poll() is not None:
        _PWSH_PROC = _start_powershell()
    return _PWSH_PROC


def _ps_invoke(cmd: str, timeout: int | None = None, proc: subprocess.Popen | None = None) -> str:
    try:
        if timeout is None:
            timeout = _ps_timeout_seconds()
        ps = proc or _ensure_powershell_running()
        if ps.stdin is None or ps.stdout is None:
            return "(powershell stream error)"
        fence = str(uuid.uuid4())
        begin = f"<<<BEGIN:{fence}>>>"
        end = f"<<<END:{fence}>>>"
        with _PWSH_LOCK:
            ps.stdin.write(f"Write-Output '{begin}'\n")
            ps.stdin.write(cmd + ("\n" if not cmd.endswith("\n") else ""))
            ps.stdin.write(f"\nWrite-Output '{end}'\n")
            ps.stdin.flush()

            t0 = time.time()
            lines: list[str] = []
            begun = False
            while True:
                if timeout and (time.time() - t0) > timeout:
                    return "(timeout)"
                line = ps.stdout.readline()
                if line == "":
                    try:
                        ps.kill()
                    except Exception:
                        pass
                    global _PWSH_PROC
                    _PWSH_PROC = None
                    return "(powershell exited)"
                s = line.rstrip("\r\n")
                if not begun:
                    if s.strip() == begin:
                        begun = True
                    continue
                if s.strip() == end:
                    break
                lines.append(s)
            out = "\n".join(lines).strip()
            return _clip(out if out else "(no output)")
    except FileNotFoundError:
        return "(powershell not found)"
    except Exception as e:
        return _clip(f"(error) {type(e).__name__}: {e}")


def _run_powershell(cmd: str, timeout: int | None = None) -> str:
    return _ps_invoke(cmd, timeout=timeout)


# ----------------------------------------------------------------------------
# Config loading and precedence helpers
# ----------------------------------------------------------------------------

GLOBALS: Dict[str, object] = {}
PDV_META: Dict[str, dict] = {}
PDVS: Dict[str, float] = {}
CLASSES: Dict[str, dict] = {}
AGENTS: List[dict] = []
AGENTS_BY_NAME: Dict[str, dict] = {}
AGENTS_BY_GROUP_IN: Dict[str, set] = {}
STATE: Dict[str, object] = {}
CONTEXT: str = ""

DEFER_CONFIG_LOADING: bool = False
_CONFIGS_LOADED: bool = False

UI: Optional[FenraUI] = None


async def inject_external_message(text: str, meta: dict | None = None):
    """UI compatibility shim.

    This used to enqueue messages and update the UI's Messages tab.
    That tab and the queue have been removed, but the UI may still call
    this function. Keep the name, but make it just write the message
    straight to the appropriate chatlog file(s).
    """

    meta = meta or {}
    ts = meta.get("timestamp") or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    sender = meta.get("author") or meta.get("sender") or "user"
    groups = meta.get("groups") or []

    entry = f"[{ts}] {sender}: {text}\n{'-'*80}\n\n"

    # If no groups were specified, write to a default/current context log.
    if not groups:
        path = os.path.join("chatlogs", "context_current.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry)
    else:
        # Otherwise, write to each group's chat log
        for g in groups:
            safe_group = str(g)
            path = os.path.join("chatlogs", f"chat_log_{safe_group}.txt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(entry)



# --- One-time inject helpers ---
def get_one_time_inject() -> str:
    return str((STATE or {}).get("one_time_inject") or "").strip()


def set_one_time_inject(text: str) -> None:
    from config_loader import save_state

    global STATE
    STATE["one_time_inject"] = str(text or "")
    save_state(STATE)
    if UI is not None and hasattr(UI, "refresh_inject_pending"):
        try:
            UI.refresh_inject_pending()
        except Exception:
            logger.exception("UI refresh_inject_pending failed")


def delete_one_time_inject() -> str:
    """Remove pending message and return its text (for UI to repopulate editor)."""

    from config_loader import save_state

    global STATE
    msg = str((STATE or {}).get("one_time_inject") or "")
    STATE["one_time_inject"] = ""
    save_state(STATE)
    if UI is not None and hasattr(UI, "refresh_inject_pending"):
        try:
            UI.refresh_inject_pending()
        except Exception:
            logger.exception("UI refresh_inject_pending failed")
    return msg

# ----------------------------------------------------------------------------
# Run control (Start/Stop)
# ----------------------------------------------------------------------------
_RUN_EVENT = threading.Event()


def _reset_config_state() -> None:
    """Reset in-memory config structures to harmless defaults."""

    global GLOBALS, PDV_META, PDVS, CLASSES, AGENTS, AGENTS_BY_NAME
    global AGENTS_BY_GROUP_IN, STATE, CONTEXT, MODEL, _CONFIGS_LOADED, _PWSH_PROC

    GLOBALS = {}
    PDV_META = {}
    PDVS = {}
    CLASSES = {}
    AGENTS = []
    AGENTS_BY_NAME = {}
    AGENTS_BY_GROUP_IN = {}
    STATE = {}
    CONTEXT = ""
    MODEL = None
    _CONFIGS_LOADED = False
    _RUN_EVENT.clear()

    try:
        if _PWSH_PROC and _PWSH_PROC.poll() is None:
            _PWSH_PROC.kill()
    except Exception:
        pass
    _PWSH_PROC = None


def start_processing() -> None:
    """Enable the agent loop to run."""

    ensure_configs_loaded()
    logger.info("[RunControl] START pressed; enabling processing")
    _RUN_EVENT.set()


def stop_processing() -> None:
    """Pause the agent loop."""

    logger.info("[RunControl] STOP pressed; pausing processing")
    _RUN_EVENT.clear()


def is_processing() -> bool:
    """Return True if processing is currently enabled."""

    return _RUN_EVENT.is_set()


def _server_list() -> list[dict]:
    with STATE_LOCK:
        servers = STATE.get("ollama_servers")
        if not isinstance(servers, list):
            return []
        return [dict(s) for s in servers if isinstance(s, dict)]


def _server_by_id(server_id: str) -> Optional[dict]:
    if not server_id:
        return None
    for entry in _server_list():
        if entry.get("id") == server_id:
            return entry
    return None


def _ensure_server_context_entry(server_id: str) -> None:
    with SERVER_CONTEXT_LOCK:
        SERVER_CONTEXTS.setdefault(server_id, "")


def _ensure_servers_initialized() -> None:
    changed = False
    with STATE_LOCK:
        servers_raw = STATE.get("ollama_servers")
        if not isinstance(servers_raw, list):
            servers_raw = []
            changed = True
        normalized: list[dict] = []
        seen: set[str] = set()
        for entry in servers_raw:
            if not isinstance(entry, dict):
                changed = True
                continue
            data = dict(entry)
            sid = str(data.get("id") or "").strip()
            if not sid:
                sid = str(uuid.uuid4())
                changed = True
            if sid in seen:
                sid = str(uuid.uuid4())
                changed = True
            name = str(data.get("name") or "").strip() or f"Server {len(normalized)+1}"
            host = str(data.get("host") or "").strip() or DEFAULT_OLLAMA_SERVER["host"]
            try:
                port_val = int(data.get("port", DEFAULT_OLLAMA_SERVER["port"]))
            except Exception:
                port_val = DEFAULT_OLLAMA_SERVER["port"]
                changed = True
            removable = bool(data.get("removable", True)) and sid != DEFAULT_OLLAMA_SERVER["id"]
            normalized.append(
                {
                    "id": sid,
                    "name": name,
                    "host": host,
                    "port": port_val,
                    "removable": removable,
                }
            )
            seen.add(sid)
        if DEFAULT_OLLAMA_SERVER["id"] not in seen:
            normalized.insert(0, dict(DEFAULT_OLLAMA_SERVER))
            seen.add(DEFAULT_OLLAMA_SERVER["id"])
            changed = True
        ctx_map = STATE.get("agent_context_by_server")
        if not isinstance(ctx_map, dict):
            ctx_map = {}
            changed = True
        for sid in list(ctx_map):
            if sid not in seen:
                ctx_map.pop(sid, None)
                changed = True
        for sid in seen:
            ctx_map.setdefault(sid, "")
        STATE["agent_context_by_server"] = ctx_map
        cur_map = STATE.get("current_agent_by_server")
        if not isinstance(cur_map, dict):
            cur_map = {}
            changed = True
        for sid in list(cur_map):
            if sid not in seen:
                cur_map.pop(sid, None)
                changed = True
        default_agent = STATE.get("current_agent")
        for sid in seen:
            cur_map.setdefault(sid, default_agent)
        STATE["current_agent_by_server"] = cur_map
        STATE["ollama_servers"] = normalized
    with SERVER_CONTEXT_LOCK:
        for sid in seen:
            SERVER_CONTEXTS.setdefault(sid, ctx_map.get(sid, ""))
        for sid in list(SERVER_CONTEXTS):
            if sid not in seen:
                SERVER_CONTEXTS.pop(sid, None)
    if changed:
        save_state(STATE)


def _set_current_agent_for_server(server_id: str, agent_name: Optional[str]) -> None:
    if not server_id:
        return
    with STATE_LOCK:
        cur_map = STATE.setdefault("current_agent_by_server", {})
        if agent_name:
            cur_map[server_id] = agent_name
            STATE["current_agent"] = agent_name
        else:
            cur_map.pop(server_id, None)
        save_state(STATE)


def _get_current_agent_for_server(server_id: str) -> Optional[str]:
    if not server_id:
        return None
    with STATE_LOCK:
        cur_map = STATE.get("current_agent_by_server")
        if not isinstance(cur_map, dict):
            return None
        value = cur_map.get(server_id)
    return value if isinstance(value, str) else None


def _set_server_context(server_id: str, text: str) -> None:
    with SERVER_CONTEXT_LOCK:
        SERVER_CONTEXTS[server_id] = text
    with STATE_LOCK:
        ctx_map = STATE.setdefault("agent_context_by_server", {})
        ctx_map[server_id] = text
    if UI is not None and hasattr(UI, "update_server_context"):
        try:
            UI.update_server_context(server_id, text)
        except Exception:
            logger.exception("UI update_server_context failed")


def _append_server_context(server_id: str, text: str) -> None:
    with SERVER_CONTEXT_LOCK:
        combined = f"{SERVER_CONTEXTS.get(server_id, '')}{text}"
        if len(combined) > 50000:
            combined = combined[-50000:]
        SERVER_CONTEXTS[server_id] = combined
    with STATE_LOCK:
        ctx_map = STATE.setdefault("agent_context_by_server", {})
        ctx_map[server_id] = combined
    if UI is not None and hasattr(UI, "update_server_context"):
        try:
            UI.update_server_context(server_id, combined)
        except Exception:
            logger.exception("UI update_server_context failed")


def get_server_context(server_id: str) -> str:
    with SERVER_CONTEXT_LOCK:
        return SERVER_CONTEXTS.get(server_id, "")


def _broadcast_server_list() -> None:
    if UI is not None and hasattr(UI, "set_ollama_servers"):
        try:
            UI.set_ollama_servers(_server_list())
        except Exception:
            logger.exception("UI set_ollama_servers failed")


def get_ollama_servers() -> list[dict]:
    ensure_configs_loaded()
    return _server_list()


def add_ollama_server(
    name: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> dict:
    ensure_configs_loaded()
    default_name = (name or "New Server").strip() or "New Server"
    with STATE_LOCK:
        servers = list(STATE.get("ollama_servers") or [])
        existing_names = {str(s.get("name") or "") for s in servers}
        base_name = default_name
        if default_name in existing_names:
            suffix = 2
            while f"{base_name} {suffix}" in existing_names:
                suffix += 1
            default_name = f"{base_name} {suffix}"
        entry = {
            "id": str(uuid.uuid4()),
            "name": default_name,
            "host": (host or DEFAULT_OLLAMA_SERVER["host"]).strip() or DEFAULT_OLLAMA_SERVER["host"],
            "port": int(port or DEFAULT_OLLAMA_SERVER["port"]),
            "removable": True,
        }
        servers.append(entry)
        STATE["ollama_servers"] = servers
        cur_map = STATE.setdefault("current_agent_by_server", {})
        cur_map[entry["id"]] = STATE.get("current_agent")
        STATE.setdefault("agent_context_by_server", {})[entry["id"]] = ""
        save_state(STATE)
    _ensure_server_context_entry(entry["id"])
    _broadcast_server_list()
    _ensure_server_runner(entry["id"])
    return entry


def update_ollama_server(
    server_id: str,
    *,
    name: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> dict:
    ensure_configs_loaded()
    updated: dict | None = None
    with STATE_LOCK:
        servers = list(STATE.get("ollama_servers") or [])
        for idx, entry in enumerate(servers):
            if entry.get("id") != server_id:
                continue
            new_entry = dict(entry)
            if name is not None:
                proposed = name.strip()
                if proposed:
                    new_entry["name"] = proposed
            if host is not None:
                proposed_host = host.strip()
                if proposed_host:
                    new_entry["host"] = proposed_host
            if port is not None:
                try:
                    new_entry["port"] = int(port)
                except Exception as exc:
                    raise ValueError("port must be an integer") from exc
            servers[idx] = new_entry
            STATE["ollama_servers"] = servers
            save_state(STATE)
            updated = dict(new_entry)
            break
    if updated is None:
        raise KeyError(f"Unknown server {server_id}")
    _broadcast_server_list()
    return updated


def remove_ollama_server(server_id: str) -> None:
    ensure_configs_loaded()
    removed = False
    with STATE_LOCK:
        servers = list(STATE.get("ollama_servers") or [])
        for idx, entry in enumerate(servers):
            if entry.get("id") != server_id:
                continue
            if not entry.get("removable", True):
                raise ValueError("This server cannot be removed")
            servers.pop(idx)
            STATE["ollama_servers"] = servers
            STATE.setdefault("current_agent_by_server", {}).pop(server_id, None)
            STATE.setdefault("agent_context_by_server", {}).pop(server_id, None)
            save_state(STATE)
            removed = True
            break
    if not removed:
        raise KeyError(f"Unknown server {server_id}")
    with SERVER_CONTEXT_LOCK:
        SERVER_CONTEXTS.pop(server_id, None)
    _broadcast_server_list()
    _stop_server_runner(server_id)


def _ensure_server_runner(server_id: str, steps: Optional[int] = None) -> None:
    if not server_id:
        return
    with SERVER_THREAD_LOCK:
        thread = SERVER_THREADS.get(server_id)
        if thread and thread.is_alive():
            if steps is not None:
                SERVER_STEPS[server_id] = steps
            return
        stop_event = threading.Event()
        SERVER_STOP_EVENTS[server_id] = stop_event
        SERVER_STEPS[server_id] = steps
        thread = threading.Thread(
            target=_run_loop_for_server,
            args=(server_id, stop_event),
            name=f"FenraServer-{server_id}",
            daemon=True,
        )
        SERVER_THREADS[server_id] = thread
        thread.start()


def _stop_server_runner(server_id: str) -> None:
    if not server_id:
        return
    with SERVER_THREAD_LOCK:
        stop_event = SERVER_STOP_EVENTS.pop(server_id, None)
        thread = SERVER_THREADS.pop(server_id, None)
        SERVER_STEPS.pop(server_id, None)
    if stop_event:
        stop_event.set()
    if thread and thread.is_alive():
        try:
            thread.join(timeout=2.0)
        except Exception:
            pass


def _refresh_server_runners(steps: Optional[int] = None) -> None:
    servers = _server_list()
    active_ids = {s.get("id") for s in servers if s.get("id")}
    for sid in active_ids:
        _ensure_server_runner(sid, steps)
    with SERVER_THREAD_LOCK:
        for sid in list(SERVER_THREADS):
            if sid not in active_ids:
                _stop_server_runner(sid)


def _run_loop_for_server(server_id: str, stop_event: threading.Event) -> None:
    cur: Optional[str] = None
    hist: List[str] = []
    count = 0
    while not stop_event.is_set():
        limit = SERVER_STEPS.get(server_id)
        if limit is not None and count >= limit:
            break
        if not _RUN_EVENT.is_set():
            time.sleep(0.1)
            continue
        if not _CONFIGS_LOADED:
            ensure_configs_loaded()
        server_cfg = _server_by_id(server_id)
        if server_cfg is None:
            logger.info("Server %s removed; stopping runner", server_id)
            break
        if cur is None:
            candidate = _get_current_agent_for_server(server_id)
            if isinstance(candidate, str) and candidate in AGENTS_BY_NAME:
                cur = candidate
            else:
                cur = next(iter(AGENTS_BY_NAME), None)
            if cur is None:
                logger.error(
                    "No current_agent available after config load; check confs/state.json and agents."
                )
                break
            hist = [cur]
            _set_current_agent_for_server(server_id, cur)
        if UI is not None:
            try:
                UI.set_active_agent(server_id, cur)
                UI.set_group_contexts(_read_group_contexts())
            except Exception:
                logger.exception("UI pre-step update failed")
        logger.info(
            "Running agent %s on server %s",
            cur,
            server_cfg.get("name") or server_id,
        )
        try:
            nxt = step_agent(cur, server_cfg)
        except OllamaServerError as exc:
            logger.error("Generation failed on server %s: %s", server_id, exc)
            _set_server_context(server_id, f"ERROR: {exc}")
            break
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent loop crashed on server %s", server_id)
            _set_server_context(server_id, f"UNEXPECTED ERROR: {exc}")
            break
        logger.info("Next agent on %s: %s", server_id, nxt)
        count += 1
        time.sleep(0.2)
        if nxt:
            cur = nxt
            _set_current_agent_for_server(server_id, cur)
            hist.append(cur)
            continue
        cur_agent = AGENTS_BY_NAME.get(cur)
        if cur_agent:
            _flag_no_downstream(cur_agent, cur_agent.get("groups_out", []))
        while hist:
            dead = hist.pop()
            if not hist:
                logger.error("All downstream paths dead-end. Please wire groups.")
                stop_event.set()
                return
            prev = hist[-1]
            alt = select_next_agent(prev)
            if alt and alt["name"] != dead:
                cur = alt["name"]
                _set_current_agent_for_server(server_id, cur)
                hist.append(cur)
                break
        else:
            logger.error("All downstream paths dead-end. Please wire groups.")
            stop_event.set()
            return
    with SERVER_THREAD_LOCK:
        SERVER_STOP_EVENTS.pop(server_id, None)
        SERVER_THREADS.pop(server_id, None)
        SERVER_STEPS.pop(server_id, None)


def load_all_configs() -> None:
    """Load global config data into module-level structures."""
    global GLOBALS, PDV_META, PDVS, CLASSES, AGENTS, AGENTS_BY_NAME, AGENTS_BY_GROUP_IN, STATE
    GLOBALS = load_globals()
    raw_pdvs = load_pdvs()
    PDV_META.clear()
    PDV_META.update(raw_pdvs)
    PDVS = {name: cfg.get("value", 0.5) for name, cfg in raw_pdvs.items()}
    # Persist a clean live snapshot at startup so the UI pie is correct after restart.
    _persist_pdvs_live()
    CLASSES = load_classes()
    AGENTS = load_agents()
    changed = False
    for a in AGENTS:
        if not a.get("created_at"):
            a["created_at"] = datetime.utcnow().isoformat() + "Z"
            changed = True
    if changed:
        save_agents(AGENTS)
    AGENTS_BY_NAME = {a["name"]: a for a in AGENTS}
    AGENTS_BY_GROUP_IN = {}
    for agent in AGENTS:
        for grp in agent.get("groups_in", []):
            AGENTS_BY_GROUP_IN.setdefault(grp, set()).add(agent["name"])
    try:
        STATE = load_state()
    except FileNotFoundError:
        logger.warning("state.json missing; initializing new state")
        STATE = {}
    except Exception as exc:
        logger.warning("state.json invalid; initializing new state: %s", exc)
        STATE = {}
    if not STATE.get("current_agent") and AGENTS:
        earliest = min(AGENTS, key=lambda a: a.get("created_at", ""))
        STATE["current_agent"] = earliest["name"]
        STATE.setdefault("pdv_history_path", os.path.join("chatlogs", "pdv_history.jsonl"))
        save_state(STATE)
    _ensure_servers_initialized()


def _persist_pdvs_live() -> None:
    """Write current PDV values to chatlogs/pdvs_live.json for the UI."""
    try:
        os.makedirs("chatlogs", exist_ok=True)
        with open("chatlogs/pdvs_live.json", "w", encoding="utf-8") as f:
            json.dump(PDVS, f)
    except Exception as exc:
        logger.warning("failed to write pdvs_live.json: %s", exc)


def _refresh_pdvs_from_disk() -> None:
    """Synchronize in-memory PDV state with confs/pdvs.json."""
    global PDV_META, PDVS
    try:
        raw_pdvs = load_pdvs()
    except Exception:
        return
    PDV_META = dict(raw_pdvs)
    PDVS = {name: cfg.get("value", 0.5) for name, cfg in raw_pdvs.items()}
    # Ensure the UI's Live Metrics reflects the latest config immediately.
    _persist_pdvs_live()


def _expand_context_macros(text: str, *, agent: dict, model_id: str) -> str:
    if not isinstance(text, str) or not text:
        return text or ""

    name = agent.get("name", "") or ""
    groups_in = ", ".join(
        [g for g in (agent.get("groups_in") or []) if isinstance(g, str)]
    ) or ""
    groups_out = ", ".join(
        [g for g in (agent.get("groups_out") or []) if isinstance(g, str)]
    ) or ""
    try:
        cls = CLASSES.get(agent.get("agent_class", ""), {}) or {}
        trig_pdv = str(cls.get("triggering_pdv", "") or "")
    except Exception:
        trig_pdv = ""

    try:
        datestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    except Exception:
        datestamp = ""

    out = text
    out = out.replace("%d", datestamp)
    out = out.replace("%a", name)
    out = out.replace("%m", model_id)
    out = out.replace("%i", groups_in)
    out = out.replace("%o", groups_out)
    out = out.replace("%p", trig_pdv)
    return out


def effective_params(agent: dict):
    cls = CLASSES[agent["agent_class"]]
    model = agent.get("model") or cls.get("model") or GLOBALS.get("model")
    temp = agent.get("temperature")
    if temp is None:
        temp = cls.get("temperature")
    if temp is None:
        temp = GLOBALS.get("temperature")
    ignore_global_system = bool(cls.get("ignore_global_system", False)) or bool(
        agent.get("ignore_global_system", False)
    )
    ignore_global_pre = bool(cls.get("ignore_global_pre", False)) or bool(
        agent.get("ignore_global_pre", False)
    )
    ignore_global_post = bool(cls.get("ignore_global_post", False)) or bool(
        agent.get("ignore_global_post", False)
    )

    ignore_class_system = bool(agent.get("ignore_class_system", False))
    ignore_class_pre = bool(agent.get("ignore_class_pre", False))
    ignore_class_post = bool(agent.get("ignore_class_post", False))

    system_text = "\n".join(
        [
            ("" if ignore_global_system else GLOBALS.get("system_prompt", "")),
            ("" if ignore_class_system else cls.get("system_prompt", "")),
            agent.get("system_prompt", ""),
        ]
    ).strip()
    pre_text = "\n".join(
        [
            ("" if ignore_global_pre else GLOBALS.get("pre_context_message", "")),
            ("" if ignore_class_pre else cls.get("pre_context_message", "")),
            agent.get("pre_context_message", ""),
        ]
    ).strip()
    post_text = "\n".join(
        [
            ("" if ignore_global_post else GLOBALS.get("post_context_message", "")),
            ("" if ignore_class_post else cls.get("post_context_message", "")),
            agent.get("post_context_message", ""),
        ]
    ).strip()
    if not model:
        raise RuntimeError(f"No model resolved for agent '{agent['name']}'. Set agent/class/global model.")
    system_text = _expand_context_macros(system_text, agent=agent, model_id=model)
    pre_text = _expand_context_macros(pre_text, agent=agent, model_id=model)
    post_text = _expand_context_macros(post_text, agent=agent, model_id=model)
    return model, temp, system_text, pre_text, post_text


def trim_message_for_budget(model: str, system_text: str, pre_text: str, msg_text: str, post_text: str, max_tokens: int) -> str:
    """Trim the MESSAGE portion so the full prompt fits ``max_tokens``."""
    def count(txt: str) -> int:
        try:
            return len(tokenize_text(model, txt or ""))
        except Exception:
            return len((txt or "").split())
    while (
        count(system_text) + count(pre_text) + count(msg_text) + count(post_text)
    ) > max_tokens:
        lines = msg_text.splitlines()
        if len(lines) <= 10:
            # keep the last ~10 lines even if over budget
            msg_text = "\n".join(lines[-10:])
            break
        drop = max(1, len(lines) // 10)
        msg_text = "\n".join(lines[drop:])
    return msg_text


def _discord_chunks(text: str, limit: int = 1900):
    text = text or ""
    while text:
        cut = text.rfind("\n", 0, limit)
        if cut == -1:
            cut = min(len(text), limit)
        yield text[:cut]
        text = text[cut:].lstrip("\n")


def post_to_discord_via_webhook(content: str) -> None:
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return
    for part in _discord_chunks(content):
        if not part.strip():
            continue
        try:
            requests.post(url, json={"content": part}, timeout=10)
        except Exception:
            logger.exception("Discord post failed")


# ----------------------------------------------------------------------------
# PDV mechanics
# ----------------------------------------------------------------------------

def apply_pdv_adjustments(adjs: List[dict], *, scale: float = 1.0) -> dict[str, float]:
    """Apply linear PDV updates with a floor at zero and no upper bound."""
    _refresh_pdvs_from_disk()
    changed = False
    for adj in adjs or []:
        name = adj["name"]
        coeff = float(adj.get("delta", 0.0))
        if name not in PDVS:
            PDV_META.setdefault(name, {"name": name, "description": "", "value": 0.0})
            PDVS[name] = float(PDV_META[name].get("value", 0.0))
            changed = True
        current = float(PDVS.get(name, 0.0))
        delta = coeff * float(scale)
        updated = current + delta
        if updated < 0.0:
            updated = 0.0
        if updated != current:
            PDVS[name] = updated
            PDV_META.setdefault(name, {"name": name, "description": ""})
            PDV_META[name]["value"] = updated
            changed = True
    if changed:
        save_pdvs(
            {
                n: {
                    "name": n,
                    "description": PDV_META.get(n, {}).get("description", ""),
                    "value": v,
                }
                for n, v in PDVS.items()
            }
        )
        os.makedirs("chatlogs", exist_ok=True)
        with open("chatlogs/pdv_history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "pdvs": PDVS}, ensure_ascii=False) + "\n")
        # Live snapshot after any adjustments as well.
        _persist_pdvs_live()
    return dict(PDVS)


# ----------------------------------------------------------------------------
# Selection helpers
# ----------------------------------------------------------------------------

def has_downstream(agent_name: str) -> bool:
    a = AGENTS_BY_NAME[agent_name]
    outs = set(a.get("groups_out", []))
    for b in AGENTS:
        if b["name"] == agent_name:
            continue
        if outs & set(b.get("groups_in", [])):
            return True
    return False


def downstream_candidates(curr_name: str) -> List[dict]:
    # Be resilient if an agent was renamed/removed mid-run
    cur = AGENTS_BY_NAME.get(curr_name)
    if cur is None:
        logger.warning(
            "downstream_candidates: '%s' not found (likely renamed/removed).", curr_name
        )
        return []
    outs = set(cur.get("groups_out", []))
    return [
        a for a in AGENTS
        if a["name"] != curr_name and outs & set(a.get("groups_in", []))
    ]


def _flag_no_downstream(agent: dict, groups: Iterable[str]) -> None:
    agent["flag_no_downstream"] = True
    agent["missing_out_groups"] = list(groups)
    save_agents(AGENTS)


def select_next_agent(curr_name: str) -> Optional[dict]:
    """Select the next agent using weighted randomness over downstream classes."""

    # If current name disappeared (e.g., renamed), just say "no selection"
    if curr_name not in AGENTS_BY_NAME:
        logger.warning("select_next_agent: '%s' not found; returning None", curr_name)
        return None

    D = downstream_candidates(curr_name)
    if not D:
        # Guard again in case it vanished between calls
        cur = AGENTS_BY_NAME.get(curr_name)
        if cur:
            _flag_no_downstream(cur, cur.get("groups_out", []))
        return None

    class_to_agents: Dict[str, List[dict]] = {}
    for agent in D:
        class_to_agents.setdefault(agent["agent_class"], []).append(agent)

    queue_empty = not _INCOMING_QUEUE
    if queue_empty:
        filtered: Dict[str, List[dict]] = {}
        for class_name, _agents in class_to_agents.items():
            if CLASSES.get(class_name, {}).get("reads_message_queue"):
                continue
            filtered[class_name] = _agents
        if filtered:
            class_to_agents = filtered

    classes: List[str] = []
    weights: List[float] = []
    for class_name, _agents in class_to_agents.items():
        trig = CLASSES[class_name]["triggering_pdv"]
        weight = max(0.0, float(PDVS.get(trig, 0.0)))
        classes.append(class_name)
        weights.append(weight)

    def _weighted_choice(options: List[str], probs: List[float]) -> str:
        total = sum(probs)
        if total <= 0.0:
            return random.choice(options)
        r = random.random() * total
        acc = 0.0
        for opt, weight in zip(options, probs):
            acc += weight
            if r <= acc:
                return opt
        return options[-1]

    chosen_class = _weighted_choice(classes, weights)
    return random.choice(class_to_agents[chosen_class])


def _discord_transcript(limit: int) -> str:
    """Pull last N Discord messages and format as transcript text."""
    try:
        fe = importlib.import_module("fenra_ui")
        if hasattr(fe, "fetch_recent_discord_messages"):
            msgs = fe.fetch_recent_discord_messages(int(limit)) or []
            # Oldest first
            msgs = list(reversed(msgs))
            lines: List[str] = []
            for it in msgs:
                sender = it.get("author") or it.get("sender") or "user"
                msg = it.get("text") or it.get("message") or ""
                ts = it.get("timestamp", "")
                lines.append(f"[{ts}] {sender}: {msg}")
            return "\n".join(lines)
    except Exception:
        logger.exception("Discord history fetch failed")
    return ""


def _read_group_contexts() -> Dict[str, str]:
    ctxs: Dict[str, str] = {}
    os.makedirs("chatlogs", exist_ok=True)
    seen_groups = set()
    for a in AGENTS:
        for g in (a.get("groups_in") or []) + (a.get("groups_out") or []):
            if g:
                seen_groups.add(g)
    for g in sorted(seen_groups):
        path = os.path.join("chatlogs", f"chat_log_{g}.txt")
        try:
            with open(path, "r", encoding="utf-8") as f:
                ctxs[g] = f.read()
        except Exception:
            ctxs[g] = ""
    return ctxs


def find_archivist_downstream(agent: dict) -> Optional[dict]:
    for cand in downstream_candidates(agent["name"]):
        if CLASSES[cand["agent_class"]].get("is_archivist"):
            return cand
    return None


# ----------------------------------------------------------------------------
# Model and loop
# ----------------------------------------------------------------------------

MODEL: Optional[AIModel] = None


def ensure_models_available(model_ids: List[str]) -> None:
    """Verify models are installed locally, pulling them if missing."""
    for attempt in range(3):
        try:
            resp = requests.get(TAGS_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as exc:
            if attempt == 2:
                logger.error("Failed to query local models from Ollama: %s", exc)
                raise
            time.sleep(2 ** attempt)

    local = {m.get("name") for m in data.get("models", [])}
    for mid in model_ids:
        if not mid or mid in local:
            continue
        for attempt in range(3):
            try:
                pull = requests.post(
                    PULL_URL, json={"name": mid, "stream": False}, timeout=60
                )
                pull.raise_for_status()
                _ = pull.json()
                logger.info("Ensured model %s is available", mid)
                break
            except Exception as exc:
                if attempt == 2:
                    logger.error("Failed to pull model %s: %s", mid, exc)
                    raise
                time.sleep(2 ** attempt)


def apply_globals_update(new_cfg: dict) -> None:
    """Apply updated global configuration without restarting the agent loop."""

    if not isinstance(new_cfg, dict):
        logger.warning("apply_globals_update ignored non-dict configuration")
        return

    global GLOBALS, MODEL

    previous = dict(GLOBALS)
    GLOBALS = dict(new_cfg)

    changed_keys = sorted(
        key
        for key in set(previous) | set(GLOBALS)
        if previous.get(key) != GLOBALS.get(key)
    )

    level_name = GLOBALS.get("debug_level", "INFO")
    try:
        level = parse_log_level(level_name)
    except Exception:
        level = parse_log_level("INFO")
    init_global_logging(level)

    if MODEL is None:
        if changed_keys:
            logger.info(
                "Applied globals update (%s) but base model not yet initialized",
                ", ".join(changed_keys),
            )
        else:
            logger.info("Applied globals update (no changes) but base model not yet initialized")
        return

    new_model_id = GLOBALS.get("model")
    if isinstance(new_model_id, str):
        new_model_id = new_model_id.strip() or None
    elif new_model_id is not None:
        new_model_id = str(new_model_id)

    old_model_id = getattr(MODEL, "model_id", None)
    if new_model_id and new_model_id != old_model_id:
        try:
            ensure_models_available([new_model_id])
        except Exception:
            logger.exception("Failed to ensure model %s is available", new_model_id)
        else:
            MODEL.model_id = new_model_id
    elif new_model_id:
        MODEL.model_id = new_model_id

    temp_val = GLOBALS.get("temperature")
    try:
        if temp_val is not None:
            MODEL.temperature = float(temp_val)
    except Exception:
        pass

    if "system_prompt" in GLOBALS:
        try:
            MODEL.system_prompt = GLOBALS.get("system_prompt", MODEL.system_prompt)
        except Exception:
            pass

    wd_val = GLOBALS.get("watchdog_timeout")
    try:
        if wd_val is None:
            MODEL.watchdog_timeout = None
        else:
            wd_num = int(float(wd_val))
            MODEL.watchdog_timeout = None if wd_num <= 0 else wd_num
    except Exception:
        pass

    max_tokens_val = GLOBALS.get("max_context_tokens")
    try:
        if max_tokens_val is not None:
            MODEL.max_tokens = int(max_tokens_val)
    except Exception:
        pass

    if changed_keys:
        logger.info("Applied globals update (%s)", ", ".join(changed_keys))
    else:
        logger.info("Applied globals update (no changes)")


def reload_globals_from_disk() -> None:
    """Refresh globals from disk and apply them immediately."""

    try:
        cfg = load_globals()
    except Exception:
        logger.exception("Failed to reload globals from disk")
        return
    apply_globals_update(cfg)


def _initialize_runtime_from_configs() -> None:
    """Finalize runtime initialization once configs are available."""

    if GLOBALS.get("model") in (None, ""):
        raise RuntimeError(
            "Global model is required. Set it in confs/globals.json or via the UI."
        )
    if GLOBALS.get("temperature") is None:
        raise RuntimeError(
            "Global temperature is required. Set it in confs/globals.json or via the UI."
        )

    level = parse_log_level(GLOBALS.get("debug_level", "INFO"))
    init_global_logging(level)

    models = set()
    for agent in AGENTS:
        model, _, _, _, _ = effective_params(agent)
        if model:
            models.add(model)
    ensure_models_available(list(models))

    global MODEL, CONTEXT
    base_model = GLOBALS.get("model") or next(iter(models))
    wd = GLOBALS.get("watchdog_timeout", 900)
    try:
        wd = None if wd is None else int(wd)
    except Exception:
        wd = 900
    MODEL = AIModel(
        name="fenra",
        model_id=base_model,
        topic_prompt="",
        role_prompt="",
        temperature=float(GLOBALS.get("temperature", 0.7)),
        max_tokens=int(GLOBALS.get("max_context_tokens", 8192)),
        system_prompt=GLOBALS.get("system_prompt", ""),
        watchdog_timeout=wd,
    )
    try:
        with open(os.path.join("chatlogs", "context_current.txt"), "r", encoding="utf-8") as f:
            CONTEXT = f.read()
    except FileNotFoundError:
        CONTEXT = ""


def ensure_configs_loaded() -> None:
    """Load configuration files on demand when deferred."""

    global _CONFIGS_LOADED

    if _CONFIGS_LOADED:
        return

    load_all_configs()
    _initialize_runtime_from_configs()
    _CONFIGS_LOADED = True


def setup(lazy_configs: bool = False) -> None:
    global DEFER_CONFIG_LOADING

    DEFER_CONFIG_LOADING = bool(lazy_configs)
    if DEFER_CONFIG_LOADING:
        _reset_config_state()
        return

    ensure_configs_loaded()


def step_agent(agent_name: str, server: dict) -> Optional[str]:
    # Pick up any PDV changes applied by UI/Discord before we compute/emit.
    _refresh_pdvs_from_disk()
    global CONTEXT
    os.makedirs("chatlogs", exist_ok=True)
    agent = AGENTS_BY_NAME[agent_name]
    server_id = str(server.get("id") or "")
    model_id, temp, system_text, pre, post = effective_params(agent)
    inject = get_one_time_inject()
    if inject:
        set_one_time_inject("")
        injected_block = (
            "\n\n----- Injected One-Time Message (will not persist) -----\n"
            f"{inject}\n"
            "----- End Injected One-Time Message -----"
        )
        post = (post or "") + injected_block
    # Ensure Directed Memories are appended as the very last context segment.
    try:
        _dm_block = directed_memory_block_for_agent(agent["name"], agent["agent_class"])
        if _dm_block:
            post = "\n\n".join(filter(None, [post, _dm_block]))
    except Exception:
        # Do not fail the run due to DM issues
        pass
    # When an agent reads the message queue, it must see ONLY the queue as its context.
    # No prior transcript or other context is included.
    reads_q = bool(CLASSES[agent["agent_class"]].get("reads_message_queue"))
    if reads_q:
        limit = int(GLOBALS.get("discord_history_limit", 10))
        msg = _discord_transcript(limit)
        if not msg.strip():
            logger.debug("Queue empty for %s; skipping generation", agent["name"])
            nxt = select_next_agent(agent_name)
            return nxt["name"] if nxt else None
    else:
        msg = CONTEXT
    msg = trim_message_for_budget(
        model_id,
        system_text,
        pre,
        msg,
        post,
        GLOBALS.get("max_context_tokens", 8192),
    )
    prompt = "\n".join(filter(None, [pre, msg, post]))
    if UI is not None:
        # Do not write the pre-gen “overview” blob to Agent Context.
        # The runtime_utils JSON watcher will overwrite the panel with the *exact*
        # payload that is POSTed to Ollama, which is what we want to display.
        try:
            UI.set_active_agent(server_id, agent["name"])
        except Exception:
            logger.exception("UI set_active_agent failed")
    try:
        reply = MODEL.generate_from_prompt(
            prompt,
            override_model=model_id,
            override_temperature=temp,
            system_text=system_text,
            server=server,
        )
    except OllamaServerError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Generation failed for %s: %s", agent["name"], exc)
        raise
    # Make the agent's *visible* output (with Fenra call markup stripped) available to Fenra functions.
    try:
        raw = reply or ""
        visible = re.sub(r"~.*?~", "", raw, flags=re.DOTALL).strip()
    except Exception:
        visible = (reply or "").strip()

    # Expose to fenra_functions via the conductor module namespace
    globals()["_LAST_VISIBLE_OUTPUT"] = visible
    # Execute any Fenra function calls emitted by the agent as ~...~ blocks.
    commands = _extract_pwsh_commands(reply)
    if commands:
        for cmd in commands:
            expr = (cmd or "").strip()
            fn_name, _found, result, params_string = fenra_functions.dispatch_expression(expr)
            name_display = fn_name or "<unknown>"
            params_display = params_string or ""
            header_line = f"\\-----{name_display} Output Begins-----/"
            footer_line = f"\\-----{name_display} Output Ends-----/"
            if not isinstance(result, str):
                result_text = json.dumps(result, ensure_ascii=False)
            else:
                result_text = result
            if params_display:
                header_line = f"\\-----{name_display}({params_display}) Output Begins-----/"
                footer_line = f"\\-----{name_display}({params_display}) Output Ends-----/"
            formatted_result = "\n".join([header_line, result_text, footer_line])
            reply = f"{reply.rstrip()}\n\n{formatted_result}" if reply else formatted_result

    cls = CLASSES[agent["agent_class"]]
    groups_target = list(agent.get("groups_out") or agent.get("groups_in") or [])
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Only post to Discord if the class (or agent) opts in AND the webhook is configured.
    should_post = bool(cls.get("outputs_to_discord") or agent.get("outputs_to_discord"))
    if should_post and os.getenv("DISCORD_WEBHOOK_URL"):
        post_to_discord_via_webhook(reply)
    # Preserve the running transcript when this was a queue-only read.
    if reads_q:
        CONTEXT = "\n".join(filter(None, [CONTEXT, reply]))
    else:
        CONTEXT = "\n".join(filter(None, [msg, reply]))
    text_block = f"[{timestamp}] {agent['name']}: {reply}\n{'-'*80}\n\n"
    if server_id:
        _append_server_context(server_id, text_block)
    for group in groups_target:
        path = os.path.join("chatlogs", f"chat_log_{group}.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(text_block)
    apply_pdv_adjustments(cls.get("pdv_adjustments", []), scale=len(reply))
    if cls.get("is_archivist"):
        targets = agent.get("groups_out") or [None]
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dst_dir = os.path.join("chatlogs", "summarized")
        os.makedirs(dst_dir, exist_ok=True)
        for grp in targets:
            if grp:
                src = os.path.join("chatlogs", f"chat_log_{grp}.txt")
            else:
                src = os.path.join("chatlogs", "context_current.txt")
            os.makedirs(os.path.dirname(src), exist_ok=True)
            if not os.path.exists(src):
                with open(src, "w", encoding="utf-8") as f:
                    f.write("")
            base = f"chat_log_{grp}_{ts}.txt" if grp else f"context_current_{ts}.txt"
            dst = os.path.join(dst_dir, base)
            shutil.copy(src, dst)
            with open(src, "w", encoding="utf-8") as f:
                f.write(reply)
        CONTEXT = reply
    with open(os.path.join("chatlogs", "context_current.txt"), "w", encoding="utf-8") as f:
        f.write(CONTEXT)
    # Token accounting should reflect the prompt that was sent.
    full_prompt = "\n".join(filter(None, [system_text, pre, msg, post]))
    try:
        used = len(tokenize_text(model_id, full_prompt))
    except Exception:
        used = len((full_prompt or "").split())
    with open(os.path.join("chatlogs", "token_usage.json"), "w", encoding="utf-8") as f:
        json.dump({"used": used, "limit": GLOBALS.get("max_context_tokens", 8192)}, f)
    if UI is not None:
        try:
            UI.log({"timestamp": timestamp, "sender": agent["name"], "message": reply})
            # Show the exact prompt that was sent (pre + msg + post), not CONTEXT.
            UI.update_agent_payload(
                server_id,
                agent["name"],
                {
                    "model": model_id,
                    "temperature": temp,
                    "system_text": system_text,
                    "pre_text": pre,
                    "post_text": post,
                    "prompt_tail": prompt[-4000:],
                },
            )
            UI.set_group_contexts(_read_group_contexts())
        except Exception:
            logger.exception("UI post-gen update failed")
    with STATE_LOCK:
        override = STATE.pop("force_next_agent", None)
    if isinstance(override, str) and override in AGENTS_BY_NAME:
        return override
    if used > GLOBALS.get("max_context_tokens", 8192):
        arch_cand = find_archivist_downstream(agent)
        if arch_cand:
            return arch_cand["name"]
    nxt = select_next_agent(agent_name)
    return nxt["name"] if nxt else None


def run_loop(steps: Optional[int] = None) -> None:
    _refresh_server_runners(steps)
    try:
        while True:
            with SERVER_THREAD_LOCK:
                active = [t for t in SERVER_THREADS.values() if t.is_alive()]
            if not active:
                break
            if steps is not None:
                limits = [SERVER_STEPS.get(sid) for sid in SERVER_THREADS]
                if not any(limit is None for limit in limits):
                    all_done = True
                    for sid, thread in list(SERVER_THREADS.items()):
                        limit = SERVER_STEPS.get(sid)
                        if limit is None:
                            all_done = False
                            break
                        if thread.is_alive():
                            all_done = False
                            break
                    if all_done:
                        break
            time.sleep(0.2)
    finally:
        if steps is not None:
            for sid in list(SERVER_THREADS):
                _stop_server_runner(sid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run a single agent step")
    parser.add_argument("--steps", type=int, default=None, help="Run N steps then exit")
    parser.add_argument("--ui", action="store_true", help="Launch Tk UI")
    args = parser.parse_args()
    setup(lazy_configs=args.ui)
    steps = 1 if args.once else args.steps
    if args.ui:
        try:
            UI = FenraUI(agents=AGENTS, on_apply_globals=apply_globals_update)

            def _ui_payload_watcher(p: dict) -> None:
                try:
                    server_id = p.get("__server") or DEFAULT_OLLAMA_SERVER["id"]
                    agent = None
                    with STATE_LOCK:
                        cur_map = STATE.get("current_agent_by_server")
                        if isinstance(cur_map, dict):
                            agent = cur_map.get(server_id)
                    if not agent:
                        agent = p.get("__agent")
                    if UI and agent:
                        payload = dict(p)
                        payload.pop("__agent", None)
                        payload.pop("__server", None)
                        payload.pop("__server_name", None)
                        UI.update_agent_payload(server_id, agent, payload)
                except Exception:
                    pass

            add_json_watcher(_ui_payload_watcher)

            cur = STATE.get("current_agent")
            servers = _server_list()
            server_choice = servers[0]["id"] if servers else DEFAULT_OLLAMA_SERVER["id"]
            if isinstance(cur, str) and cur in AGENTS_BY_NAME:
                UI.set_active_agent(server_choice, cur)
            UI.set_group_contexts(_read_group_contexts())

            def _loop() -> None:
                try:
                    run_loop(steps)
                except Exception:
                    logger.exception("Agent loop crashed")

            t = threading.Thread(target=_loop, daemon=True)
            t.start()
            UI.start()
        finally:
            UI = None
    else:
        # Non-UI mode should behave as before: start immediately.
        start_processing()
        run_loop(steps)
