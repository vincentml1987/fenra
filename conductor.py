import os
import json
import time
import random
import shutil
import argparse
import threading
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
)

logger = create_object_logger("Conductor")

TAGS_URL = "http://localhost:11434/api/tags"
PULL_URL = "http://localhost:11434/api/pull"

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
    """Return True if the span has no leading or trailing whitespace."""

    if not isinstance(span, str) or not span:
        return False
    return not span[0].isspace() and not span[-1].isspace()


def _extract_pwsh_commands(text: str) -> list[str]:
    """Extract *~...~* call spans that keep whitespace away from the markers."""

    spans = re.findall(r"\*~(.*?)~\*", text or "", flags=re.DOTALL)
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

# Queue deprecated. Kept for compatibility but unused.
_INCOMING_QUEUE: List[Dict[str, object]] = []

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


def _queue_empty() -> bool:
    return len(_INCOMING_QUEUE) == 0


async def inject_external_message(text: str, meta: dict | None = None):
    """Accept an externally sourced message and enqueue it for processing."""
    meta = meta or {}
    entry = {
        "timestamp": meta.get("timestamp")
        or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "sender": meta.get("author") or meta.get("sender") or "user",
        "message": text,
        "raw_message": text,
    }
    _INCOMING_QUEUE.append(entry)
    if UI is not None:
        try:
            UI.update_queue(_INCOMING_QUEUE)
        except Exception:
            pass


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

def apply_pdv_adjustments(adjs: List[dict], *, scale: float = 1.0) -> None:
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
    cur = AGENTS_BY_NAME[curr_name]
    outs = set(cur.get("groups_out", []))
    return [a for a in AGENTS if a["name"] != curr_name and outs & set(a.get("groups_in", []))]


def _flag_no_downstream(agent: dict, groups: Iterable[str]) -> None:
    agent["flag_no_downstream"] = True
    agent["missing_out_groups"] = list(groups)
    save_agents(AGENTS)


def select_next_agent(curr_name: str) -> Optional[dict]:
    """Select the next agent using weighted randomness over downstream classes."""

    D = downstream_candidates(curr_name)
    if not D:
        cur = AGENTS_BY_NAME[curr_name]
        _flag_no_downstream(cur, cur.get("groups_out", []))
        return None

    class_to_agents: Dict[str, List[dict]] = {}
    for agent in D:
        class_to_agents.setdefault(agent["agent_class"], []).append(agent)

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

    attempted: set[str] = set()
    while len(attempted) < len(classes):
        chosen_class = _weighted_choice(classes, weights)
        attempted.add(chosen_class)
        candidates = list(class_to_agents.get(chosen_class, []))
        random.shuffle(candidates)
        for cand in candidates:
            outs = set(cand.get("groups_out", []))
            consumers = [
                other
                for other in AGENTS
                if other["name"] != cand["name"] and outs & set(other.get("groups_in", []))
            ]
            if consumers:
                return cand
            _flag_no_downstream(cand, outs)
        idx = classes.index(chosen_class)
        weights[idx] = 0.0

    cur = AGENTS_BY_NAME[curr_name]
    _flag_no_downstream(cur, cur.get("groups_out", []))
    return None


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


def _load_messages_to_humans(path: str = os.path.join("chatlogs", "messages_to_humans.json")) -> List[Dict[str, object]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_messages_to_humans(items: List[Dict[str, object]], path: str = os.path.join("chatlogs", "messages_to_humans.json")) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


def _append_human_log(entry: Dict[str, object], path: str = os.path.join("chatlogs", "messages_to_humans.log")) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = f"[{entry.get('timestamp','')}] {entry.get('sender','')}: {entry.get('message','')}\n{'-'*80}\n\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


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


def step_agent(agent_name: str) -> Optional[str]:
    # Pick up any PDV changes applied by UI/Discord before we compute/emit.
    _refresh_pdvs_from_disk()
    global CONTEXT
    os.makedirs("chatlogs", exist_ok=True)
    agent = AGENTS_BY_NAME[agent_name]
    model_id, temp, system_text, pre, post = effective_params(agent)
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
           UI.set_active_agent(agent["name"])
       except Exception:
           logger.exception("UI set_active_agent failed")
    try:
        reply = MODEL.generate_from_prompt(
            prompt,
            override_model=model_id,
            override_temperature=temp,
            system_text=system_text,
        )
    except Exception as exc:  # keep loop alive on Ollama/network errors
        logger.exception("Generation failed for %s: %s", agent["name"], exc)
        nxt = select_next_agent(agent_name)
        return nxt["name"] if nxt else None
    # Execute any Fenra function calls emitted by the agent as *~...~* blocks.
    commands = _extract_pwsh_commands(reply)
    if commands:
        for cmd in commands:
            expr = (cmd or "").strip()
            fn_name, _found, result = fenra_functions.dispatch_expression(expr)
            if UI is not None and hasattr(UI, "append_ps"):
                try:
                    UI.append_ps(f"Function called: {fn_name}")
                    UI.append_ps(f"Function result: {result}")
                except Exception:
                    logger.exception("UI append_ps failed for function dispatch")
            reply += f"\nFunction called: {fn_name}\nFunction result: {result}\n"

    cls = CLASSES[agent["agent_class"]]
    groups_target = list(agent.get("groups_out") or agent.get("groups_in") or [])
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Always record the message to the “humans” log so the UI shows it.
    entry = {
        "sender": agent["name"],
        "timestamp": timestamp,
        "message": reply,
        "groups": groups_target,
    }
    msgs = _load_messages_to_humans()
    msgs.append(entry)
    _save_messages_to_humans(msgs)
    _append_human_log(entry)

    # Only post to Discord if the class (or agent) opts in AND the webhook is configured.
    should_post = bool(cls.get("outputs_to_discord") or agent.get("outputs_to_discord"))
    if should_post and os.getenv("DISCORD_WEBHOOK_URL"):
        post_to_discord_via_webhook(reply)
    # Preserve the running transcript when this was a queue-only read.
    if reads_q:
        CONTEXT = "\n".join(filter(None, [CONTEXT, f"{agent['name']}: {reply}"]))
    else:
        CONTEXT = "\n".join(filter(None, [msg, f"{agent['name']}: {reply}"]))
    text_block = f"[{timestamp}] {agent['name']}: {reply}\n{'-'*80}\n\n"
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
    if used > GLOBALS.get("max_context_tokens", 8192):
        arch_cand = find_archivist_downstream(agent)
        if arch_cand:
            return arch_cand["name"]
    nxt = select_next_agent(agent_name)
    return nxt["name"] if nxt else None


def run_loop(steps: Optional[int] = None) -> None:
    # Defer reading state until Start is pressed and configs are loaded.
    cur: Optional[str] = None
    hist: List[str] = []
    count = 0
    while steps is None or count < steps:
        # Respect Start/Stop toggle: idle until started.
        if not _RUN_EVENT.is_set():
            time.sleep(0.1)
            continue
        # First tick after Start: ensure configs are present and seed current agent.
        if not _CONFIGS_LOADED:
            ensure_configs_loaded()
        if cur is None:
            candidate = STATE.get("current_agent")
            if isinstance(candidate, str) and candidate in AGENTS_BY_NAME:
                cur = candidate
            else:
                # Pick a reasonable default or bail with a clear error.
                cur = next(iter(AGENTS_BY_NAME), None)
            if cur is None:
                logger.error("No current_agent available after config load; check confs/state.json and agents.")
                return
            hist = [cur]
        if UI is not None:
            try:
                UI.set_active_agent(cur)
                UI.update_topology(AGENTS_BY_NAME[cur], AGENTS)
                UI.set_group_contexts(_read_group_contexts())
            except Exception:
                logger.exception("UI pre-step update failed")
        logger.info("Running agent %s", cur)
        nxt = step_agent(cur)
        logger.info("Next agent: %s", nxt)
        count += 1
        time.sleep(0.2)
        if nxt:
            cur = nxt
            STATE["current_agent"] = cur
            save_state(STATE)
            hist.append(cur)
            continue
        # flag current agent as dead-end and backtrack
        cur_agent = AGENTS_BY_NAME.get(cur)
        if cur_agent:
            _flag_no_downstream(cur_agent, cur_agent.get("groups_out", []))
        while hist:
            dead = hist.pop()
            if not hist:
                logger.error("All downstream paths dead-end. Please wire groups.")
                return
            prev = hist[-1]
            alt = select_next_agent(prev)
            if alt and alt["name"] != dead:
                cur = alt["name"]
                STATE["current_agent"] = cur
                save_state(STATE)
                hist.append(cur)
                break


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
                    # Prefer the conductor's current agent; fall back to payload tag.
                    agent = STATE.get("current_agent") or p.get("__agent")
                    if UI and agent:
                        payload = dict(p)
                        payload.pop("__agent", None)
                        UI.update_agent_payload(agent, payload)
                except Exception:
                    pass

            add_json_watcher(_ui_payload_watcher)

            cur = STATE.get("current_agent")
            if isinstance(cur, str) and cur in AGENTS_BY_NAME:
                UI.set_active_agent(cur)
                UI.update_topology(AGENTS_BY_NAME[cur], AGENTS)
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
