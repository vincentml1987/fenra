import os
import json
import time
import random
import shutil
import argparse
import threading
from datetime import datetime
from typing import Dict, List, Optional, Iterable

import requests
from fenra_ui import FenraUI

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
)

logger = create_object_logger("Conductor")

TAGS_URL = "http://localhost:11434/api/tags"
PULL_URL = "http://localhost:11434/api/pull"
QUEUE_PATH = os.path.join("chatlogs", "queued_messages.json")

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

UI: Optional[FenraUI] = None


def load_all_configs() -> None:
    """Load global config data into module-level structures."""
    global GLOBALS, PDV_META, PDVS, CLASSES, AGENTS, AGENTS_BY_NAME, AGENTS_BY_GROUP_IN, STATE
    GLOBALS = load_globals()
    raw_pdvs = load_pdvs()
    PDV_META.clear()
    PDV_META.update(raw_pdvs)
    PDVS = {name: cfg.get("value", 0.5) for name, cfg in raw_pdvs.items()}
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


def effective_params(agent: dict):
    cls = CLASSES[agent["agent_class"]]
    model = agent.get("model") or cls.get("model") or GLOBALS.get("model")
    temp = agent.get("temperature")
    if temp is None:
        temp = cls.get("temperature")
    if temp is None:
        temp = GLOBALS.get("temperature")
    system_text = "\n".join(
        [
            GLOBALS.get("system_prompt", ""),
            cls.get("system_prompt", ""),
            agent.get("system_prompt", ""),
        ]
    ).strip()
    pre_text = "\n".join(
        [
            GLOBALS.get("pre_context_message", ""),
            cls.get("pre_context_message", ""),
            agent.get("pre_context_message", ""),
        ]
    ).strip()
    post_text = "\n".join(
        [
            GLOBALS.get("post_context_message", ""),
            cls.get("post_context_message", ""),
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

def apply_pdv_adjustments(adjs: List[dict]) -> None:
    gamma = float(GLOBALS.get("pdv_gamma", 2.0))
    changed = False
    for adj in adjs:
        name = adj["name"]
        if name not in PDVS:
            PDV_META.setdefault(name, {"name": name, "description": "", "value": 0.5})
            PDVS[name] = PDV_META[name].get("value", 0.5)
            changed = True
        m = float(adj.get("delta", 0.0))
        x = float(PDVS.get(name, 0.5))
        g = (4 * x * (1 - x)) ** gamma
        x2 = min(1.0, max(0.0, x + m * g))
        if abs(x2 - x) > 1e-9:
            PDVS[name] = x2
            PDV_META.setdefault(name, {"name": name, "description": ""})
            PDV_META[name]["value"] = x2
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
        with open("chatlogs/pdvs_live.json", "w", encoding="utf-8") as f:
            json.dump(PDVS, f)


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
    D = downstream_candidates(curr_name)
    if not D:
        cur = AGENTS_BY_NAME[curr_name]
        _flag_no_downstream(cur, cur.get("groups_out", []))
        return None
    pdvs = {CLASSES[a["agent_class"]]["triggering_pdv"] for a in D}
    target = max(pdvs, key=lambda p: PDVS.get(p, 0.0))
    C = [a for a in D if CLASSES[a["agent_class"]]["triggering_pdv"] == target] or D
    random.shuffle(C)
    for cand in C:
        outs = set(cand.get("groups_out", []))
        consumers = [b for b in AGENTS if b["name"] != cand["name"] and outs & set(b.get("groups_in", []))]
        if consumers:
            return cand
        _flag_no_downstream(cand, outs)
    cur = AGENTS_BY_NAME[curr_name]
    _flag_no_downstream(cur, cur.get("groups_out", []))
    return None


def merge_and_clear_queue(path: str = QUEUE_PATH) -> str:
    """Merge queued messages into newline text and clear the queue."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        items = []
    items = items or []
    lines: List[str] = []
    for it in sorted(items, key=lambda x: x.get("timestamp", "")):
        sender = it.get("sender", "user")
        msg = it.get("message", "")
        ts = it.get("timestamp", "")
        lines.append(f"[{ts}] {sender}: {msg}")
    if items:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f)
        except Exception:
            pass
    return "\n".join(lines)


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


def setup() -> None:
    load_all_configs()
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
    MODEL = AIModel(
        name="fenra",
        model_id=base_model,
        topic_prompt="",
        role_prompt="",
        temperature=float(GLOBALS.get("temperature", 0.7)),
        max_tokens=int(GLOBALS.get("max_context_tokens", 8192)),
        system_prompt=GLOBALS.get("system_prompt", ""),
    )
    try:
        with open(os.path.join("chatlogs", "context_current.txt"), "r", encoding="utf-8") as f:
            CONTEXT = f.read()
    except FileNotFoundError:
        CONTEXT = ""


def step_agent(agent_name: str) -> Optional[str]:
    global CONTEXT
    os.makedirs("chatlogs", exist_ok=True)
    agent = AGENTS_BY_NAME[agent_name]
    model_id, temp, system_text, pre, post = effective_params(agent)
    msg = CONTEXT
    if CLASSES[agent["agent_class"]].get("reads_message_queue"):
        q = merge_and_clear_queue()
        msg = "\n".join(filter(None, [msg, q]))
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
        try:
            UI.set_active_agent(agent["name"])
            UI.update_agent_payload(
                agent["name"],
                {
                    "model": model_id,
                    "temperature": temp,
                    "system_text": system_text,
                    "pre_text": pre,
                    "post_text": post,
                    "message_preview": (msg or "")[-2000:],
                },
            )
        except Exception:
            logger.exception("UI payload pre-gen update failed")
    reply = MODEL.generate_from_prompt(
        prompt,
        override_model=model_id,
        override_temperature=temp,
        system_text=system_text,
    )
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
    CONTEXT = "\n".join(filter(None, [msg, f"{agent['name']}: {reply}"]))
    text_block = f"[{timestamp}] {agent['name']}: {reply}\n{'-'*80}\n\n"
    for group in groups_target:
        path = os.path.join("chatlogs", f"chat_log_{group}.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(text_block)
    apply_pdv_adjustments(cls.get("pdv_adjustments", []))
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
    full_prompt = "\n".join(filter(None, [system_text, pre, CONTEXT, post]))
    try:
        used = len(tokenize_text(model_id, full_prompt))
    except Exception:
        used = len((full_prompt or "").split())
    with open(os.path.join("chatlogs", "token_usage.json"), "w", encoding="utf-8") as f:
        json.dump({"used": used, "limit": GLOBALS.get("max_context_tokens", 8192)}, f)
    if UI is not None:
        try:
            UI.log({"timestamp": timestamp, "sender": agent["name"], "message": reply})
            full_prompt = "\n".join(filter(None, [system_text, pre, CONTEXT, post]))
            UI.update_agent_payload(
                agent["name"],
                {
                    "model": model_id,
                    "temperature": temp,
                    "system_text": system_text,
                    "pre_text": pre,
                    "post_text": post,
                    "prompt_tail": full_prompt[-4000:],
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
    cur = STATE["current_agent"]
    hist: List[str] = [cur]
    count = 0
    while steps is None or count < steps:
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
    setup()
    steps = 1 if args.once else args.steps
    if args.ui:
        try:
            UI = FenraUI(agents=AGENTS)
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
        run_loop(steps)
