import json
import sys
import time
from datetime import datetime
from typing import List, Dict, Tuple
import configparser
import threading
import os
import re
import shutil
import random
import math

import logging
import requests

from ai_model import Agent, Ruminator, Archivist, ToolAgent, Listener, Speaker
from fenra_ui import FenraUI
from runtime_utils import init_global_logging, parse_log_level, create_object_logger


logger = create_object_logger("Conductor")

TAGS_URL = "http://localhost:11434/api/tags"
PULL_URL = "http://localhost:11434/api/pull"


def load_config(path: str):
    """Parse fenra_config.txt and return instantiated agent objects."""
    logger.debug("Entering load_config path=%s", path)
    parser = configparser.ConfigParser()
    '''if not parser.read(path):
        raise RuntimeError(f"Failed to read config file {path}")'''
        
    with open(path, 'r', encoding='utf-8') as f:
        parser.read_file(f)

    logger.info("Loaded configuration from %s", path)

    sections = parser.sections()
    global_present = parser.has_section("global")
    topic_in_models = any(
        parser.has_option(sec, "topic_prompt") for sec in sections if sec != "global"
    )
    if not global_present and not topic_in_models:
        raise RuntimeError(
            "Config error: topic_prompt not found in global or model sections."
        )

    if global_present:
        topic_prompt_global = parser.get("global", "topic_prompt")
        temperature_global = parser.getfloat("global", "temperature", fallback=0.7)
        max_tokens_global_str = parser.get("global", "max_tokens", fallback=None)
        max_tokens_global = int(max_tokens_global_str) if max_tokens_global_str else None
        chat_style_global = parser.get("global", "chat_style", fallback=None)
        watchdog_global = parser.getint("global", "watchdog_timeout", fallback=300)
        debug_level_str = parser.get("global", "debug_level", fallback="INFO")
        init_global_logging(parse_log_level(debug_level_str))
    else:
        topic_prompt_global = None
        temperature_global = 0.7
        max_tokens_global = None
        chat_style_global = None
        watchdog_global = 300
        init_global_logging(logging.INFO)

    agents = []
    for section in sections:
        if section == "global":
            continue

        if not parser.has_option(section, "model"):
            raise RuntimeError(f"Config error: model missing for AI '{section}'")

        active = parser.getboolean(section, "active", fallback=True)
        if not active:
            continue

        model_id = parser.get(section, "model")
        role_prompt = parser.get(section, "role_prompt", fallback="")
        groups_str = parser.get(section, "groups", fallback="general")
        groups = [g.strip() for g in groups_str.split(',') if g.strip()]
        temperature = parser.getfloat(section, "temperature", fallback=temperature_global)
        max_tokens_str = parser.get(section, "max_tokens", fallback=None)
        if max_tokens_str is not None:
            max_tokens = int(max_tokens_str)
        else:
            max_tokens = max_tokens_global
        chat_style = parser.get(section, "chat_style", fallback=chat_style_global)
        watchdog = parser.getint(section, "watchdog_timeout", fallback=watchdog_global)
        system_prompt = parser.get(section, "system_prompt", fallback=None)

        topic_prompt = parser.get(section, "topic_prompt", fallback=topic_prompt_global)
        if topic_prompt is None:
            raise RuntimeError(
                f"Config error: topic_prompt missing for AI '{section}' and no global default."
            )

        role = parser.get(section, "role", fallback="ruminator").lower()

        cfg = {
            "topic_prompt": topic_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_style": chat_style,
            "watchdog_timeout": watchdog,
            "system_prompt": system_prompt,
        }

        if role == "archivist":
            agents.append(
                Archivist(
                    name=section,
                    model_name=model_id,
                    role_prompt=role_prompt,
                    config=cfg,
                    groups=groups,
                )
            )
        elif role in ("tool", "toolagent", "tools"):
            agent = ToolAgent(
                name=section,
                model_name=model_id,
                role_prompt=role_prompt,
                config=cfg,
                groups=groups,
            )
            agents.append(agent)
        elif role == "listener":
            agents.append(
                Listener(
                    name=section,
                    model_name=model_id,
                    role_prompt=role_prompt,
                    config=cfg,
                    groups=groups,
                )
            )
        elif role == "speaker":
            agents.append(
                Speaker(
                    name=section,
                    model_name=model_id,
                    role_prompt=role_prompt,
                    config=cfg,
                    groups=groups,
                )
            )
        else:
            agents.append(
                Ruminator(
                    name=section,
                    model_name=model_id,
                    role_prompt=role_prompt,
                    config=cfg,
                    groups=groups,
                )
            )

    if not agents:
        raise RuntimeError("No active AI models found in config.")

    logger.debug("Exiting load_config")
    return agents


def load_global_defaults(path: str) -> Dict[str, object]:
    """Return default model configuration from the global section."""
    logger.debug("Entering load_global_defaults path=%s", path)
    parser = configparser.ConfigParser()
    with open(path, "r", encoding="utf-8") as f:
        parser.read_file(f)

    if parser.has_section("global"):
        sec = parser["global"]
        max_tok_str = sec.get("max_tokens", fallback=None)
        max_tok_val = int(max_tok_str) if max_tok_str else None
        result = {
            "topic_prompt": sec.get("topic_prompt", ""),
            "temperature": sec.getfloat("temperature", fallback=0.7),
            "max_tokens": max_tok_val,
            "chat_style": sec.get("chat_style", fallback=None),
            "watchdog_timeout": sec.getint("watchdog_timeout", fallback=300),
            "system_prompt": sec.get("system_prompt", fallback=None),
        }
        logger.debug("Exiting load_global_defaults")
        return result

    result = {
        "topic_prompt": "",
        "temperature": 0.7,
        "max_tokens": None,
        "chat_style": None,
        "watchdog_timeout": 300,
        "system_prompt": None,
    }
    logger.debug("Exiting load_global_defaults")
    return result


def ensure_models_available(model_ids: List[str]) -> None:
    """Verify models are installed locally, pulling them if missing."""
    logger.debug("Entering ensure_models_available model_ids=%s", model_ids)
    try:
        resp = requests.get(TAGS_URL)
    except requests.RequestException as exc:
        logger.error("Error contacting Ollama server: %s", exc)
        sys.exit(1)
    if resp.status_code != 200:
        logger.error("Failed to list models: %s %s", resp.status_code, resp.text)
        sys.exit(1)

    try:
        tags_info = resp.json()
    except json.JSONDecodeError:
        logger.error("Invalid response from tags endpoint.")
        sys.exit(1)

    local_models = {m.get("name") for m in tags_info.get("models", [])}

    for mid in model_ids:
        if mid in local_models:
            continue
        logger.info("Model %s not found locally. Downloading...", mid)
        try:
            payload = {"name": mid, "stream": False}
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Payload to Ollama:\n%s", json.dumps(payload, indent=2))
            pull_resp = requests.post(
                PULL_URL,
                json=payload,
            )
        except requests.RequestException as exc:
            logger.error("Failed to pull model %s: %s", mid, exc)
            sys.exit(1)
        if pull_resp.status_code != 200:
            logger.error(
                "Error pulling model %s: %s %s", mid, pull_resp.status_code, pull_resp.text
            )
            sys.exit(1)
        try:
            result = pull_resp.json()
        except json.JSONDecodeError:
            logger.error("Unexpected response pulling model %s: %s", mid, pull_resp.text)
            sys.exit(1)
        status = result.get("status")
        if status != "success":
            logger.error("Model pull failed for %s: %s", mid, result)
            sys.exit(1)
    logger.debug("Exiting ensure_models_available")


def parse_model_ids(path: str) -> List[str]:
    """Return a list of model IDs for all active agents in the config."""
    logger.debug("Entering parse_model_ids path=%s", path)
    parser = configparser.ConfigParser()
    with open(path, "r", encoding="utf-8") as f:
        parser.read_file(f)

    ids: List[str] = []
    for section in parser.sections():
        if section == "global":
            continue
        if not parser.has_option(section, "model"):
            raise RuntimeError(f"Config error: model missing for AI '{section}'")
        active = parser.getboolean(section, "active", fallback=True)
        if not active:
            continue
        ids.append(parser.get(section, "model"))
    logger.debug("Exiting parse_model_ids with %s", ids)
    return ids


def iter_load_config(path: str):
    """Yield Agent objects from the configuration file one by one."""
    logger.debug("Entering iter_load_config path=%s", path)
    parser = configparser.ConfigParser()
    with open(path, "r", encoding="utf-8") as f:
        parser.read_file(f)

    sections = parser.sections()
    global_present = parser.has_section("global")
    topic_in_models = any(
        parser.has_option(sec, "topic_prompt") for sec in sections if sec != "global"
    )
    if not global_present and not topic_in_models:
        raise RuntimeError(
            "Config error: topic_prompt not found in global or model sections."
        )

    if global_present:
        topic_prompt_global = parser.get("global", "topic_prompt")
        temperature_global = parser.getfloat("global", "temperature", fallback=0.7)
        max_tokens_global_str = parser.get("global", "max_tokens", fallback=None)
        max_tokens_global = int(max_tokens_global_str) if max_tokens_global_str else None
        chat_style_global = parser.get("global", "chat_style", fallback=None)
        watchdog_global = parser.getint("global", "watchdog_timeout", fallback=300)
        debug_level_str = parser.get("global", "debug_level", fallback="INFO")
        init_global_logging(parse_log_level(debug_level_str))
    else:
        topic_prompt_global = None
        temperature_global = 0.7
        max_tokens_global = None
        chat_style_global = None
        watchdog_global = 300
        init_global_logging(logging.INFO)

    for section in sections:
        if section == "global":
            continue

        if not parser.has_option(section, "model"):
            raise RuntimeError(f"Config error: model missing for AI '{section}'")

        active = parser.getboolean(section, "active", fallback=True)
        if not active:
            continue

        model_id = parser.get(section, "model")
        role_prompt = parser.get(section, "role_prompt", fallback="")
        groups_str = parser.get(section, "groups", fallback="general")
        groups = [g.strip() for g in groups_str.split(',') if g.strip()]
        temperature = parser.getfloat(section, "temperature", fallback=temperature_global)
        max_tokens_str = parser.get(section, "max_tokens", fallback=None)
        if max_tokens_str is not None:
            max_tokens = int(max_tokens_str)
        else:
            max_tokens = max_tokens_global
        chat_style = parser.get(section, "chat_style", fallback=chat_style_global)
        watchdog = parser.getint(section, "watchdog_timeout", fallback=watchdog_global)
        system_prompt = parser.get(section, "system_prompt", fallback=None)

        topic_prompt = parser.get(section, "topic_prompt", fallback=topic_prompt_global)
        if topic_prompt is None:
            raise RuntimeError(
                f"Config error: topic_prompt missing for AI '{section}' and no global default."
            )

        role = parser.get(section, "role", fallback="ruminator").lower()

        cfg = {
            "topic_prompt": topic_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_style": chat_style,
            "watchdog_timeout": watchdog,
            "system_prompt": system_prompt,
        }

        if role == "archivist":
            yield Archivist(
                name=section,
                model_name=model_id,
                role_prompt=role_prompt,
                config=cfg,
                groups=groups,
            )
        elif role in ("tool", "toolagent", "tools"):
            yield ToolAgent(
                name=section,
                model_name=model_id,
                role_prompt=role_prompt,
                config=cfg,
                groups=groups,
            )
        elif role == "listener":
            yield Listener(
                name=section,
                model_name=model_id,
                role_prompt=role_prompt,
                config=cfg,
                groups=groups,
            )
        elif role == "speaker":
            yield Speaker(
                name=section,
                model_name=model_id,
                role_prompt=role_prompt,
                config=cfg,
                groups=groups,
            )
        else:
            yield Ruminator(
                name=section,
                model_name=model_id,
                role_prompt=role_prompt,
                config=cfg,
                groups=groups,
            )

    logger.debug("Exiting iter_load_config")


def _load_chat_history_for_group(path: str, group: str) -> List[Dict[str, str]]:
    """Return chat history parsed from a single group log."""
    logger.debug(
        "Entering _load_chat_history_for_group path=%s group=%s", path, group
    )
    history: List[Dict[str, str]] = []
    if not os.path.exists(path):
        return history

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
    except OSError as exc:
        logger.error("Failed to read %s: %s", path, exc)
        return history

    sep = "-" * 80
    pattern = re.compile(r"\[(.*?)\]\s*(.*?):\s*(.*)")
    blocks = data.split(f"\n{sep}\n\n")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if not lines:
            continue
        match = pattern.match(lines[0])
        if not match:
            continue
        ts, sender, first = match.groups()
        message = first
        if len(lines) > 1:
            message += "\n" + "\n".join(lines[1:])
        history.append(
            {"sender": sender, "timestamp": ts, "message": message, "groups": [group]}
        )



    logger.debug("Exiting _load_chat_history_for_group")
    return history


def load_all_chat_histories() -> List[Dict[str, str]]:
    """Load chat history from all chatlogs/chat_log_[group].txt files."""
    logger.debug("Entering load_all_chat_histories")
    history: List[Dict[str, str]] = []
    log_dir = "chatlogs"
    pattern = re.compile(r"chat_log_(.+)\.txt$")
    if os.path.isdir(log_dir):
        for fname in os.listdir(log_dir):
            m = pattern.match(fname)
            if m:
                group = m.group(1)
                path = os.path.join(log_dir, fname)
                history.extend(_load_chat_history_for_group(path, group))

    # Backwards compatibility with old location
    for fname in os.listdir('.'):
        m = pattern.match(fname)
        if m:
            group = m.group(1)
            history.extend(_load_chat_history_for_group(fname, group))

    if os.path.exists(os.path.join(log_dir, "chat_log.txt")):
        history.extend(
            _load_chat_history_for_group(os.path.join(log_dir, "chat_log.txt"), "general")
        )

    if os.path.exists("chat_log.txt"):
        history.extend(_load_chat_history_for_group("chat_log.txt", "general"))

    logger.debug("Exiting load_all_chat_histories")
    return history


def load_message_queue() -> List[Dict[str, object]]:
    """Return queued user messages from disk."""
    path = os.path.join("chatlogs", "queued_messages.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load queued messages: %s", exc)
    return []


def save_message_queue(queue: List[Dict[str, object]]) -> None:
    """Persist queued user messages to disk."""
    os.makedirs("chatlogs", exist_ok=True)
    path = os.path.join("chatlogs", "queued_messages.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(queue, f)
    except OSError as exc:
        logger.error("Failed to save queued messages: %s", exc)


def load_messages_to_humans() -> List[Dict[str, object]]:
    """Return past messages sent to humans from disk."""
    path = os.path.join("chatlogs", "messages_to_humans.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load messages to humans: %s", exc)
    return []


def save_messages_to_humans(messages: List[Dict[str, object]]) -> None:
    """Persist messages sent to humans to disk."""
    os.makedirs("chatlogs", exist_ok=True)
    path = os.path.join("chatlogs", "messages_to_humans.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f)
    except OSError as exc:
        logger.error("Failed to save messages to humans: %s", exc)


def append_human_log(entry: Dict[str, object]) -> None:
    """Append a text record of the message sent to humans."""
    os.makedirs("chatlogs", exist_ok=True)
    path = os.path.join("chatlogs", "messages_to_humans.log")
    text = f"[{entry['timestamp']}] {entry['sender']}: {entry['message']}\n{'-'*80}\n\n"
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
    except OSError as exc:
        logger.error("Failed to write human log: %s", exc)


def main() -> None:
    logger.debug("Entering main")
    logger.info("Starting conductor")
    config_path = "fenra_config.txt"
    load_global_defaults(config_path)
    model_ids = parse_model_ids(config_path)
    ensure_models_available(model_ids)

    agents: List[Agent] = []
    archivists: List[Archivist] = []
    listeners: List[Listener] = []
    speakers: List[Speaker] = []
    ruminators: List[Agent] = []
    all_groups: List[str] = []

    agent_lock = threading.Lock()
    ready_event = threading.Event()

    def loader() -> None:
        nonlocal all_groups
        for agent in iter_load_config(config_path):
            with agent_lock:
                agents.append(agent)
                if isinstance(agent, Archivist):
                    archivists.append(agent)
                elif isinstance(agent, Listener):
                    listeners.append(agent)
                elif isinstance(agent, Speaker):
                    speakers.append(agent)
                else:
                    ruminators.append(agent)
                all_groups = sorted({g for a in agents for g in a.groups})
                if archivists and listeners and speakers and ruminators:
                    ready_event.set()

    threading.Thread(target=loader, daemon=True).start()
    logger.info("Loading agents in background...")

    ready_event.wait()

    # At least one of each agent type loaded

    chat_log: List[Dict[str, str]] = load_all_chat_histories()
    inject_queue: List[Dict[str, str]] = []
    message_queue: List[Dict[str, object]] = load_message_queue()
    messages_to_humans: List[Dict[str, object]] = load_messages_to_humans()
    chat_lock = threading.Lock()

    def conversation_loop() -> None:
        logger.debug("Entering conversation_loop")
        while True:
            with chat_lock:
                if inject_queue:
                    pending = list(inject_queue)
                    inject_queue.clear()
                    chat_log.extend(pending)
                else:
                    pending = []
                with agent_lock:
                    active_listeners = [a for a in listeners if a.active]
                    active_ruminators = [a for a in ruminators if a.active]
                    active_archivists = [a for a in archivists if a.active]
                    active_speakers = [a for a in speakers if a.active]
                queue_empty = not message_queue

            for msg in pending:
                text = (
                    f"[{msg['timestamp']}] {msg['sender']}: {msg['message']}\n"
                    f"{'-' * 80}\n\n"
                )
                for group in msg.get("groups", ["general"]):
                    fname = os.path.join("chatlogs", f"chat_log_{group}.txt")
                    os.makedirs(os.path.dirname(fname), exist_ok=True)
                    with open(fname, "a", encoding="utf-8") as log_file:
                        log_file.write(text)
                print(text)
                ui.root.after(0, ui.log, msg)

            if not (active_listeners and active_ruminators and active_speakers):
                time.sleep(0.5)
                continue

            if queue_empty:
                available: List[Tuple[str, Agent]] = []
                for r in active_ruminators:
                    available.append(("ruminator", r))
                for a in active_archivists:
                    available.append(("archivist", a))
                for s in active_speakers:
                    available.append(("speaker", s))
                if not available:
                    time.sleep(0.5)
                    continue
                role, agent = random.choice(available)
                with chat_lock:
                    context = [
                        m
                        for m in chat_log
                        if set(m.get("groups", ["general"])) & set(agent.groups)
                    ]
                if role == "ruminator":
                    try:
                        r_reply = agent.step(context)
                    except requests.Timeout:
                        logger.error("%s timed out", agent.name)
                        time.sleep(0.5)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Error from %s: %s", agent.name, exc)
                        time.sleep(0.5)
                        continue
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with chat_lock:
                        entry = {
                            "sender": agent.name,
                            "timestamp": timestamp,
                            "message": r_reply,
                            "groups": agent.groups,
                            "epoch": time.time(),
                        }
                        chat_log.append(entry)
                    text = f"[{timestamp}] {agent.name}: {r_reply}\n{'-' * 80}\n\n"
                    for group in agent.groups:
                        fname = os.path.join("chatlogs", f"chat_log_{group}.txt")
                        os.makedirs(os.path.dirname(fname), exist_ok=True)
                        with open(fname, "a", encoding="utf-8") as log_file:
                            log_file.write(text)
                    print(text)
                    ui.root.after(0, ui.log, entry)
                    time.sleep(0.5)
                    continue

                if role == "archivist":
                    try:
                        summary = agent.step(context)
                    except requests.Timeout:
                        logger.error("%s timed out", agent.name)
                        time.sleep(0.5)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Error from %s: %s", agent.name, exc)
                        time.sleep(0.5)
                        continue
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if summary:
                        ts_display = timestamp
                        ts_file = datetime.now().strftime("%Y%m%d%H%M%S")
                        text = (
                            f"[{ts_display}] {agent.name} archived transcript and wrote summary.\n{'-' * 80}\n\n"
                        )
                        print(text)
                        ui.root.after(
                            0,
                            ui.log,
                            {
                                "sender": agent.name,
                                "timestamp": ts_display,
                                "message": "archived transcript and wrote summary.",
                                "groups": agent.groups,
                            },
                        )
                        summary_text = f"[{ts_display}] {agent.name}: {summary}\n{'-' * 80}\n\n"
                        for group in agent.groups:
                            fname = os.path.join("chatlogs", f"chat_log_{group}.txt")
                            if os.path.exists(fname):
                                os.makedirs(os.path.join("chatlogs", "summarized"), exist_ok=True)
                                dest = os.path.join(
                                    "chatlogs",
                                    "summarized",
                                    f"chat_log_{group}_{ts_file}.txt",
                                )
                                shutil.copy2(fname, dest)
                            os.makedirs(os.path.dirname(fname), exist_ok=True)
                            with open(fname, "w", encoding="utf-8") as log_file:
                                log_file.write(summary_text)
                        with chat_lock:
                            chat_log[:] = [
                                m
                                for m in chat_log
                                if not (
                                    set(m.get("groups", ["general"])) & set(agent.groups)
                                )
                            ]
                            entry = {
                                "sender": agent.name,
                                "timestamp": ts_display,
                                "message": summary,
                                "groups": agent.groups,
                                "epoch": time.time(),
                            }
                            chat_log.append(entry)
                    time.sleep(0.5)
                    continue

                if role == "speaker":
                    try:
                        s_reply = agent.step(context)
                    except requests.Timeout:
                        logger.error("%s timed out", agent.name)
                        time.sleep(0.5)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Error from %s: %s", agent.name, exc)
                        time.sleep(0.5)
                        continue
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with chat_lock:
                        entry = {
                            "sender": agent.name,
                            "timestamp": timestamp,
                            "message": s_reply,
                            "groups": agent.groups,
                            "epoch": time.time(),
                        }
                        chat_log.append(entry)
                        messages_to_humans.append(entry)
                        save_messages_to_humans(messages_to_humans)
                        append_human_log(entry)
                    text = f"[{timestamp}] {agent.name}: {s_reply}\n{'-' * 80}\n\n"
                    print(text)
                    for group in agent.groups:
                        fname = os.path.join("chatlogs", f"chat_log_{group}.txt")
                        os.makedirs(os.path.dirname(fname), exist_ok=True)
                        with open(fname, "a", encoding="utf-8") as log_file:
                            log_file.write(text)
                    ui.root.after(0, ui.log, entry)
                    ui.root.after(0, ui.update_sent, list(messages_to_humans))
                    time.sleep(0.5)
                    continue

            listener = random.choice(active_listeners)
            with chat_lock:
                msg = message_queue.pop(0)
                save_message_queue(message_queue)
                ui.root.after(0, ui.update_queue, list(message_queue))
            payload_message = msg["message"]

            prev_groups = listener.groups
            num_rums = random.randint(2, 4)
            selected_rums = []
            for _ in range(num_rums):
                candidates = [r for r in active_ruminators if set(r.groups) & set(prev_groups)]
                if not candidates:
                    candidates = active_ruminators
                rum = random.choice(candidates)
                selected_rums.append(rum)
                prev_groups = rum.groups

            if active_archivists:
                a_candidates = [
                    a for a in active_archivists if set(a.groups) & set(prev_groups)
                ]
                if not a_candidates:
                    a_candidates = active_archivists
                archivist_ai = random.choice(a_candidates)
                prev_groups = archivist_ai.groups
            else:
                archivist_ai = None

            candidates = [s for s in active_speakers if set(s.groups) & set(prev_groups)]
            if not candidates:
                candidates = active_speakers
            speaker_ai = random.choice(candidates)

            chain_names = [listener.name] + [r.name for r in selected_rums]
            if archivist_ai:
                chain_names.append(archivist_ai.name)
            chain_names.append(speaker_ai.name)
            logger.info("Agent chain: %s", " > ".join(chain_names))

            lines = [
                f"[{m['timestamp']}] {m['sender']}: {m['message']}" for m in messages_to_humans
            ]
            ruminations = "\n".join(lines)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                reply = listener.prompt_ais(ruminations, payload_message)
            except requests.Timeout:
                logger.error("%s timed out", listener.name)
                time.sleep(0.5)
                continue
            entry = {
                "sender": listener.name,
                "timestamp": timestamp,
                "message": reply,
                "groups": listener.groups,
                "epoch": time.time(),
            }
            with chat_lock:
                chat_log.append(entry)
            text = f"[{timestamp}] {listener.name}: {reply}\n{'-' * 80}\n\n"
            for group in listener.groups:
                fname = os.path.join("chatlogs", f"chat_log_{group}.txt")
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                with open(fname, "a", encoding="utf-8") as log_file:
                    log_file.write(text)
            print(text)
            ui.root.after(0, ui.log, entry)

            for rum in selected_rums:
                with chat_lock:
                    context = [
                        m
                        for m in chat_log
                        if set(m.get("groups", ["general"])) & set(rum.groups)
                    ]
                try:
                    r_reply = rum.step(context)
                except requests.Timeout:
                    logger.error("%s timed out", rum.name)
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.error("Error from %s: %s", rum.name, exc)
                    continue
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with chat_lock:
                    entry = {
                        "sender": rum.name,
                        "timestamp": timestamp,
                        "message": r_reply,
                        "groups": rum.groups,
                        "epoch": time.time(),
                    }
                    chat_log.append(entry)
                text = f"[{timestamp}] {rum.name}: {r_reply}\n{'-' * 80}\n\n"
                for group in rum.groups:
                    fname = os.path.join("chatlogs", f"chat_log_{group}.txt")
                    os.makedirs(os.path.dirname(fname), exist_ok=True)
                    with open(fname, "a", encoding="utf-8") as log_file:
                        log_file.write(text)
                print(text)
                ui.root.after(0, ui.log, entry)

            if archivist_ai:
                with chat_lock:
                    context = [
                        m
                        for m in chat_log
                        if set(m.get("groups", ["general"])) & set(archivist_ai.groups)
                    ]
                try:
                    summary = archivist_ai.step(context)
                except requests.Timeout:
                    logger.error("%s timed out", archivist_ai.name)
                    summary = ""
                except Exception as exc:  # noqa: BLE001
                    logger.error("Error from %s: %s", archivist_ai.name, exc)
                    summary = ""
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if summary:
                    ts_display = timestamp
                    ts_file = datetime.now().strftime("%Y%m%d%H%M%S")
                    text = (
                        f"[{ts_display}] {archivist_ai.name} archived transcript and wrote summary.\n{'-' * 80}\n\n"
                    )
                    print(text)
                    ui.root.after(
                        0,
                        ui.log,
                        {
                            "sender": archivist_ai.name,
                            "timestamp": ts_display,
                            "message": "archived transcript and wrote summary.",
                            "groups": archivist_ai.groups,
                        },
                    )
                    summary_text = f"[{ts_display}] {archivist_ai.name}: {summary}\n{'-' * 80}\n\n"
                    for group in archivist_ai.groups:
                        fname = os.path.join("chatlogs", f"chat_log_{group}.txt")
                        if os.path.exists(fname):
                            os.makedirs(os.path.join("chatlogs", "summarized"), exist_ok=True)
                            dest = os.path.join(
                                "chatlogs",
                                "summarized",
                                f"chat_log_{group}_{ts_file}.txt",
                            )
                            shutil.copy2(fname, dest)
                        os.makedirs(os.path.dirname(fname), exist_ok=True)
                        with open(fname, "w", encoding="utf-8") as log_file:
                            log_file.write(summary_text)
                    with chat_lock:
                        chat_log[:] = [
                            m
                            for m in chat_log
                            if not (
                                set(m.get("groups", ["general"])) & set(archivist_ai.groups)
                            )
                        ]
                        entry = {
                            "sender": archivist_ai.name,
                            "timestamp": ts_display,
                            "message": summary,
                            "groups": archivist_ai.groups,
                            "epoch": time.time(),
                        }
                        chat_log.append(entry)

            with chat_lock:
                context = [
                    m
                    for m in chat_log
                    if set(m.get("groups", ["general"])) & set(speaker_ai.groups)
                ]
            try:
                s_reply = speaker_ai.step(context)
            except requests.Timeout:
                logger.error("%s timed out", speaker_ai.name)
                time.sleep(0.5)
                continue
            except Exception as exc:  # noqa: BLE001
                logger.error("Error from %s: %s", speaker_ai.name, exc)
                time.sleep(0.5)
                continue
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with chat_lock:
                entry = {
                    "sender": speaker_ai.name,
                    "timestamp": timestamp,
                    "message": s_reply,
                    "groups": speaker_ai.groups,
                    "epoch": time.time(),
                }
                chat_log.append(entry)
                messages_to_humans.append(entry)
                save_messages_to_humans(messages_to_humans)
                append_human_log(entry)
            text = f"[{timestamp}] {speaker_ai.name}: {s_reply}\n{'-' * 80}\n\n"
            print(text)
            for group in speaker_ai.groups:
                fname = os.path.join("chatlogs", f"chat_log_{group}.txt")
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                with open(fname, "a", encoding="utf-8") as log_file:
                    log_file.write(text)
            ui.root.after(0, ui.log, entry)
            ui.root.after(0, ui.update_sent, list(messages_to_humans))

            time.sleep(0.5)
        logger.debug("Exiting conversation_loop")

    ui = FenraUI(agents, inject_callback=None, send_callback=None)

    def send_message(message: str) -> None:
        logger.debug("Entering send_message message=%s", message)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {"message": message, "timestamp": ts, "epoch": time.time()}
        with chat_lock:
            message_queue.append(entry)
            save_message_queue(message_queue)
        ui.root.after(0, ui.update_queue, list(message_queue))
        logger.debug("Exiting send_message")

    def inject_message(group: str, message: str) -> None:
        logger.debug("Entering inject_message group=%s message=%s", group, message)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        target_groups = all_groups if group == "All Groups" else [group]
        with chat_lock:
            inject_queue.append(
                {
                    "sender": "Human",
                    "timestamp": timestamp,
                    "message": message,
                    "groups": target_groups,
                }
            )
        logger.debug("Exiting inject_message")

    ui.inject_callback = inject_message
    ui.send_callback = send_message
    ui.update_queue(message_queue)
    ui.update_sent(messages_to_humans)

    t = threading.Thread(target=conversation_loop, daemon=True)
    t.start()

    ui.start()
    logger.debug("Exiting main")


if __name__ == "__main__":
    main()
