import json
import sys
import time
from datetime import datetime
from typing import List, Dict
import configparser
import threading
import os
import re
import shutil
import random
import math

import logging
import requests

from ai_model import Ruminator, Archivist, ToolAgent, Listener
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
        max_tokens_global = parser.getint("global", "max_tokens", fallback=300)
        chat_style_global = parser.get("global", "chat_style", fallback=None)
        watchdog_global = parser.getint("global", "watchdog_timeout", fallback=300)
        debug_level_str = parser.get("global", "debug_level", fallback="INFO")
        init_global_logging(parse_log_level(debug_level_str))
    else:
        topic_prompt_global = None
        temperature_global = 0.7
        max_tokens_global = 300
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
        max_tokens = parser.getint(section, "max_tokens", fallback=max_tokens_global)
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
        result = {
            "topic_prompt": sec.get("topic_prompt", ""),
            "temperature": sec.getfloat("temperature", fallback=0.7),
            "max_tokens": sec.getint("max_tokens", fallback=300),
            "chat_style": sec.get("chat_style", fallback=None),
            "watchdog_timeout": sec.getint("watchdog_timeout", fallback=300),
            "system_prompt": sec.get("system_prompt", fallback=None),
        }
        logger.debug("Exiting load_global_defaults")
        return result

    result = {
        "topic_prompt": "",
        "temperature": 0.7,
        "max_tokens": 300,
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


def _load_chat_history_for_group(path: str, group: str) -> List[Dict[str, str]]:
    """Return chat history parsed from a single group log and archive it."""
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

    try:
        os.makedirs("chatlogs", exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        base = os.path.basename(path)
        dest = os.path.join("chatlogs", f"{base}-{stamp}")
        shutil.copy2(path, dest)
    except OSError as exc:
        logger.error("Failed to archive chat log %s: %s", path, exc)

    logger.debug("Exiting _load_chat_history_for_group")
    return history


def load_all_chat_histories() -> List[Dict[str, str]]:
    """Load chat history from all chat_log_[group].txt files."""
    logger.debug("Entering load_all_chat_histories")
    history: List[Dict[str, str]] = []
    pattern = re.compile(r"chat_log_(.+)\.txt$")
    for fname in os.listdir('.'):
        m = pattern.match(fname)
        if m:
            group = m.group(1)
            history.extend(_load_chat_history_for_group(fname, group))

    # Backwards compatibility with single chat_log.txt file
    if os.path.exists("chat_log.txt"):
        history.extend(_load_chat_history_for_group("chat_log.txt", "general"))

    logger.debug("Exiting load_all_chat_histories")
    return history


def main() -> None:
    logger.debug("Entering main")
    logger.info("Starting conductor")
    config_path = "fenra_config.txt"
    agents = load_config(config_path)
    defaults = load_global_defaults(config_path)
    ensure_models_available([a.model_name for a in agents])

    archivists = [a for a in agents if isinstance(a, Archivist)]
    archivist = archivists[0] if archivists else None
    listeners = [a for a in agents if isinstance(a, Listener)]
    participants = [a for a in agents if a not in archivists + listeners]
    all_groups = sorted({g for a in agents for g in a.groups})

    chat_log: List[Dict[str, str]] = load_all_chat_histories()
    inject_queue: List[Dict[str, str]] = []
    message_queue: List[Dict[str, object]] = []
    sent_messages: List[Dict[str, object]] = []
    messages_to_humans: List[Dict[str, object]] = []
    chat_lock = threading.Lock()
    threads: List[threading.Thread] = []

    def conversation_loop() -> None:
        logger.debug("Entering conversation_loop")
        msg_count = 0
        listener_counter = 0
        BASE_LOOPS = 50
        while True:
            
            pending: List[Dict[str, str]] = []
            with chat_lock:
                if inject_queue:
                    pending = list(inject_queue)
                    inject_queue.clear()
                    chat_log.extend(pending)
                active_participants = [a for a in participants if a.active]
                active_archivists = [a for a in archivists if a.active]
                active_listeners = [a for a in listeners if a.active]
                current_log = list(chat_log)
                current_queue = list(message_queue)
                current_sent = list(sent_messages)
                current_human = list(messages_to_humans)
            for msg in pending:
                text = (
                    f"[{msg['timestamp']}] {msg['sender']}: {msg['message']}\n"
                    f"{'-' * 80}\n\n"
                )
                for group in msg.get("groups", ["general"]):
                    fname = f"chat_log_{group}.txt"
                    with open(fname, "a", encoding="utf-8") as log_file:
                        log_file.write(text)
                print(text)
                ui.root.after(0, ui.log, text)

            if current_queue and active_listeners:
                now_ms = time.time() * 1000
                oldest_age = now_ms - current_queue[0]["epoch"] * 1000
                newest_age = now_ms - current_queue[-1]["epoch"] * 1000
                count = len(current_queue)
                priority = count * (oldest_age / (newest_age + 1))
                if listener_counter <= 0:
                    listener_ai = random.choice(active_listeners)
                    msg = current_queue[0]
                    outputs = [m["message"] for m in current_human if m["epoch"] >= msg["epoch"]]
                    print(outputs)
                    if listener_ai.check_answered(msg["message"], outputs):
                        with chat_lock:
                            message_queue.pop(0)
                        ui.root.after(0, ui.update_queue, list(message_queue))
                    else:
                        lines = [
                            f"[{m['timestamp']}] {m['sender']}: {m['message']}" for m in current_log
                        ]
                        ruminations = "\n".join(lines)
                        reply = listener_ai.prompt_ais(ruminations, msg["message"])
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        entry = {
                            "sender": listener_ai.name,
                            "timestamp": ts,
                            "message": reply,
                            "groups": all_groups,
                            "epoch": time.time(),
                        }
                        with chat_lock:
                            chat_log.append(entry)
                            sent_messages.append(entry)
                        text = f"[{ts}] {listener_ai.name}: {reply}\n{'-' * 80}\n\n"
                        for group in all_groups:
                            fname = f"chat_log_{group}.txt"
                            with open(fname, "a", encoding="utf-8") as log_file:
                                log_file.write(text)
                        print(text)
                        ui.root.after(0, ui.log, text)
                        
                    listener_counter = max(1, math.ceil(BASE_LOOPS / priority)) if priority > 0 else BASE_LOOPS
                else:
                    listener_counter -= 1
                
            active_choices = active_participants + active_archivists
            if not active_choices:
                time.sleep(0.5)
                continue
            ai = random.choice(active_choices)
            context = [
                m
                for m in current_log
                if set(m.get("groups", ["general"])) & set(ai.groups)
            ]
            try:
                result = ai.step(context)
            except requests.Timeout:
                logger.error("%s timed out", ai.name)
                continue
            except Exception as exc:  # noqa: BLE001
                logger.error("Error from %s: %s", ai.name, exc)
                continue

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if isinstance(ai, Archivist):
                summary = result
                logger.info("%s archived transcript", ai.name)
                ts_display = timestamp
                ts_file = datetime.now().strftime("%Y%m%d%H%M%S")
                text = (
                    f"[{ts_display}] {ai.name} archived transcript and wrote summary.\n{'-' * 80}\n\n"
                )
                print(text)
                ui.root.after(0, ui.log, text)
                summary_text = f"[{ts_display}] {ai.name}: {summary}\n{'-' * 80}\n\n"
                for group in ai.groups:
                    fname = f"chat_log_{group}.txt"
                    if os.path.exists(fname):
                        os.makedirs("chatlogs", exist_ok=True)
                        dest = os.path.join(
                            "chatlogs",
                            f"chat_log_{group}_summarized_{ts_file}.txt",
                        )
                        shutil.copy2(fname, dest)
                    with open(fname, "w", encoding="utf-8") as log_file:
                        log_file.write(summary_text)
                with chat_lock:
                    chat_log[:] = [
                        m
                        for m in chat_log
                        if not (
                            set(m.get("groups", ["general"])) & set(ai.groups)
                        )
                    ]
                    entry = {
                        "sender": ai.name,
                        "timestamp": ts_display,
                        "message": summary,
                        "groups": ai.groups,
                        "epoch": time.time(),
                    }
                    chat_log.append(entry)
                    sent_messages.append(entry)
                msg_count = 0
                continue

            reply = result
            with chat_lock:
                entry = {
                    "sender": ai.name,
                    "timestamp": timestamp,
                    "message": reply,
                    "groups": ai.groups,
                    "epoch": time.time(),
                }
                chat_log.append(entry)
                sent_messages.append(entry)
            logger.info("%s: generated response", ai.name)
            text = f"[{timestamp}] {ai.name}: {reply}\n{'-' * 80}\n\n"
            print(text)
            for group in ai.groups:
                fname = f"chat_log_{group}.txt"
                with open(fname, "a", encoding="utf-8") as log_file:
                    log_file.write(text)
            ui.root.after(0, ui.log, text)
            msg_count += 1

            if msg_count >= len(active_participants) and archivist and archivist.active:
                with chat_lock:
                    current_log = list(chat_log)
                context = [
                    m
                    for m in current_log
                    if set(m.get("groups", ["general"])) & set(archivist.groups)
                ]
                try:
                    summary = archivist.step(context)
                except requests.Timeout:
                    logger.error("%s timed out", archivist.name)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Error from %s: %s", archivist.name, exc)
                else:
                    logger.info("%s archived transcript", archivist.name)
                    ts_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ts_file = datetime.now().strftime("%Y%m%d%H%M%S")
                    text = (
                        f"[{ts_display}] {archivist.name} archived transcript and wrote summary.\n{'-' * 80}\n\n"
                    )
                    print(text)
                    ui.root.after(0, ui.log, text)
                    summary_text = f"[{ts_display}] {archivist.name}: {summary}\n{'-' * 80}\n\n"
                    for group in archivist.groups:
                        fname = f"chat_log_{group}.txt"
                        if os.path.exists(fname):
                            os.makedirs("chatlogs", exist_ok=True)
                            dest = os.path.join(
                                "chatlogs",
                                f"chat_log_{group}_summarized_{ts_file}.txt",
                            )
                            shutil.copy2(fname, dest)
                        with open(fname, "w", encoding="utf-8") as log_file:
                            log_file.write(summary_text)
                    with chat_lock:
                        chat_log[:] = [
                            m
                            for m in chat_log
                            if not (
                                set(m.get("groups", ["general"]))
                                & set(archivist.groups)
                            )
                        ]
                        entry = {
                            "sender": archivist.name,
                            "timestamp": ts_display,
                            "message": summary,
                            "groups": archivist.groups,
                            "epoch": time.time(),
                        }
                        chat_log.append(entry)
                        sent_messages.append(entry)
                    msg_count = 0

            time.sleep(0.5)
        logger.debug("Exiting conversation_loop")

    ui = FenraUI(agents, inject_callback=None, send_callback=None)

    def send_message(message: str) -> None:
        logger.debug("Entering send_message message=%s", message)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {"message": message, "timestamp": ts, "epoch": time.time()}
        with chat_lock:
            message_queue.append(entry)
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

    t = threading.Thread(target=conversation_loop, daemon=True)
    t.start()
    threads.append(t)

    ui.start()
    logger.debug("Exiting main")


if __name__ == "__main__":
    main()
