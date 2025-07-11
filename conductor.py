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

import logging
import requests

from ai_model import Ruminator, Archivist, ToolAgent
from fenra_ui import FenraUI
from runtime_utils import init_global_logging, parse_log_level, create_object_logger


logger = create_object_logger("Conductor")

TAGS_URL = "http://localhost:11434/api/tags"
PULL_URL = "http://localhost:11434/api/pull"


def load_config(path: str):
    """Parse fenra_config.txt and return instantiated agent objects."""
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
                )
            )
        elif role in ("tool", "toolagent", "tools"):
            agent = ToolAgent(
                name=section,
                model_name=model_id,
                role_prompt=role_prompt,
                config=cfg,
            )
            agents.append(agent)
        else:
            agents.append(
                Ruminator(
                    name=section,
                    model_name=model_id,
                    role_prompt=role_prompt,
                    config=cfg,
                )
            )

    if not agents:
        raise RuntimeError("No active AI models found in config.")

    return agents


def load_global_defaults(path: str) -> Dict[str, object]:
    """Return default model configuration from the global section."""
    parser = configparser.ConfigParser()
    with open(path, "r", encoding="utf-8") as f:
        parser.read_file(f)

    if parser.has_section("global"):
        sec = parser["global"]
        return {
            "topic_prompt": sec.get("topic_prompt", ""),
            "temperature": sec.getfloat("temperature", fallback=0.7),
            "max_tokens": sec.getint("max_tokens", fallback=300),
            "chat_style": sec.get("chat_style", fallback=None),
            "watchdog_timeout": sec.getint("watchdog_timeout", fallback=300),
            "system_prompt": sec.get("system_prompt", fallback=None),
        }

    return {
        "topic_prompt": "",
        "temperature": 0.7,
        "max_tokens": 300,
        "chat_style": None,
        "watchdog_timeout": 300,
        "system_prompt": None,
    }


def ensure_models_available(model_ids: List[str]) -> None:
    """Verify models are installed locally, pulling them if missing."""
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
            if logger.isEnabledFor(logging.DEBUG):
                payload = {"name": mid, "stream": False}
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


def load_chat_history(path: str) -> List[Dict[str, str]]:
    """Return chat history parsed from a log file and archive it."""
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
        history.append({"sender": sender, "timestamp": ts, "message": message})

    try:
        os.makedirs("chatlogs", exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        dest = os.path.join("chatlogs", f"chatlog-{stamp}.txt")
        shutil.copy2(path, dest)
    except OSError as exc:
        logger.error("Failed to archive chat log: %s", exc)

    return history


def main() -> None:
    logger.info("Starting conductor")
    config_path = "fenra_config.txt"
    agents = load_config(config_path)
    defaults = load_global_defaults(config_path)
    ensure_models_available([a.model_name for a in agents])

    ruminators = [a for a in agents if isinstance(a, Ruminator)]
    archivists = [a for a in agents if isinstance(a, Archivist)]
    archivist = archivists[0] if archivists else None

    chat_log: List[Dict[str, str]] = load_chat_history("chat_log.txt")
    chat_lock = threading.Lock()
    threads: List[threading.Thread] = []

    def conversation_loop() -> None:
        idx = 0
        while True:
            with chat_lock:
                active_ruminators = [a for a in ruminators if a.active]
                context = list(chat_log)
            if not active_ruminators:
                time.sleep(0.5)
                continue
            ai = active_ruminators[idx % len(active_ruminators)]
            try:
                reply = ai.step(context)
            except requests.Timeout:
                logger.error("%s timed out", ai.name)
                idx = (idx + 1) % len(active_ruminators)
                continue
            except Exception as exc:  # noqa: BLE001
                logger.error("Error from %s: %s", ai.name, exc)
                idx = (idx + 1) % len(active_ruminators)
                continue

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with chat_lock:
                chat_log.append(
                    {
                        "sender": ai.name,
                        "timestamp": timestamp,
                        "message": reply,
                    }
                )
            logger.info("%s: generated response", ai.name)
            text = f"[{timestamp}] {ai.name}: {reply}\n{'-' * 80}\n\n"
            print(text)
            with open("chat_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(text)
            ui.root.after(0, ui.log, text)

            idx = (idx + 1) % len(active_ruminators)

            if idx == 0 and archivist and archivist.active:
                with chat_lock:
                    context = list(chat_log)
                try:
                    summary = archivist.step(context)
                except requests.Timeout:
                    logger.error("%s timed out", archivist.name)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Error from %s: %s", archivist.name, exc)
                else:
                    logger.info("%s archived transcript", archivist.name)
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    text = (
                        f"[{ts}] {archivist.name} archived transcript and wrote summary.\n{'-' * 80}\n\n"
                    )
                    print(text)
                    ui.root.after(0, ui.log, text)
                    with chat_lock:
                        chat_log.clear()
                        chat_log.append(
                            {
                                "sender": archivist.name,
                                "timestamp": ts,
                                "message": summary,
                            }
                        )

            time.sleep(0.5)

    def add_agent(name: str, model_id: str, role_prompt: str):
        cfg = {
            "topic_prompt": defaults.get("topic_prompt", ""),
            "temperature": defaults.get("temperature", 0.7),
            "max_tokens": defaults.get("max_tokens", 300),
            "chat_style": defaults.get("chat_style"),
            "watchdog_timeout": defaults.get("watchdog_timeout", 300),
            "system_prompt": defaults.get("system_prompt"),
        }

        agent = Ruminator(
            name=name,
            model_name=model_id,
            role_prompt=role_prompt,
            config=cfg,
        )
        ensure_models_available([model_id])
        agents.append(agent)
        ruminators.append(agent)
        return agent

    def remove_agent(agent):
        nonlocal archivist
        if agent not in agents:
            return False
        agent.active = False
        if isinstance(agent, Ruminator) and agent in ruminators:
            ruminators.remove(agent)
        if isinstance(agent, Archivist) and archivist is agent:
            archivist = None
        agents.remove(agent)
        return True

    ui = FenraUI(agents, add_agent_callback=add_agent, remove_agent_callback=remove_agent)

    t = threading.Thread(target=conversation_loop, daemon=True)
    t.start()
    threads.append(t)

    ui.start()


if __name__ == "__main__":
    main()
