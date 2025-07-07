import json
import sys
import time
from datetime import datetime
from typing import List, Dict
import configparser
import threading

import logging
import requests

from ai_model import Ruminator, Archivist
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


def main() -> None:
    logger.info("Starting conductor")
    agents = load_config("fenra_config.txt")
    ensure_models_available([a.model_name for a in agents])

    ui = FenraUI(agents)

    ruminators = [a for a in agents if isinstance(a, Ruminator)]
    archivists = [a for a in agents if isinstance(a, Archivist)]
    archivist = archivists[0] if archivists else None

    chat_log: List[Dict[str, str]] = []

    def loop() -> None:
        try:
            while True:
                for ai in ruminators:
                    reply = ai.step(chat_log)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

                if archivist:
                    summary = archivist.step(chat_log)
                    logger.info("%s archived transcript", archivist.name)
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    text = (
                        f"[{ts}] {archivist.name} archived transcript and wrote summary.\n{'-' * 80}\n\n"
                    )
                    print(text)
                    ui.root.after(0, ui.log, text)

                    chat_log.clear()
                    chat_log.append(
                        {
                            "sender": archivist.name,
                            "timestamp": ts,
                            "message": summary,
                        }
                    )

                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Conversation ended by user")
            print("\nConversation ended.")

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()

    ui.start()


if __name__ == "__main__":
    main()
