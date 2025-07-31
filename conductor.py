import json
import sys
import time
from datetime import datetime
from typing import List, Dict, Tuple, Callable
import configparser
import threading
import os
import re
import shutil
import random
import math

import logging
import requests
import subprocess
from subprocess import Popen, TimeoutExpired

from ai_model import Agent, Ruminator, Archivist, ToolAgent, Listener, Speaker
from fenra_ui import FenraUI
from runtime_utils import init_global_logging, parse_log_level, create_object_logger


def _parse_debug_level(path: str) -> int:
    """Return a logging level parsed from the config without logging."""
    parser = configparser.ConfigParser()
    with open(path, "r", encoding="utf-8") as f:
        parser.read_file(f)
    level_name = parser.get("global", "debug_level", fallback="INFO")
    return parse_log_level(level_name)


logger = create_object_logger("Conductor")

TAGS_URL = "http://localhost:11434/api/tags"
PULL_URL = "http://localhost:11434/api/pull"

# Handle starting and restarting the Ollama server

def start_ollama_server() -> Popen:
    """Start the Ollama server as a background process."""
    return Popen(
        ["ollama", "serve", "--daemon"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )


# Current running Ollama process
server_proc: Popen | None = None

# Discord integration (outbound only)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")
DISCORD_BOT_TOKEN = (
    os.getenv("DISCORD_BOT_TOKEN")
    or os.getenv("FENRA_DISCORD_TOKEN")
    or os.getenv("fenra_token")
)


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
        watchdog_global = parser.getint("global", "watchdog_timeout", fallback=900)
        debug_level_str = parser.get("global", "debug_level", fallback="INFO")
        init_global_logging(parse_log_level(debug_level_str))
    else:
        topic_prompt_global = None
        temperature_global = 0.7
        max_tokens_global = None
        chat_style_global = None
        watchdog_global = 900
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
        groups_str = parser.get(section, "groups", fallback="")
        groups = [g.strip() for g in groups_str.split(',') if g.strip()]
        groups_in_str = parser.get(section, "groups_in", fallback="")
        groups_in = [g.strip() for g in groups_in_str.split(',') if g.strip()]
        groups_out_str = parser.get(section, "groups_out", fallback="")
        groups_out = [g.strip() for g in groups_out_str.split(',') if g.strip()]
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
                    groups_in=groups_in,
                    groups_out=groups_out,
                )
            )
        elif role in ("tool", "toolagent", "tools"):
            agent = ToolAgent(
                name=section,
                model_name=model_id,
                role_prompt=role_prompt,
                config=cfg,
                groups=groups,
                groups_in=groups_in,
                groups_out=groups_out,
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
                    groups_in=groups_in,
                    groups_out=groups_out,
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
                    groups_in=groups_in,
                    groups_out=groups_out,
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
                    groups_in=groups_in,
                    groups_out=groups_out,
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
            "watchdog_timeout": sec.getint("watchdog_timeout", fallback=900),
            "system_prompt": sec.get("system_prompt", fallback=None),
        }
        logger.debug("Exiting load_global_defaults")
        return result

    result = {
        "topic_prompt": "",
        "temperature": 0.7,
        "max_tokens": None,
        "chat_style": None,
        "watchdog_timeout": 900,
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
        logging.shutdown()
        sys.exit(1)
    if resp.status_code != 200:
        logger.error("Failed to list models: %s %s", resp.status_code, resp.text)
        logging.shutdown()
        sys.exit(1)

    try:
        tags_info = resp.json()
    except json.JSONDecodeError:
        logger.error("Invalid response from tags endpoint.")
        logging.shutdown()
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
            logging.shutdown()
            sys.exit(1)
        if pull_resp.status_code != 200:
            logger.error(
                "Error pulling model %s: %s %s", mid, pull_resp.status_code, pull_resp.text
            )
            logging.shutdown()
            sys.exit(1)
        try:
            result = pull_resp.json()
        except json.JSONDecodeError:
            logger.error("Unexpected response pulling model %s: %s", mid, pull_resp.text)
            logging.shutdown()
            sys.exit(1)
        status = result.get("status")
        if status != "success":
            logger.error("Model pull failed for %s: %s", mid, result)
            logging.shutdown()
            sys.exit(1)
    logger.debug("Exiting ensure_models_available")


def parse_model_ids(path: str) -> List[str]:
    """Return a list of **unique** model IDs for all active agents."""
    logger.debug("Entering parse_model_ids path=%s", path)
    parser = configparser.ConfigParser()
    with open(path, "r", encoding="utf-8") as f:
        parser.read_file(f)

    ids: List[str] = []
    seen = set()
    for section in parser.sections():
        if section == "global":
            continue
        if not parser.has_option(section, "model"):
            raise RuntimeError(f"Config error: model missing for AI '{section}'")
        active = parser.getboolean(section, "active", fallback=True)
        if not active:
            continue
        model = parser.get(section, "model")
        if model not in seen:
            ids.append(model)
            seen.add(model)
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
        watchdog_global = parser.getint("global", "watchdog_timeout", fallback=900)
        debug_level_str = parser.get("global", "debug_level", fallback="INFO")
        init_global_logging(parse_log_level(debug_level_str))
    else:
        topic_prompt_global = None
        temperature_global = 0.7
        max_tokens_global = None
        chat_style_global = None
        watchdog_global = 900
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
        groups_str = parser.get(section, "groups", fallback="")
        groups = [g.strip() for g in groups_str.split(',') if g.strip()]
        groups_in_str = parser.get(section, "groups_in", fallback="")
        groups_in = [g.strip() for g in groups_in_str.split(',') if g.strip()]
        groups_out_str = parser.get(section, "groups_out", fallback="")
        groups_out = [g.strip() for g in groups_out_str.split(',') if g.strip()]
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
                groups_in=groups_in,
                groups_out=groups_out,
            )
        elif role in ("tool", "toolagent", "tools"):
            yield ToolAgent(
                name=section,
                model_name=model_id,
                role_prompt=role_prompt,
                config=cfg,
                groups=groups,
                groups_in=groups_in,
                groups_out=groups_out,
            )
        elif role == "listener":
            yield Listener(
                name=section,
                model_name=model_id,
                role_prompt=role_prompt,
                config=cfg,
                groups=groups,
                groups_in=groups_in,
                groups_out=groups_out,
            )
        elif role == "speaker":
            yield Speaker(
                name=section,
                model_name=model_id,
                role_prompt=role_prompt,
                config=cfg,
                groups=groups,
                groups_in=groups_in,
                groups_out=groups_out,
            )
        else:
            yield Ruminator(
                name=section,
                model_name=model_id,
                role_prompt=role_prompt,
                config=cfg,
                groups=groups,
                groups_in=groups_in,
                groups_out=groups_out,
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


def _discord_chunks(text: str, limit: int = 1900):
    """Yield message chunks within Discord's ~2000 char cap."""
    text = text or ""
    while text:
        cut = text.rfind("\n", 0, limit)
        if cut == -1:
            cut = min(len(text), limit)
        yield text[:cut]
        text = text[cut:].lstrip("\n")


def _post_to_discord_via_webhook(content: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        return
    for part in _discord_chunks(content):
        try:
            resp = requests.post(
                DISCORD_WEBHOOK_URL,
                json={"content": part},
                timeout=10,
            )
            if resp.status_code == 429:
                data = resp.json()
                time.sleep(float(data.get("retry_after", 1.0)))
                requests.post(DISCORD_WEBHOOK_URL, json={"content": part}, timeout=10)
        except Exception as exc:  # noqa: BLE001
            logger.error("Discord webhook failed: %s", exc)


def _post_to_discord_via_bot(content: str) -> None:
    if not (DISCORD_CHANNEL_ID and DISCORD_BOT_TOKEN):
        return
    url = f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages"
    headers = {
        "Authorization": f"Bot {DISCORD_BOT_TOKEN}",
        "Content-Type": "application/json",
    }
    for part in _discord_chunks(content):
        try:
            resp = requests.post(url, headers=headers, json={"content": part}, timeout=10)
            if resp.status_code == 429:
                data = resp.json()
                time.sleep(float(data.get("retry_after", 1.0)))
                requests.post(url, headers=headers, json={"content": part}, timeout=10)
            elif resp.status_code >= 400:
                logger.error("Discord bot post failed: %s %s", resp.status_code, resp.text)
        except Exception as exc:  # noqa: BLE001
            logger.error("Discord bot post exception: %s", exc)


def post_to_discord(content: str) -> None:
    """Send a message to Discord using webhook if available, else bot token."""
    if not content:
        return
    if DISCORD_WEBHOOK_URL:
        _post_to_discord_via_webhook(content)
    else:
        _post_to_discord_via_bot(content)


def restart_ollama_server() -> None:
    """Restart the Ollama server process."""
    global server_proc
    if server_proc is not None:
        try:
            server_proc.kill()
            server_proc.wait(timeout=10)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to kill Ollama server: %s", exc)
    server_proc = start_ollama_server()
    time.sleep(5)


def step_with_retry(agent: Agent, func: Callable[[], str]) -> str:
    """Call an agent function, retrying once after restarting Ollama on timeout."""
    base_timeout = agent.model.watchdog_timeout
    agent.model.watchdog_timeout = base_timeout
    try:
        return func()
    except (requests.Timeout, TimeoutExpired):
        logger.error("%s timed out; restarting Ollama server", agent.name)
        restart_ollama_server()
        agent.model.watchdog_timeout = base_timeout
        return func()
    finally:
        agent.model.watchdog_timeout = base_timeout


def main() -> None:

    global server_proc
    server_proc = start_ollama_server()
    time.sleep(5)
    config_path = "fenra_config.txt"
    level = _parse_debug_level(config_path)
    init_global_logging(level)
    global logger
    logger = create_object_logger("Conductor")
    logger.debug("Entering main")
    logger.info("Starting conductor")
    load_global_defaults(config_path)
    model_ids = parse_model_ids(config_path)
    ensure_models_available(model_ids)

    parser = configparser.ConfigParser()
    with open(config_path, "r", encoding="utf-8") as f:
        parser.read_file(f)

    forgetfulness_weight = parser.getfloat("global", "forgetfulness", fallback=1.0)
    talkativeness_weight = parser.getfloat("global", "talkativeness", fallback=1.0)
    rumination_weight = parser.getfloat("global", "rumination", fallback=1.0)
    attentiveness_weight = parser.getfloat("global", "attentiveness", fallback=1.0)

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
        state_current = random.choice(agents)
        epoch = 0
        while True:
            with chat_lock:
                message_queue[:] = load_message_queue()
                ui.root.after(0, ui.update_queue, list(message_queue))
                pending_inject = list(inject_queue)
                inject_queue.clear()

            for msg in pending_inject:
                groups = msg.get("groups", ["general"])
                entry = {
                    "sender": msg.get("sender", "Human"),
                    "timestamp": msg.get(
                        "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ),
                    "message": msg.get("message", ""),
                    "groups": groups,
                    "epoch": time.time(),
                }
                with chat_lock:
                    chat_log.append(entry)
                text = (
                    f"[{entry['timestamp']}] {entry['sender']}: {entry['message']}\n"
                    f"{'-' * 80}\n\n"
                )
                for group in groups:
                    fname = os.path.join("chatlogs", f"chat_log_{group}.txt")
                    os.makedirs(os.path.dirname(fname), exist_ok=True)
                    with open(fname, "a", encoding="utf-8") as log_file:
                        log_file.write(text)
                logger.debug(text.strip())
                ui.root.after(0, ui.log, entry)

            with agent_lock:
                active_agents = [a for a in agents if a.active]
            if not active_agents:
                time.sleep(0.5)
                continue
            if state_current not in active_agents:
                state_current = random.choice(active_agents)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if isinstance(state_current, Listener):
                with chat_lock:
                    message_queue[:] = load_message_queue()
                    if message_queue:
                        msg = message_queue.pop(0)
                        save_message_queue(message_queue)
                        ui.root.after(0, ui.update_queue, list(message_queue))
                        human_entry = {
                            "sender": msg.get("sender", "Human"),
                            "timestamp": msg.get("timestamp", timestamp),
                            "message": msg.get("message", ""),
                            "groups": msg.get("groups", list(state_current.groups_in) or ["general"]),
                            "epoch": time.time(),
                        }
                    else:
                        human_entry = {
                            "sender": "System",
                            "timestamp": timestamp,
                            "message": "There has been no message sent from anyone outside of Fenra.",
                            "groups": list(state_current.groups_in) or ["general"],
                            "epoch": time.time(),
                        }
                    chat_log.append(human_entry)
                text = f"[{human_entry['timestamp']}] {human_entry['sender']}: {human_entry['message']}\n{'-' * 80}\n\n"
                for group in human_entry["groups"]:
                    fname = os.path.join("chatlogs", f"chat_log_{group}.txt")
                    os.makedirs(os.path.dirname(fname), exist_ok=True)
                    with open(fname, "a", encoding="utf-8") as log_file:
                        log_file.write(text)
                logger.debug(text.strip())
                ui.root.after(0, ui.log, human_entry)

            with chat_lock:
                context = [
                    m
                    for m in chat_log
                    if set(m.get("groups", ["general"])) & set(state_current.groups_in)
                ]
            try:
                reply = step_with_retry(state_current, lambda: state_current.step(context))
            except Exception as exc:  # noqa: BLE001
                logger.error("Error from %s: %s", state_current.name, exc)
                reply = ""

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            epoch += 1
            entry = {
                "sender": state_current.name,
                "timestamp": timestamp,
                "message": reply,
                "groups": list(state_current.groups_out),
                "epoch": epoch,
            }
            with chat_lock:
                chat_log.append(entry)
            text = f"[{timestamp}] {state_current.name}: {reply}\n{'-' * 80}\n\n"
            for group in state_current.groups_out or ["general"]:
                fname = os.path.join("chatlogs", f"chat_log_{group}.txt")
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                with open(fname, "a", encoding="utf-8") as log_file:
                    log_file.write(text)
            logger.debug(text.strip())
            ui.root.after(0, ui.log, entry)
            if isinstance(state_current, Speaker):
                messages_to_humans.append(entry)
                save_messages_to_humans(messages_to_humans)
                append_human_log(entry)
                try:
                    discord_text = f"**{entry['sender']}** — {entry['timestamp']}\n{entry['message']}"
                    post_to_discord(discord_text)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to post Speaker message to Discord: %s", exc)
                ui.root.after(0, ui.update_sent, list(messages_to_humans))

            with agent_lock:
                active_agents = [a for a in agents if a.active]

            candidates = [
                b
                for b in active_agents
                if b is not state_current or state_current.allow_self_consume
            ]
            candidates = [b for b in candidates if b.groups_in & state_current.groups_out]

            if candidates:
                role_buckets = {
                    "speaker": [a for a in candidates if isinstance(a, Speaker)],
                    "ruminator": [
                        a
                        for a in candidates
                        if isinstance(a, Ruminator) or isinstance(a, ToolAgent)
                    ],
                    "archivist": [a for a in candidates if isinstance(a, Archivist)],
                    "listener": [a for a in candidates if isinstance(a, Listener)],
                }
                available = {
                    role: agents_list
                    for role, agents_list in role_buckets.items()
                    if agents_list
                }
                weights = {
                    "speaker": talkativeness_weight,
                    "ruminator": rumination_weight,
                    "archivist": forgetfulness_weight,
                    "listener": attentiveness_weight,
                }

                roles = list(available.keys())
                if len(roles) > 1 and state_current in candidates:
                    candidates = [c for c in candidates if c is not state_current]
                    for role in roles:
                        available[role] = [a for a in available[role] if a is not state_current] or available[role]

                weight_values = [weights[r] for r in roles]
                total = sum(weight_values)
                if total <= 0:
                    chosen_role = random.choice(roles)
                else:
                    rnd = random.random() * total
                    for r, w in zip(roles, weight_values):
                        if rnd < w:
                            chosen_role = r
                            break
                        rnd -= w
                    else:
                        chosen_role = roles[-1]

                state_current = random.choice(available[chosen_role])
            else:
                pool = [b for b in active_agents if b is not state_current]
                if not pool:
                    pool = active_agents
                state_current = random.choice(pool)

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
