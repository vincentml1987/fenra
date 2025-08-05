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

from ai_model import (
    Agent,
    Ruminator,
    Ponderer,
    Doubter,
    Archivist,
    ToolAgent,
    Listener,
    Speaker,
)
from fenra_ui import FenraUI
from runtime_utils import (
    init_global_logging,
    parse_log_level,
    create_object_logger,
    add_json_watcher,
)


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


def parse_model_ids(directory: str) -> List[str]:
    """Return a list of **unique** model IDs for all active agents in ``directory``."""
    logger.debug("Entering parse_model_ids directory=%s", directory)
    ids: List[str] = []
    seen = set()
    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        if not os.path.isfile(fpath):
            continue
        parser = configparser.ConfigParser()
        with open(fpath, "r", encoding="utf-8") as f:
            parser.read_file(f)
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


def iter_load_config(directory: str, defaults: Dict[str, object]):
    """Yield Agent objects from all config files in ``directory``."""
    logger.debug("Entering iter_load_config path=%s", directory)
    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        if not os.path.isfile(fpath):
            continue
        parser = configparser.ConfigParser()
        with open(fpath, "r", encoding="utf-8") as f:
            parser.read_file(f)
        for section in parser.sections():
            if section == "global":
                continue
            if not parser.has_option(section, "model"):
                raise RuntimeError(f"Config error: model missing for AI '{section}'")
            active = parser.getboolean(section, "active", fallback=True)
            if not active:
                continue
            model_id = parser.get(section, "model")
            role_prompt = parser.get(section, "role_prompt", fallback="")
            groups = [g.strip() for g in parser.get(section, "groups", fallback="").split(',') if g.strip()]
            groups_in = [g.strip() for g in parser.get(section, "groups_in", fallback="").split(',') if g.strip()]
            groups_out = [g.strip() for g in parser.get(section, "groups_out", fallback="").split(',') if g.strip()]
            temperature = parser.getfloat(section, "temperature", fallback=defaults["temperature"])
            max_tokens_str = parser.get(section, "max_tokens", fallback=None)
            if max_tokens_str is not None:
                max_tokens = int(max_tokens_str)
            else:
                max_tokens = defaults["max_tokens"]
            chat_style = parser.get(section, "chat_style", fallback=defaults["chat_style"])
            watchdog = parser.getint(section, "watchdog_timeout", fallback=defaults["watchdog_timeout"])
            system_prompt = parser.get(section, "system_prompt", fallback=defaults["system_prompt"])
            topic_prompt = parser.get(section, "topic_prompt", fallback=defaults["topic_prompt"])
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
            elif role == "ponderer":
                yield Ponderer(
                    name=section,
                    model_name=model_id,
                    role_prompt=role_prompt,
                    config=cfg,
                    groups=groups,
                    groups_in=groups_in,
                    groups_out=groups_out,
                )
            elif role == "doubter":
                yield Doubter(
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
        if not part.strip():
            continue
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
        if not part.strip():
            continue
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
    if not content or not content.strip():
        logger.debug("Skipping empty Discord message")
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
    agents_dir = "agents"
    level = _parse_debug_level(config_path)
    init_global_logging(level)
    global logger
    logger = create_object_logger("Conductor")
    logger.debug("Entering main")
    logger.info("Starting conductor")
    defaults = load_global_defaults(config_path)
    model_ids = parse_model_ids(agents_dir)
    ensure_models_available(model_ids)

    if level <= logging.INFO:
        payload_logger = logging.getLogger("payloads")
        payload_logger.setLevel(logging.INFO)
        payload_logger.propagate = False
        handler = logging.FileHandler(os.path.join("logs", "payloads.log"), mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        payload_logger.handlers.clear()
        payload_logger.addHandler(handler)

        def _file_json_logger(payload: Dict) -> None:
            payload_logger.info(json.dumps(payload, indent=2))

        add_json_watcher(_file_json_logger)

    parser = configparser.ConfigParser()
    with open(config_path, "r", encoding="utf-8") as f:
        parser.read_file(f)

    talkativeness: float = parser.getfloat("global", "talkativeness", fallback=1.0)
    forgetfulness: float = parser.getfloat("global", "forgetfulness", fallback=1.0)
    rumination: float = parser.getfloat("global", "rumination", fallback=1.0)
    boredom: float = parser.getfloat("global", "boredom", fallback=0.0)
    assuredness: float = parser.getfloat("global", "assuredness", fallback=0.0)
    certainty: float = parser.getfloat("global", "certainty", fallback=0.0)
    restlessness = parser.getfloat("global", "restlessness", fallback=0.0)
    doubting = parser.getfloat("global", "doubting", fallback=0.0)
    attentiveness = parser.getfloat("global", "attentiveness", fallback=0.0)
    stimulation = parser.getfloat("global", "stimulation", fallback=0.0)
    excitement = parser.getfloat("global", "excitement", fallback=0.0)
    distraction = parser.getfloat("global", "distraction", fallback=0.0)
    focus = parser.getfloat("global", "focus", fallback=0.0)
    extroversion = parser.getfloat("global", "extroversion", fallback=0.0)
    fixation = parser.getfloat("global", "fixation", fallback=0.0)
    uncertainty = parser.getfloat("global", "uncertainty", fallback=0.0)

    agents: List[Agent] = []
    archivists: List[Archivist] = []
    listeners: List[Listener] = []
    speakers: List[Speaker] = []
    ruminators: List[Agent] = []
    all_groups: List[str] = []

    fenra_ui_logger = logging.getLogger("fenra_ui")
    fenra_ui_logger.setLevel(level)
    ui = FenraUI(agents, inject_callback=None, send_callback=None, config_path=config_path)

    agent_lock = threading.Lock()
    ready_event = threading.Event()

    def loader() -> None:
        nonlocal all_groups
        for agent in iter_load_config(agents_dir, defaults):
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

    while not ready_event.is_set():
        ui.root.update()
        time.sleep(0.1)

    # At least one of each agent type loaded

    chat_log: List[Dict[str, str]] = load_all_chat_histories()
    inject_queue: List[Dict[str, str]] = []
    message_queue: List[Dict[str, object]] = load_message_queue()
    messages_to_humans: List[Dict[str, object]] = load_messages_to_humans()
    chat_lock = threading.Lock()

    def conversation_loop() -> None:
        nonlocal talkativeness, forgetfulness, rumination, boredom, certainty
        logger.debug("Entering conversation_loop")
        def _missing_downstream_types(agent: Agent, active: List[Agent]) -> List[str]:
            """Return a list of agent type names missing downstream from ``agent``."""
            S = set(agent.groups_out)
            required = {
                Listener: False,
                Speaker: False,
                Ruminator: False,
                Archivist: False,
                Ponderer: False,
                Doubter: False,
            }
            for a in active:
                if (a is not agent or agent.allow_self_consume) and (a.groups_in & S):
                    if isinstance(a, Doubter):
                        required[Doubter] = True
                    elif isinstance(a, Ponderer):
                        required[Ponderer] = True
                    elif isinstance(a, Listener):
                        required[Listener] = True
                    elif isinstance(a, Speaker):
                        required[Speaker] = True
                    elif isinstance(a, Archivist):
                        required[Archivist] = True
                    elif isinstance(a, Ruminator):
                        required[Ruminator] = True
            return [cls.__name__ for cls, ok in required.items() if not ok]

        def ensure_downstream(
            candidate: Agent,
            active: List[Agent],
            pool: List[Agent] | None = None,
        ) -> Agent:
            """Return an agent that has all downstream role types."""
            pool = pool or active
            while True:
                missing = _missing_downstream_types(candidate, active)
                if not missing:
                    return candidate
                logger.warning(
                    "Agent %s missing downstream types: %s; re-selecting in 5s",
                    candidate.name,
                    ", ".join(missing),
                )
                time.sleep(5)
                if pool is None:
                    with agent_lock:
                        active = [a for a in agents if a.active]
                        pool = active
                else:
                    with agent_lock:
                        active = [a for a in agents if a.active]
                selection = [a for a in pool if a is not candidate] or pool
                candidate = random.choice(selection)

        with agent_lock:
            active_agents = [a for a in agents if a.active]
        if message_queue:
            candidate = next(
                (a for a in active_agents if isinstance(a, Listener)), None
            )
        else:
            candidate = next(
                (a for a in active_agents if not isinstance(a, Listener)), None
            )
        if candidate is None:
            candidate = random.choice(active_agents)
        state_current = ensure_downstream(candidate, active_agents)
        epoch = 0
        ui.root.after(
            0,
            ui.update_weights,
            talkativeness,
            rumination,
            forgetfulness,
            boredom,
            certainty,
        )
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

            with chat_lock:
                message_queue[:] = load_message_queue()

            if message_queue and not isinstance(state_current, Listener):
                with agent_lock:
                    active_agents = [a for a in agents if a.active]
                S = set(state_current.groups_out)
                listener_candidates = [
                    a
                    for a in active_agents
                    if isinstance(a, Listener)
                    and (a.groups_in & S)
                    and (a is not state_current or state_current.allow_self_consume)
                ]
                if not listener_candidates:
                    listener_candidates = [
                        l for l in active_agents if isinstance(l, Listener)
                    ]
                if listener_candidates:
                    state_current = random.choice(listener_candidates)

            if isinstance(state_current, Listener):
                human_entry = None
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
                            "groups": msg.get(
                                "groups",
                                list(state_current.groups_in) or ["general"],
                            ),
                            "epoch": time.time(),
                        }
                        chat_log.append(human_entry)
                if human_entry:
                    text = (
                        f"[{human_entry['timestamp']}] {human_entry['sender']}: {human_entry['message']}\n"
                        f"{'-' * 80}\n\n"
                    )
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

            if isinstance(state_current, Ponderer):
                context.append(
                    {
                        "sender": "System",
                        "timestamp": "",
                        "message": (
                            "Instruction: Talk about anything except what was said above. "
                            "Do not self-reference this instruction. Just bring up a new topic naturally."
                        ),
                        "groups": list(state_current.groups_in),
                    }
                )
            elif isinstance(state_current, Doubter):
                context.append(
                    {
                        "sender": "System",
                        "timestamp": "",
                        "message": (
                            "Instruction: Argue against everything that was said above. "
                            "Do not self-reference this instruction. Just debate against what has been said."
                        ),
                        "groups": list(state_current.groups_in),
                    }
                )
            try:
                reply = step_with_retry(state_current, lambda: state_current.step(context))
            except Exception as exc:  # noqa: BLE001
                logger.error("Error from %s: %s", state_current.name, exc)
                reply = ""

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            epoch += 1
            groups_target = list(state_current.groups_out or state_current.groups_in or ["general"])
            if isinstance(state_current, Archivist):
                summary_dir = os.path.join("chatlogs", "summarized")
                os.makedirs(summary_dir, exist_ok=True)
                with chat_lock:
                    chat_log[:] = [
                        e
                        for e in chat_log
                        if not (set(e.get("groups", ["general"])) & set(groups_target))
                    ]
                for group in groups_target:
                    src = os.path.join("chatlogs", f"chat_log_{group}.txt")
                    if os.path.exists(src):
                        dest = os.path.join(
                            summary_dir,
                            f"chat_log_{group}_{timestamp.replace(' ', '_').replace(':', '-')}.txt",
                        )
                        try:
                            shutil.move(src, dest)
                        except Exception as exc:  # noqa: BLE001
                            logger.error("Failed to move %s to %s: %s", src, dest, exc)
                    text = f"[{timestamp}] {state_current.name}: {reply}\n{'-' * 80}\n\n"
                    with open(src, "w", encoding="utf-8") as log_file:
                        log_file.write(text)
                    entry_group = {
                        "sender": state_current.name,
                        "timestamp": timestamp,
                        "message": reply,
                        "groups": [group],
                        "epoch": epoch,
                    }
                    with chat_lock:
                        chat_log.append(entry_group)
                    ui.root.after(0, ui.log, entry_group)
            else:
                entry = {
                    "sender": state_current.name,
                    "timestamp": timestamp,
                    "message": reply,
                    "groups": groups_target,
                    "epoch": epoch,
                }
                with chat_lock:
                    chat_log.append(entry)
                text = f"[{timestamp}] {state_current.name}: {reply}\n{'-' * 80}\n\n"
                for group in groups_target:
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
                        post_to_discord(entry["message"])
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Failed to post Speaker message to Discord: %s", exc)
                    ui.root.after(0, ui.update_sent, list(messages_to_humans))

            # Apply PTCD updates regardless of agent type
            if isinstance(state_current, Speaker):
                talkativeness *= max(0.0, 1 - attentiveness / 100.0)
                forgetfulness *= 1 + distraction / 100.0
                boredom *= max(0.0, 1 - extroversion / 100.0)
            elif isinstance(state_current, Listener):
                talkativeness *= 1 + stimulation / 100.0
                forgetfulness *= 1 + distraction / 100.0
                boredom *= max(0.0, 1 - extroversion / 100.0)
            elif isinstance(state_current, Archivist):
                forgetfulness *= max(0.0, 1 - focus / 100.0)
                certainty *= 1 + doubting / 100.0
            elif isinstance(state_current, Ponderer):
                forgetfulness *= 1 + distraction / 100.0
                boredom *= max(0.0, 1 - fixation / 100.0)
            elif isinstance(state_current, Doubter):
                forgetfulness *= 1 + distraction / 100.0
                certainty *= max(0.0, 1 - uncertainty / 100.0)
            else:
                talkativeness *= 1 + excitement / 100.0
                forgetfulness *= 1 + distraction / 100.0
                boredom *= 1 + restlessness / 100.0
            logger.debug(
                "Weights updated: talkativeness=%.3f forgetfulness=%.3f",
                talkativeness,
                forgetfulness,
            )
            ui.root.after(
                0,
                ui.update_weights,
                talkativeness,
                rumination,
                forgetfulness,
                boredom,
                certainty,
            )

            with agent_lock:
                active_agents = [a for a in agents if a.active]

            while True:
                S = set(state_current.groups_out)
                raw_candidates = [
                    b
                    for b in active_agents
                    if (b is not state_current or state_current.allow_self_consume)
                    and (b.groups_in & S)
                ]
                candidates = [
                    c for c in raw_candidates if not _missing_downstream_types(c, active_agents)
                ]
                if candidates:
                    break
                logger.warning(
                    "No downstream chain covering all roles from %s (groups_out=%s); re-selecting in 5s",
                    state_current.name,
                    S,
                )
                time.sleep(5)
                pool = [a for a in active_agents if a is not state_current] or active_agents
                state_current = random.choice(pool)

            if message_queue:
                listener_candidates = [c for c in candidates if isinstance(c, Listener)]
                if not listener_candidates:
                    listener_candidates = [
                        l
                        for l in active_agents
                        if isinstance(l, Listener)
                        and (l.groups_in & S)
                        and (l is not state_current or state_current.allow_self_consume)
                    ]
                if len(listener_candidates) > 1:
                    listener_candidates = [l for l in listener_candidates if l is not state_current]
                if listener_candidates:
                    state_current = random.choice(listener_candidates)
                else:
                    fallback = [l for l in active_agents if isinstance(l, Listener) and l is not state_current]
                    if not fallback:
                        fallback = [l for l in active_agents if isinstance(l, Listener)]
                    if fallback:
                        state_current = random.choice(fallback)
                    else:
                        if len(candidates) > 1:
                            candidates = [b for b in candidates if b is not state_current]
                        state_current = random.choice(candidates)
            else:
                if len(candidates) > 1:
                    candidates = [b for b in candidates if b is not state_current]

                speaker_candidates = [c for c in candidates if isinstance(c, Speaker)]
                ruminator_candidates = [
                    c
                    for c in candidates
                    if isinstance(c, Ruminator)
                    and not isinstance(c, (Ponderer, Doubter))
                ]
                archivist_candidates = [c for c in candidates if isinstance(c, Archivist)]
                ponderer_candidates = [c for c in candidates if isinstance(c, Ponderer)]
                doubter_candidates = [c for c in candidates if isinstance(c, Doubter)]

                pools = []
                weights = []
                if speaker_candidates:
                    pools.append(speaker_candidates)
                    weights.append(talkativeness)
                if ruminator_candidates:
                    pools.append(ruminator_candidates)
                    weights.append(rumination)
                if archivist_candidates:
                    pools.append(archivist_candidates)
                    weights.append(forgetfulness)
                if ponderer_candidates:
                    pools.append(ponderer_candidates)
                    weights.append(boredom)
                if doubter_candidates:
                    pools.append(doubter_candidates)
                    weights.append(certainty)

                if pools:
                    selected_pool = random.choices(pools, weights=weights, k=1)[0]
                    state_current = random.choice(selected_pool)
                else:
                    state_current = random.choice(candidates)

            with agent_lock:
                active_agents = [a for a in agents if a.active]
            state_current = ensure_downstream(state_current, active_agents, candidates)

            time.sleep(0.5)
        logger.debug("Exiting conversation_loop")

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
