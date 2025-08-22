import json
import sys
import time
from datetime import datetime
from typing import List, Dict, Tuple, Callable, Set
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
from collections import defaultdict

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
    TransientModelError,
)


def _parse_debug_level(path: str) -> int:
    """Return a logging level parsed from the config without logging."""
    parser = configparser.ConfigParser()
    with open(path, "r", encoding="utf-8") as f:
        parser.read_file(f)
    level_name = parser.get("global", "debug_level", fallback="INFO")
    return parse_log_level(level_name)


logger = create_object_logger("Conductor")


def _agent_to_ui(agent: Agent) -> dict:
    role = getattr(agent, "role_prompt", "").strip().lower()
    if not role:
        role = agent.__class__.__name__.lower()
    return {
        "name": agent.name,
        "role": role,
        "groups_in": sorted(getattr(agent, "groups_in", [])),
        "groups_out": sorted(getattr(agent, "groups_out", [])),
    }

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

# Track current agent per thread for UI attribution
_ACTIVE_AGENT = threading.local()
_ACTIVE_AGENT.name = None

_DEPRECATED_GROUP = "gen" + "eral"

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
    try:
        return func()
    except (
        requests.Timeout,
        requests.ReadTimeout,
        requests.ConnectionError,
        TimeoutExpired,
        TransientModelError,
    ):
        logger.error("%s timed out; restarting Ollama server", agent.name)
        restart_ollama_server()
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

    # Efficient lookup structures
    active_agents: Set[Agent] = set()
    agents_by_role: Dict[type, Set[Agent]] = {}
    roles_by_group: Dict[str, Set[type]] = {}

    fenra_ui_logger = logging.getLogger("fenra_ui")
    fenra_ui_logger.setLevel(level)
    ui = FenraUI(agents, inject_callback=None, send_callback=None, config_path=config_path)

    def _ui_json_watcher(payload: Dict) -> None:
        name = getattr(_ACTIVE_AGENT, "name", None) or payload.get("model") or "Unknown"
        try:
            ui.update_agent_payload(str(name), payload)
        except Exception:
            logger.exception("UI watcher failed")

    add_json_watcher(_ui_json_watcher)

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
                if agent.active:
                    active_agents.add(agent)
                    agents_by_role.setdefault(type(agent), set()).add(agent)
                    for g in agent.groups_in:
                        roles_by_group.setdefault(g, set()).add(type(agent))
                if archivists and listeners and speakers and ruminators:
                    ready_event.set()


    threading.Thread(target=loader, daemon=True).start()
    logger.info("Loading agents in background...")

    while not ready_event.is_set():
        ui.root.update()
        time.sleep(0.1)

    for ag in agents:
        if not getattr(ag, "groups_in", None):
            raise ValueError(f"Agent {ag.name} has no groups_in")
        if _DEPRECATED_GROUP in ag.groups_in or _DEPRECATED_GROUP in getattr(ag, "groups_out", []):
            raise ValueError(f"Agent {ag.name} references deprecated group '{_DEPRECATED_GROUP}'")

    # At least one of each agent type loaded

    chat_log: List[Dict[str, str]] = load_all_chat_histories()
    inject_queue: List[Dict[str, str]] = []
    message_queue: List[Dict[str, object]] = load_message_queue()
    messages_to_humans: List[Dict[str, object]] = load_messages_to_humans()
    chat_lock = threading.Lock()

    def _format_entry(entry: Dict[str, str]) -> str:
        return f"[{entry['timestamp']}] {entry['sender']}: {entry['message']}\n{'-' * 80}\n\n"

    def _build_group_contexts(history: List[Dict[str, str]]) -> Dict[str, str]:
        ctx: Dict[str, str] = {}
        for e in history:
            groups = e.get("groups")
            if not groups:
                raise ValueError("Log entry missing 'groups'")
            if _DEPRECATED_GROUP in groups:
                raise ValueError(f"Log entry references deprecated group '{_DEPRECATED_GROUP}'")
            text = _format_entry(e)
            for g in groups:
                ctx[g] = ctx.get(g, "") + text
        return ctx

    group_contexts: Dict[str, str] = _build_group_contexts(chat_log)
    ordered_gc: Dict[str, str] = {g: group_contexts.get(g, "") for g in (all_groups or [])}
    for g, txt in group_contexts.items():
        if g not in ordered_gc:
            ordered_gc[g] = txt
    ui.set_group_contexts(ordered_gc)

    def conversation_loop() -> None:
        nonlocal talkativeness, forgetfulness, rumination, boredom, certainty
        logger.debug("Entering conversation_loop")
        timeout_secs: defaultdict[str, float] = defaultdict(lambda: 900.0)
        def missing_downstream_roles(agent: Agent) -> List[str]:
            """Return a list of role names missing downstream from ``agent``."""
            S = set(agent.groups_in)
            present: Set[type] = set()
            for grp in S:
                roles = roles_by_group.get(grp, set()).copy()
                if not agent.allow_self_consume:
                    roles.discard(type(agent))
                present |= roles
            required = {Listener, Speaker, Ruminator, Archivist, Ponderer, Doubter}
            return [cls.__name__ for cls in required if cls not in present]

        def ensure_downstream(candidate: Agent, pool: Set[Agent] | None = None) -> Agent:
            """Return an agent that has all downstream role types."""
            while True:
                missing = missing_downstream_roles(candidate)
                if not missing:
                    return candidate
                logger.warning(
                    "Agent %s missing downstream types: %s; re-selecting in 5s",
                    candidate.name,
                    ", ".join(missing),
                )
                time.sleep(5)
                choices = list((pool or active_agents) - {candidate}) or list(pool or active_agents)
                candidate = random.choice(choices)

        def simulate_ptcd(
            agent: Agent | type,
            t: float,
            r: float,
            f: float,
            b: float,
            c: float,
        ) -> Tuple[float, float, float, float, float]:
            """Return PTCD values after simulating ``agent`` execution."""
            if isinstance(agent, Speaker) or agent == Speaker:
                t *= max(0.0, 1 - attentiveness / 100.0)
                f *= 1 + distraction / 100.0
                b *= max(0.0, 1 - extroversion / 100.0)
            elif isinstance(agent, Listener) or agent == Listener:
                t *= 1 + stimulation / 100.0
                f *= 1 + distraction / 100.0
                b *= max(0.0, 1 - extroversion / 100.0)
            elif isinstance(agent, Archivist) or agent == Archivist:
                f *= max(0.0, 1 - focus / 100.0)
                c *= 1 + doubting / 100.0
            elif isinstance(agent, Ponderer) or agent == Ponderer:
                f *= 1 + distraction / 100.0
                b *= max(0.0, 1 - fixation / 100.0)
            elif isinstance(agent, Doubter) or agent == Doubter:
                f *= 1 + distraction / 100.0
                c *= max(0.0, 1 - uncertainty / 100.0)
            else:
                t *= 1 + excitement / 100.0
                f *= 1 + distraction / 100.0
                b *= 1 + restlessness / 100.0
            return t, r, f, b, c

        def _run_agent_once(agent: Agent) -> None:
            """Execute ``agent`` one step and update logs/PTCD."""
            nonlocal epoch, talkativeness, rumination, forgetfulness, boredom, certainty

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with chat_lock:
                context = [
                    m
                    for m in chat_log
                    if set(m.get("groups", ())) & set(agent.groups_in)
                ]

            if isinstance(agent, Ponderer):
                context.append(
                    {
                        "sender": "System",
                        "timestamp": "",
                        "message": (
                            "Instruction: Talk about anything except what was said above. "
                            "Do not self-reference this instruction. Just bring up a new topic naturally."
                        ),
                        "groups": list(agent.groups_in),
                    }
                )
            elif isinstance(agent, Doubter):
                context.append(
                    {
                        "sender": "System",
                        "timestamp": "",
                        "message": (
                            "Instruction: Argue against everything that was said above. "
                            "Do not self-reference this instruction. Just debate against what has been said."
                        ),
                        "groups": list(agent.groups_in),
                    }
                )
            try:
                if ui and hasattr(ui, "update_topology"):
                    active = _agent_to_ui(agent)
                    roster = [_agent_to_ui(a) for a in list(active_agents) or agents]
                    ui.update_topology(active, roster)
            except Exception:  # noqa: BLE001
                logger.exception("update_topology failed (non-fatal)")
            try:
                _ACTIVE_AGENT.name = agent.name
                if ui:
                    ui.set_active_agent(agent.name)
                agent.model.watchdog_timeout = timeout_secs[agent.name]
                reply = step_with_retry(agent, lambda: agent.step(context))
            except Exception as exc:  # noqa: BLE001
                name = agent.name
                timeout_secs[name] *= 1.1
                delay = 2.0
                logger.error(
                    "Error from %s: %s — retrying same agent after %.1fs (timeout %.1fs)",
                    name,
                    exc,
                    delay,
                    timeout_secs[name],
                )
                time.sleep(delay)
                return
            else:
                timeout_secs[agent.name] = 900.0
                agent.model.watchdog_timeout = 900.0
            finally:
                _ACTIVE_AGENT.name = None
                if ui:
                    ui.set_active_agent("None")

            epoch += 1
            if not agent.groups_in:
                raise ValueError(f"{agent.name} has no groups_in; cannot route context")
            groups_target = list(agent.groups_in)
            if isinstance(agent, Archivist):
                summary_dir = os.path.join("chatlogs", "summarized")
                os.makedirs(summary_dir, exist_ok=True)

                with chat_lock:
                    chat_log[:] = [
                        e
                        for e in chat_log
                        if not (set(e.get("groups", ())) & set(groups_target))
                    ]

                text = f"[{timestamp}] {agent.name}: {reply}\n{'-' * 80}\n\n"
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
                    with open(src, "w", encoding="utf-8") as log_file:
                        log_file.write(text)

                entry = {
                    "sender": agent.name,
                    "timestamp": timestamp,
                    "message": reply,
                    "groups": groups_target,
                    "epoch": epoch,
                }
                with chat_lock:
                    chat_log.append(entry)
                for group in groups_target:
                    group_contexts[group] = text
                    ui.update_group_context(group, text)
                ui.root.after(0, ui.log, entry)
            else:
                entry = {
                    "sender": agent.name,
                    "timestamp": timestamp,
                    "message": reply,
                    "groups": groups_target,
                    "epoch": epoch,
                }
                with chat_lock:
                    chat_log.append(entry)
                text = f"[{timestamp}] {agent.name}: {reply}\n{'-' * 80}\n\n"
                for group in groups_target:
                    fname = os.path.join("chatlogs", f"chat_log_{group}.txt")
                    os.makedirs(os.path.dirname(fname), exist_ok=True)
                    with open(fname, "a", encoding="utf-8") as log_file:
                        log_file.write(text)
                logger.debug(text.strip())
                for group in groups_target:
                    group_contexts[group] = group_contexts.get(group, "") + text
                    ui.update_group_context(group, group_contexts[group])
                ui.root.after(0, ui.log, entry)
                if isinstance(agent, Speaker):
                    messages_to_humans.append(entry)
                    save_messages_to_humans(messages_to_humans)
                    append_human_log(entry)
                    try:
                        post_to_discord(entry["message"])
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Failed to post Speaker message to Discord: %s", exc)
                    ui.root.after(0, ui.update_sent, list(messages_to_humans))

            talkativeness, rumination, forgetfulness, boredom, certainty = simulate_ptcd(
                agent, talkativeness, rumination, forgetfulness, boredom, certainty
            )
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

        def plan_chain(human_groups: List[str]) -> List[Agent]:
            """Plan a chain of agents in response to a human message."""
            t_tmp = talkativeness
            r_tmp = rumination
            f_tmp = forgetfulness
            b_tmp = boredom
            c_tmp = certainty

            listener_candidates = [
                l
                for l in agents_by_role.get(Listener, set())
                if set(l.groups_in) & set(human_groups)
                and not missing_downstream_roles(l)
            ]
            if not listener_candidates:
                listener_candidates = [
                    l
                    for l in agents_by_role.get(Listener, set())
                    if not missing_downstream_roles(l)
                ]
            if not listener_candidates:
                return []
            listener = ensure_downstream(random.choice(listener_candidates))
            chain = [listener]
            t_tmp, r_tmp, f_tmp, b_tmp, c_tmp = simulate_ptcd(
                listener, t_tmp, r_tmp, f_tmp, b_tmp, c_tmp
            )
            current = listener
            steps = 0
            while steps < 20:
                S = set(current.groups_in)
                raw_candidates_set: Set[Agent] = set()
                for agent2 in active_agents:
                    if agent2 is current:
                        continue
                    if not getattr(agent2, "groups_out", None):
                        continue
                    if set(agent2.groups_out) & S:
                        raw_candidates_set.add(agent2)
                if not current.allow_self_consume:
                    raw_candidates_set.discard(current)
                candidates_set = {
                    c for c in raw_candidates_set if not missing_downstream_roles(c)
                }
                speaker_candidates = {c for c in candidates_set if isinstance(c, Speaker)}
                ruminator_candidates = {
                    c
                    for c in candidates_set
                    if isinstance(c, Ruminator)
                    and not isinstance(c, (Ponderer, Doubter))
                }
                doubter_candidates = {c for c in candidates_set if isinstance(c, Doubter)}
                pools: List[List[Agent]] = []
                weights: List[float] = []
                if speaker_candidates:
                    pools.append(list(speaker_candidates))
                    weights.append(t_tmp)
                if ruminator_candidates:
                    pools.append(list(ruminator_candidates))
                    weights.append(r_tmp)
                if doubter_candidates:
                    pools.append(list(doubter_candidates))
                    weights.append(c_tmp)
                if pools:
                    selected_pool = random.choices(pools, weights=weights, k=1)[0]
                    nxt = random.choice(selected_pool)
                elif candidates_set:
                    nxt = random.choice(list(candidates_set))
                else:
                    break
                chain.append(nxt)
                t_tmp, r_tmp, f_tmp, b_tmp, c_tmp = simulate_ptcd(
                    nxt, t_tmp, r_tmp, f_tmp, b_tmp, c_tmp
                )
                current = nxt
                if isinstance(nxt, Speaker):
                    S = set(nxt.groups_in)
                    arch_candidates: Set[Agent] = set()
                    for agent2 in active_agents:
                        if not isinstance(agent2, Archivist):
                            continue
                        if not getattr(agent2, "groups_out", None):
                            continue
                        if set(agent2.groups_out) & S:
                            arch_candidates.add(agent2)
                    if not nxt.allow_self_consume:
                        arch_candidates.discard(nxt)
                    if not arch_candidates:
                        arch_candidates = set(agents_by_role.get(Archivist, set()))
                    if arch_candidates:
                        arch = random.choice(list(arch_candidates))
                        chain.append(arch)
                    break
                steps += 1
            return chain

        if message_queue:
            candidate = next(iter(agents_by_role.get(Listener, set())), None)
        else:
            non_listeners = active_agents - agents_by_role.get(Listener, set())
            candidate = next(iter(non_listeners), None)
        if candidate is None and active_agents:
            candidate = random.choice(tuple(active_agents))
        state_current = ensure_downstream(candidate)
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
                groups = msg.get("groups")
                if not groups:
                    raise ValueError("Injected message missing 'groups'")
                if _DEPRECATED_GROUP in groups:
                    raise ValueError(f"Injected message references deprecated group '{_DEPRECATED_GROUP}'")
                entry = {
                    "sender": msg.get("sender", "Human"),
                    "timestamp": msg.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
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
                for group in groups:
                    group_contexts[group] = group_contexts.get(group, "") + text
                    ui.update_group_context(group, group_contexts[group])
                ui.root.after(0, ui.log, entry)

            if message_queue:
                with chat_lock:
                    batch = list(message_queue)
                    message_queue.clear()
                    save_message_queue(message_queue)
                ui.root.after(0, ui.update_queue, list(message_queue))

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                lines = []
                groups_union: Set[str] = set()
                for msg in batch:
                    ts = msg.get("timestamp", timestamp)
                    sender = msg.get("sender", "Human")
                    text = msg.get("message", "")
                    lines.append(f"[{ts}] {sender}: {text}")
                    groups = msg.get("groups")
                    if not groups:
                        raise ValueError("Queued message missing 'groups'")
                    if _DEPRECATED_GROUP in groups:
                        raise ValueError(f"Queued message references deprecated group '{_DEPRECATED_GROUP}'")
                    groups_union.update(groups)
                if not groups_union:
                    raise ValueError("Batch has no groups")
                human_entry = {
                    "sender": "Human",
                    "timestamp": timestamp,
                    "message": "\n".join(lines),
                    "groups": sorted(groups_union),
                    "epoch": time.time(),
                }
                with chat_lock:
                    chat_log.append(human_entry)
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
                for group in human_entry["groups"]:
                    group_contexts[group] = group_contexts.get(group, "") + text
                    ui.update_group_context(group, group_contexts[group])
                ui.root.after(0, ui.log, human_entry)

                chain = plan_chain(human_entry["groups"])
                if chain:
                    logger.info(
                        "Planned chain: %s",
                        " \u2192 ".join(
                            f"{a.__class__.__name__}({a.name})" for a in chain
                        ),
                    )
                    for ag in chain:
                        _run_agent_once(ag)
                    state_current = chain[-1]
                continue

            if not active_agents:
                time.sleep(0.5)
                continue
            if state_current not in active_agents:
                state_current = random.choice(tuple(active_agents))

            _run_agent_once(state_current)

            while True:
                S = set(state_current.groups_in)
                raw_candidates_set: Set[Agent] = set()
                for agent2 in active_agents:
                    if agent2 is state_current:
                        continue
                    if not getattr(agent2, "groups_out", None):
                        continue
                    if set(agent2.groups_out) & S:
                        raw_candidates_set.add(agent2)
                if not state_current.allow_self_consume:
                    raw_candidates_set.discard(state_current)
                candidates_set = {
                    c for c in raw_candidates_set if not missing_downstream_roles(c)
                }
                if candidates_set:
                    break
                logger.warning(
                    "No downstream chain covering all roles from %s (groups_in=%s); re-selecting in 5s",
                    state_current.name,
                    S,
                )
                time.sleep(5)
                pool = (active_agents - {state_current}) or active_agents
                state_current = random.choice(tuple(pool))

            if len(candidates_set) > 1:
                candidates_set.discard(state_current)

            speaker_candidates = agents_by_role.get(Speaker, set()) & candidates_set
            ruminator_candidates = (
                agents_by_role.get(Ruminator, set())
                - agents_by_role.get(Ponderer, set())
                - agents_by_role.get(Doubter, set())
            ) & candidates_set
            archivist_candidates = agents_by_role.get(Archivist, set()) & candidates_set
            ponderer_candidates = agents_by_role.get(Ponderer, set()) & candidates_set
            doubter_candidates = agents_by_role.get(Doubter, set()) & candidates_set

            pools = []
            weights = []
            if speaker_candidates:
                pools.append(list(speaker_candidates))
                weights.append(talkativeness)
            if ruminator_candidates:
                pools.append(list(ruminator_candidates))
                weights.append(rumination)
            if archivist_candidates:
                pools.append(list(archivist_candidates))
                weights.append(forgetfulness)
            if ponderer_candidates:
                pools.append(list(ponderer_candidates))
                weights.append(boredom)
            if doubter_candidates:
                pools.append(list(doubter_candidates))
                weights.append(certainty)

            if pools:
                selected_pool = random.choices(pools, weights=weights, k=1)[0]
                state_current = random.choice(selected_pool)
            else:
                state_current = random.choice(list(candidates_set))

            state_current = ensure_downstream(state_current, candidates_set)

            time.sleep(0.5)
        logger.debug("Exiting conversation_loop")

    def send_message(message: str, groups: List[str]) -> None:
        logger.debug("Entering send_message message=%s groups=%s", message, groups)
        if not groups:
            raise ValueError("send_message requires at least one group")
        if _DEPRECATED_GROUP in groups:
            raise ValueError(f"send_message references deprecated group '{_DEPRECATED_GROUP}'")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {"message": message, "timestamp": ts, "epoch": time.time(), "groups": list(groups)}
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
