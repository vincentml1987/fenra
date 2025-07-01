import json
import sys
import time
from datetime import datetime
from typing import List, Dict
import configparser

import requests

from ai_model import AIModel

TAGS_URL = "http://localhost:11434/api/tags"
PULL_URL = "http://localhost:11434/api/pull"


def load_config(path: str) -> List[AIModel]:
    """Parse fenra_config.txt and return instantiated AIModel objects."""
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise RuntimeError(f"Failed to read config file {path}")

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
    else:
        topic_prompt_global = None
        temperature_global = 0.7
        max_tokens_global = 300
        chat_style_global = None

    models: List[AIModel] = []
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

        topic_prompt = parser.get(section, "topic_prompt", fallback=topic_prompt_global)
        if topic_prompt is None:
            raise RuntimeError(
                f"Config error: topic_prompt missing for AI '{section}' and no global default."
            )

        models.append(
            AIModel(
                name=section,
                model_id=model_id,
                topic_prompt=topic_prompt,
                role_prompt=role_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_style=chat_style,
            )
        )

    if not models:
        raise RuntimeError("No active AI models found in config.")

    return models


def ensure_models_available(model_ids: List[str]) -> None:
    """Verify models are installed locally, pulling them if missing."""
    try:
        resp = requests.get(TAGS_URL)
    except requests.RequestException as exc:
        print(f"Error contacting Ollama server: {exc}")
        sys.exit(1)
    if resp.status_code != 200:
        print(f"Failed to list models: {resp.status_code} {resp.text}")
        sys.exit(1)

    try:
        tags_info = resp.json()
    except json.JSONDecodeError:
        print("Invalid response from tags endpoint.")
        sys.exit(1)

    local_models = {m.get("name") for m in tags_info.get("models", [])}

    for mid in model_ids:
        if mid in local_models:
            continue
        print(f"Model {mid} not found locally. Downloading...")
        try:
            pull_resp = requests.post(
                PULL_URL,
                json={"name": mid, "stream": False},
            )
        except requests.RequestException as exc:
            print(f"Failed to pull model {mid}: {exc}")
            sys.exit(1)
        if pull_resp.status_code != 200:
            print(f"Error pulling model {mid}: {pull_resp.status_code} {pull_resp.text}")
            sys.exit(1)
        try:
            result = pull_resp.json()
        except json.JSONDecodeError:
            print(f"Unexpected response pulling model {mid}: {pull_resp.text}")
            sys.exit(1)
        status = result.get("status")
        if status != "success":
            print(f"Model pull failed for {mid}: {result}")
            sys.exit(1)


def main() -> None:
    ai_models = load_config("fenra_config.txt")
    ensure_models_available([m.model_id for m in ai_models])

    chat_log: List[Dict[str, str]] = []

    index = 0
    try:
        while True:
            ai = ai_models[index]
            reply = ai.generate_response(chat_log)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            chat_log.append(
                {
                    "sender": ai.name,
                    "timestamp": timestamp,
                    "message": reply,
                }
            )
            print(f"[{timestamp}] {ai.name}: {reply}\n{'-' * 80}\n\n")

            with open("chat_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"[{timestamp}] {ai.name}: {reply}\n{'-' * 80}\n\n")
                
            index = (index + 1) % len(ai_models)
            print(index)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nConversation ended.")


if __name__ == "__main__":
    main()
