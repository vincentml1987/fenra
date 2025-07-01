import json
import sys
import time
from datetime import datetime
from typing import List, Dict, Tuple

import requests

from ai_model import AIModel

TAGS_URL = "http://localhost:11434/api/tags"
PULL_URL = "http://localhost:11434/api/pull"


def load_config(path: str) -> List[Tuple[str, str]]:
    """Load config file and return list of (name, model_id)."""
    models = []
    try:
        with open(path, "r", encoding="utf-8") as cfg:
            for line in cfg:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    name, model_id = line.split("=", 1)
                    name = name.strip()
                    model_id = model_id.strip()
                    if name and model_id:
                        models.append((name, model_id))
    except OSError as exc:
        print(f"Failed to read config file {path}: {exc}")
        sys.exit(1)

    if not models:
        print("No models found in config file.")
        sys.exit(1)
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
    config = load_config("fenra_config.txt")
    names, ids = zip(*config)
    ensure_models_available(list(ids))

    ai_models = [AIModel(name, mid) for name, mid in config]
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
