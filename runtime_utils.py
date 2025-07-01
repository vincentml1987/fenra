import json
import re
import threading
import time
from collections import deque
from typing import Dict

import requests


class AITimeTracker:
    """Track recent successful generation times in AI Seconds."""

    def __init__(self, rolling_window: int = 10) -> None:
        self.data = deque(maxlen=rolling_window)

    def record(self, wall_time: float, model_size_b: float) -> None:
        """Convert wall time to AI Seconds (normalized to 7b) and store."""
        factor = (model_size_b / 7) ** 1.2
        ai_seconds = wall_time / factor
        self.data.append(ai_seconds)

    def average(self) -> float:
        """Return the average AI Seconds; fallback to 1.0 if none."""
        if not self.data:
            return 1.0
        return sum(self.data) / len(self.data)


def parse_model_size(model_id: str) -> float:
    """Extract model size in billions from the model id string."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", model_id)
    if not match:
        return 7.0
    try:
        return float(match.group(1))
    except ValueError:
        return 7.0


def parse_response(resp: requests.Response) -> str:
    """Parse a non-streaming Ollama response into text."""
    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid JSON from Ollama") from exc
    return data.get("response", "")


def generate_with_watchdog(
    payload: Dict,
    model_size_b: float,
    tracker: AITimeTracker,
    timeout_cushion: float = 2.0,
) -> str:
    """Call Ollama with a watchdog timeout based on normalized averages."""

    model_id = payload.get("model", "unknown")
    result: Dict[str, object] = {"response": None, "exception": None}

    def worker() -> None:
        start = time.time()
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=None,
            )
            if resp.status_code != 200:
                result["exception"] = RuntimeError(
                    f"Ollama API error: {resp.status_code} {resp.text}"
                )
                return
            text = parse_response(resp)
            wall = time.time() - start
            tracker.record(wall, model_size_b)
            result["response"] = text
        except Exception as exc:  # noqa: BLE001
            result["exception"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    avg_ai = tracker.average()
    expected_wall = avg_ai * ((model_size_b / 7) ** 1.2)
    timeout = expected_wall * timeout_cushion

    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(f"{model_id} exceeded timeout of {timeout:.1f}s")

    if result["exception"] is not None:
        raise result["exception"]  # type: ignore[arg-type]

    return str(result.get("response", ""))
