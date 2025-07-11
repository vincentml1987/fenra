import json
import os
import re
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Optional
import ctypes

import logging
import requests


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
GLOBAL_LOG_LEVEL = logging.INFO


def parse_log_level(level_name: str) -> int:
    """Return a logging level constant from a name."""
    return getattr(logging, level_name.upper(), logging.INFO)


def init_global_logging(level: int) -> None:
    """Initialize root logging and store the global level."""
    global GLOBAL_LOG_LEVEL
    GLOBAL_LOG_LEVEL = level
    os.makedirs("logs", exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "global.log"), mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers)


def create_object_logger(class_name: str) -> logging.Logger:
    """Create a per-object logger writing to a timestamped file."""
    os.makedirs("logs", exist_ok=True)
    ts = datetime.utcnow()
    while True:
        stamp = ts.strftime("%Y%m%dT%H%M%SZ")
        fname = f"{class_name}-{stamp}.log"
        path = os.path.join("logs", fname)
        if not os.path.exists(path):
            break
        ts += timedelta(seconds=1)

    logger = logging.getLogger(f"{class_name}-{stamp}")
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(GLOBAL_LOG_LEVEL)
    logger.propagate = False
    return logger


class AITimeTracker:
    """Track recent successful generation times in AI Seconds."""

    def __init__(self, rolling_window: int = 10) -> None:
        self.data = deque(maxlen=rolling_window)
        self.logger = create_object_logger(self.__class__.__name__)
        self.logger.info("Initialized AITimeTracker")

    def record(self, wall_time: float, model_size_gb: float) -> None:
        """Convert wall time to AI Seconds (normalized to 7GB) and store."""
        factor = (model_size_gb / 7) ** 1.2
        ai_seconds = wall_time / factor
        self.data.append(ai_seconds)
        self.logger.debug(
            "Recorded %.2fs wall time (%.2fGB) -> %.2f AI seconds",
            wall_time,
            model_size_gb,
            ai_seconds,
        )

    def average(self) -> float:
        """Return the rolling average AI Seconds with a 300s baseline."""
        total = sum(self.data) + 300
        count = len(self.data) + 1
        avg = total / count
        self.logger.debug("Average AI seconds: %.2f", avg)
        return avg


class ThreadTimeTracker:
    """Manage per-thread time trackers and compute a global average."""

    def __init__(self, rolling_window: int = 10) -> None:
        self.trackers: Dict[int, AITimeTracker] = {}
        self.rolling_window = rolling_window
        self.logger = create_object_logger(self.__class__.__name__)
        self.logger.info("Initialized ThreadTimeTracker")

    def _tracker_for(self, tid: int) -> AITimeTracker:
        if tid not in self.trackers:
            self.trackers[tid] = AITimeTracker(self.rolling_window)
        return self.trackers[tid]

    def record(self, tid: int, wall_time: float, model_size_gb: float) -> None:
        self._tracker_for(tid).record(wall_time, model_size_gb)

    def average(self) -> float:
        total = 300
        count = 1
        for tracker in self.trackers.values():
            total += sum(tracker.data)
            count += len(tracker.data)
        avg = total / count
        self.logger.debug("Global average AI seconds: %.2f", avg)
        return avg


WATCHDOG_TRACKER = ThreadTimeTracker()


def parse_model_size(model_id: str) -> float:
    """Return the disk size of the model in gigabytes."""
    logger = logging.getLogger("ModelSize")
    try:
        import subprocess

        result = subprocess.run(
            ["ollama", "list", model_id], capture_output=True, text=True, check=True
        )
        lines = result.stdout.splitlines()
        for line in lines:
            if model_id in line:
                match = re.search(r"(\d+(?:\.\d+)?)\s*(KB|MB|GB|TB)", line, re.IGNORECASE)
                if not match:
                    break
                value = float(match.group(1))
                unit = match.group(2).upper()
                if unit == "KB":
                    return value / (1024 * 1024)
                if unit == "MB":
                    return value / 1024
                if unit == "GB":
                    return value
                if unit == "TB":
                    return value * 1024
        logger.error("Failed to parse model size from ollama output for %s", model_id)
    except FileNotFoundError:
        logger.error("ollama executable not found")
    except subprocess.CalledProcessError as exc:
        logger.error("Error running ollama list for %s: %s", model_id, exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error obtaining size for %s: %s", model_id, exc)
    return 7.0


def parse_response(resp: requests.Response) -> str:
    """Parse a non-streaming Ollama response into text."""
    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid JSON from Ollama") from exc
    return data.get("response", "")


def strip_think_markup(text: str) -> str:
    """Return text with any <think>...</think> sections removed."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def kill_thread(thread: threading.Thread) -> None:
    """Forcefully stop a thread by raising SystemExit within it."""
    ident: Optional[int] = thread.ident
    if ident is None:
        return
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(ident), ctypes.py_object(SystemExit)
    )
    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(ident), 0)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def generate_with_watchdog(
    payload: Dict,
    model_size_gb: float,
    tracker: ThreadTimeTracker,
    timeout_cushion: float = 2.0,
) -> str:
    """Call Ollama with a watchdog timeout based on normalized averages."""

    model_id = payload.get("model", "unknown")
    logger = create_object_logger(f"Watchdog-{model_id}")
    logger.info("Starting generation with watchdog")

    start = time.time()
    avg_ai = tracker.average()
    expected_wall = avg_ai * ((model_size_gb / 7) ** 1.2)
    timeout = expected_wall * timeout_cushion
    logger.debug("Timeout set to %.2fs", timeout)
    try:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Payload to Ollama:\n%s", json.dumps(payload, indent=2))
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama API error: {resp.status_code} {resp.text}")
        text = parse_response(resp)
        wall = time.time() - start
        tracker.record(threading.get_ident(), wall, model_size_gb)
        logger.info("Generation complete")
        return str(text)
    except requests.Timeout as exc:
        logger.error("Timeout exceeded")
        raise exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Exception during generation: %s", exc)
        raise
