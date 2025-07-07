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
    logging.basicConfig(level=level, format=LOG_FORMAT)


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
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(GLOBAL_LOG_LEVEL)
    logger.propagate = False
    return logger


class AITimeTracker:
    """Track recent successful generation times in AI Seconds."""

    def __init__(self, rolling_window: int = 10) -> None:
        self.data = deque(maxlen=rolling_window)
        self.logger = create_object_logger(self.__class__.__name__)
        self.logger.info("Initialized AITimeTracker")

    def record(self, wall_time: float, model_size_b: float) -> None:
        """Convert wall time to AI Seconds (normalized to 7b) and store."""
        factor = (model_size_b / 7) ** 1.2
        ai_seconds = wall_time / factor
        self.data.append(ai_seconds)
        self.logger.debug(
            "Recorded %.2fs wall time (%.2fB) -> %.2f AI seconds",
            wall_time,
            model_size_b,
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

    def record(self, tid: int, wall_time: float, model_size_b: float) -> None:
        self._tracker_for(tid).record(wall_time, model_size_b)

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
    model_size_b: float,
    tracker: ThreadTimeTracker,
    timeout_cushion: float = 2.0,
) -> str:
    """Call Ollama with a watchdog timeout based on normalized averages."""

    model_id = payload.get("model", "unknown")
    logger = create_object_logger(f"Watchdog-{model_id}")
    logger.info("Starting generation with watchdog")
    result: Dict[str, object] = {"response": None, "exception": None}

    def worker() -> None:
        start = time.time()
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Payload to Ollama:\n%s", json.dumps(payload, indent=2))
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
            tracker.record(threading.get_ident(), wall, model_size_b)
            result["response"] = text
        except Exception as exc:  # noqa: BLE001
            result["exception"] = exc
            logger.error("Exception during generation: %s", exc)

    while True:
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        avg_ai = tracker.average()
        expected_wall = avg_ai * ((model_size_b / 7) ** 1.2)
        timeout = expected_wall * timeout_cushion
        logger.debug("Timeout set to %.2fs", timeout)

        thread.join(timeout=timeout)

        if thread.is_alive():
            logger.error("Timeout exceeded, restarting worker")
            kill_thread(thread)
            continue

        if result["exception"] is not None:
            logger.error("Raising exception from worker")
            raise result["exception"]  # type: ignore[arg-type]

        logger.info("Generation complete")
        return str(result.get("response", ""))
