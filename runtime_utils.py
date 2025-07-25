import json
import os
import re
import threading
import time
from collections import deque

from typing import Dict, Optional
import ctypes

import logging
import requests


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
GLOBAL_LOG_LEVEL = logging.INFO

logger = logging.getLogger(__name__)


def parse_log_level(level_name: str) -> int:
    """Return a logging level constant from a name."""
    logger.debug("Entering parse_log_level level_name=%s", level_name)
    result = getattr(logging, level_name.upper(), logging.INFO)
    logger.debug("Exiting parse_log_level")
    return result


def init_global_logging(level: int) -> None:
    """Initialize root logging and store the global level."""
    logger.debug("Entering init_global_logging level=%s", level)
    global GLOBAL_LOG_LEVEL
    GLOBAL_LOG_LEVEL = level
    os.makedirs("logs", exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "fenra.log"), mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers)
    logger.debug("Exiting init_global_logging")


def create_object_logger(class_name: str) -> logging.Logger:
    """Return a logger that propagates to the global handlers."""
    # Remove characters that could result in invalid names.
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "", class_name)

    logger = logging.getLogger(safe_name)
    logger.handlers.clear()
    logger.setLevel(GLOBAL_LOG_LEVEL)
    logger.propagate = True
    return logger


class AITimeTracker:
    """Track recent successful generation times in AI Seconds."""

    def __init__(self, rolling_window: int = 10) -> None:
        logger.debug("Entering AITimeTracker.__init__ rolling_window=%s", rolling_window)
        self.data = deque(maxlen=rolling_window)
        self.logger = create_object_logger(self.__class__.__name__)
        self.logger.info("Initialized AITimeTracker")
        logger.debug("Exiting AITimeTracker.__init__")

    def record(self, wall_time: float, model_size_gb: float) -> None:
        self.logger.debug(
            "Entering AITimeTracker.record wall_time=%s model_size_gb=%s",
            wall_time,
            model_size_gb,
        )
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
        self.logger.debug("Exiting AITimeTracker.record")

    def average(self) -> float:
        self.logger.debug("Entering AITimeTracker.average")
        """Return the rolling average AI Seconds with a 300s baseline."""
        total = sum(self.data) + 300
        count = len(self.data) + 1
        avg = total / count
        self.logger.debug("Average AI seconds: %.2f", avg)
        self.logger.debug("Exiting AITimeTracker.average")
        return avg


class ThreadTimeTracker:
    """Manage per-thread time trackers and compute a global average."""

    def __init__(self, rolling_window: int = 10) -> None:
        logger.debug("Entering ThreadTimeTracker.__init__ rolling_window=%s", rolling_window)
        self.trackers: Dict[int, AITimeTracker] = {}
        self.rolling_window = rolling_window
        self.logger = create_object_logger(self.__class__.__name__)
        self.logger.info("Initialized ThreadTimeTracker")
        logger.debug("Exiting ThreadTimeTracker.__init__")

    def _tracker_for(self, tid: int) -> AITimeTracker:
        self.logger.debug("Entering ThreadTimeTracker._tracker_for tid=%s", tid)
        if tid not in self.trackers:
            self.trackers[tid] = AITimeTracker(self.rolling_window)
        tracker = self.trackers[tid]
        self.logger.debug("Exiting ThreadTimeTracker._tracker_for")
        return tracker

    def record(self, tid: int, wall_time: float, model_size_gb: float) -> None:
        self.logger.debug(
            "Entering ThreadTimeTracker.record tid=%s wall_time=%s model_size_gb=%s",
            tid,
            wall_time,
            model_size_gb,
        )
        self._tracker_for(tid).record(wall_time, model_size_gb)
        self.logger.debug("Exiting ThreadTimeTracker.record")

    def average(self) -> float:
        self.logger.debug("Entering ThreadTimeTracker.average")
        total = 300
        count = 1
        for tracker in self.trackers.values():
            total += sum(tracker.data)
            count += len(tracker.data)
        avg = total / count
        self.logger.debug("Global average AI seconds: %.2f", avg)
        self.logger.debug("Exiting ThreadTimeTracker.average")
        return avg


WATCHDOG_TRACKER = ThreadTimeTracker()


def parse_model_size(model_id: str) -> float:
    """Return the disk size of the model in gigabytes."""
    logger = logging.getLogger("ModelSize")
    logger.debug("Entering parse_model_size model_id=%s", model_id)
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
    logger.debug("Exiting parse_model_size")
    return 7.0


def parse_response(resp: requests.Response) -> str:
    """Parse a non-streaming Ollama response into text."""
    logger.debug("Entering parse_response")
    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid JSON from Ollama") from exc
    logger.debug("Exiting parse_response")
    return data.get("response", "")


def strip_think_markup(text: str) -> str:
    """Return text with any <think>...</think> sections removed."""
    logger.debug("Entering strip_think_markup")
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    logger.debug("Exiting strip_think_markup")
    return result


def kill_thread(thread: threading.Thread) -> None:
    """Forcefully stop a thread by raising SystemExit within it."""
    logger.debug("Entering kill_thread thread=%s", thread)
    ident: Optional[int] = thread.ident
    if ident is None:
        logger.debug("Exiting kill_thread: no ident")
        return
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(ident), ctypes.py_object(SystemExit)
    )
    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(ident), 0)
        raise SystemError("PyThreadState_SetAsyncExc failed")
    logger.debug("Exiting kill_thread")


def generate_with_watchdog(
    payload: Dict,
    model_size_gb: float,
    tracker: ThreadTimeTracker,
    timeout_cushion: float = 2.0,
) -> str:
    """Call Ollama with a watchdog timeout based on normalized averages.

    If a timeout occurs, retry indefinitely while gradually increasing the
    timeout cushion.
    """
    logger.debug(
        "Entering generate_with_watchdog model_size_gb=%s timeout_cushion=%s",
        model_size_gb,
        timeout_cushion,
    )

    model_id = payload.get("model", "unknown")
    wd_logger = create_object_logger(f"Watchdog-{model_id}")
    attempt = 1

    while True:
        wd_logger.info("Starting generation with watchdog (attempt %d)", attempt)

        start = time.time()
        avg_ai = tracker.average()
        expected_wall = avg_ai * ((model_size_gb / 7) ** 1.2)
        timeout = expected_wall * timeout_cushion
        wd_logger.debug("Timeout set to %.2fs", timeout)
        try:
            if wd_logger.isEnabledFor(logging.DEBUG):
                wd_logger.debug(
                    "Payload to Ollama:\n%s", json.dumps(payload, indent=2)
                )
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=timeout,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Ollama API error: {resp.status_code} {resp.text}"
                )
            text = parse_response(resp)
            wall = time.time() - start
            tracker.record(threading.get_ident(), wall, model_size_gb)
            wd_logger.info("Generation complete")
            logger.debug("Exiting generate_with_watchdog")
            return str(text)
        except requests.Timeout:
            wd_logger.error(
                "Timeout exceeded on attempt %d; increasing cushion", attempt
            )
            timeout_cushion *= 1.5
            attempt += 1
            continue
        except Exception as exc:  # noqa: BLE001
            wd_logger.error("Exception during generation: %s", exc)
            raise
