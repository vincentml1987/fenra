"""ClientState plus the two long-lived daemon loops: heartbeat and
work-poll/relay. The two must stay independent - a slow local model
(command-r:35b, mistral-small:22b) blocking the work loop must never
cause the heartbeat loop to miss its window and get this host falsely
marked offline.

All UI updates happen through a caller-supplied on_update(fn) callback
so this module never touches Tkinter directly - app.py wires that up as
self.root.after(0, fn), matching fenra.py's own discipline of never
mutating widgets from a background thread.
"""
import threading
import time

from . import net, ollama_relay


def _sleep_in_slices(total_seconds, should_stop):
    """Sleeps in 0.1s slices, matching fenra.py's own convention, so a
    stop/pause request is noticed promptly instead of after one long
    time.sleep(interval)."""
    remaining = total_seconds
    while remaining > 0 and not should_stop():
        step = min(0.1, remaining)
        time.sleep(step)
        remaining -= step


class ClientState:
    def __init__(self, config_state):
        self.config = config_state
        self.lock = threading.Lock()
        self.running_flag = False
        self.paused = False
        self.status = "idle"  # idle | running | paused
        self.running_model = None
        self.active_process = None
        self.kill_requested = False

    def set_paused(self, value):
        with self.lock:
            self.paused = bool(value)
            self.status = "paused" if self.paused else "idle"

    def request_kill(self):
        """Sets a flag that ollama_relay.run_job's own polling loop
        checks and acts on (terminating its child process) - this
        method does not terminate anything itself, since the Process
        object is only safely touched from within run_job's own
        lock-guarded access."""
        with self.lock:
            self.kill_requested = True

    def snapshot(self):
        with self.lock:
            return {
                "status": self.status,
                "running_model": self.running_model,
                "paused": self.paused,
            }


class ClientWorker:
    def __init__(self, state, on_update=None, on_error=None):
        self.state = state
        self.on_update = on_update or (lambda: None)
        self.on_error = on_error or (lambda msg: None)
        self._threads = []

    def start(self):
        self.state.running_flag = True
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="fenra-client-heartbeat"
        )
        work_thread = threading.Thread(
            target=self._work_loop, daemon=True, name="fenra-client-work"
        )
        self._threads = [heartbeat_thread, work_thread]
        heartbeat_thread.start()
        work_thread.start()

    def stop(self):
        self.state.running_flag = False

    def _should_stop(self):
        return not self.state.running_flag

    def _heartbeat_loop(self):
        while self.state.running_flag:
            try:
                models = ollama_relay.list_models(self.state.config["ollama_host"])
                snap = self.state.snapshot()
                net.send_heartbeat(
                    self.state.config, models, snap["status"], snap["running_model"]
                )
            except Exception as exc:
                self.on_error(f"Heartbeat failed: {exc}")
            self.on_update()
            _sleep_in_slices(
                self.state.config["heartbeat_interval_sec"], self._should_stop
            )

    def _work_loop(self):
        while self.state.running_flag:
            with self.state.lock:
                paused = self.state.paused
            if paused:
                _sleep_in_slices(
                    self.state.config["work_poll_interval_sec"], self._should_stop
                )
                continue

            job = None
            try:
                job = net.poll_work(self.state.config)
            except Exception as exc:
                self.on_error(f"Work poll failed: {exc}")

            if job is not None:
                self._run_job(job)
            else:
                _sleep_in_slices(
                    self.state.config["work_poll_interval_sec"], self._should_stop
                )

    def _run_job(self, job):
        job_id = job["job_id"]
        kind = job["kind"]
        ollama_request = job["ollama_request"]

        with self.state.lock:
            self.state.status = "running"
            self.state.running_model = ollama_request.get("model")
            self.state.kill_requested = False
        self.on_update()

        outcome_kwargs = {}
        try:
            response = ollama_relay.run_job(
                self.state.config["ollama_host"], kind, ollama_request, self.state
            )
            outcome_kwargs = {"outcome": "ok", "ollama_response": response}
        except ollama_relay.KilledError as exc:
            outcome_kwargs = {
                "outcome": "error",
                "error_kind": "killed",
                "error_detail": str(exc),
            }
        except ollama_relay.MalformedResponseError as exc:
            outcome_kwargs = {
                "outcome": "error",
                "error_kind": "malformed_response",
                "error_detail": str(exc),
            }
        except Exception as exc:
            outcome_kwargs = {
                "outcome": "error",
                "error_kind": "ollama_error",
                "error_detail": str(exc),
            }

        with self.state.lock:
            self.state.status = "idle" if not self.state.paused else "paused"
            self.state.running_model = None
        self.on_update()

        try:
            net.submit_result(self.state.config, job_id, **outcome_kwargs)
        except net.StaleJobError:
            pass  # server already gave up on this job - expected, not an error
        except Exception as exc:
            self.on_error(f"Result submission failed: {exc}")
