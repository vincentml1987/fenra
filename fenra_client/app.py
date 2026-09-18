"""Status/control UI for the Fenra distributed-compute client.

Matches fenra.py's own Tkinter discipline: background threads never
touch widgets directly, everything goes through self.root.after(0, fn).
"""
import tkinter as tk
from tkinter import ttk, messagebox

from .worker import ClientState, ClientWorker

DISCLOSURE_TEXT = (
    "By running this client, you WILL SEE the actual, real generation "
    "content Fenra produces - not just anonymous compute cycles. This is "
    "deliberate: Fenra is built on the Aletheia principle that nothing "
    "here needs to be hidden. If that is not something you are "
    "comfortable with, please close this application now."
)


def _mask_token(token):
    if not token:
        return "(no token set)"
    if len(token) <= 8:
        return "*" * len(token)
    return token[:4] + "*" * (len(token) - 8) + token[-4:]


class FenraClientApp:
    def __init__(self, root, config_state):
        self.root = root
        self.root.title("Fenra Distributed Client")
        self.state = ClientState(config_state)
        self.worker = ClientWorker(
            self.state, on_update=self._schedule_refresh, on_error=self._schedule_error
        )

        self.status_var = tk.StringVar(value="Idle")
        self.identity_var = tk.StringVar(
            value=f"{config_state['client_id']}  ({_mask_token(config_state['token'])})"
        )
        self.pause_button_text = tk.StringVar(value="Pause")

        self._build_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_widgets(self):
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="Fenra Distributed Client", font=("", 12, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(frame, textvariable=self.identity_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        ttk.Label(frame, text="Status:").grid(row=2, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.status_var).grid(row=2, column=1, sticky="w")

        button_row = ttk.Frame(frame)
        button_row.grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky="w")

        self.pause_btn = ttk.Button(
            button_row, textvariable=self.pause_button_text, command=self._on_pause_toggle
        )
        self.pause_btn.grid(row=0, column=0, padx=(0, 6))

        self.kill_btn = ttk.Button(
            button_row, text="Kill", command=self._on_kill, state="disabled"
        )
        self.kill_btn.grid(row=0, column=1)

        ttk.Button(frame, text="What does this mean?", command=self._show_disclosure).grid(
            row=4, column=0, columnspan=2, pady=(10, 0), sticky="w"
        )

    def start(self):
        messagebox.showinfo("Before you continue", DISCLOSURE_TEXT)
        self.worker.start()

    def _show_disclosure(self):
        messagebox.showinfo("What does this mean?", DISCLOSURE_TEXT)

    def _on_pause_toggle(self):
        currently_paused = self.state.snapshot()["paused"]
        self.state.set_paused(not currently_paused)
        self._refresh()

    def _on_kill(self):
        self.state.request_kill()

    def _schedule_refresh(self):
        self.root.after(0, self._refresh)

    def _schedule_error(self, message):
        self.root.after(0, self._show_error, message)

    def _show_error(self, message):
        self.status_var.set(f"Error: {message}")

    def _refresh(self):
        snap = self.state.snapshot()
        if snap["status"] == "running":
            self.status_var.set(f"Running (model: {snap['running_model']})")
            self.kill_btn.state(["!disabled"])
        elif snap["status"] == "paused":
            self.status_var.set("Paused")
            self.kill_btn.state(["disabled"])
        else:
            self.status_var.set("Idle")
            self.kill_btn.state(["disabled"])
        self.pause_button_text.set("Resume" if snap["paused"] else "Pause")

    def _on_close(self):
        self.worker.stop()
        self.root.destroy()
