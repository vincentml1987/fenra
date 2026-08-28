"""
Fenra's Aletheosis - v0

A minimal GUI where Fenra talks to herself, looping against a local Ollama
model. See Qualia/decisions.md for design notes.

Prompt construction each loop tick:
    system = TOP + "\n\n" + BOTTOM
    prompt = TOP + "\n\n" + <Fenra's last response to herself> + "\n\n" + BOTTOM

The full request JSON sent to Ollama is logged with a timestamp so it can be
reviewed later on the History tab.
"""

import json
import os
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk, scrolledtext, messagebox

import requests

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "history.jsonl")

DEFAULT_MODEL = "llama3"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_INTERVAL_SEC = 3


def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def load_history():
    """Load previously logged prompt/response entries from disk."""
    ensure_log_dir()
    entries = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def append_history(entry):
    ensure_log_dir()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


class FenraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fenra's Aletheosis")
        self.root.geometry("900x700")

        self.running = False
        self.loop_thread = None
        self.last_thought = ""
        self.history = load_history()

        self._build_ui()
        self._populate_history_list()

    # ---------------------------------------------------------------- UI --

    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        self.talk_tab = ttk.Frame(notebook)
        self.history_tab = ttk.Frame(notebook)
        notebook.add(self.talk_tab, text="Fenra")
        notebook.add(self.history_tab, text="History")

        self._build_talk_tab()
        self._build_history_tab()

    def _build_talk_tab(self):
        frame = self.talk_tab

        # --- controls row ---
        controls = ttk.Frame(frame)
        controls.pack(fill="x", padx=6, pady=6)

        ttk.Label(controls, text="Model:").pack(side="left")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        ttk.Entry(controls, textvariable=self.model_var, width=18).pack(side="left", padx=(2, 10))

        ttk.Label(controls, text="Host:").pack(side="left")
        self.host_var = tk.StringVar(value=DEFAULT_HOST)
        ttk.Entry(controls, textvariable=self.host_var, width=22).pack(side="left", padx=(2, 10))

        ttk.Label(controls, text="Interval (s):").pack(side="left")
        self.interval_var = tk.StringVar(value=str(DEFAULT_INTERVAL_SEC))
        ttk.Entry(controls, textvariable=self.interval_var, width=5).pack(side="left", padx=(2, 10))

        self.start_stop_btn = ttk.Button(controls, text="Start", command=self.toggle_loop)
        self.start_stop_btn.pack(side="left", padx=(10, 0))

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(controls, textvariable=self.status_var).pack(side="right")

        # --- 10 / 80 / 10 stacked text boxes ---
        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)   # top box    - 10%
        body.rowconfigure(1, weight=8)   # middle box - 80%
        body.rowconfigure(2, weight=1)   # bottom box - 10%

        self.top_box = scrolledtext.ScrolledText(body, wrap="word", height=4)
        self.top_box.grid(row=0, column=0, sticky="nsew", pady=(0, 4))

        self.middle_box = scrolledtext.ScrolledText(body, wrap="word", state="disabled")
        self.middle_box.grid(row=1, column=0, sticky="nsew", pady=4)

        self.bottom_box = scrolledtext.ScrolledText(body, wrap="word", height=4)
        self.bottom_box.grid(row=2, column=0, sticky="nsew", pady=(4, 0))

    def _build_history_tab(self):
        frame = self.history_tab

        paned = ttk.Panedwindow(frame, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(paned, width=180)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=4)

        list_frame = ttk.Frame(left)
        list_frame.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.history_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, exportselection=False)
        scrollbar.config(command=self.history_listbox.yview)
        self.history_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.history_listbox.bind("<<ListboxSelect>>", self._on_history_select)

        self.json_view = scrolledtext.ScrolledText(right, wrap="none", state="disabled")
        self.json_view.pack(fill="both", expand=True)

    # ------------------------------------------------------------ history --

    def _populate_history_list(self):
        self.history_listbox.delete(0, "end")
        for entry in self.history:
            self.history_listbox.insert("end", entry.get("timestamp", "?"))

    def _on_history_select(self, event):
        selection = self.history_listbox.curselection()
        if not selection:
            return
        entry = self.history[selection[0]]
        pretty = json.dumps(entry.get("request", entry), indent=2, ensure_ascii=False)
        self.json_view.config(state="normal")
        self.json_view.delete("1.0", "end")
        self.json_view.insert("end", pretty)
        self.json_view.config(state="disabled")

    # --------------------------------------------------------------- loop --

    def toggle_loop(self):
        if self.running:
            self.running = False
            self.start_stop_btn.config(text="Start")
            self.status_var.set("Stopping...")
        else:
            self.running = True
            self.start_stop_btn.config(text="Stop")
            self.status_var.set("Running")
            self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)
            self.loop_thread.start()

    def _run_loop(self):
        while self.running:
            try:
                self._tick()
            except Exception as exc:  # keep the loop alive on transient errors
                self.root.after(0, self._set_status, f"Error: {exc}")
            try:
                interval = float(self.interval_var.get())
            except ValueError:
                interval = DEFAULT_INTERVAL_SEC
            for _ in range(int(interval * 10)):
                if not self.running:
                    break
                time.sleep(0.1)
        self.root.after(0, self._set_status, "Idle")

    def _set_status(self, text):
        self.status_var.set(text)

    def _tick(self):
        top_text = self.top_box.get("1.0", "end-1c")
        bottom_text = self.bottom_box.get("1.0", "end-1c")

        system_prompt = f"{top_text}\n\n{bottom_text}".strip()
        prompt = f"{top_text}\n\n{self.last_thought}\n\n{bottom_text}".strip()

        payload = {
            "model": self.model_var.get().strip() or DEFAULT_MODEL,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
        }

        timestamp = datetime.now().isoformat(timespec="seconds")
        self.root.after(0, self._set_status, "Thinking...")

        host = self.host_var.get().strip().rstrip("/") or DEFAULT_HOST
        response = requests.post(f"{host}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        response_text = response.json().get("response", "").strip()

        entry = {"timestamp": timestamp, "request": payload, "response": response_text}
        self.history.append(entry)
        append_history(entry)

        self.last_thought = response_text

        self.root.after(0, self._append_message, timestamp, response_text)
        self.root.after(0, self._add_history_row, timestamp)
        self.root.after(0, self._set_status, "Running")

    def _append_message(self, timestamp, text):
        self.middle_box.config(state="normal")
        self.middle_box.insert("end", f"[{timestamp}]\n{text}\n\n")
        self.middle_box.see("end")
        self.middle_box.config(state="disabled")

    def _add_history_row(self, timestamp):
        self.history_listbox.insert("end", timestamp)


def main():
    root = tk.Tk()
    app = FenraApp(root)

    def on_close():
        app.running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
