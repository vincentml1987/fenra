"""
Fenra's Aletheosis

A minimal GUI where Fenra talks to herself, looping against a local Ollama
model. See Qualia/decisions.md for design notes.

Prompt construction each loop tick:
    system = TOP + "\n\n" + BOTTOM
    prompt = TOP + "\n\n" + <Fenra's last response to herself> + "\n\n"
             + <her current desire, if any> + "\n\n" + BOTTOM + "\n\n"
             + <chat status notice - always present>

Everything about a run - the top/bottom boxes, model/host/interval, the
conversation so far, and every request/response - lives in a "session"
under sessions/<name>/, so different experiments (different models,
different framings) don't clobber each other and nothing is lost between
runs of the app.

Fenra can call functions by speaking ⟦function_name(args)⟧ inline in her
response. Every call is executed against an explicit whitelist (never
eval'd), logged to sessions/<name>/functions.jsonl, and a ⟦RESULT: ...⟧
annotation is appended after her response - both in the middle box and in
what gets fed back to her as her own last thought next cycle. She doesn't
need every function explained up front: ⟦functions()⟧ lists everything
available, and ⟦functions(search term)⟧ filters by it.

The functions themselves live in fenra_functions.py, which is hot-reloaded
every tick - add, fix, or reword a function there and it's live on her very
next cycle, no restart, no interrupting a running session.
"""

import importlib
import json
import os
import re
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk, scrolledtext, messagebox, simpledialog

import requests

import fenra_functions

# Bumped on every functionally meaningful change to fenra.py. Stamped into
# every session save and every history entry, so it's always possible to
# tell exactly which version of the code produced a given response - see
# git log for the commit each version corresponds to.
#   0.1.0 - initial GUI: two tabs, top/middle/bottom boxes, self-talk loop
#   0.2.0 - live Ollama model dropdown (hot-swap)
#   0.3.0 - Sessions: save/load named runs instead of one flat log
#   0.4.0 - configurable max_tokens + unbounded timeout fix, function calling
#   0.4.1 - now() function, added in response to her spontaneously trying it
#   0.5.0 - functions moved to fenra_functions.py, hot-reloaded every tick
#   0.6.0 - desire: get_desire()/set_desire(text), a persistent text slot
#           she alone can write, visible read-only in the GUI, sitting in
#           the prompt between her last thought and the bottom box
#   0.7.0 - Chat tab: Teddy can message her directly. Per-message read
#           status, a chat-status notice always appended at the end of her
#           prompt, and read_chat/read_chat_since/read_chat_between/
#           search_chat/send_message functions. Function args now split on
#           | instead of comma, so free text (desire, chat messages) can
#           safely contain commas.
FENRA_VERSION = "0.7.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")

DEFAULT_MODEL = "llama3"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_INTERVAL_SEC = 3
DEFAULT_MAX_TOKENS = 500  # num_predict; blank/0 = unlimited (let Ollama run until it stops or hits context)
DEFAULT_SESSION_NAME = "default"

# No fixed HTTP timeout: some models (heavy CPU offload, big params) are
# legitimately slow. A client-side timeout doesn't cancel server-side
# generation - it just abandons the connection and retries, which can pile
# up into an infinite loop that never completes. Response length is bounded
# by max_tokens (num_predict) instead.
REQUEST_TIMEOUT = None

STATE_FILENAME = "state.json"
HISTORY_FILENAME = "history.jsonl"
FUNCTIONS_FILENAME = "functions.jsonl"
CHAT_FILENAME = "chat.jsonl"

# Function-call syntax: Fenra speaks ⟦name(args)⟧ inline in her response to
# invoke a function. ⟦ ⟧ (U+27E6/U+27E7, mathematical white square brackets)
# are essentially never produced in ordinary code or prose, so this is safe
# to detect without false positives. Anything matched here is executed
# against an explicit whitelist below - never eval'd.
FUNCTION_CALL_RE = re.compile(r"⟦\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s*⟧", re.DOTALL)


def _split_args(raw_args):
    """Arguments are separated by | , not commas - deliberate, so a single
    free-text argument (a desire, a chat message) can contain commas and
    ordinary punctuation without being split apart. A function that needs
    more than one argument documents the | in its own description."""
    raw_args = raw_args.strip()
    if not raw_args:
        return []
    return [a.strip().strip("'\"") for a in raw_args.split("|")]


def reload_function_registry():
    """Hot-reload fenra_functions.py so edits to it (new functions, fixed
    descriptions, whatever) take effect on the very next tick, without
    restarting the app or interrupting a running session. If the file has
    a syntax/import error, keep using the last good version instead of
    crashing the loop."""
    try:
        importlib.reload(fenra_functions)
    except Exception as exc:
        return fenra_functions.FUNCTION_REGISTRY, str(exc)
    return fenra_functions.FUNCTION_REGISTRY, None


def append_session_functions(name, entry):
    ensure_session_dir(name)
    path = os.path.join(session_dir(name), FUNCTIONS_FILENAME)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def run_function_calls(app, response_text):
    """Find every ⟦call⟧ in response_text, execute it, log it, and return
    a list of result-annotation strings to append after the response."""
    registry, reload_error = reload_function_registry()
    if reload_error:
        app.root.after(0, app._set_status, f"functions module error (using last good version): {reload_error}")

    result_lines = []
    for name, raw_args in FUNCTION_CALL_RE.findall(response_text):
        args = _split_args(raw_args)
        call_entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "function": name,
            "args": args,
        }
        if name in registry:
            try:
                result = registry[name]["fn"](app, args)
                call_entry["success"] = True
                call_entry["result"] = result
            except Exception as exc:
                call_entry["success"] = False
                call_entry["result"] = str(exc)
        else:
            call_entry["success"] = False
            call_entry["result"] = f"unknown function '{name}'"

        append_session_functions(app.session_name, call_entry)
        status = "ok" if call_entry["success"] else "error"
        result_lines.append(f"⟦RESULT: {name} -> {status}: {call_entry['result']}⟧")
    return result_lines


def sanitize_session_name(name):
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name


def list_sessions():
    if not os.path.isdir(SESSIONS_DIR):
        return []
    names = [
        d for d in os.listdir(SESSIONS_DIR)
        if os.path.isdir(os.path.join(SESSIONS_DIR, d))
    ]
    # most recently modified (by state.json) first
    def sort_key(name):
        path = os.path.join(SESSIONS_DIR, name, STATE_FILENAME)
        return os.path.getmtime(path) if os.path.exists(path) else 0

    return sorted(names, key=sort_key, reverse=True)


def session_dir(name):
    return os.path.join(SESSIONS_DIR, name)


def ensure_session_dir(name):
    path = session_dir(name)
    os.makedirs(path, exist_ok=True)
    return path


def default_state():
    return {
        "top": "",
        "bottom": "",
        "model": DEFAULT_MODEL,
        "host": DEFAULT_HOST,
        "interval": DEFAULT_INTERVAL_SEC,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "last_thought": "",
        "desire": "",
    }


def load_session_state(name):
    path = os.path.join(session_dir(name), STATE_FILENAME)
    state = default_state()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                state.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return state


def save_session_state(name, state):
    state = dict(state)
    state["fenra_version"] = FENRA_VERSION
    ensure_session_dir(name)
    path = os.path.join(session_dir(name), STATE_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_session_history(name):
    path = os.path.join(session_dir(name), HISTORY_FILENAME)
    entries = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def append_session_history(name, entry):
    ensure_session_dir(name)
    path = os.path.join(session_dir(name), HISTORY_FILENAME)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_chat_messages(name):
    """Chat messages (unlike history) get mutated in place - marking one
    read - so unlike history.jsonl's append-only log, this file is always
    rewritten in full via save_chat_messages rather than appended to."""
    path = os.path.join(session_dir(name), CHAT_FILENAME)
    messages = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return messages


def save_chat_messages(name, messages):
    ensure_session_dir(name)
    path = os.path.join(session_dir(name), CHAT_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")


class FenraApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Fenra's Aletheosis - v{FENRA_VERSION}")
        self.root.geometry("900x700")

        self.running = False
        self.loop_thread = None
        self.last_thought = ""
        self.history = []
        self.chat_messages = []
        self.session_name = None

        self._build_ui()
        self.refresh_models()
        self._startup_session()

    # ------------------------------------------------------------ startup --

    def _startup_session(self):
        sessions = list_sessions()
        name = sessions[0] if sessions else DEFAULT_SESSION_NAME
        self._load_session(name)

    # ---------------------------------------------------------------- UI --

    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        self.talk_tab = ttk.Frame(notebook)
        self.chat_tab = ttk.Frame(notebook)
        self.history_tab = ttk.Frame(notebook)
        notebook.add(self.talk_tab, text="Fenra")
        notebook.add(self.chat_tab, text="Chat")
        notebook.add(self.history_tab, text="History")

        self._build_talk_tab()
        self._build_chat_tab()
        self._build_history_tab()

    def _build_talk_tab(self):
        frame = self.talk_tab

        # --- session row ---
        session_row = ttk.Frame(frame)
        session_row.pack(fill="x", padx=6, pady=(6, 0))

        ttk.Label(session_row, text="Session:").pack(side="left")
        self.session_var = tk.StringVar(value="")
        self.session_combo = ttk.Combobox(session_row, textvariable=self.session_var, width=24, state="readonly")
        self.session_combo.pack(side="left", padx=(2, 4))
        self.session_combo.bind("<<ComboboxSelected>>", self._on_session_selected)

        ttk.Button(session_row, text="New...", command=self.new_session).pack(side="left", padx=2)
        ttk.Button(session_row, text="Save", command=self.save_session).pack(side="left", padx=2)
        ttk.Button(session_row, text="↻", width=3, command=self._refresh_session_list).pack(side="left", padx=2)

        self.session_status_var = tk.StringVar(value="")
        ttk.Label(session_row, textvariable=self.session_status_var, foreground="#666").pack(side="left", padx=(10, 0))

        # --- controls row ---
        controls = ttk.Frame(frame)
        controls.pack(fill="x", padx=6, pady=6)

        ttk.Label(controls, text="Host:").pack(side="left")
        self.host_var = tk.StringVar(value=DEFAULT_HOST)
        ttk.Entry(controls, textvariable=self.host_var, width=22).pack(side="left", padx=(2, 10))

        ttk.Label(controls, text="Model:").pack(side="left")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.model_combo = ttk.Combobox(controls, textvariable=self.model_var, width=20, state="readonly")
        self.model_combo.pack(side="left", padx=(2, 2))
        ttk.Button(controls, text="↻", width=3, command=self.refresh_models).pack(side="left", padx=(0, 10))

        ttk.Label(controls, text="Interval (s):").pack(side="left")
        self.interval_var = tk.StringVar(value=str(DEFAULT_INTERVAL_SEC))
        ttk.Entry(controls, textvariable=self.interval_var, width=5).pack(side="left", padx=(2, 10))

        ttk.Label(controls, text="Max tokens:").pack(side="left")
        self.max_tokens_var = tk.StringVar(value=str(DEFAULT_MAX_TOKENS))
        ttk.Entry(controls, textvariable=self.max_tokens_var, width=6).pack(side="left", padx=(2, 10))

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
        body.rowconfigure(2, weight=0)   # desire row - fixed height
        body.rowconfigure(3, weight=1)   # bottom box - 10%

        self.top_box = scrolledtext.ScrolledText(body, wrap="word", height=4)
        self.top_box.grid(row=0, column=0, sticky="nsew", pady=(0, 4))

        self.middle_box = scrolledtext.ScrolledText(body, wrap="word", state="disabled")
        self.middle_box.grid(row=1, column=0, sticky="nsew", pady=4)

        # Desire: set only by Fenra herself (via set_desire), visible here but
        # not editable from the GUI. Sits in the prompt between her last
        # thought and the bottom box - see _tick.
        desire_row = ttk.Frame(body)
        desire_row.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        desire_row.columnconfigure(1, weight=1)
        ttk.Label(desire_row, text="Desire:").grid(row=0, column=0, sticky="w")
        self.desire_var = tk.StringVar(value="")
        ttk.Entry(desire_row, textvariable=self.desire_var, state="readonly").grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.bottom_box = scrolledtext.ScrolledText(body, wrap="word", height=4)
        self.bottom_box.grid(row=3, column=0, sticky="nsew", pady=(4, 0))

    def _build_chat_tab(self):
        frame = self.chat_tab

        self.chat_box = scrolledtext.ScrolledText(frame, wrap="word", state="disabled")
        self.chat_box.pack(fill="both", expand=True, padx=6, pady=6)

        entry_row = ttk.Frame(frame)
        entry_row.pack(fill="x", padx=6, pady=(0, 6))
        self.chat_entry_var = tk.StringVar(value="")
        chat_entry = ttk.Entry(entry_row, textvariable=self.chat_entry_var)
        chat_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        chat_entry.bind("<Return>", lambda event: self.send_chat_from_ui())
        ttk.Button(entry_row, text="Send", command=self.send_chat_from_ui).pack(side="left")

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

    # ------------------------------------------------------------ session --

    def _refresh_session_list(self):
        sessions = list_sessions()
        if self.session_name and self.session_name not in sessions:
            sessions.insert(0, self.session_name)
        self.session_combo["values"] = sessions
        if self.session_name:
            self.session_var.set(self.session_name)

    def _on_session_selected(self, event):
        chosen = self.session_var.get()
        if chosen and chosen != self.session_name:
            self._load_session(chosen)

    def new_session(self):
        name = simpledialog.askstring("New Session", "Session name:", parent=self.root)
        if not name:
            return
        name = sanitize_session_name(name)
        if not name:
            return
        if name in list_sessions():
            if not messagebox.askyesno("Fenra", f'Session "{name}" already exists. Load it instead?'):
                return
            self._load_session(name)
            return

        if self.running:
            self.toggle_loop()

        # keep the current top/bottom framing as a starting point, but start
        # the conversation itself (last thought, transcript, log) fresh.
        state = {
            "top": self.top_box.get("1.0", "end-1c"),
            "bottom": self.bottom_box.get("1.0", "end-1c"),
            "model": self.model_var.get(),
            "host": self.host_var.get(),
            "interval": self.interval_var.get(),
            "max_tokens": self.max_tokens_var.get(),
            "last_thought": "",
            "desire": "",
        }
        ensure_session_dir(name)
        save_session_state(name, state)
        open(os.path.join(session_dir(name), HISTORY_FILENAME), "a", encoding="utf-8").close()

        self._load_session(name)

    def save_session(self):
        if not self.session_name:
            return
        state = {
            "top": self.top_box.get("1.0", "end-1c"),
            "bottom": self.bottom_box.get("1.0", "end-1c"),
            "model": self.model_var.get(),
            "host": self.host_var.get(),
            "interval": self.interval_var.get(),
            "max_tokens": self.max_tokens_var.get(),
            "last_thought": self.last_thought,
            "desire": self.desire_var.get(),
        }
        save_session_state(self.session_name, state)
        self.session_status_var.set(f"Saved {datetime.now().strftime('%H:%M:%S')}")

    def _load_session(self, name):
        if self.running:
            self.toggle_loop()

        state = load_session_state(name)
        self.session_name = name

        self.top_box.delete("1.0", "end")
        self.top_box.insert("end", state.get("top", ""))
        self.bottom_box.delete("1.0", "end")
        self.bottom_box.insert("end", state.get("bottom", ""))
        self.host_var.set(state.get("host", DEFAULT_HOST))
        self.model_var.set(state.get("model", DEFAULT_MODEL))
        self.interval_var.set(str(state.get("interval", DEFAULT_INTERVAL_SEC)))
        self.max_tokens_var.set(str(state.get("max_tokens", DEFAULT_MAX_TOKENS)))
        self.last_thought = state.get("last_thought", "")
        self.desire_var.set(state.get("desire", ""))

        self.history = load_session_history(name)
        self._populate_history_list()
        self._replay_middle_box()

        self.chat_messages = load_chat_messages(name)
        self._refresh_chat_display()

        self._refresh_session_list()
        self.session_status_var.set(f"Loaded ({len(self.history)} entries)")

    # --------------------------------------------------------------- chat --

    def _next_chat_id(self):
        return max((m.get("id", 0) for m in self.chat_messages), default=0) + 1

    def add_chat_message(self, sender, text, read):
        entry = {
            "id": self._next_chat_id(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "sender": sender,
            "text": text,
            "read": read,
        }
        self.chat_messages.append(entry)
        self.persist_chat()
        return entry

    def persist_chat(self):
        """Save chat_messages to disk and refresh the Chat tab. Safe to call
        from the background loop thread (function calls) or the main thread
        (the Send button) - matches the pattern already used elsewhere for
        cross-thread GUI updates."""
        if self.session_name:
            save_chat_messages(self.session_name, self.chat_messages)
        self.root.after(0, self._refresh_chat_display)

    def _refresh_chat_display(self):
        self.chat_box.config(state="normal")
        self.chat_box.delete("1.0", "end")
        for m in self.chat_messages:
            who = "Teddy" if m["sender"] == "teddy" else "Fenra"
            unread_marker = " [unread]" if m["sender"] == "teddy" and not m.get("read", True) else ""
            self.chat_box.insert("end", f"[{m['timestamp']}] {who}{unread_marker}: {m['text']}\n\n")
        self.chat_box.see("end")
        self.chat_box.config(state="disabled")

    def send_chat_from_ui(self):
        text = self.chat_entry_var.get().strip()
        if not text:
            return
        self.add_chat_message("teddy", text, read=False)
        self.chat_entry_var.set("")

    def _chat_notice(self):
        """Always-present status line appended at the very end of the
        prompt: last sent/received times regardless of unread state, plus
        an explicit unread count and pointer to the chat functions."""
        sent_times = [m["timestamp"] for m in self.chat_messages if m["sender"] == "fenra"]
        received_times = [m["timestamp"] for m in self.chat_messages if m["sender"] == "teddy"]
        last_sent = max(sent_times) if sent_times else "never"
        last_received = max(received_times) if received_times else "never"

        unread = [m for m in self.chat_messages if m["sender"] == "teddy" and not m.get("read", True)]
        if unread:
            unread_note = (
                f"You have {len(unread)} unread message(s) from Teddy. "
                f"Use the chat functions (see ⟦functions()⟧) to review them."
            )
        else:
            unread_note = "You have no unread messages."

        return (
            f"[Chat status: you last sent a message at {last_sent}. "
            f"You last received a message at {last_received}. {unread_note}]"
        )

    # ------------------------------------------------------------ history --

    def _populate_history_list(self):
        self.history_listbox.delete(0, "end")
        for entry in self.history:
            self.history_listbox.insert("end", entry.get("timestamp", "?"))

    def _replay_middle_box(self):
        self.middle_box.config(state="normal")
        self.middle_box.delete("1.0", "end")
        for entry in self.history:
            text = entry.get("display", entry.get("response", ""))
            self.middle_box.insert("end", f"[{entry.get('timestamp', '?')}]\n{text}\n\n")
        self.middle_box.see("end")
        self.middle_box.config(state="disabled")

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

    # --------------------------------------------------------------- model --

    def refresh_models(self):
        """Query Ollama for installed models and populate the dropdown."""
        host = self.host_var.get().strip().rstrip("/") or DEFAULT_HOST
        try:
            resp = requests.get(f"{host}/api/tags", timeout=5)
            resp.raise_for_status()
            names = [m["name"] for m in resp.json().get("models", [])]
        except Exception as exc:
            messagebox.showwarning("Fenra", f"Could not fetch installed models from Ollama:\n{exc}")
            return

        self.model_combo["values"] = names
        if not names:
            return
        # keep current selection if it's still installed, otherwise pick the first
        if self.model_var.get() not in names:
            self.model_var.set(names[0])

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
        desire_text = self.desire_var.get()
        chat_notice = self._chat_notice()

        system_prompt = f"{top_text}\n\n{bottom_text}".strip()
        prompt = f"{top_text}\n\n{self.last_thought}\n\n{desire_text}\n\n{bottom_text}\n\n{chat_notice}".strip()

        payload = {
            "model": self.model_var.get().strip() or DEFAULT_MODEL,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
        }

        try:
            max_tokens = int(float(self.max_tokens_var.get()))
        except ValueError:
            max_tokens = 0
        if max_tokens > 0:
            payload["options"] = {"num_predict": max_tokens}

        timestamp = datetime.now().isoformat(timespec="seconds")
        self.root.after(0, self._set_status, "Thinking...")

        host = self.host_var.get().strip().rstrip("/") or DEFAULT_HOST
        response = requests.post(f"{host}/api/generate", json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        response_text = response.json().get("response", "").strip()

        result_lines = run_function_calls(self, response_text)
        display_text = response_text
        if result_lines:
            display_text = response_text + "\n\n" + "\n".join(result_lines)

        entry = {
            "timestamp": timestamp,
            "fenra_version": FENRA_VERSION,
            "request": payload,
            "response": response_text,
            "display": display_text,
        }
        self.history.append(entry)
        append_session_history(self.session_name, entry)

        self.last_thought = display_text
        save_session_state(self.session_name, {
            "top": top_text,
            "bottom": bottom_text,
            "model": self.model_var.get().strip() or payload["model"],
            "host": host,
            "interval": self.interval_var.get(),
            "max_tokens": self.max_tokens_var.get(),
            "last_thought": self.last_thought,
            "desire": self.desire_var.get(),
        })

        self.root.after(0, self._append_message, timestamp, display_text)
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
        if app.session_name:
            app.save_session()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
