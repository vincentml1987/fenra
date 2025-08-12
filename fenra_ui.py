import json
import logging
import os
import tkinter as tk
from datetime import datetime
from tkinter import scrolledtext, simpledialog
from tkinter import ttk
import configparser

logger = logging.getLogger(__name__)

CHATLOG_DIR = "chatlogs"
QUEUED_FILE = os.path.join(CHATLOG_DIR, "queued_messages.json")
SENT_FILE = os.path.join(CHATLOG_DIR, "messages_to_humans.json")


class FenraUI:
    """Simple UI for displaying output and listing AIs."""

    def __init__(self, agents, inject_callback=None, send_callback=None, config_path="fenra_config.txt"):
        logger.debug(
            "Entering FenraUI.__init__ with agents=%s inject_callback=%s send_callback=%s",
            agents,
            inject_callback,
            send_callback,
        )
        self.root = tk.Tk()
        self.root.title("Fenra")
        self.agents = agents
        self.inject_callback = inject_callback
        self.send_callback = send_callback
        self.config_path = config_path

        self.sent_messages = []
        self.log_messages = []

        parser = configparser.ConfigParser()
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                parser.read_file(f)
        except Exception:  # noqa: BLE001
            parser.read(config_path)
        self.global_config = dict(parser.items("global")) if parser.has_section("global") else {}

        tv = float(self.global_config.get("talkativeness", 0.0))
        rum = float(self.global_config.get("rumination", 0.0))
        fg = float(self.global_config.get("forgetfulness", 0.0))
        bd = float(self.global_config.get("boredom", 0.0))
        ct = float(self.global_config.get("certainty", 0.0))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # ----- Configuration Tab -----
        config_tab = ttk.Frame(self.notebook)
        self.notebook.add(config_tab, text="Configuration")
        btn_frame = ttk.Frame(config_tab)
        btn_frame.pack(fill=tk.X, anchor="e")
        ttk.Button(btn_frame, text="Reload", command=self.reload_config_snapshot).pack(
            side=tk.RIGHT, padx=5, pady=5
        )
        ttk.Button(btn_frame, text="Edit…", command=self.open_config_editor_dialog).pack(
            side=tk.RIGHT, padx=5, pady=5
        )
        self.config_output = scrolledtext.ScrolledText(config_tab, state="disabled", height=12)
        self.config_output.pack(fill=tk.BOTH, expand=True)
        self.reload_config_snapshot()

        # ----- Live Metrics Tab -----
        metrics_tab = ttk.Frame(self.notebook)
        self.notebook.add(metrics_tab, text="Live Metrics")
        self.metric_bars = {}
        self.metric_labels = {}
        metrics = [
            ("Talkativeness", "talkativeness"),
            ("Rumination", "rumination"),
            ("Forgetfulness", "forgetfulness"),
            ("Boredom", "boredom"),
            ("Certainty", "certainty"),
        ]
        for name, key in metrics:
            row = ttk.Frame(metrics_tab)
            row.pack(fill=tk.X, padx=10, pady=5)
            ttk.Label(row, text=name, font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
            bar = ttk.Progressbar(row, maximum=100, mode="determinate")
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            val_label = ttk.Label(row, text="0.00")
            val_label.pack(side=tk.LEFT, padx=5)
            self.metric_bars[key] = bar
            self.metric_labels[key] = val_label

        self.thought_label = self.metric_labels["talkativeness"]
        self.rumination_label = self.metric_labels["rumination"]
        self.forget_label = self.metric_labels["forgetfulness"]
        self.boredom_label = self.metric_labels["boredom"]
        self.certainty_label = self.metric_labels["certainty"]

        # ----- Internal Thoughts Tab -----
        thoughts_tab = ttk.Frame(self.notebook)
        self.notebook.add(thoughts_tab, text="Internal Thoughts")
        paned = ttk.Panedwindow(thoughts_tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        left_frame = ttk.Frame(paned)
        right_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        paned.add(right_frame, weight=1)
        self.thought_text = scrolledtext.ScrolledText(left_frame, state="disabled")
        self.thought_text.pack(fill=tk.BOTH, expand=True)
        self.event_text = scrolledtext.ScrolledText(right_frame, state="disabled")
        self.event_text.pack(fill=tk.BOTH, expand=True)
        self.output = self.thought_text
        self.base_timeout = (
            agents[0].watchdog_timeout if agents and hasattr(agents[0], "watchdog_timeout") else 900
        )
        self.timeout_label = ttk.Label(thoughts_tab, text=f"Base Timeout: {self.base_timeout}s")
        self.timeout_label.pack(anchor="w")

        # ----- Messages Tab -----
        messages_tab = ttk.Frame(self.notebook)
        self.notebook.add(messages_tab, text="Messages")
        top = ttk.Frame(messages_tab)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Refresh", command=self.update_queue_and_sent).pack(
            side=tk.RIGHT, padx=5, pady=5
        )
        queued_frame = ttk.LabelFrame(messages_tab, text="Queued (from humans)")
        queued_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        self.queued_text = scrolledtext.ScrolledText(queued_frame, state="disabled")
        self.queued_text.pack(fill=tk.BOTH, expand=True)
        sent_frame = ttk.LabelFrame(messages_tab, text="Sent (to humans)")
        sent_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        self.sent_text = scrolledtext.ScrolledText(sent_frame, state="disabled")
        self.sent_text.pack(fill=tk.BOTH, expand=True)

        self._refresh_log_display()
        logger.debug(
            "Seeding UI with config weights: talkativeness=%s, rumination=%s, "
            "forgetfulness=%s, boredom=%s, certainty=%s",
            tv,
            rum,
            fg,
            bd,
            ct,
        )
        self.update_weights(tv, rum, fg, bd, ct)
        self.update_queue_and_sent()
        logger.debug("Exiting FenraUI.__init__")

    def _set_text(self, widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.yview(tk.END)
        widget.configure(state="disabled")

    def _append_text(self, widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.insert(tk.END, text)
        widget.yview(tk.END)
        widget.configure(state="disabled")

    def reload_config_snapshot(self) -> None:
        logger.debug("Entering reload_config_snapshot")

        def do() -> None:
            parser = configparser.ConfigParser()
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    parser.read_file(f)
            except Exception:  # noqa: BLE001
                parser.read(self.config_path)
            self.global_config = (
                dict(parser.items("global")) if parser.has_section("global") else {}
            )
            lines = ["Global Configuration:"]
            for k, v in self.global_config.items():
                lines.append(f"{k} = {v}")
            self._set_text(self.config_output, "\n".join(lines) + "\n")

        self.root.after(0, do)
        logger.debug("Exiting reload_config_snapshot")

    def open_config_editor_dialog(self) -> None:
        logger.debug("Entering open_config_editor_dialog")

        def do() -> None:
            dialog = tk.Toplevel(self.root)
            dialog.title("Edit Configuration")
            text = scrolledtext.ScrolledText(dialog, width=60, height=20)
            text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            cfg_lines = ["[global]"] + [f"{k} = {v}" for k, v in self.global_config.items()]
            text.insert("1.0", "\n".join(cfg_lines))
            btn_frame = ttk.Frame(dialog)
            btn_frame.pack(fill=tk.X, pady=5)
            ttk.Button(btn_frame, text="Save (disabled)", state=tk.DISABLED).pack(
                side=tk.RIGHT, padx=5
            )
            ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(
                side=tk.RIGHT, padx=5
            )
            dialog.transient(self.root)
            dialog.grab_set()
            dialog.wait_window()

        self.root.after(0, do)
        logger.debug("Exiting open_config_editor_dialog")

    def update_queue_and_sent(self, queued: list | None = None, sent: list | None = None) -> None:
        logger.debug("Entering update_queue_and_sent queued=%s sent=%s", queued, sent)

        def do() -> None:
            q_data = queued
            if q_data is None:
                try:
                    with open(QUEUED_FILE, "r", encoding="utf-8") as f:
                        q_data = json.load(f)
                except Exception:  # noqa: BLE001
                    q_data = None
            if isinstance(q_data, list) and q_data:
                lines = []
                for entry in q_data:
                    ts = entry.get("timestamp", "?")
                    msg = entry.get("message", entry)
                    lines.append(f"[{ts}] {msg}")
                q_content = "\n".join(lines) + "\n"
            else:
                q_content = "No queued messages.\n"
            self._set_text(self.queued_text, q_content)

            s_data = sent
            if s_data is None:
                try:
                    with open(SENT_FILE, "r", encoding="utf-8") as f:
                        s_data = json.load(f)
                except Exception:  # noqa: BLE001
                    s_data = None
            if isinstance(s_data, list) and s_data:
                lines = []
                for entry in s_data:
                    ts = entry.get("timestamp", "?")
                    sender = entry.get("sender", "?")
                    msg = entry.get("message", "")
                    lines.append(f"[{ts}] {sender}: {msg}")
                s_content = "\n".join(lines) + "\n"
            else:
                s_content = "No sent messages.\n"
            self._set_text(self.sent_text, s_content)

        self.root.after(0, do)
        logger.debug("Exiting update_queue_and_sent")

    def append_thought(self, text: str, timestamp: str | None = None) -> None:
        logger.debug("Entering append_thought text=%s timestamp=%s", text, timestamp)
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M:%S")

        def do() -> None:
            self._append_text(self.thought_text, f"[{timestamp}] {text}\n")

        self.root.after(0, do)
        logger.debug("Exiting append_thought")

    def append_event(self, text: str, timestamp: str | None = None) -> None:
        logger.debug("Entering append_event text=%s timestamp=%s", text, timestamp)
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M:%S")

        def do() -> None:
            self._append_text(self.event_text, f"[{timestamp}] {text}\n")

        self.root.after(0, do)
        logger.debug("Exiting append_event")


    class _InjectDialog(simpledialog.Dialog):
        """Dialog for entering a message to inject."""

        def __init__(self, parent, group_name: str):
            logger.debug("Entering _InjectDialog.__init__ group_name=%s", group_name)
            self.group_name = group_name
            self.message = ""
            super().__init__(parent, title="Inject Message")
            logger.debug("Exiting _InjectDialog.__init__")

        def body(self, master):
            logger.debug("Entering _InjectDialog.body")
            tk.Label(master, text=f"Send message to {self.group_name}:").grid(row=0, column=0, sticky="w")
            self.text = scrolledtext.ScrolledText(master, width=40, height=10)
            self.text.grid(row=1, column=0, sticky="nsew")
            master.grid_rowconfigure(1, weight=1)
            master.grid_columnconfigure(0, weight=1)
            logger.debug("Exiting _InjectDialog.body")
            return self.text

        def buttonbox(self):
            box = tk.Frame(self)
            send = tk.Button(box, text="Send", width=10, command=self.ok, default=tk.ACTIVE)
            send.pack(side=tk.LEFT, padx=5, pady=5)
            cancel = tk.Button(box, text="Cancel", width=10, command=self.cancel)
            cancel.pack(side=tk.LEFT, padx=5, pady=5)
            self.bind("<Escape>", self.cancel)
            box.pack()

        def apply(self):
            logger.debug("Entering _InjectDialog.apply")
            self.message = self.text.get("1.0", tk.END).rstrip()
            self.result = self.message
            logger.debug("Exiting _InjectDialog.apply")

    class _SendDialog(simpledialog.Dialog):
        """Dialog for entering a message for the listeners."""

        def body(self, master):
            logger.debug("Entering _SendDialog.body")
            tk.Label(master, text="Message to user:").grid(row=0, column=0, sticky="w")
            self.text = scrolledtext.ScrolledText(master, width=40, height=10)
            self.text.grid(row=1, column=0, sticky="nsew")
            master.grid_rowconfigure(1, weight=1)
            master.grid_columnconfigure(0, weight=1)
            logger.debug("Exiting _SendDialog.body")
            return self.text

        def buttonbox(self):
            box = tk.Frame(self)
            send = tk.Button(box, text="Send", width=10, command=self.ok, default=tk.ACTIVE)
            send.pack(side=tk.LEFT, padx=5, pady=5)
            cancel = tk.Button(box, text="Cancel", width=10, command=self.cancel)
            cancel.pack(side=tk.LEFT, padx=5, pady=5)
            self.bind("<Escape>", self.cancel)
            box.pack()

        def apply(self):
            logger.debug("Entering _SendDialog.apply")
            self.message = self.text.get("1.0", tk.END).rstrip()
            self.result = self.message
            logger.debug("Exiting _SendDialog.apply")

    def _inject_message(self):
        logger.debug("Entering _inject_message")
        if not self.inject_callback:
            logger.debug("Exiting _inject_message: no callback")
            return
        group_name = "All Groups"
        dialog = self._InjectDialog(self.root, group_name)
        result = dialog.result
        if result:
            self.inject_callback(group_name, result)
        logger.debug("Exiting _inject_message")

    def _send_message(self):
        logger.debug("Entering _send_message")
        if not self.send_callback:
            logger.debug("Exiting _send_message: no callback")
            return
        dialog = self._SendDialog(self.root)
        result = dialog.result
        if result:
            self.send_callback(result)
        logger.debug("Exiting _send_message")

    def update_queue(self, messages):
        logger.debug("Entering update_queue messages=%s", messages)
        self.update_queue_and_sent(queued=messages)
        logger.debug("Exiting update_queue")

    def update_sent(self, messages):
        logger.debug("Entering update_sent messages=%s", messages)
        self.sent_messages = list(messages)
        self.update_queue_and_sent(sent=messages)
        logger.debug("Exiting update_sent")

    def update_weights(
        self,
        talkativeness: float,
        rumination: float,
        forgetfulness: float,
        boredom: float = 0.0,
        certainty: float = 0.0,
    ) -> None:
        logger.debug(
            "Entering update_weights talkativeness=%s rumination=%s forgetfulness=%s boredom=%s certainty=%s",
            talkativeness,
            rumination,
            forgetfulness,
            boredom,
            certainty,
        )

        def do() -> None:
            values = {
                "talkativeness": talkativeness,
                "rumination": rumination,
                "forgetfulness": forgetfulness,
                "boredom": boredom,
                "certainty": certainty,
            }
            for key, val in values.items():
                if key in self.metric_labels:
                    self.metric_labels[key].config(text=f"{val:.2f}")
                if key in self.metric_bars:
                    prog = max(0.0, min(100.0, val * 100.0))
                    self.metric_bars[key]["value"] = prog

        self.root.after(0, do)
        logger.debug("Exiting update_weights")

    def _expand_all(self):
        logger.debug("_expand_all called but tree view removed")

    def _collapse_all(self):
        logger.debug("_collapse_all called but tree view removed")

    def _send_from_box(self):
        logger.debug("_send_from_box called but message box removed")

    def _refresh_chat_display(self):
        logger.debug("_refresh_chat_display called but chat display removed")

    def _refresh_log_display(self):
        logger.debug("Entering _refresh_log_display")
        content = ""
        for m in self.log_messages:
            content += (
                f"[{m['timestamp']}] {m['sender']}: {m['message']}\n{'-'*80}\n\n"
            )
        self._set_text(self.output, content)
        logger.debug("Exiting _refresh_log_display")


    def log(self, entry):
        logger.debug("Entering log entry=%s", entry)
        self.log_messages.append(entry)
        self._refresh_log_display()
        logger.debug("Exiting log")


    def start(self):
        logger.debug("Entering start")
        self.root.mainloop()
        logger.debug("Exiting start")
