import json
import logging
import os
import threading
import time
from typing import Optional
import tkinter as tk
from tkinter import scrolledtext, simpledialog
from tkinter import ttk
import configparser

logger = logging.getLogger(__name__)

CHATLOG_DIR = "chatlogs"
QUEUED_MESSAGES_PATH = os.path.join(CHATLOG_DIR, "queued_messages.json")
SENT_MESSAGES_PATH = os.path.join(CHATLOG_DIR, "messages_to_humans.json")

ROLE_COLORS = {
    "listener":  "#ff4d4d",  # red
    "ruminator": "#ff8c1a",  # orange
    "doubter":   "#ffd11a",  # yellow
    "ponderer":  "#2ecc71",  # green
    "archivist": "#3498db",  # blue
    "speaker":   "#8e44ad",  # violet
}


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

        # Extract initial weights so the UI reflects configured values immediately
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

        cfg_top = ttk.Frame(config_tab)
        cfg_top.pack(fill=tk.X, pady=2)
        ttk.Button(cfg_top, text="Reload", command=self.reload_config_snapshot).pack(
            side=tk.RIGHT, padx=2
        )
        ttk.Button(cfg_top, text="Edit…", command=self.open_config_editor_dialog).pack(
            side=tk.RIGHT, padx=2
        )

        self.config_output = scrolledtext.ScrolledText(config_tab, state="disabled", height=12)
        self.config_output.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self.reload_config_snapshot()

        # ----- Live Metrics Tab -----
        metrics_tab = ttk.Frame(self.notebook)
        self.notebook.add(metrics_tab, text="Live Metrics")

        self.metric_bars = {}
        self.metric_labels = {}
        metric_names = [
            "Talkativeness",
            "Rumination",
            "Forgetfulness",
            "Boredom",
            "Certainty",
        ]
        for name in metric_names:
            frame = ttk.Frame(metrics_tab)
            frame.pack(fill=tk.X, padx=4, pady=2)
            ttk.Label(frame, text=name + ":", font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
            bar = ttk.Progressbar(frame, maximum=100, mode="determinate")
            bar.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
            val = ttk.Label(frame, text="0.00")
            val.pack(side=tk.LEFT, padx=4)
            self.metric_bars[name] = bar
            self.metric_labels[name] = val

        # ----- Internal Thoughts Tab -----
        thoughts_tab = ttk.Frame(self.notebook)
        self.notebook.add(thoughts_tab, text="Internal Thoughts")

        paned = ttk.Panedwindow(thoughts_tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        thought_frame = ttk.Frame(paned)
        event_frame = ttk.Frame(paned)
        paned.add(thought_frame, weight=1)
        paned.add(event_frame, weight=1)

        self.thought_stream = scrolledtext.ScrolledText(thought_frame, state="disabled")
        self.thought_stream.pack(fill=tk.BOTH, expand=True)

        self.events_stream = scrolledtext.ScrolledText(event_frame, state="disabled")
        self.events_stream.pack(fill=tk.BOTH, expand=True)

        # Backward compatibility
        self.output = self.thought_stream

        self.base_timeout = (
            agents[0].watchdog_timeout if agents and hasattr(agents[0], "watchdog_timeout") else 900
        )
        self.timeout_label = ttk.Label(thoughts_tab, text=f"Base Timeout: {self.base_timeout}s")
        self.timeout_label.pack(anchor="w", padx=4, pady=2)

        self._refresh_log_display()

        # ----- Messages Tab -----
        messages_tab = ttk.Frame(self.notebook)
        self.notebook.add(messages_tab, text="Messages")

        msg_top = ttk.Frame(messages_tab)
        msg_top.pack(fill=tk.X, pady=2)
        ttk.Button(msg_top, text="Refresh", command=self.update_queue_and_sent).pack(
            side=tk.RIGHT, padx=2
        )

        queued_frame = ttk.LabelFrame(messages_tab, text="Queued (from humans)")
        queued_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self.queued_text = scrolledtext.ScrolledText(queued_frame, state="disabled", height=10)
        self.queued_text.pack(fill=tk.BOTH, expand=True)

        sent_frame = ttk.LabelFrame(messages_tab, text="Sent (to humans)")
        sent_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self.sent_text = scrolledtext.ScrolledText(sent_frame, state="disabled", height=10)
        self.sent_text.pack(fill=tk.BOTH, expand=True)

        self.update_queue_and_sent()

        # ----- Topology Tab -----
        topology_tab = ttk.Frame(self.notebook)
        self.notebook.add(topology_tab, text="Topology")

        self.topology_header = ttk.Label(topology_tab, text="Active Agent: None")
        self.topology_header.pack(anchor="w", padx=4, pady=2)

        self.topology_canvas = tk.Canvas(topology_tab, background="white")
        self.topology_canvas.pack(fill=tk.BOTH, expand=True)
        self.topology_canvas.bind("<Configure>", lambda e: self._redraw_topology())

        legend = ttk.Frame(topology_tab)
        legend.pack(anchor="e", padx=4, pady=2)
        for role, color in ROLE_COLORS.items():
            tk.Label(legend, bg=color, width=2).pack(side=tk.LEFT, padx=2)
            ttk.Label(legend, text=role.title()).pack(side=tk.LEFT, padx=2)

        self._topology_active = None
        self._topology_agents = []
        self._topology_node_items = {}
        self._topology_tooltip = None

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
        logger.debug("Exiting FenraUI.__init__")


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

        def _update():
            metrics = [
                ("Talkativeness", talkativeness),
                ("Rumination", rumination),
                ("Forgetfulness", forgetfulness),
                ("Boredom", boredom),
                ("Certainty", certainty),
            ]
            for name, value in metrics:
                bar = self.metric_bars.get(name)
                label = self.metric_labels.get(name)
                if bar and label:
                    pct = max(0.0, min(100.0, value * 100.0))
                    bar["value"] = pct
                    label.config(text=f"{value:.2f}")

        self._threadsafe(_update)
        logger.debug("Exiting update_weights")

    def append_thought(self, text: str, timestamp: Optional[str] = None) -> None:
        logger.debug("Entering append_thought text=%s timestamp=%s", text, timestamp)

        def _append():
            ts = timestamp or time.strftime("%H:%M:%S")
            self._append_text(self.thought_stream, f"[{ts}] {text}\n")

        self._threadsafe(_append)
        logger.debug("Exiting append_thought")

    def append_event(self, text: str, timestamp: Optional[str] = None) -> None:
        logger.debug("Entering append_event text=%s timestamp=%s", text, timestamp)

        def _append():
            ts = timestamp or time.strftime("%H:%M:%S")
            self._append_text(self.events_stream, f"[{ts}] {text}\n")

        self._threadsafe(_append)
        logger.debug("Exiting append_event")

    def update_queue_and_sent(self, queued: Optional[list] = None, sent: Optional[list] = None) -> None:
        logger.debug("Entering update_queue_and_sent queued=%s sent=%s", queued, sent)

        def _load(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:  # noqa: BLE001
                return None

        if queued is None:
            queued = _load(QUEUED_MESSAGES_PATH)
        if sent is None:
            sent = _load(SENT_MESSAGES_PATH)

        def _render():
            # Queued messages
            self.queued_text.configure(state="normal")
            self.queued_text.delete("1.0", tk.END)
            if isinstance(queued, list) and queued:
                for entry in queued:
                    if isinstance(entry, dict):
                        ts = entry.get("timestamp", "unknown")
                        msg = entry.get("raw_message") or entry.get("message") or str(entry)
                    else:
                        ts = "unknown"
                        msg = str(entry)
                    self.queued_text.insert(tk.END, f"[{ts}] {msg}\n")
            else:
                self.queued_text.insert(tk.END, "No queued messages.\n")
            self.queued_text.configure(state="disabled")
            self.queued_text.see(tk.END)

            # Sent messages
            self.sent_text.configure(state="normal")
            self.sent_text.delete("1.0", tk.END)
            if isinstance(sent, list) and sent:
                for entry in sent:
                    if isinstance(entry, dict):
                        ts = entry.get("timestamp", "unknown")
                        sender = entry.get("sender", "unknown")
                        msg = entry.get("message", "")
                    else:
                        ts = "unknown"
                        sender = "unknown"
                        msg = str(entry)
                    self.sent_text.insert(tk.END, f"[{ts}] {sender}: {msg}\n")
            else:
                self.sent_text.insert(tk.END, "No sent messages.\n")
            self.sent_text.configure(state="disabled")
            self.sent_text.see(tk.END)

        self._threadsafe(_render)
        logger.debug("Exiting update_queue_and_sent")

    def reload_config_snapshot(self) -> None:
        logger.debug("Entering reload_config_snapshot")
        parser = configparser.ConfigParser()
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                parser.read_file(f)
        except Exception:  # noqa: BLE001
            parser.read(self.config_path)
        self.global_config = dict(parser.items("global")) if parser.has_section("global") else {}

        def _update():
            self.config_output.configure(state="normal")
            self.config_output.delete("1.0", tk.END)
            self.config_output.insert(tk.END, "Global Configuration:\n")
            for k, v in self.global_config.items():
                self.config_output.insert(tk.END, f"{k} = {v}\n")
            self.config_output.configure(state="disabled")
            self.config_output.see(tk.END)

        self._threadsafe(_update)
        logger.debug("Exiting reload_config_snapshot")

    def open_config_editor_dialog(self) -> None:
        logger.debug("Entering open_config_editor_dialog")

        def _open():
            dialog = tk.Toplevel(self.root)
            dialog.title("Edit Configuration")
            text = tk.Text(dialog, width=60, height=15)
            text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            content = "[global]\n" + "\n".join(
                f"{k} = {v}" for k, v in self.global_config.items()
            )
            text.insert("1.0", content)
            btn_frame = ttk.Frame(dialog)
            btn_frame.pack(fill=tk.X, padx=4, pady=4)
            ttk.Button(btn_frame, text="Save", state=tk.DISABLED).pack(side=tk.RIGHT, padx=2)
            ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(
                side=tk.RIGHT, padx=2
            )

        self._threadsafe(_open)
        logger.debug("Exiting open_config_editor_dialog")

    def update_topology(self, active_agent: dict, agents: list[dict]) -> None:
        logger.debug("Entering update_topology active_agent=%s agents=%s", active_agent, agents)

        def _update():
            self._topology_active = active_agent
            self._topology_agents = agents
            self._redraw_topology()

        self._threadsafe(_update)
        logger.debug("Exiting update_topology")

    def _compute_neighbors(self, active: dict, agents: list[dict]) -> tuple[list[dict], list[dict], int, int]:
        upstream: list[dict] = []
        downstream: list[dict] = []
        active_in = set(active.get("groups_in", []) or [])
        active_out = set(active.get("groups_out", []) or [])
        for ag in agents:
            if ag.get("name") == active.get("name"):
                continue
            ag_in = set(ag.get("groups_in", []) or [])
            ag_out = set(ag.get("groups_out", []) or [])
            if ag_out & active_in:
                upstream.append(ag)
            if active_out & ag_in:
                downstream.append(ag)
        extra_up = max(0, len(upstream) - 25)
        extra_down = max(0, len(downstream) - 25)
        return upstream[:25], downstream[:25], extra_up, extra_down

    def _redraw_topology(self) -> None:
        active = self._topology_active
        self._hide_tooltip()
        self.topology_canvas.delete("all")
        if not active:
            self.topology_header.config(text="Active Agent: None")
            return
        self.topology_header.config(
            text=f"Active Agent: {active.get('name')} ({active.get('role', '').title()})"
        )
        agents = self._topology_agents
        upstream, downstream, extra_up, extra_down = self._compute_neighbors(active, agents)
        width = self.topology_canvas.winfo_width() or 1
        height = self.topology_canvas.winfo_height() or 1
        cx_up, cx_act, cx_down = width * 0.2, width * 0.5, width * 0.8
        active_y = height / 2

        def positions(n: int) -> list[float]:
            pad = 40
            if n <= 0:
                return []
            step = (height - pad * 2) / n
            return [pad + step / 2 + i * step for i in range(n)]

        up_pos = positions(len(upstream))
        down_pos = positions(len(downstream))
        self._topology_node_items.clear()

        for ag, y in zip(upstream, up_pos):
            self._draw_node(cx_up, y, ag, 16)
            self._draw_arrow(cx_up, y, cx_act, active_y)
        if extra_up:
            self.topology_canvas.create_text(cx_up, height - 20, text=f"+{extra_up} more…")
        elif not upstream:
            self.topology_canvas.create_text(
                cx_up, active_y, text="No likely sources", fill="#888888"
            )

        for ag, y in zip(downstream, down_pos):
            self._draw_node(cx_down, y, ag, 16)
            self._draw_arrow(cx_act, active_y, cx_down, y)
        if extra_down:
            self.topology_canvas.create_text(cx_down, height - 20, text=f"+{extra_down} more…")
        elif not downstream:
            self.topology_canvas.create_text(
                cx_down, active_y, text="No likely targets", fill="#888888"
            )

        self._draw_node(cx_act, active_y, active, 24)

    def _draw_node(self, x: float, y: float, agent: dict, radius: int) -> None:
        role = (agent.get("role") or "").lower()
        color = ROLE_COLORS.get(role, "#cccccc")
        circle = self.topology_canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=color,
            outline="black",
        )
        name = agent.get("name", "")
        display = name if len(name) <= 18 else name[:17] + "…"
        text = self.topology_canvas.create_text(x, y + radius + 12, text=display)
        for item in (circle, text):
            self.topology_canvas.tag_bind(
                item,
                "<Enter>",
                lambda e, a=agent: self._show_tooltip(e.x_root, e.y_root, a),
            )
            self.topology_canvas.tag_bind(item, "<Leave>", lambda e: self._hide_tooltip())
            self.topology_canvas.tag_bind(
                item,
                "<Double-1>",
                lambda e, a=agent: self.update_topology(a, self._topology_agents),
            )
            self._topology_node_items[item] = agent

    def _draw_arrow(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.topology_canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST)

    def _show_tooltip(self, x: int, y: int, agent: dict) -> None:
        self._hide_tooltip()
        tip = tk.Toplevel(self.root)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x + 10}+{y + 10}")
        info = (
            f"{agent.get('name')}\nRole: {agent.get('role')}\n"
            f"in: {len(agent.get('groups_in', []))}  out: {len(agent.get('groups_out', []))}"
        )
        ttk.Label(tip, text=info, relief=tk.SOLID, borderwidth=1, padding=2).pack()
        self._topology_tooltip = tip

    def _hide_tooltip(self) -> None:
        if self._topology_tooltip:
            self._topology_tooltip.destroy()
            self._topology_tooltip = None

    def _append_text(self, widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.configure(state="disabled")

    def _threadsafe(self, func, *args, **kwargs) -> None:
        if threading.current_thread() is threading.main_thread():
            func(*args, **kwargs)
        else:
            self.root.after(0, lambda: func(*args, **kwargs))

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

        def _update():
            self.thought_stream.configure(state="normal")
            self.thought_stream.delete("1.0", tk.END)
            for m in self.log_messages:
                text = f"[{m['timestamp']}] {m['sender']}: {m['message']}\n{'-'*80}\n\n"
                self.thought_stream.insert(tk.END, text)
            self.thought_stream.configure(state="disabled")
            self.thought_stream.see(tk.END)

        self._threadsafe(_update)
        logger.debug("Exiting _refresh_log_display")

    def log(self, entry):
        logger.debug("Entering log entry=%s", entry)
        self.log_messages.append(entry)
        text = f"[{entry['timestamp']}] {entry['sender']}: {entry['message']}\n{'-'*80}\n\n"
        self._threadsafe(self._append_text, self.thought_stream, text)
        logger.debug("Exiting log")

    def start(self):
        logger.debug("Entering start")
        self.root.mainloop()
        logger.debug("Exiting start")
