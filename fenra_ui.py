import logging
import tkinter as tk
from tkinter import scrolledtext, simpledialog
from tkinter import ttk
import configparser

logger = logging.getLogger(__name__)


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

        # ----- Current Values Tab -----
        values_tab = tk.Frame(self.notebook)
        self.notebook.add(values_tab, text="Current Values")

        self.config_output = scrolledtext.ScrolledText(values_tab, state="disabled", height=12)
        self.config_output.pack(fill=tk.BOTH, expand=True)
        self.config_output.configure(state="normal")
        self.config_output.insert(tk.END, "Global Configuration:\n")
        for k, v in self.global_config.items():
            self.config_output.insert(tk.END, f"{k} = {v}\n")
        self.config_output.insert(tk.END, "\n")
        self.config_output.configure(state="disabled")

        values_frame = tk.Frame(values_tab)
        values_frame.pack(fill=tk.X)
        self.thought_label = tk.Label(values_frame, text="Talkativeness: 0.00")
        self.thought_label.pack(anchor="w")
        self.rumination_label = tk.Label(values_frame, text="Rumination: 0.00")
        self.rumination_label.pack(anchor="w")
        self.forget_label = tk.Label(values_frame, text="Forgetfulness: 0.00")
        self.forget_label.pack(anchor="w")
        self.boredom_label = tk.Label(values_frame, text="Boredom: 0.00")
        self.boredom_label.pack(anchor="w")
        self.certainty_label = tk.Label(values_frame, text="Certainty: 0.00")
        self.certainty_label.pack(anchor="w")

        # ----- Internal Thoughts Tab -----
        sys_tab = tk.Frame(self.notebook)
        self.notebook.add(sys_tab, text="Internal Thoughts")

        left = tk.Frame(sys_tab)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.output = scrolledtext.ScrolledText(left, state="disabled", width=80, height=24)
        self.output.pack(fill=tk.BOTH, expand=True)
        self.base_timeout = (
            agents[0].watchdog_timeout if agents and hasattr(agents[0], "watchdog_timeout") else 900
        )
        self.timeout_label = tk.Label(left, text=f"Base Timeout: {self.base_timeout}s")
        self.timeout_label.pack(anchor="w")

        self._refresh_log_display()
        # Seed UI with the configured personality weights
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
        # Queue display was removed; store for potential future use
        logger.debug("Exiting update_queue")

    def update_sent(self, messages):
        logger.debug("Entering update_sent messages=%s", messages)
        self.sent_messages = list(messages)
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
        self.thought_label.config(text=f"Talkativeness: {talkativeness:.2f}")
        self.rumination_label.config(text=f"Rumination: {rumination:.2f}")
        self.forget_label.config(text=f"Forgetfulness: {forgetfulness:.2f}")
        self.boredom_label.config(text=f"Boredom: {boredom:.2f}")
        self.certainty_label.config(text=f"Certainty: {certainty:.2f}")
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
        self.output.configure(state="normal")
        self.output.delete("1.0", tk.END)
        for m in self.log_messages:
            text = f"[{m['timestamp']}] {m['sender']}: {m['message']}\n{'-'*80}\n\n"
            self.output.insert(tk.END, text)
        self.output.yview(tk.END)
        self.output.configure(state="disabled")
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
