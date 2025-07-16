import logging
import tkinter as tk
from tkinter import scrolledtext, simpledialog
from tkinter import ttk

from ai_model import Ruminator, Archivist

logger = logging.getLogger(__name__)


class FenraUI:
    """Simple UI for displaying output and listing AIs."""

    def __init__(self, agents, inject_callback=None, send_callback=None):
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

        # Left side for console output
        self.output = scrolledtext.ScrolledText(self.root, state="disabled", width=80, height=24)
        self.output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right side for agent list and info
        right = tk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree = ttk.Treeview(right, show="tree")
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.group_names = []

        group_map = {}
        for a in agents:
            for g in a.groups:
                group_map.setdefault(g, []).append(a.name)

        self.all_groups_item = self.tree.insert("", tk.END, text="All Groups", open=False)
        self.group_names.append("All Groups")

        for group in sorted(group_map):
            parent = self.tree.insert("", tk.END, text=group, open=False)
            for name in sorted(group_map[group]):
                self.tree.insert(parent, tk.END, text=name)
            self.group_names.append(group)
        self.group_items = list(self.tree.get_children())

        btn_frame = tk.Frame(right)
        btn_frame.pack(fill=tk.X)
        self.expand_btn = tk.Button(btn_frame, text="Expand All", command=self._expand_all)
        self.expand_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.collapse_btn = tk.Button(btn_frame, text="Collapse All", command=self._collapse_all)
        self.collapse_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.inject_button = tk.Button(right, text="Inject Message", command=self._inject_message)
        self.inject_button.pack(fill=tk.X)

        self.send_button = tk.Button(right, text="Send message", command=self._send_message)
        self.send_button.pack(fill=tk.X)

        tk.Label(right, text="Queued Messages:").pack(fill=tk.X)
        self.queue_list = tk.Listbox(right, height=8)
        self.queue_list.pack(fill=tk.BOTH, expand=True)

        tk.Label(right, text="Sent to Humans:").pack(fill=tk.X)
        self.sent_list = tk.Listbox(right, height=8)
        self.sent_list.pack(fill=tk.BOTH, expand=True)
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
        selected = self.tree.focus() or self.all_groups_item
        group_item = selected
        parent = self.tree.parent(selected)
        if parent:
            group_item = parent
        group_name = self.tree.item(group_item, "text") or "All Groups"
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
        self.queue_list.delete(0, tk.END)
        for m in messages:
            text = f"[{m['timestamp']}] {m['message']}"
            self.queue_list.insert(tk.END, text)
        logger.debug("Exiting update_queue")

    def update_sent(self, messages):
        logger.debug("Entering update_sent messages=%s", messages)
        self.sent_list.delete(0, tk.END)
        for m in messages:
            text = f"[{m['timestamp']}] {m['sender']}: {m['message']}"
            self.sent_list.insert(tk.END, text)
        logger.debug("Exiting update_sent")

    def _expand_all(self):
        logger.debug("Entering _expand_all")
        for item in self.group_items:
            self.tree.item(item, open=True)
        logger.debug("Exiting _expand_all")

    def _collapse_all(self):
        logger.debug("Entering _collapse_all")
        for item in self.group_items:
            self.tree.item(item, open=False)
        logger.debug("Exiting _collapse_all")


    def log(self, text):
        logger.debug("Entering log text=%s", text)
        self.output.configure(state="normal")
        self.output.insert(tk.END, text)
        self.output.yview(tk.END)
        self.output.configure(state="disabled")
        logger.debug("Exiting log")

    def start(self):
        logger.debug("Entering start")
        self.root.mainloop()
        logger.debug("Exiting start")
