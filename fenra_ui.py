import logging
import tkinter as tk
from tkinter import scrolledtext, simpledialog
from tkinter import ttk

from ai_model import Ruminator, Archivist

logger = logging.getLogger(__name__)


class FenraUI:
    """Simple UI for displaying output and listing AIs."""

    def __init__(self, agents, inject_callback=None):
        logger.debug("Entering FenraUI.__init__ with agents=%s inject_callback=%s", agents, inject_callback)
        self.root = tk.Tk()
        self.root.title("Fenra")
        self.agents = agents
        self.inject_callback = inject_callback

        # Left side for console output
        self.output = scrolledtext.ScrolledText(self.root, state="disabled", width=80, height=24)
        self.output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right side for agent list and info
        right = tk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree = ttk.Treeview(right, show="tree")
        self.tree.pack(fill=tk.BOTH, expand=True)

        group_map = {}
        for a in agents:
            for g in a.groups:
                group_map.setdefault(g, []).append(a.name)

        for group in sorted(group_map):
            parent = self.tree.insert("", tk.END, text=group, open=True)
            for name in sorted(group_map[group]):
                self.tree.insert(parent, tk.END, text=name)

        self.inject_button = tk.Button(right, text="Inject Message", command=self._inject_message)
        self.inject_button.pack(fill=tk.X)
        logger.debug("Exiting FenraUI.__init__")

    class _InjectDialog(simpledialog.Dialog):
        """Dialog for injecting a message as a specific AI or human."""

        def __init__(self, parent, agents):
            logger.debug("Entering _InjectDialog.__init__ with agents=%s", agents)
            self.agents = agents
            self.selected = agents[0]
            self.message = ""
            self.send_as_human = tk.BooleanVar(value=False)
            self.human_name = tk.StringVar()
            super().__init__(parent, title="Inject Message")
            logger.debug("Exiting _InjectDialog.__init__")

        def body(self, master):
            logger.debug("Entering _InjectDialog.body")
            tk.Label(master, text="AI:").grid(row=0, column=0, sticky="w")
            self.ai_var = tk.StringVar(value=self.selected.name)
            option = tk.OptionMenu(master, self.ai_var, *[a.name for a in self.agents], command=self._update_info)
            option.grid(row=0, column=1, sticky="ew")

            self.human_check = tk.Checkbutton(
                master,
                text="Send as human",
                variable=self.send_as_human,
                command=self._toggle_human,
            )
            self.human_check.grid(row=1, column=0, sticky="w")
            self.name_entry = tk.Entry(master, textvariable=self.human_name, state="disabled")
            self.name_entry.grid(row=1, column=1, sticky="ew")

            self.info = tk.Label(master, text=self._format_info(self.selected), justify=tk.LEFT, anchor="w")
            self.info.grid(row=2, column=0, columnspan=2, sticky="w")

            tk.Label(master, text="Message:").grid(row=3, column=0, sticky="nw")
            self.text = scrolledtext.ScrolledText(master, width=40, height=10)
            self.text.grid(row=3, column=1, sticky="nsew")
            logger.debug("Exiting _InjectDialog.body")
            return self.text

        def _format_info(self, agent):
            logger.debug("Entering _InjectDialog._format_info agent=%s", agent)
            groups = ", ".join(agent.groups)
            result = f"Role Prompt:\n{agent.role_prompt}\nGroups: {groups}"
            logger.debug("Exiting _InjectDialog._format_info")
            return result

        def _update_info(self, value):
            logger.debug("Entering _InjectDialog._update_info value=%s", value)
            for a in self.agents:
                if a.name == value:
                    self.selected = a
                    break
            self.info.configure(text=self._format_info(self.selected))
            logger.debug("Exiting _InjectDialog._update_info")

        def _toggle_human(self):
            logger.debug("Entering _InjectDialog._toggle_human")
            state = "normal" if self.send_as_human.get() else "disabled"
            self.name_entry.configure(state=state)
            logger.debug("Exiting _InjectDialog._toggle_human")

        def apply(self):
            logger.debug("Entering _InjectDialog.apply")
            self.message = self.text.get("1.0", tk.END).strip()
            self.result = (
                self.selected,
                self.message,
                self.send_as_human.get(),
                self.human_name.get().strip(),
            )
            logger.debug("Exiting _InjectDialog.apply")

    def _inject_message(self):
        logger.debug("Entering _inject_message")
        if not self.inject_callback:
            logger.debug("Exiting _inject_message: no callback")
            return
        dialog = self._InjectDialog(self.root, self.agents)
        result = dialog.result
        if result and result[1]:
            self.inject_callback(*result)
        logger.debug("Exiting _inject_message")


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
