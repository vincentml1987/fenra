import tkinter as tk
from tkinter import scrolledtext, simpledialog

from ai_model import Ruminator, Archivist
from runtime_utils import parse_model_size


class FenraUI:
    """Simple UI for displaying output and listing AIs."""

    def __init__(self, agents, inject_callback=None):
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

        self.listbox = tk.Listbox(right)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        for agent in agents:
            self.listbox.insert(tk.END, agent.name)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        self.info_var = tk.StringVar()
        self.info_label = tk.Label(right, textvariable=self.info_var, justify=tk.LEFT, anchor="w")
        self.info_label.pack(fill=tk.BOTH, expand=True)

        self.inject_button = tk.Button(right, text="Inject Message", command=self._inject_message)
        self.inject_button.pack(fill=tk.X)

    class _InjectDialog(simpledialog.Dialog):
        """Dialog for injecting a message as a specific AI."""

        def __init__(self, parent, agents):
            self.agents = agents
            self.selected = agents[0]
            self.message = ""
            super().__init__(parent, title="Inject Message")

        def body(self, master):
            tk.Label(master, text="AI:").grid(row=0, column=0, sticky="w")
            self.ai_var = tk.StringVar(value=self.selected.name)
            option = tk.OptionMenu(master, self.ai_var, *[a.name for a in self.agents], command=self._update_info)
            option.grid(row=0, column=1, sticky="ew")

            self.info = tk.Label(master, text=self._format_info(self.selected), justify=tk.LEFT, anchor="w")
            self.info.grid(row=1, column=0, columnspan=2, sticky="w")

            tk.Label(master, text="Message:").grid(row=2, column=0, sticky="nw")
            self.text = scrolledtext.ScrolledText(master, width=40, height=10)
            self.text.grid(row=2, column=1, sticky="nsew")
            return self.text

        def _format_info(self, agent):
            groups = ", ".join(agent.groups)
            return f"Role Prompt:\n{agent.role_prompt}\nGroups: {groups}"

        def _update_info(self, value):
            for a in self.agents:
                if a.name == value:
                    self.selected = a
                    break
            self.info.configure(text=self._format_info(self.selected))

        def apply(self):
            self.message = self.text.get("1.0", tk.END).strip()
            self.result = (self.selected, self.message)

    def _inject_message(self):
        if not self.inject_callback:
            return
        dialog = self._InjectDialog(self.root, self.agents)
        result = dialog.result
        if result and result[1]:
            self.inject_callback(*result)

    def _on_select(self, event):
        selection = event.widget.curselection()
        if not selection:
            return
        idx = selection[0]
        agent = self.agents[idx]
        if isinstance(agent, Archivist):
            role = "archivist"
        elif isinstance(agent, Ruminator):
            role = "ruminator"
        else:
            role = agent.__class__.__name__.lower()

        params = parse_model_size(agent.model_name)
        info = (
            f"Name: {agent.name}\n"
            f"Model: {agent.model_name}\n"
            f"Role: {role}\n"
            f"Disk Size: {params} GB\n"
            f"Groups: {', '.join(agent.groups)}\n"
            f"Role Prompt:\n{agent.role_prompt}"
        )
        self.info_var.set(info)

    def log(self, text):
        self.output.configure(state="normal")
        self.output.insert(tk.END, text)
        self.output.yview(tk.END)
        self.output.configure(state="disabled")

    def start(self):
        self.root.mainloop()
