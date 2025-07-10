import tkinter as tk
from tkinter import scrolledtext, simpledialog, messagebox

from ai_model import Ruminator, Archivist
from runtime_utils import parse_model_size


class FenraUI:
    """Simple UI for displaying output and listing AIs."""

    def __init__(self, agents, add_agent_callback=None, remove_agent_callback=None):
        self.root = tk.Tk()
        self.root.title("Fenra")
        self.agents = agents
        self.add_agent_callback = add_agent_callback
        self.remove_agent_callback = remove_agent_callback

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

        self.add_button = tk.Button(right, text="Add Model", command=self._prompt_add)
        self.add_button.pack(fill=tk.X)

        self.remove_button = tk.Button(right, text="Remove Model", command=self._remove_selected)
        self.remove_button.pack(fill=tk.X)

    def _prompt_add(self):
        """Prompt the user for a new agent and add it via callback."""
        if not self.add_agent_callback:
            return

        name = simpledialog.askstring("Add Model", "Name:", parent=self.root)
        if not name:
            return

        model = simpledialog.askstring("Add Model", "Model ID:", parent=self.root)
        if not model:
            return

        role = simpledialog.askstring("Add Model", "Role Prompt:", parent=self.root)
        if role is None:
            role = ""

        try:
            agent = self.add_agent_callback(name, model, role)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc), parent=self.root)
            return

        if agent is not None:
            self.agents.append(agent)
            self.listbox.insert(tk.END, agent.name)

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
            f"Role Prompt:\n{agent.role_prompt}"
        )
        self.info_var.set(info)

    def _remove_selected(self):
        if not self.remove_agent_callback:
            return
        selection = self.listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        agent = self.agents[idx]
        try:
            removed = self.remove_agent_callback(agent)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc), parent=self.root)
            return
        if removed:
            self.listbox.delete(idx)
            self.agents.pop(idx)
            self.info_var.set("")

    def log(self, text):
        self.output.configure(state="normal")
        self.output.insert(tk.END, text)
        self.output.yview(tk.END)
        self.output.configure(state="disabled")

    def start(self):
        self.root.mainloop()
