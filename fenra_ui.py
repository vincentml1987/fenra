import tkinter as tk
from tkinter import scrolledtext, simpledialog, messagebox


class FenraUI:
    """Simple UI for displaying output and listing AIs."""

    def __init__(self, agents, add_agent_callback=None):
        self.root = tk.Tk()
        self.root.title("Fenra")
        self.agents = agents
        self.add_agent_callback = add_agent_callback

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
        info = f"Name: {agent.name}\nModel: {agent.model_name}\nRole Prompt:\n{agent.role_prompt}"
        self.info_var.set(info)

    def log(self, text):
        self.output.configure(state="normal")
        self.output.insert(tk.END, text)
        self.output.yview(tk.END)
        self.output.configure(state="disabled")

    def start(self):
        self.root.mainloop()
