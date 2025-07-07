import tkinter as tk
from tkinter import scrolledtext


class FenraUI:
    """Simple UI for displaying output and listing AIs."""

    def __init__(self, agents):
        self.root = tk.Tk()
        self.root.title("Fenra")
        self.agents = agents

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
