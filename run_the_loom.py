"""Launcher for the_loom world (worlds-rebuild branch, 2026-09-15 -
function agent severed from urges, explicit "I wish to take the
following actions:" intent signal) - same pattern as run_the_agora.py,
starts the loop automatically instead of waiting for a GUI click.
Resumes exactly where the world's on-disk state left off. 3 bare,
factual-identity voices (Gemma/Qwen/Mistral) - no personality/drives -
built specifically as a control for verifying the new architecture."""
import sys
sys.path.insert(0, '.')
import fenra
import tkinter as tk

root = tk.Tk()
app = fenra.FenraApp(root)
app._load_world("the_loom")
app.toggle_loop()
root.mainloop()
