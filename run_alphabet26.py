"""Launcher for the alphabet-26 world (worlds-rebuild branch) that starts
the loop automatically instead of waiting for a GUI click - this branch
has no start/stop-signal-file mechanism yet. Resumes exactly where the
world's on-disk state left off (rotation index, all accumulated
context) - does not reset anything. See Qualia/pickup.md and
Qualia/worlds-rebuild-notes.md for the world's own history."""
import sys
sys.path.insert(0, '.')
import fenra
import tkinter as tk

root = tk.Tk()
app = fenra.FenraApp(root)
app._load_world("alphabet-26")
app.toggle_loop()
root.mainloop()
