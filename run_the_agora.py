"""Launcher for the_agora world (worlds-rebuild branch, function-agent
redesign - urge agent -> voice -> function agent) - same pattern as
run_the_commons.py, starts the loop automatically instead of waiting for
a GUI click. Resumes exactly where the world's on-disk state left off."""
import sys
sys.path.insert(0, '.')
import fenra
import tkinter as tk

root = tk.Tk()
app = fenra.FenraApp(root)
app._load_world("the_agora")
app.toggle_loop()
root.mainloop()
