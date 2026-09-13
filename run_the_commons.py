"""Launcher for the_commons world (worlds-rebuild branch, rooms +
registers rebuild), same pattern as run_the_town.py - starts the loop
automatically instead of waiting for a GUI click. Resumes exactly
where the world's on-disk state left off."""
import sys
sys.path.insert(0, '.')
import fenra
import tkinter as tk

root = tk.Tk()
app = fenra.FenraApp(root)
app._load_world("the_commons")
app.toggle_loop()
root.mainloop()
