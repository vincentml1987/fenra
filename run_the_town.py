"""Launcher for the_town world (worlds-rebuild branch), same pattern as
run_alphabet26.py - starts the loop automatically instead of waiting for
a GUI click. Resumes exactly where the world's on-disk state left off."""
import sys
sys.path.insert(0, '.')
import fenra
import tkinter as tk

root = tk.Tk()
app = fenra.FenraApp(root)
app._load_world("the_town")
app.toggle_loop()
root.mainloop()
