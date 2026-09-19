"""Launcher for the_ledger (worlds-rebuild, v0.21.1). World built by Vero and
Teddy, reviewed by Qualia; three personality-seeded voices in their own
offices around a shared atrium. Starts the loop automatically and resumes
where the world's on-disk state left off."""
import sys
sys.path.insert(0, '.')
import fenra
import tkinter as tk

root = tk.Tk()
app = fenra.FenraApp(root)
app._load_world("the_ledger")
app.toggle_loop()
root.mainloop()
