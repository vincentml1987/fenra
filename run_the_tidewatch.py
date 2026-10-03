"""Launcher for the_tidewatch (worlds-rebuild, v0.21.1). World built by Vero
and Qualia with Teddy's approval; three voices (wren, tarn, ness) at a small
station on a shoreline, plus Teddy's Pilot Mode avatar in the window room.
Starts the loop automatically and resumes where the world's on-disk state
left off."""
import sys
sys.path.insert(0, '.')
import fenra
import tkinter as tk

root = tk.Tk()
app = fenra.FenraApp(root)
app._load_world("the_tidewatch")
app.toggle_loop()
root.mainloop()
