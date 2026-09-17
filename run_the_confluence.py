"""Launcher for the_confluence world (worlds-rebuild branch, 2026-09-15 -
v0.11.0, self-logging actions get a permanent caller record + whisper
bystander awareness). Same pattern as run_the_loom.py, starts the loop
automatically instead of waiting for a GUI click. Resumes exactly where
the world's on-disk state left off.

Unlike the_loom (a deliberately minimal 3-voice control built to verify
the new function-agent architecture in isolation), this world was built
at Teddy's explicit invitation to let the assistant make its own real
design choices - 5 voices across 5 distinct model families (gemma3:27b,
qwen2.5:14b, mistral-small:22b, command-r:35b, granite4.1:8b), named but
still told plainly what they are, no assigned personality/backstory/
goals (kept deliberately - the_loom's bare-identity approach already
produced real, unscripted emergent behavior this session, no reason to
override it with authored personality now that the architecture's
proven), starting together in a hub room with two adjacent satellite
rooms already mapped out for real spatial texture from turn one."""
import sys
sys.path.insert(0, '.')
import fenra
import tkinter as tk

root = tk.Tk()
app = fenra.FenraApp(root)
app._load_world("the_confluence")
app.toggle_loop()
root.mainloop()
