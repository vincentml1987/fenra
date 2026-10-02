"""Launcher for the_kiln world (worlds-rebuild branch, 2026-09-17 -
v0.15.0, revert to raw whole-turn dispatch + urge-informed function
agent + bounded voice history). Same pattern as run_the_confluence.py,
starts the loop automatically instead of waiting for a GUI click.
Resumes exactly where the world's on-disk state left off.

Built at Teddy's explicit invitation to start a fresh world under the
new architecture, design choices left to the assistant. Same 5 model
families as the_confluence (gemma3:27b, qwen2.5:14b, granite4.1:8b,
mistral-small:22b, command-r:35b), new names (Ash, Root, Cove, Wick,
Fen) to keep the two worlds' logs unambiguous, no assigned personality/
backstory/goals - same bare-identity approach proven out this session.

Deliberate design choice distinguishing it from the_confluence: a
single starting room ("hearth"), no satellites mapped out in advance.
The morning this world was built, Teddy found most function categories
sitting maxed and unaddressed in the_confluence's urge state -
create_room chief among them - and the whole point of tonight's revert
(dropping the bracket-item cap, feeding urges to the function agent with
real teeth) was to fix that. Starting with nowhere else to go at all is
a direct, observable test of whether it worked: if create_room is
genuinely getting exercised now, new rooms should appear organically
over time with no other rooms already scaffolded in to fall back on.
"""
import sys
sys.path.insert(0, '.')
import fenra
import tkinter as tk

root = tk.Tk()
app = fenra.FenraApp(root)
app._load_world("the_kiln")
app.toggle_loop()
root.mainloop()
