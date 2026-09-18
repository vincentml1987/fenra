"""Launcher for the Fenra distributed-compute client (worlds-rebuild
branch). Standalone from fenra.py entirely - see
Communications/client-server-plan.md. Loads
fenra_client/client_config.json (or defaults if the file does not exist
yet), starts the heartbeat and work-poll loops automatically, and opens
the status/control UI.

Edit fenra_client/client_config.json to set this machine's client_id,
token, and the server's address before running for real - the shipped
defaults point at 127.0.0.1 and an empty token, which will fail auth
against any real server.

Deliberately deviates from the repo's other run_*.py launchers by using
an `if __name__ == "__main__":` guard, unlike run_the_kiln.py and
siblings. This is required, not stylistic: the kill mechanism uses
multiprocessing.Process, and Windows multiprocessing re-imports this
entry script in every spawned worker process. Without the guard, each
worker would re-run this whole launcher (including opening its own Tk
window and spawning its own workers) recursively.
"""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.path.insert(0, '.')
    from fenra_client import config, app
    import tkinter as tk

    state = config.load_client_state()
    root = tk.Tk()
    client_app = app.FenraClientApp(root, state)
    client_app.start()
    root.mainloop()
