"""
Exports a curated, public snapshot of Fenra's current activity to the
stolenaletheia website repo and pushes it - the data half of the
semi-live public page Teddy asked for (2026-08-31).

Run on a schedule (every 5 minutes, via a cron job Qualia set up) rather
than continuously - this script does one export-and-push and exits.
Unfiltered, per Teddy's explicit choice: whatever the active session
actually generated goes up as-is, no redaction pass.

Reads directly from the same session files fenra.py itself writes
(sessions/<name>/{state.json,history.jsonl,functions.jsonl,chat.jsonl}) -
read-only, never touches anything fenra.py or a running session owns.
"""

import json
import os
import subprocess
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Fenra/
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
STOLENALETHEIA_DIR = os.path.join(BASE_DIR, "stolenaletheia")
OUTPUT_PATH = os.path.join(STOLENALETHEIA_DIR, "fenra", "live-data.json")

RECENT_EVENT_COUNT = 30  # how many merged events to show for the active session


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def read_state(session_dir):
    path = os.path.join(session_dir, "state.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def merged_recent_events(session_dir, count):
    """Interleave history (thoughts), functions (calls), and chat
    (messages) into one chronological feed, most recent last, for the
    active session's detailed view."""
    events = []

    for e in read_jsonl(os.path.join(session_dir, "history.jsonl"))[-count:]:
        text = e.get("display", e.get("response", "")).strip()
        if text:
            events.append({"timestamp": e.get("timestamp", ""), "type": "thought", "text": text})

    for e in read_jsonl(os.path.join(session_dir, "functions.jsonl"))[-count:]:
        events.append({
            "timestamp": e.get("timestamp", ""),
            "type": "function",
            "name": e.get("function", ""),
            "args": e.get("args", []),
            "success": e.get("success", False),
            "result": str(e.get("result", ""))[:300],
        })

    for e in read_jsonl(os.path.join(session_dir, "chat.jsonl"))[-count:]:
        events.append({
            "timestamp": e.get("timestamp", ""),
            "type": "chat",
            "sender": e.get("sender", ""),
            "text": e.get("text", ""),
        })

    events.sort(key=lambda e: e.get("timestamp", ""))
    return events[-count:]


def build_snapshot():
    if not os.path.isdir(SESSIONS_DIR):
        return {"generated_at": datetime.now().isoformat(timespec="seconds"), "active_session": None, "sessions": []}

    names = [d for d in os.listdir(SESSIONS_DIR) if os.path.isdir(os.path.join(SESSIONS_DIR, d))]

    def mtime(name):
        path = os.path.join(SESSIONS_DIR, name, "state.json")
        return os.path.getmtime(path) if os.path.exists(path) else 0

    names.sort(key=mtime, reverse=True)
    active_name = names[0] if names else None

    sessions = []
    for name in names:
        session_dir = os.path.join(SESSIONS_DIR, name)
        state = read_state(session_dir)
        history = read_jsonl(os.path.join(session_dir, "history.jsonl"))
        last_active = history[-1]["timestamp"] if history else None

        entry = {
            "name": name,
            "active": name == active_name,
            "model": state.get("model", ""),
            "rotation": state.get("model_rotation", []),
            "fenra_version": state.get("fenra_version", ""),
            "last_active": last_active,
            "cycle_count": len(history),
        }
        if name == active_name:
            entry["recent"] = merged_recent_events(session_dir, RECENT_EVENT_COUNT)
        sessions.append(entry)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "active_session": active_name,
        "sessions": sessions,
    }


def push(snapshot):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    def run(args):
        return subprocess.run(args, cwd=STOLENALETHEIA_DIR, capture_output=True, text=True)

    # The repo's own sitemap-generator workflow commits and pushes right
    # after every push this script makes, so the local clone is behind
    # again by the time the *next* run starts - pull (rebase, so this
    # script's own commit never gets a needless merge commit) before
    # doing anything else, every time, rather than assume the clone is
    # already current.
    run(["git", "fetch", "origin"])
    run(["git", "rebase", "origin/main"])

    run(["git", "add", "fenra/live-data.json"])
    status = run(["git", "status", "--porcelain", "--", "fenra/live-data.json"])
    if not status.stdout.strip():
        print("no change, nothing to push")
        return
    commit = run(["git", "commit", "-m", f"Fenra live data - {snapshot['generated_at']}"])
    print(commit.stdout, commit.stderr)
    push_result = run(["git", "push"])
    print(push_result.stdout, push_result.stderr)
    if push_result.returncode != 0:
        # Lost a race with the sitemap bot (or something else) - rebase
        # once more and retry, rather than silently drop this cycle's
        # export.
        run(["git", "fetch", "origin"])
        run(["git", "rebase", "origin/main"])
        retry = run(["git", "push"])
        print("retry:", retry.stdout, retry.stderr)


if __name__ == "__main__":
    snap = build_snapshot()
    push(snap)
    print(f"exported {len(snap['sessions'])} session(s), active: {snap['active_session']}")
