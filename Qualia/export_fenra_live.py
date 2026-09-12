"""
Exports a curated-scope, fully unfiltered snapshot of the_town (the
current worlds-rebuild world) to the stolenaletheia website repo and
pushes it - the data half of the public Fenra "Voices" / "Groups" view
(rebuilt 2026-09-12, replacing the pre-worlds-rebuild version of this
script and the two pages it fed).

Run on a schedule (every 5 minutes, via a cron job Qualia set up,
matching the original cadence) rather than continuously - this script
does one export-and-push and exits.

Unfiltered, per Teddy's explicit standing choice: whatever the world
actually contains goes up as-is, no redaction pass - Fenra, Teddy, and
Qualia are the only participants this data can contain. Scoped to "the
most recent thing we're working on" rather than a full archive: only
`the_town` (worlds-rebuild's current world - `alphabet-26` is stopped
and deliberately not included, per Qualia/pickup.md), and only each
voice's/group's most recent activity, not its entire history.

Reads directly from the same files fenra.py itself writes
(worlds/<world>/voices/<voice>/state.json,
worlds/<world>/groups/<group>.json) - read-only, never touches
anything fenra.py or a running world owns. Deliberately doesn't import
fenra.py itself (same posture as the original script) - reimplements
the small amount of read-only logic it needs (XLEUD, group-membership,
merged group chat) directly against the raw JSON, so this export can
never accidentally trigger any real write-side behavior.
"""

import json
import math
import os
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Fenra/
WORLD_NAME = "the_town"
WORLD_DIR = os.path.join(BASE_DIR, "worlds", WORLD_NAME)
VOICES_DIR = os.path.join(WORLD_DIR, "voices")
GROUPS_DIR = os.path.join(WORLD_DIR, "groups")
STOLENALETHEIA_DIR = os.path.join(BASE_DIR, "stolenaletheia")
OUTPUT_PATH = os.path.join(STOLENALETHEIA_DIR, "fenra", "live-data.json")

RECENT_MESSAGE_COUNT = 40  # per voice, and per group's merged chat transcript

# Same constants as fenra.py's own urge system (Qualia/worlds-rebuild-notes.md,
# 2026-09-11 design) - duplicated here rather than imported, matching this
# script's read-only-JSON-only posture. Keep in sync if fenra.py's ever change.
URGE_FUNCTIONS = ("send_message", "give_currency", "post_board", "skim_board", "read_board", "delete_board")
URGE_DESIRE = 7


def xleud(urge_value, desire=URGE_DESIRE):
    """Same saturating curve as fenra.py's xleud() - the felt-urge
    percentage a voice (and the app's own Urge Viewer) actually sees,
    not the raw persisted number."""
    return 1 - math.exp(-urge_value / desire)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def list_voice_names():
    if not os.path.isdir(VOICES_DIR):
        return []
    return sorted(d for d in os.listdir(VOICES_DIR) if os.path.isdir(os.path.join(VOICES_DIR, d)))


def list_group_names():
    if not os.path.isdir(GROUPS_DIR):
        return []
    return sorted(n[:-5] for n in os.listdir(GROUPS_DIR) if n.endswith(".json"))


def load_voice_state(name):
    return read_json(os.path.join(VOICES_DIR, name, "state.json"), {}) or {}


def load_group_state(name):
    return read_json(os.path.join(GROUPS_DIR, f"{name}.json"), {}) or {}


def build_urge_breakdown(state):
    urge = state.get("urge", {})
    understand = state.get("understand_urge", {})
    return {
        name: {
            "urge": urge.get(name, 0.0),
            "xleud_pct": round(xleud(urge.get(name, 0.0)) * 100, 1),
            "understand_urge": understand.get(name, 0.0),
            "understand_xleud_pct": round(xleud(understand.get(name, 0.0)) * 100, 1),
        }
        for name in URGE_FUNCTIONS
    }


def build_voices_snapshot(group_states):
    voices = []
    for name in list_voice_names():
        state = load_voice_state(name)
        own_groups = sorted(g for g, gs in group_states.items() if name in gs.get("members", []))
        messages = sorted(state.get("messages", []), key=lambda m: m.get("id", 0))
        voices.append({
            "name": name,
            "model": state.get("model", ""),
            "currencies": state.get("currencies", {}),
            "identity": state.get("identity", ""),
            "groups": own_groups,
            "urges": build_urge_breakdown(state),
            "understand_urge_general": state.get("understand_urge_general", 0.0),
            "recent_messages": messages[-RECENT_MESSAGE_COUNT:],
        })
    return voices


def merged_group_chat(group_name, member_names, count):
    """Every member's own self-tagged broadcasts into this group,
    merged and sorted - the real full text, matching fenra.py's own
    group_chat_transcript() (same logic, reimplemented against raw
    state.json rather than imported)."""
    entries = []
    for member in member_names:
        state = load_voice_state(member)
        for m in state.get("messages", []):
            if m.get("speaker") == member and group_name in m.get("groups", []):
                entries.append(m)
    entries.sort(key=lambda m: m.get("timestamp", ""))
    return entries[-count:]


def build_groups_snapshot():
    groups = []
    group_states = {}
    for name in list_group_names():
        gstate = load_group_state(name)
        group_states[name] = gstate
        members = gstate.get("members", [])
        groups.append({
            "name": gstate.get("name", name),
            "members": members,
            "board": sorted(gstate.get("board", []), key=lambda p: p.get("id", 0)),
            "recent_chat": merged_group_chat(name, members, RECENT_MESSAGE_COUNT),
        })
    return groups, group_states


def build_snapshot():
    if not os.path.isdir(WORLD_DIR):
        return {"generated_at": datetime.now().isoformat(timespec="seconds"), "world": WORLD_NAME, "voices": [], "groups": []}
    groups, group_states = build_groups_snapshot()
    voices = build_voices_snapshot(group_states)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "world": WORLD_NAME,
        "voices": voices,
        "groups": groups,
    }


def _dirty_paths_excluding(run, keep_paths):
    """Every path git status reports as changed, other than keep_paths -
    e.g. a draft Qualia is holding in the repo awaiting Teddy's approval
    - so it can be stashed out of the way rather than silently breaking
    the rebase (same fix as the original script, kept verbatim)."""
    status = run(["git", "status", "--porcelain"])
    paths = []
    for line in status.stdout.splitlines():
        path = line[3:].strip().strip('"')
        if path and path not in keep_paths:
            paths.append(path)
    return paths


def push(rel_path, abs_path, data):
    def run(args):
        return subprocess.run(args, cwd=STOLENALETHEIA_DIR, capture_output=True, text=True)

    # Discard any leftover uncommitted state of our own output file first -
    # safe, since it's about to be fully regenerated anyway.
    run(["git", "checkout", "--", rel_path])

    other_dirty = _dirty_paths_excluding(run, {rel_path})
    stashed = False
    if other_dirty:
        stash = run(["git", "stash", "push", "-u", "-m", "export_fenra_live: temporary", "--"] + other_dirty)
        stashed = stash.returncode == 0
        if not stashed:
            print("WARNING: could not stash unrelated pending changes, proceeding anyway:", stash.stderr)

    try:
        # The repo's own sitemap-generator workflow commits and pushes
        # right after every push this script makes, so the local clone
        # is behind again by the time the *next* run starts - pull
        # (rebase) before doing anything else, every time.
        run(["git", "fetch", "origin"])
        rebase = run(["git", "rebase", "origin/main"])
        if rebase.returncode != 0:
            run(["git", "rebase", "--abort"])
            print("WARNING: rebase failed even after stashing unrelated changes - aborted:", rebase.stderr)
            return

        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        run(["git", "add", rel_path])

        status = run(["git", "status", "--porcelain", "--", rel_path])
        if not status.stdout.strip():
            print("no change, nothing to push")
            return
        commit = run(["git", "commit", "-m", f"Fenra live data - {datetime.now().isoformat(timespec='seconds')}"])
        print(commit.stdout, commit.stderr)
        push_result = run(["git", "push"])
        print(push_result.stdout, push_result.stderr)
        if push_result.returncode != 0:
            run(["git", "fetch", "origin"])
            run(["git", "rebase", "origin/main"])
            retry = run(["git", "push"])
            print("retry:", retry.stdout, retry.stderr)
    finally:
        if stashed:
            pop = run(["git", "stash", "pop"])
            print(pop.stdout, pop.stderr)


if __name__ == "__main__":
    snap = build_snapshot()
    push("fenra/live-data.json", OUTPUT_PATH, snap)
    print(f"exported {len(snap['voices'])} voice(s), {len(snap['groups'])} group(s) from '{snap['world']}'")
