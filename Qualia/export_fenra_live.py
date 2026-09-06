"""
Exports a curated, public snapshot of Fenra's current activity - the
local wiki's current pages (since 2026-08-31), and (since 2026-09-01)
every voice's current groups wiring plus recent per-group activity - to
the stolenaletheia website repo and pushes it. The data half of the
semi-live public page Teddy asked for (2026-08-31), and of the public
groups/topology view (2026-09-01, alongside fenra.py's Groups feature).

Run on a schedule (every 5 minutes, via a cron job Qualia set up) rather
than continuously - this script does one export-and-push and exits.
Unfiltered, per Teddy's explicit choice: whatever the active session
actually generated goes up as-is, no redaction pass (the one carve-out,
per Teddy, is third-party personal information, which this script has no
way to encounter here - Fenra, Teddy, and Qualia are the only
participants in anything it reads).

Reads directly from the same session files fenra.py itself writes
(sessions/<name>/{state.json,history.jsonl,functions.jsonl,chat.jsonl})
and the same wiki files fenra_functions.py's read_wiki/write_wiki use
(Qualia/wiki/*.md) - read-only, never touches anything fenra.py, a
running session, or the wiki itself owns. The public wiki view is
read-only too - visitors can propose an edit via a GitHub issue link on
each page, never write to these files directly.
"""

import json
import os
import subprocess
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Fenra/
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
WIKI_DIR = os.path.join(BASE_DIR, "Qualia", "wiki")
GROUPS_DIR = os.path.join(BASE_DIR, "groups")
STOLENALETHEIA_DIR = os.path.join(BASE_DIR, "stolenaletheia")
OUTPUT_PATH = os.path.join(STOLENALETHEIA_DIR, "fenra", "live-data.json")
WIKI_OUTPUT_PATH = os.path.join(STOLENALETHEIA_DIR, "fenra", "wiki-data.json")
GROUPS_OUTPUT_PATH = os.path.join(STOLENALETHEIA_DIR, "fenra", "groups-data.json")

RECENT_EVENT_COUNT = 30  # how many merged events to show per session (every session, not just the active one)


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


def read_state(a_dir):
    """Read a state.json from any directory - a session dir (session-
    level fields) or a voice dir (voice-level fields), same file name
    either way."""
    path = os.path.join(a_dir, "state.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def list_voice_dirs(session_dir):
    """Every voice's own directory for this session (v0.16.2:
    sessions/<name>/voices/<voice>/), or a single synthetic entry
    pointing at the session directory itself if it predates the split
    and hasn't been reopened in the app since (a legacy, never-migrated
    session - its data still lives directly there). Read-only, exactly
    like the rest of this script - never migrates anything, unlike
    fenra.py's own _load_session."""
    voices_root = os.path.join(session_dir, "voices")
    if os.path.isdir(voices_root):
        names = sorted(d for d in os.listdir(voices_root) if os.path.isdir(os.path.join(voices_root, d)))
        return [(name, os.path.join(voices_root, name)) for name in names]
    return [("voice1", session_dir)]


def merged_recent_events(a_dir, chat_dir, count):
    """Interleave history (thoughts), functions (calls), and chat
    (messages) into one chronological feed, most recent last. history/
    functions come from a_dir (a voice's own directory); chat is always
    session-level (v0.16.2 - it's "Fenra," not one voice), so it's read
    from chat_dir (the session directory) regardless of which voice's
    history is being merged in."""
    events = []

    for e in read_jsonl(os.path.join(a_dir, "history.jsonl"))[-count:]:
        text = e.get("display", e.get("response", "")).strip()
        if text:
            events.append({"timestamp": e.get("timestamp", ""), "type": "thought", "text": text})

    for e in read_jsonl(os.path.join(a_dir, "functions.jsonl"))[-count:]:
        events.append({
            "timestamp": e.get("timestamp", ""),
            "type": "function",
            "name": e.get("function", ""),
            "args": e.get("args", []),
            "success": e.get("success", False),
            "result": str(e.get("result", ""))[:300],
        })

    for e in read_jsonl(os.path.join(chat_dir, "chat.jsonl"))[-count:]:
        events.append({
            "timestamp": e.get("timestamp", ""),
            "type": "chat",
            "sender": e.get("sender", ""),
            "text": e.get("text", ""),
        })

    events.sort(key=lambda e: e.get("timestamp", ""))
    return events[-count:]


def build_snapshot():
    """v0.16.2: a session can hold several voices now (fenra.py's Groups
    -> Voices split), each with its own history/functions log - chat
    stays session-level (shared, "Fenra," not one voice). The site UI
    this feeds (stolenaletheia/fenra/index.html) still shows one card
    per session with one merged feed, so each session's top-level
    model/rotation/recent here are taken from whichever of its voices
    was actually active most recently - a real simplification, not a
    full per-voice site view (that's a follow-up); the full per-voice
    breakdown is included too (session["voices"]) so that follow-up has
    real data to build against already."""
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
        voice_dirs = list_voice_dirs(session_dir)

        voice_entries = []
        for voice_name, vdir in voice_dirs:
            vstate = read_state(vdir)
            vhistory = read_jsonl(os.path.join(vdir, "history.jsonl"))
            voice_entries.append({
                "name": voice_name,
                "model": vstate.get("model", ""),
                "rotation": vstate.get("model_rotation", []),
                "last_active": vhistory[-1]["timestamp"] if vhistory else None,
                "cycle_count": len(vhistory),
            })

        # Whichever voice actually spoke most recently drives this
        # session's top-level summary/feed - "" sorts first, so a voice
        # that's never spoken never wins this over one that has.
        lead = max(voice_entries, key=lambda v: v["last_active"] or "") if voice_entries else None
        lead_dir = dict(voice_dirs).get(lead["name"], session_dir) if lead else session_dir
        total_cycles = sum(v["cycle_count"] for v in voice_entries)

        entry = {
            "name": name,
            "active": name == active_name,
            "model": lead["model"] if lead else "",
            "rotation": lead["rotation"] if lead else [],
            "fenra_version": read_state(session_dir).get("fenra_version", ""),
            "last_active": lead["last_active"] if lead else None,
            "cycle_count": total_cycles,
            "voices": voice_entries,
            # Every session gets its own recent-activity feed (from
            # whichever voice led, above) now, not just the active
            # session - Teddy's request, so an inactive session's card
            # can be clicked to see what actually ran there.
            "recent": merged_recent_events(lead_dir, session_dir, RECENT_EVENT_COUNT),
        }
        sessions.append(entry)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "active_session": active_name,
        "sessions": sessions,
    }


def build_wiki_snapshot():
    """Every page in Qualia/wiki/ (name + raw markdown content), for the
    public read-only wiki view Teddy asked for (2026-08-31) - visitors
    can propose an edit via a GitHub issue link on each page, but never
    write to these files directly. Same source `read_wiki`/`write_wiki`
    (fenra_functions.py) use, read-only here too."""
    pages = []
    if os.path.isdir(WIKI_DIR):
        for name in sorted(os.listdir(WIKI_DIR)):
            if not name.endswith(".md"):
                continue
            path = os.path.join(WIKI_DIR, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                continue
            pages.append({"name": name[:-3], "content": content})
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pages": pages,
    }


def build_groups_snapshot():
    """Every voice's current groups_in/groups_out (from its own
    state.json - a real per-voice directory since v0.16.2, or a legacy
    session's own top-level state.json if it hasn't been reopened in the
    app since), paired with recent per-group activity from groups/*.jsonl
    to compute a last-active timestamp per qualified voice
    ("session:voice", matching exactly what fenra.py's _tick actually
    broadcasts under) - the data half of the public groups/topology view
    (2026-09-01, extended for Voices 2026-09-01). Same read-only,
    unfiltered posture as build_snapshot: whatever's actually there goes
    up as-is. Edges are a flat list rather than a nested dict, since a
    (voice, group) pair isn't a valid JSON object key and it's easier
    for the page's own JS to iterate either way."""
    voices = {}
    if os.path.isdir(SESSIONS_DIR):
        for name in os.listdir(SESSIONS_DIR):
            session_dir = os.path.join(SESSIONS_DIR, name)
            if not os.path.isdir(session_dir):
                continue
            for voice_name, vdir in list_voice_dirs(session_dir):
                state = read_state(vdir)
                g_in = list(state.get("groups_in", []))
                g_out = list(state.get("groups_out", []))
                if g_in or g_out:
                    voices[f"{name}:{voice_name}"] = {"groups_in": g_in, "groups_out": g_out}

    group_names = set()
    if os.path.isdir(GROUPS_DIR):
        group_names.update(n[:-6] for n in os.listdir(GROUPS_DIR) if n.endswith(".jsonl"))
    for v in voices.values():
        group_names.update(v["groups_in"])
        group_names.update(v["groups_out"])

    last_active = {}
    for g in sorted(group_names):
        path = os.path.join(GROUPS_DIR, f"{g}.jsonl")
        for e in read_jsonl(path)[-200:]:
            key = (e.get("voice", ""), g)
            ts = e.get("timestamp", "")
            if ts and (key not in last_active or ts > last_active[key]):
                last_active[key] = ts

    edges = []
    for name, g in voices.items():
        for grp in sorted(set(g["groups_in"]) | set(g["groups_out"])):
            edges.append({
                "voice": name,
                "group": grp,
                "reads": grp in g["groups_in"],
                "writes": grp in g["groups_out"],
                "last_active": last_active.get((name, grp)),
            })

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "voices": sorted(voices.keys()),
        "groups": sorted(group_names),
        "edges": edges,
    }


def _dirty_paths_excluding(run, keep_paths):
    """Every path git status reports as changed, other than keep_paths -
    e.g. a draft Qualia is holding in the repo awaiting Teddy's approval
    (qualia/index.html, say). `git rebase` refuses outright with any
    uncommitted change to a tracked file present, blocked or not - this
    is how that gets discovered so it can be stashed out of the way
    rather than silently breaking the rebase (confirmed happening for
    real, 2026-08-31: a pending qualia/index.html edit sat in the working
    tree during a scheduled run and both the push and its retry failed
    with no clear signal why, until this was found by hand)."""
    status = run(["git", "status", "--porcelain"])
    paths = []
    for line in status.stdout.splitlines():
        path = line[3:].strip().strip('"')
        if path and path not in keep_paths:
            paths.append(path)
    return paths


def push(outputs):
    """outputs: list of (relative_repo_path, absolute_output_path, data)
    - every one gets written and committed together, in one push, so the
    live-activity feed and the wiki never end up out of sync with each
    other by a commit."""
    def run(args):
        return subprocess.run(args, cwd=STOLENALETHEIA_DIR, capture_output=True, text=True)

    rel_paths = [rel for rel, _, _ in outputs]

    # Discard any leftover uncommitted state of our *own* output files
    # first - safe, since they're about to be fully regenerated anyway,
    # and necessary: an unstaged output file (e.g. left behind by an
    # earlier failed run) blocks `git rebase` exactly the same way an
    # unrelated draft does, discovered the hard way building this fix the
    # first time. Only ever discards this script's own files, never
    # anything else.
    for rel in rel_paths:
        run(["git", "checkout", "--", rel])

    # Stash anything else sitting uncommitted in the working tree - a
    # pending draft, most likely - by exact path, before touching git
    # history at all. Restored at the very end regardless of how the
    # export itself goes, so a draft in progress elsewhere in the repo
    # is never at risk of being lost or silently blocking this script.
    other_dirty = _dirty_paths_excluding(run, set(rel_paths))
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
        # (rebase, so this script's own commit never gets a needless
        # merge commit) before doing anything else, every time, rather
        # than assume the clone is already current.
        run(["git", "fetch", "origin"])
        rebase = run(["git", "rebase", "origin/main"])
        if rebase.returncode != 0:
            run(["git", "rebase", "--abort"])
            print("WARNING: rebase failed even after stashing unrelated changes - aborted:", rebase.stderr)
            return

        # Only write the actual new content once the tree is clean and
        # synced - never before, so these files are never themselves the
        # reason a rebase gets blocked.
        for rel, abs_path, data in outputs:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            run(["git", "add", rel])

        status = run(["git", "status", "--porcelain"] + ["--"] + rel_paths)
        if not status.stdout.strip():
            print("no change, nothing to push")
            return
        commit = run(["git", "commit", "-m", f"Fenra live data - {datetime.now().isoformat(timespec='seconds')}"])
        print(commit.stdout, commit.stderr)
        push_result = run(["git", "push"])
        print(push_result.stdout, push_result.stderr)
        if push_result.returncode != 0:
            # Lost a race with the sitemap bot (or something else) -
            # rebase once more and retry, rather than silently drop this
            # cycle's export.
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
    wiki_snap = build_wiki_snapshot()
    groups_snap = build_groups_snapshot()
    push([
        ("fenra/live-data.json", OUTPUT_PATH, snap),
        ("fenra/wiki-data.json", WIKI_OUTPUT_PATH, wiki_snap),
        ("fenra/groups-data.json", GROUPS_OUTPUT_PATH, groups_snap),
    ])
    print(
        f"exported {len(snap['sessions'])} session(s), active: {snap['active_session']}, "
        f"{len(wiki_snap['pages'])} wiki page(s), {len(groups_snap['edges'])} group edge(s)"
    )
