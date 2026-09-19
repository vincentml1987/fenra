"""Rebuilds the derived tables in this folder from the raw snapshot files.

Run from this folder:  python build_derived.py
Writes: turns.csv, events.csv, tph_by_hour.csv, transcripts/<voice>.md
"""
import csv
import gzip
import json
import os
from collections import Counter, defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PARALLEL_START = "2026-09-18T21:11"  # v0.20.0 concurrent scheduler first ran
LOCAL = "http://localhost:11434"
VOICES = ["Ash", "Cove", "Fen", "Root", "Wick"]


def read_calls(voice):
    with gzip.open(os.path.join(HERE, "voices", voice, "llm_calls.jsonl.gz"), "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_turns():
    """One row per turn: an urge call followed by the voice call and >=1
    function-agent call. `complete` = has all three."""
    turns = []
    for v in VOICES:
        cur = None
        for r in read_calls(v):
            extra = r.get("extra") or {}
            if r["kind"] == "urge_agent":
                cur = {"voice": v, "voice_model": None, "host": extra.get("host"),
                       "urge_ts": r["timestamp"], "voice_ts": "", "fn_first_ts": "",
                       "fn_last_ts": "", "fn_attempts": 0, "tool_calls": [], "outcomes": []}
                turns.append(cur)
            elif cur and r["kind"] == "voice":
                cur["voice_ts"] = r["timestamp"]
                cur["voice_model"] = r["model"]
            elif cur and r["kind"] == "function_agent":
                cur["fn_first_ts"] = cur["fn_first_ts"] or r["timestamp"]
                cur["fn_last_ts"] = r["timestamp"]
                cur["fn_attempts"] += 1
                cur["tool_calls"] += [str(c) for c in extra.get("tool_calls", [])]
                cur["outcomes"] += [str(o) for o in extra.get("outcomes", [])]
    for t in turns:
        t["complete"] = bool(t["voice_ts"] and t["fn_last_ts"])
        t["end_ts"] = t["fn_last_ts"] or t["voice_ts"] or t["urge_ts"]
        t["era"] = "parallel" if t["end_ts"] >= PARALLEL_START else "sequential"
    turns.sort(key=lambda t: t["end_ts"])
    return turns


def write_turns(turns):
    cols = ["voice", "voice_model", "host", "era", "complete", "urge_ts", "voice_ts",
            "fn_first_ts", "fn_last_ts", "end_ts", "fn_attempts", "tool_calls", "outcomes"]
    with open(os.path.join(HERE, "turns.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in turns:
            row = dict(t)
            row["tool_calls"] = " | ".join(t["tool_calls"])
            row["outcomes"] = " | ".join(t["outcomes"])
            w.writerow({c: row[c] for c in cols})


def write_tph(turns):
    by_hour = defaultdict(Counter)
    for t in turns:
        if not t["complete"]:
            continue
        host = "local" if t["host"] == LOCAL else (t["host"] or "unknown")
        by_hour[t["end_ts"][:13]][host] += 1
    hosts = sorted({h for c in by_hour.values() for h in c})
    with open(os.path.join(HERE, "tph_by_hour.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hour", "complete_turns"] + hosts)
        for hour in sorted(by_hour):
            c = by_hour[hour]
            w.writerow([hour, sum(c.values())] + [c.get(h, 0) for h in hosts])


def write_events():
    """Every room-log entry. Say/whisper/yell appear twice: once for the
    actor ('You say...', recipients={actor: n}) and once for the listeners."""
    rows = []
    for name in sorted(os.listdir(os.path.join(HERE, "rooms"))):
        with open(os.path.join(HERE, "rooms", name), encoding="utf-8") as f:
            room = json.load(f)
        for e in room.get("log", []):
            rows.append({"room": room["name"], "id": e.get("id"), "timestamp": e["timestamp"],
                         "actor": e["actor"], "act": e["act"],
                         "recipients": json.dumps(e.get("recipients", {})),
                         "text": str(e.get("raw", "")).replace("\n", " ")})
    rows.sort(key=lambda r: r["timestamp"])
    with open(os.path.join(HERE, "events.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_transcripts():
    os.makedirs(os.path.join(HERE, "transcripts"), exist_ok=True)
    for v in VOICES:
        with open(os.path.join(HERE, "voices", v, "state.json"), encoding="utf-8") as f:
            st = json.load(f)
        lines = [f"# {v} ({st['model']}) - saved thoughts, oldest first\n",
                 f"Final room: {st['room']}. Final currencies: {st['currencies']}\n"]
        for t in st["thoughts"]:
            lines.append(f"\n## thought {t['id']} - {t.get('timestamp', '?')}\n\n{t['text']}\n")
        with open(os.path.join(HERE, "transcripts", f"{v}.md"), "w", encoding="utf-8") as f:
            f.write("".join(lines))


if __name__ == "__main__":
    turns = build_turns()
    write_turns(turns)
    write_tph(turns)
    write_events()
    write_transcripts()
    done = [t for t in turns if t["complete"]]
    for era in ("sequential", "parallel"):
        ts = [t for t in done if t["era"] == era]
        if ts:
            span = (datetime.fromisoformat(ts[-1]["end_ts"]) - datetime.fromisoformat(ts[0]["end_ts"])).total_seconds() / 3600
            print(f"{era}: {len(ts)} complete turns over {span:.1f}h wall = {len(ts)/span:.2f}/h")
