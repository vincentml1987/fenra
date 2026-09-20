"""Timing benchmark for the_ledger's model settings (Qualia, 2026-09-19).

Runs one realistic call per model against an Ollama host with a chosen
num_ctx / num_predict, thinking left on, and records what Ollama itself
reports (load, prefill and generation time and token counts). The prompts are
built from the real thoughts and prompt tails of the stopped run in
Communications/the_ledger-teddys-arrogance-2026-09-19/, so it needs nothing
else. Nothing here touches a world.

  python ctx_bench.py --num-ctx 12288 --num-predict 3000 --out results-dir
  python ctx_bench.py --dry-run          # build prompts, print sizes, call nothing

Change --host to point at another machine's Ollama (e.g. Vero's), and
--voice-tests to the models installed there.
"""
import argparse, datetime, gzip, json, os, subprocess, sys, time
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(HERE, "..", "the_ledger-teddys-arrogance-2026-09-19")
DEFAULT_VOICE_TESTS = "qwen3.8:27b=sable,muse-glimmer:30b=quill,nemotron-3.5-lightning:latest=marrow"


def calls(voice):
    path = os.path.join(SNAP, "voices", voice, "llm_calls.jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_history(voice, chars):
    """Filler padding for ONE voice's own real thoughts only - a real voice's
    prompt never contains another voice's raw first-person thoughts under a
    name label, so mixing all three (the original approach) biased every
    tested voice toward whichever voice had the most real thoughts in the
    stopped run (marrow, at 7) regardless of which voice's own tail followed.
    Fixed 2026-09-19 after that showed up as identity confusion in both the
    Stheno and nemo-gutenberg benchmark runs."""
    with open(os.path.join(SNAP, "voices", voice, "state.json"), encoding="utf-8") as f:
        thoughts = [t["text"] for t in json.load(f)["thoughts"]]
    if not thoughts:
        return ""
    out, n, i = [], 0, 0
    while n < chars:
        piece = thoughts[i % len(thoughts)] + "\n\n"
        out.append(piece)
        n += len(piece)
        i += 1
    return "".join(out)[:chars]


def voice_prompt(voice, model, history_chars):
    marker = "Everything above this line"
    tails = [r["prompt"][r["prompt"].index(marker):]
             for r in calls(voice) if r["kind"] == "voice" and marker in r["prompt"]]
    tail = min(tails, key=len)
    tail = tail.replace("Model: ornith-1.5:35b", f"Model: {model}")
    return build_history(voice, history_chars) + tail


def real_prompt(kind):
    best = ""
    for v in ("sable", "marrow", "quill"):
        for r in calls(v):
            if r["kind"] == kind and len(r["prompt"]) > len(best):
                best = r["prompt"]
    return best


def free_gb():
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return round(int(out) / 1048576, 1)
    except Exception:
        return None


def ps_line(host):
    try:
        models = requests.get(f"{host}/api/ps", timeout=10).json().get("models", [])
        return [{"name": m["name"], "size_gb": round(m["size"] / 1e9, 1),
                 "in_vram_gb": round(m.get("size_vram", 0) / 1e9, 1)} for m in models]
    except Exception as exc:
        return str(exc)


def run(host, label, model, prompt, options, outdir):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {label}: {model}, prompt {len(prompt)} chars", flush=True)
    before = free_gb()
    t0 = time.time()
    req = {"model": model, "prompt": prompt, "stream": False, "options": options}
    resp = requests.post(f"{host}/api/generate", json=req, timeout=None)
    wall = time.time() - t0
    rec = {"label": label, "model": model, "options": options, "wall_s": round(wall, 1),
           "prompt_chars": len(prompt), "free_gb_before": before, "free_gb_after": free_gb(),
           "ps_after": ps_line(host), "http": resp.status_code}
    if resp.status_code == 200:
        j = resp.json()
        pe, ev = j.get("prompt_eval_count", 0), j.get("eval_count", 0)
        pd, ed = j.get("prompt_eval_duration", 0) / 1e9, j.get("eval_duration", 0) / 1e9
        rec.update({
            "load_s": round(j.get("load_duration", 0) / 1e9, 1),
            "prompt_tokens": pe, "prefill_s": round(pd, 1),
            "prefill_tok_s": round(pe / pd, 1) if pd else None,
            "gen_tokens": ev, "gen_s": round(ed, 1),
            "gen_tok_s": round(ev / ed, 2) if ed else None,
            "done_reason": j.get("done_reason"),
            "thinking_chars": len(j.get("thinking") or ""),
            "reply_chars": len(j.get("response") or ""),
        })
        with open(os.path.join(outdir, f"{label}.txt"), "w", encoding="utf-8") as f:
            f.write(f"# {label} ({model})\n\n## thinking\n{j.get('thinking') or ''}\n\n## reply\n{j.get('response') or ''}\n")
    else:
        rec["error"] = resp.text[:500]
    with open(os.path.join(outdir, "results.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--num-ctx", type=int, default=12288)
    ap.add_argument("--num-predict", type=int, default=3000)
    ap.add_argument("--history-chars", type=int, default=28000)
    ap.add_argument("--voice-tests", default=DEFAULT_VOICE_TESTS, help="model=voice,model=voice")
    ap.add_argument("--function-model", default="qwen3:30b")
    ap.add_argument("--urge-model", default="phi4-mini")
    ap.add_argument("--repeat-penalty", type=float, default=1.3)
    ap.add_argument("--stop", action="append", default=[],
                     help="Extra stop sequence for voice calls only (repeatable). "
                          "e.g. --stop \"Everything below is your HUD\"")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    voice_opts = {"num_ctx": a.num_ctx, "num_predict": a.num_predict, "repeat_penalty": a.repeat_penalty}
    if a.stop:
        voice_opts["stop"] = a.stop

    tests = []
    for pair in a.voice_tests.split(","):
        model, voice = pair.split("=")
        tests.append((f"voice-{voice}", model, voice_prompt(voice, model, a.history_chars), dict(voice_opts)))
    tests.append(("function-agent", a.function_model, real_prompt("function_agent"),
                  {"num_ctx": a.num_ctx, "num_predict": 1000, "repeat_penalty": a.repeat_penalty}))
    tests.append(("urge-agent", a.urge_model, real_prompt("urge_agent"),
                  {"num_ctx": a.num_ctx, "num_predict": 250, "repeat_penalty": a.repeat_penalty}))

    if a.dry_run:
        for label, model, prompt, opts in tests:
            print(label, model, len(prompt), "chars (~", len(prompt) // 4, "tokens)", opts)
        return
    started = datetime.datetime.now()
    print(f"started {started:%Y-%m-%d %H:%M:%S}; host {a.host}; num_ctx {a.num_ctx}; num_predict {a.num_predict}", flush=True)
    for label, model, prompt, opts in tests:
        run(a.host, label, model, prompt, opts, a.out)
    print(f"finished {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)


if __name__ == "__main__":
    main()
