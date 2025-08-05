#!/usr/bin/env python3
import argparse, random, secrets, sys, os, math, subprocess

# Constants
ROLES = ["ruminator", "listener", "speaker", "archivist", "ponderer", "doubter"]
LIMIT_BYTES = 250 * 1024**2  # 250 MiB per file
DECAY_RATE = 0.01  # Exponential decay parameter for group count bias

# Helper functions

def build_groups():
    # Group000–Group999 (1000 total groups)
    return [f"Group{n:03d}" for n in range(0, 1000)]


def pick_group_list(rng: random.Random, universe):
    """
    Selects a list size k with an exponential bias (1 is most likely, larger k are exponentially rarer),
    then samples k distinct groups from the universe.
    """
    u = rng.random() or sys.float_info.min
    k = int(-math.log(u) / DECAY_RATE) + 1
    k = max(1, min(k, len(universe)))
    return rng.sample(universe, k)


def format_list(items):
    return ", ".join(items)


def emit_agent(fid, role, agent_id, model, groups_in, groups_out):
    fid.write(f"[{role}{agent_id:06d}]\n")
    fid.write(f"model = {model}\n")
    fid.write(f"groups_in = {format_list(groups_in)}\n")
    fid.write(f"groups_out = {format_list(groups_out)}\n")
    fid.write(f"role = {role}\n\n")


def load_models():
    """
    Runs `ollama list` and parses the model names (first column of each row, skipping header).
    """
    try:
        out = subprocess.check_output(["ollama", "list"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running `ollama list`: {e}", file=sys.stderr)
        sys.exit(1)
    lines = [l for l in out.splitlines() if l.strip()]
    if len(lines) < 2:
        print("No models found in `ollama list` output.", file=sys.stderr)
        sys.exit(1)
    models = []
    for line in lines[1:]:  # skip header
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def main():
    parser = argparse.ArgumentParser(description="Generate Fenra agents with file rotation on size limit.")
    parser.add_argument("--start", type=int, default=0, help="First agent ID (default 0)")
    parser.add_argument("--end", type=int, default=999999, help="Last agent ID (default 999999)")
    parser.add_argument("--seed", type=str, default=None, help="Set for reproducible randomness")
    parser.add_argument("-o", "--outfile", default="agents.conf", help="Base output filename")
    args = parser.parse_args()

    # Initialize RNG
    rng = random.Random()
    if args.seed:
        rng.seed(args.seed)
    else:
        rng.seed(secrets.randbits(128))

    # Load available models via `ollama list`
    models = load_models()

    groups = build_groups()
    base, ext = os.path.splitext(args.outfile)
    def make_filename(part):
        return args.outfile if part == 0 else f"{base}_part{part}{ext}"

    part = 0
    current_path = make_filename(part)
    f = open(current_path, "w", encoding="utf-8")

    def can_continue():
        try:
            out_size = os.path.getsize(current_path)
        except FileNotFoundError:
            out_size = 0
        # Check per-file limit
        return out_size <= LIMIT_BYTES

    for agent_id in range(args.start, args.end + 1):
        if not can_continue():
            f.close()
            part += 1
            current_path = make_filename(part)
            print(f"Switching to new file: {current_path} at agent {agent_id:06d}")
            f = open(current_path, "w", encoding="utf-8")
        role = rng.choice(ROLES)
        model = rng.choice(models)
        gi = pick_group_list(rng, groups)
        go = pick_group_list(rng, groups)
        emit_agent(f, role, agent_id, model, gi, go)

    f.close()

if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
