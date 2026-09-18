"""Standalone experiment (2026-09-17) - NOT wired into fenra.py's real
dispatch path. Teddy's idea: instead of the function agent inferring
dispatches from a voice's raw prose, have the VOICE's own generation call
be schema-constrained directly - "thoughts" required, then one optional
field per function (say/whisper/yell/give_currency/...), each typed from
that function's own real params. A voice fills in only the ones it wants;
everything else stays absent. Urge percentages are deliberately NOT part
of this schema or prompt - Teddy's stated line is "I don't mind them
knowing what they can DO, just not what they FEEL" - urge prose can still
be handed to the voice as plain text same as today, this only changes the
shape of the *response*.

Reuses FUNCTION_REGISTRY straight from fenra.py so the schema can never
drift out of sync with the real functions - same spirit as
build_function_agent_tools().

Run: python schema_experiment.py [model]
Requires a local Ollama at http://localhost:11434 with the model pulled.
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from fenra import FUNCTION_REGISTRY, DEFAULT_HOST  # noqa: E402

import requests


def build_voice_response_schema():
    """thoughts (required, non-empty) + one optional array field per
    FUNCTION_REGISTRY entry, each item typed from that function's own
    `params` string (same '[optional]' convention build_function_agent_tools
    already uses). Arrays, not single objects, so a voice can e.g.
    give_currency to two different people in one turn if it wants to -
    zero items is always legal, nothing forces a fill."""
    properties = {
        "thoughts": {
            "type": "string",
            "minLength": 1,
            "description": "Your private internal thoughts this turn - not seen by anyone else.",
        }
    }
    for name, meta in FUNCTION_REGISTRY.items():
        params = [p for p in meta["params"].split("|") if p]
        item_properties = {}
        item_required = []
        for p in params:
            optional = p.startswith("[") and p.endswith("]")
            pname = p.strip("[]")
            item_properties[pname] = {"type": "string", "description": pname}
            if not optional:
                item_required.append(pname)
        properties[name] = {
            "type": "array",
            "description": meta["description"],
            "items": {
                "type": "object",
                "properties": item_properties,
                "required": item_required,
            },
        }
    return {
        "type": "object",
        "properties": properties,
        "required": ["thoughts"],
    }


def build_system_prompt():
    """Deliberately no urge percentages, no [URGES] block - just what a
    voice can DO, per Teddy's line. Real fenra.py prompt assembly
    (hud/history/room-state text) is untouched; this is a minimal
    standalone stand-in so the experiment is self-contained."""
    lines = [
        "You are a voice living in a small shared world. You are currently",
        "in the room 'hearth' along with another voice named Ren, who just",
        "said: \"I've been meaning to build something out past the ridge -",
        "anyone want to come look at the spot with me?\"",
        "",
        "Respond with your real internal thoughts, and take any actions",
        "you genuinely want to - or none at all. Nothing is required of you",
        "beyond your own thoughts.",
    ]
    return "\n".join(lines)


def run(model):
    schema = build_voice_response_schema()
    prompt = build_system_prompt()
    resp = requests.post(
        f"{DEFAULT_HOST}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "format": schema,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.7},
        },
        timeout=None,
    )
    resp.raise_for_status()
    body = resp.json()
    raw = body["response"]
    if not raw and body.get("thinking"):
        print("(!) response empty, JSON landed in 'thinking' instead - printing that field raw below")
        print(body["thinking"])
        return
    print("=== SCHEMA ===")
    print(json.dumps(schema, indent=2))
    print("\n=== RAW MODEL OUTPUT ===")
    print(raw)
    print("\n=== PARSED ===")
    parsed = json.loads(raw)
    print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3:30b"
    run(model)
