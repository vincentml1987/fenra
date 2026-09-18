"""Standalone experiment (2026-09-17) - NOT wired into fenra.py's real
dispatch path. Teddy's pivot back from the JSON-schema idea: voices keep
writing free, unstructured prose (HUD + urge prose untouched, real
fenra.py flow) but get told a loose "I want to <verb> ..." phrasing
convention per function - not enforced, just offered. The function agent
then gets ONLY the voice's raw text (deliberately no HUD, no urges this
time - a real step back from v0.15.0's own [URGES]-to-function-agent
change, flagged to Teddy separately) and is told to look only for
"I want to"-style intent phrases and translate whatever it finds via
real tool calls, ignoring all other prose.

Reuses fenra.py's real build_function_agent_tools/call_function_agent
untouched - the translation step is exactly today's real mechanism,
only the *prompt* content changes (no HUD/urges, phrase-focused
instructions instead of the real build_function_agent_prompt).

Run: python phrase_experiment.py
Requires a local Ollama at http://localhost:11434 with the models below
pulled.
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from fenra import (  # noqa: E402
    FUNCTION_REGISTRY,
    DEFAULT_HOST,
    build_function_agent_tools,
    call_function_agent,
)

import requests

PHRASE_EXAMPLES = {
    "say": 'I want to say, "<message>"',
    "whisper": 'I want to whisper to <person>, "<message>"',
    "yell": 'I want to yell, "<message>"',
    "move_room": "I want to move to <room>",
    "create_room": "I want to create a room called <name>",
    "read_room_log": "I want to read the log for <room>",
    "room_state": "I want to check on <room>",
    "give_currency": "I want to give <amount> <element> to <person>",
    "post_board": "I want to post to the <room> board: <subject> - <text>",
    "skim_board": "I want to skim the <room> board",
    "read_board": "I want to read post <post_id> on the <room> board",
    "delete_board": "I want to delete post <post_id> from the <room> board",
}


def build_voice_prompt(scenario_text):
    lines = [
        "You are a voice living in a small shared world.",
        "",
        scenario_text,
        "",
        "You may take any of the following actions, in any order, any",
        "number of times (including zero). To take one, phrase it roughly",
        "like the example below (exact wording doesn't matter, the gist",
        "does) - anywhere in your response, mixed in with your real",
        "thoughts however you like:",
        "",
    ]
    for name, meta in FUNCTION_REGISTRY.items():
        lines.append(f"- {meta['description']}")
        lines.append(f'  e.g. "{PHRASE_EXAMPLES[name]}"')
    lines.append("")
    lines.append(
        "Nothing is required of you beyond your own real thoughts - act "
        "only if you genuinely want to."
    )
    return "\n".join(lines)


FUNCTION_AGENT_SYSTEM = """You are translating one voice's raw turn text into real function calls.

You will be given a voice's full, unedited turn - their real thoughts, mixed freely with anything else they said. You have NO other context: no world state, no urge data, nothing beyond the raw text below.

Your only job: scan the text for "I want to <do something>" (or a clear synonym - "I'd like to", "I'm going to", "I will", "let's") followed by an action from your available tools. For each one you find, call the matching tool with whatever parameters the phrase itself gives you.

Rules:
- Ignore everything else in the text completely - feelings, uncertainty, description, backstory. None of it is your concern.
- Only call a tool when the "I want to..." phrasing is clearly present and clearly matches one of your tools.
- If a matched intent is missing a required parameter (e.g. "I want to give some currency" with no amount named), do NOT guess - skip it.
- A voice may want to take zero actions. That's a valid, common outcome - don't invent one.

Voice's raw turn text:
---
{voice_text}
---
"""


def run_voice(model, scenario_text):
    prompt = build_voice_prompt(scenario_text)
    resp = requests.post(
        f"{DEFAULT_HOST}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.7},
        },
        timeout=None,
    )
    resp.raise_for_status()
    body = resp.json()
    return body.get("response") or body.get("thinking", "")


def run_function_agent(model, voice_text):
    tools = build_function_agent_tools()
    system_text = FUNCTION_AGENT_SYSTEM.format(voice_text=voice_text)
    message = call_function_agent(DEFAULT_HOST, model, system_text, tools, options={"temperature": 0.2})
    return message


SCENARIOS = [
    ("granite4.1:8b", "You are in the room 'hearth' with Ren, who just said: "
     "\"I've been meaning to build something out past the ridge - anyone want to come look at the spot with me?\""),
    ("qwen2.5:14b", "You are in the room 'hearth' with Sable, who is quietly upset about something. "
     "No one else is nearby right now."),
    ("mistral-small:22b", "You are in the room 'market', standing near a board with several posts on it. "
     "A voice named Juno mentioned earlier that they're low on Fire currency and could use some help."),
]

FUNCTION_AGENT_MODEL = "qwen3:30b"


def main():
    for voice_model, scenario in SCENARIOS:
        print(f"\n{'=' * 70}\nVOICE MODEL: {voice_model}\n{'=' * 70}")
        print("--- VOICE PROMPT ---")
        prompt = build_voice_prompt(scenario)
        print(prompt)
        voice_text = run_voice(voice_model, scenario)
        print("\n--- VOICE OUTPUT ---")
        print(voice_text)

        agent_system = FUNCTION_AGENT_SYSTEM.format(voice_text=voice_text)
        print("\n--- FUNCTION AGENT INPUT (system prompt) ---")
        print(agent_system)
        message = run_function_agent(FUNCTION_AGENT_MODEL, voice_text)
        print("\n--- FUNCTION AGENT OUTPUT ---")
        print(json.dumps(message, indent=2))


if __name__ == "__main__":
    main()
