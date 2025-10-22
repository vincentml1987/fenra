# Fenra

Fenra now uses JSON configuration files under `confs/` to define globals, PDVs, agent classes and agents.

## Configuration
- `confs/globals.json` – global defaults (model, temperature, prompts).
- `confs/pdvs.json` – list of PDVs with descriptions and values.
- `confs/agent_classes.json` – class definitions and PDV adjustments.
- `confs/agents.json` – concrete agents referencing a class and their group wiring.

All edits are saved atomically. The conductor reads configs on startup; hot-reload is planned but not yet implemented.

User messages are queued by appending JSON objects to `chatlogs/queued_messages.json`. The UI writes to this file directly when no callbacks are wired.

`generate_agents.py` has been removed; add agents by editing the JSON files instead.

## Quick Start
1. Edit the JSON files under `confs/` to define globals, classes, PDVs and agents.
2. Start the conductor:
   ```bash
   python conductor.py            # infinite loop
   python conductor.py --once     # run a single step
   python conductor.py --steps N  # run N steps then exit
   ```
3. Launch the UI with `python fenra_ui.py`.

## Notes
- If `confs/state.json` is missing, it will be created on first run with the earliest agent set as `current_agent`.
- Speaker-class outputs are mirrored to `chatlogs/messages_to_humans.json` (for the UI "Sent (to humans)" tab).
- The message queue lives at `chatlogs/queued_messages.json` and is cleared when a queue-reading agent consumes it.
- Hot reload of configs is planned but not yet implemented.

## Codex Prompt (system instructions)

Use the following global/system prompt so that agents emit Fenra function calls correctly:

```
You can call Fenra runtime functions by outputting EXACTLY one line containing ONLY a call wrapped like:

*~function_name(arg1,arg2,kw="val")~*

HARD RULES:
- The character immediately after `*~` and the character immediately before `~*` must not be whitespace. Whitespace inside the span is allowed.
- All string arguments MUST be in double quotes.
- If a human-readable name contains spaces, you may keep the spaces or replace them with underscores (e.g., "New_Name"). The runtime will translate underscores back to spaces.
- After you emit a function call, immediately echo the exact call on the next line without the *~ ~* markers, prefixed by `CALL:` and with no spaces. Example:

*~rename_agent("New_Name")~*
CALL:rename_agent("New_Name")

- Do not insert any other text between the call and the CALL echo. If you do not need to call a function, reply normally.
- Never output more than one *~ ... ~* call per turn unless explicitly instructed.

Examples (valid):
*~list_functions()~*
CALL:list_functions()

*~rename_agent("Old_Name","New_Name")~*
CALL:rename_agent("Old_Name","New_Name")

Why these constraints? Fenra extracts function-call spans strictly as `*~ ... ~*` and validates only that the characters touching the markers are non-whitespace before dispatching them via `fenra_functions.dispatch_expression(...)`. The Conductor also logs the function name and result back into the message stream. The CALL echo makes the exact call visible too.
```
