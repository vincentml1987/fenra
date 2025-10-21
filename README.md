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
**Fenra function-call protocol (strict)**

* Use `*~` and `~*` to invoke Fenra functions: `*~function_name(args)~*`.
* **No-whitespace rule:** There must be **zero** whitespace characters (spaces, tabs, or newlines) **anywhere** between the opening `*~` and the closing `~*`.
  If any whitespace would appear between the markers, the entire sequence must be treated as **plain text**, not a function call.
* Emit each call on its own line. Never place calls inside code fences or other markup. Do not add commentary inside the markers.
* Examples:

  * ✅ Valid: `*~list_agents()~*`
  * ✅ Valid: `*~duplicate_self()~*`
  * ❌ Ignore: `*~ list_agents()~*` (leading space)
  * ❌ Ignore: `*~list_agents ()~*` (space before parentheses)
  * ❌ Ignore: `*~list agents()~*` (space in name)
  * ❌ Ignore: `*~some_call("two words")~*` (space inside the arguments)
* If you cannot express something without spaces inside the markers, **do not** wrap it in `*~…~*`; write normal text instead.

(This prompt governs how you write calls; extraction of calls is done by the Conductor after generation.)
```
