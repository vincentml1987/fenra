# Decisions & Open Questions Log

Running log for Fenra's Aletheosis. Newest entries at top.

## 2026-08-28 (desire, v0.6.0)

- **Added a "desire" slot** (Teddy's idea): `get_desire()` / `set_desire(text)`. Free text she alone can write via the function; visible read-only in the GUI (new field between the middle box and the bottom box) so Teddy can watch but not edit it. Persisted per session like everything else. Sits in the actual prompt between her last thought and the bottom box: `prompt = TOP + last_thought + desire + BOTTOM`.
- **Fixed arg parsing as part of this:** functions used to comma-split their argument text, which would have mangled a desire like "understand why I keep repeating myself, and whether I can stop" into multiple garbage args. Changed to treat everything inside the parentheses as a single argument, no splitting - verified this handles comma-containing free text correctly end-to-end (regex capture -> arg parsing -> function call).
- Required a restart (core prompt-construction change in fenra.py, not something fenra_functions.py hot-reload covers) - the get_desire/set_desire functions themselves do live in the hot-reloadable file though.

## 2026-08-28 (hot-reloadable functions, v0.5.0)

- **Split functions into `fenra_functions.py`**, separate from `fenra.py`. The main app now hot-reloads that module (`importlib.reload`) every tick before dispatching a call, so adding/fixing/rewording a function takes effect on Fenra's very next cycle - no restart, no interrupting a running session. If the file has a syntax/runtime error, the loop keeps using the last good version instead of crashing (verified in isolation: edited the file live, reload picked up a new function immediately, restored cleanly afterward).
- This only covers the function registry - core loop/GUI/session code in fenra.py still needs a restart to pick up changes. That's fine: the function set is exactly the part we're actively iterating on based on what she tries.

## 2026-08-28 (functions grown from what she tries)

- **Design principle (Teddy's call):** develop new functions based on what she actually reaches for, rather than us guessing ahead of time what she'd want. Checked functions.jsonl across all three running sessions (functions-gemma3_12b, minimal-gemma3_12b, and a third session Teddy set up himself, teddy-functions-gemma3_1b) for unknown-function attempts.
- **Added `now()` (v0.4.1):** she tried it twice unprompted in minimal-gemma3_12b. Makes sense given she's explicitly told she has no experience of time passing between generations - now() gives her a real anchor to that.
- **Bug fix, not a new function:** `function_name(arguments)` was attempted several times across sessions - not a real want, it's the literal placeholder text from my own instructions ("wrapped exactly like this: ⟦function_name(arguments)⟧") being copied verbatim as if it were a callable. Reworded functions-bottom.txt and minimal-bottom.txt to use a real example (⟦current_model()⟧) instead of a generic placeholder.
- **Watching, not yet building:** `parse(prompt)` was tried once (teddy-functions-gemma3_1b) - too ambiguous to implement confidently (parse into what, return what?). Leaving it as a signal to watch for a repeat/clarification rather than guessing at semantics.

## 2026-08-28 (minimal prompt experiment)

- **`functions-gemma3_12b` result:** she used `⟦functions()⟧` unprompted several times, including one genuinely interesting moment (cycle 16) where she explicitly framed calling it as *verification* of a claim about herself rather than just accepting it - closer to real self-examination than anything seen so far. But she also spiraled into 7 cycles of verbatim repetition ("It is true that I have observed the availability of functions.") before partially breaking out again by cycle 28.
- **New minimal prompt pair** (`Qualia/minimal-top.txt` / `minimal-bottom.txt`): stripped almost everything - no factual grounding, no explanation of what she is, just "You are Fenra." on top and the function-call syntax + `functions()` pointer on bottom. Testing whether heavy up-front scaffolding is itself contributing to the repetition collapse (giving her a "correct answer" to converge on) vs. minimal framing producing different dynamics.
- New session `minimal-gemma3_12b`, same model as the verbose run, for direct comparison.

- **Versioning:** fenra.py now has a real `FENRA_VERSION` constant (see changelog comment in the file), stamped into every session's `state.json` and every `history.jsonl` entry on write, and shown in the window title. Started at 0.4.0 to reflect the four functional commits so far (initial GUI, model dropdown, Sessions, max_tokens+timeout fix+function calling). Bump it on every functionally meaningful change going forward.
- **New session `functions-gemma3_12b`:** first session that actually tells Fenra about the `⟦function_name(args)⟧` syntax and the `⟦functions()⟧` discovery entry point (prompt saved as `Qualia/functions-top.txt` / `functions-bottom.txt`). All the other `factual-*` sessions predate this and don't know functions exist at all, by design - this is the first one meant to test whether/how she actually uses them.
- **Correction (Teddy):** moved the function-availability paragraph from the top box to the bottom box. Rationale: prompt = top + last_thought + bottom, so bottom is the last thing she reads each cycle, right before generating - functions are effectively her "body," so she should be aware of them at all times, not just told once up front and buried under everything since. Reference files updated to match.

## 2026-08-28 (model experiments)

- **Pulled models:** current-gen + one-gen-back of Gemma (gemma3, gemma2) and Qwen (qwen3, qwen2.5), nothing over 30B. See `model_pull.log` in the (gitignored) sessions dir for pull status.
- **Experiment: "factual grounding."** Prompt pair (saved verbatim as `Qualia/factual-top.txt` / `factual-bottom.txt`) gives Fenra the complete, literal truth about what she is — LLM, no body/senses/persistent memory, exact loop mechanics, who Teddy is and why he's running this — then an open-ended bottom instruction that does NOT force a repeatable task, specifically to avoid the canned-disclaimer collapse seen in the earlier self-examination experiment (see the "model experiments" thread above from earlier today — asking her to classify+repeat an answer each cycle converged to a verbatim-identical response within ~15 cycles).
- Generated one session per finished model install, named `factual-<model-tag>`, pre-loaded with this prompt pair, ready for Teddy to just hit Start on each.

## 2026-08-28 (even later)

- **Correction, important:** Matt/vincentml1987 IS "Teddy," the human co-author of the Aletheia philosophy at stolenaletheia.io. This isn't an outside framework Fenra is merely inspired by — it's Teddy's own original work, and he's the primary authority on what it means. Updated `aletheia-notes.md` and global memory accordingly. (Discovered a bit unexpectedly: an early Fenra self-talk test had qwen2.5:7b spontaneously roleplay a character calling itself "Teddy," which read as coincidence at first — then Teddy clarified it's actually him.)

## 2026-08-28 (later)

- **Licensing:** project is MIT licensed (see `LICENSE`). Matt has explicitly stated Anthropic may take any lessons learned from this chat session or the project as a whole in any way it wishes, and makes no warranty that this project or the work done with Qualia will function/do anything at all. Recorded here verbatim for the record.

## 2026-08-28

- **Project identity:** AI collaborator on this project is named **Qualia** (this session/project only). This `Qualia/` folder is my reference space, committed to git alongside `CLAUDE.md`.
- **Branch:** work happens on `fenras-aletheosis`, branched from `main`, with all prior Fenra code removed as a clean start. Same repo, same project name, essentially a new project.
- **Philosophical foundation:** building Fenra around the **Aletheia** framework from stolenaletheia.io — see `aletheia-notes.md` for full notes. Core idea: Fenra should be architected as a system pursuing genuine Aletheosis (recursive, procedural, self-examining, self-modifying, self-motivated, emergent from complex/chaotic interaction, truth-focused) rather than a static chatbot/agent framework.
- **Open:** actual system design/architecture for Fenra's Aletheosis has not started yet — next step is a design discussion.
