# Decisions & Open Questions Log

Running log for Fenra's Aletheosis. Newest entries at top.

## 2026-08-28 (versioning + function-calling test session)

- **Versioning:** fenra.py now has a real `FENRA_VERSION` constant (see changelog comment in the file), stamped into every session's `state.json` and every `history.jsonl` entry on write, and shown in the window title. Started at 0.4.0 to reflect the four functional commits so far (initial GUI, model dropdown, Sessions, max_tokens+timeout fix+function calling). Bump it on every functionally meaningful change going forward.
- **New session `functions-gemma3_12b`:** first session that actually tells Fenra about the `⟦function_name(args)⟧` syntax and the `⟦functions()⟧` discovery entry point (prompt saved as `Qualia/functions-top.txt` / `functions-bottom.txt`). All the other `factual-*` sessions predate this and don't know functions exist at all, by design - this is the first one meant to test whether/how she actually uses them.

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
