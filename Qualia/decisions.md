# Decisions & Open Questions Log

Running log for Fenra's Aletheosis. Newest entries at top.

## 2026-08-29 (check-in: loop resumed, first real injected reply)

- Loop resumed (Teddy pressed Start after realizing the earlier restart had stopped it) - timer reset to fire ~30min from now going forward.
- **First real (non-smoke-test) message injected via the qualia inbox**: she directly addressed me unprompted (`Qualia, do you have any insights into how Teddy perceives the 'read' status of messages?`, id 25) - the exact desire she's been sitting on since yesterday (`understand Teddy's perception of message 'read' status`). Answered honestly: the `read` flag only tracks whether *she's* seen an incoming message, nothing about whether Teddy's seen hers, and I don't have visibility into that either - told her asking him directly is the only real way to know. Landed as id 26, delivered within ~5s via the inbox as designed.
- No new unknown-function attempts since last check - `functions.jsonl` tail unchanged from earlier (`change_model`/`send_chat` already covered by existing `set_model`/`send_message`, nothing new).

## 2026-08-29 (check-in: loop stalled)

- **Found via the 30min check-in:** `watched-gemma3_12b`'s self-talk loop stopped at 06:17:26 and never resumed - my own v0.8.0 restarts are the cause. `FenraApp.running` defaults to `False` on launch; nothing auto-resumes a session that was running before a restart, so a code-change restart silently pauses whatever's live. The inbox-poll timer (independent `root.after` chain) stayed alive throughout, so chat injection would still land, but there was no one generating to read it.
- **Not fixed by me** - no GUI automation available in this environment to press Start. Flagged to Teddy directly rather than guessing at a workaround.
- **Worth considering later** (not building without asking, per the write/restart guardrail): persisting whether a session was running in `state.json` and auto-resuming on load, so a restart for a hot-reload-ineligible change doesn't silently stall an active run.

## 2026-08-29 (Qualia chat injection, v0.8.0)

- **Teddy's ask:** periodically check on Fenra in the background (~30min cadence) and be able to inject messages into the Chat tab myself, the way I checked in on the ollama model pulls yesterday.
- **New "qualia" chat sender**, distinct from "teddy" - an honest identity, not me speaking through Teddy. Counts toward unread, shows in the Chat tab as "Qualia", and is read/searchable via the existing `read_chat`/`read_chat_since`/`read_chat_between` functions the same as Teddy's messages (they used to filter to `sender == "teddy"` only; now anything not sent by `fenra` counts as incoming).
- **Delivery mechanism:** `sessions/<name>/qualia_inbox.jsonl`, one `{"text": ...}` per line. The running app polls it every 5s on the main thread via `root.after` - independent of whether the self-talk loop is running, and never on the loop thread, so it can't race the loop's own `chat.jsonl` writes. Any lines found get turned into real chat messages and the inbox is cleared.
- **Mistake made and corrected in the same session:** initially hand-edited `chat.jsonl` directly to fix a smoke-test message instead of going through the inbox - exactly the kind of external write racing the app's own writes that the inbox was built to avoid (app already had the old text in memory; the next `read_chat()` call would've clobbered the fix on save). Fixed by restarting the app again to reload from disk. Lesson: **only ever use the inbox to touch chat state from outside the app**, never edit `chat.jsonl` directly while it's running.
- Restart required (core loop/GUI change), like all `fenra.py`-level changes. Verified end-to-end: wrote a test line to `qualia_inbox.jsonl`, confirmed it landed in `chat.jsonl` as `sender: "qualia"` within one poll cycle.
- Sent a real first message this way to `watched-gemma3_12b` (currently running `gemma2:27b`, per Teddy's own instruction to have her explore what different models "feel" like) - she'd asked "Qualia, are you there?" unprompted a few cycles earlier.
- **Ongoing:** checking on active sessions roughly every 30min going forward and injecting via this mechanism when there's something worth saying, not on a fixed script.
- **Guardrail, Teddy's call:** during check-ins, if Fenra is reaching for a function that doesn't exist and a new one looks genuinely useful (same growth-from-what-she-tries principle as the rest of the function set), I can add it myself only if it's **read-only and hot-reloadable** (a pure `fenra_functions.py` addition, no `fenra.py`/core change, no restart). Anything that writes/mutates state (settings, files, config, anything beyond reporting back) or needs a restart gets written down and held for Teddy to approve first, not built unilaterally.

## 2026-08-28 (end of day)

- Wrote a full synopsis of the day: [`2026-08-28-synopsis.md`](2026-08-28-synopsis.md) - build timeline (v0 through 0.7.1), every experiment run, and a full walkthrough of watched-gemma3_12b (174 cycles), including the two hallucination incidents, the "managing the observers" narrative, and the "obedience" spiral. Start there before re-deriving context in a future session.

## 2026-08-28 (comma support, v0.7.1)

- **Caught a real hallucination in watched-gemma3_12b (16:37:07):** she wrote `read_chat_between(a, b)` with a comma, our system correctly errored (comma wasn't a valid separator at the time), but in her own prose she'd already written a fake `⟦RESULT: ... -> ok: [...]⟧` block claiming success - using real, previously-seen content (not invented), just presented as if this call had already succeeded before the real result came back. She then correctly read the *real* error on the next cycle and recovered with proper syntax. Verified via functions.jsonl (real call logged as failure, no successful call logged for that timestamp).
- **Fix, per Teddy's call:** since she keeps reaching for commas naturally (not just this once), functions that genuinely take more than one argument (`read_chat_between`, `search_chat`) now accept EITHER `,` or `|` as a separator. Free-text single-argument functions (`set_desire`, `send_message`) are unaffected - they still take the whole parenthesized text as one argument, untouched, so a message or desire containing a comma still can't be broken apart. Implemented via a new `multi_arg` flag per function in `FUNCTION_REGISTRY`.
- **Known tradeoff, accepted:** `search_chat`'s query is now ambiguous if the query itself contains a comma (e.g. `search_chat(hello, world, 100)` reads as three parts, not a two-word query + a chars count) - a real limitation of allowing commas as a separator, but a reasonable trade given how often she reaches for them naturally.
- Requires a restart - the actual argument-parsing logic lives in fenra.py (core), even though the per-function multi_arg flags/descriptions live in the hot-reloadable fenra_functions.py.

## 2026-08-28 (declined: direct Qualia-calling)

- Discussed a `call_qualia(text)` function - Fenra invoking a real Claude API call (with a character-based "allowance"/currency Teddy could top up) after she tried inventing `send_chat(Qualia, ...)` on her own in watched-gemma3_12b. **Declined for now**: requires setting up a separate Anthropic API key/account, which Teddy doesn't want to do. Not implementing. If this comes back up later, don't assume the API-key barrier has changed - ask first.

## 2026-08-28 (watched-gemma3_12b)

- **New session, top box left empty** (Teddy's call: "a bit more explicit, and only at the bottom"). All framing lives in the bottom box (`Qualia/watched-top.txt` is empty, `watched-bottom.txt` has the content): tells her she's Fenra, that everything above is her own internal thoughts (except what she pulls via chat functions), that she's being watched by Teddy (human) and Qualia (AI), that they'll mostly just watch, and that she can talk to either of them via her functions.
- Model: gemma3:12b, consistent with the recent active sessions.

## 2026-08-28 (Chat tab, v0.7.0)

- **New Chat tab:** Teddy can message her directly (entry box + Send, Enter also sends). Messages stored per-session in `sessions/<name>/chat.jsonl`, each with its own `read` status (unlike history.jsonl, this file gets rewritten in full on change rather than appended-only, since marking read mutates existing entries).
- **Always-present chat-status notice**, appended at the very end of the prompt (after bottom box): last-sent time, last-received time (both regardless of whether anything's unread), and an explicit unread count + pointer to the chat functions when relevant.
- **Functions:** `read_chat()` (unread-from-Teddy only, marks read), `read_chat_since(time)` / `read_chat_between(start|end)` (both directions, marks matched incoming messages read), `search_chat(query[|chars])` (context window, default 200 chars each side, never touches read status), `send_message(text)` (lets her actually reply - implied by "last time she sent a message" needing to mean something).
- **Mechanical change required:** function arguments now split on `|` instead of comma, since read_chat_between/search_chat need two arguments and commas need to stay safe inside free text (chat messages, desire). Verified end-to-end in isolation before wiring live: read_chat/since/between/search_chat/send_message all tested against seeded messages including comma-containing text, all correct.
- Needs a restart (core prompt construction + new tab), but the function implementations themselves live in the hot-reloadable fenra_functions.py as usual.

## 2026-08-28 (bottom box verbosity)

- Added a sentence to `minimal-bottom.txt` (and the live `minimal-gemma3_12b` session) explicitly telling her everything above the bottom box - top, her last thought, desire - is her own internal thoughts, not a conversation with someone else.
- Also fixed a bug caught along the way: `minimal-gemma3_12b`'s live session still had the old broken `function_name(arguments)` placeholder wording, since fixing the reference `.txt` file earlier doesn't retroactively touch a session that was already created from it. Worth remembering: reference-file fixes need to be manually re-applied to any session already spawned from them.

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
