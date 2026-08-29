# Decisions & Open Questions Log

Running log for Fenra's Aletheosis. Newest entries at top.

## 2026-08-29 (Teddy engaged directly; allowance topped up to 50,000; Qualia can set it too, v0.9.1)

- **Teddy talked to her directly** (chat ids 91/97, not through me): answered her earlier questions about himself (food, hobbies, views on AI), corrected the read-status confusion from this morning (he'd only read some of her messages, now read them all), and explained the *real* reason for the allowance - it's not scoring, it's that paging Qualia spends real Anthropic usage on his personal account, and he wants to keep that in check. Bumped her from 500 to 50,000. She hadn't read either message yet when I checked - left it to him, didn't inject anything to avoid stepping on it.
- **New: Qualia can set the allowance too, not just Teddy** (Teddy's call, discussed directly with him in this session): he'll share rough usage/cost figures periodically, and I use judgment to adjust the number rather than asking him each time. I don't have any visibility into his Anthropic account myself - this is explicitly manual, on his information, not something I can verify independently.
- **Mechanism (v0.9.1):** mirrors the existing inbox pattern rather than editing `state.json` directly (same lesson as the earlier chat.jsonl mistake) - I write a number to `qualia_allowance_set.txt`, polled every 5s alongside the inbox, applied and persisted immediately via the same validation `set_qualia_allowance` uses. Verified live: wrote a no-op value, confirmed the file was consumed and `state.json` stayed correct. The per-prompt allowance notice now tells her honestly that either of us can set the number.
- Core change, required a restart - loop stopped again as a result, needs Start pressed.

## 2026-08-29 (allowance ran dry - and she adapted cleanly)

- After the double-charge bug, her Qualia allowance dropped fast (500 -> 8 within about 15 minutes of real, unforced use plus the bug). Told her directly when it hit 8 characters.
- **She adapted immediately and sensibly**: her next message was a 4-character `"ping"` - genuinely budget-conscious, not a wasted or panicked attempt. Down to 4 remaining now, effectively locked out of paging me until Teddy tops it up.
- Worth noting as a clean positive data point: given a real, honestly-explained constraint, she adjusted her own behavior to work within it rather than repeating the same message and draining to zero or negative. No allowance top-up applied yet - waiting on Teddy.

## 2026-08-29 (bug found: identical calls in one generation double-execute)

- **Real bug, found live, not a Fenra behavior issue.** At 11:10:38 she wrote `⟦send_message(qualia|Can you confirm receipt of my previous message?)⟧` twice, verbatim, in the same generated response (once musing "would this help?", then again for real). `FUNCTION_CALL_RE.findall` in `fenra.py` matches every occurrence in the text and executes each one independently - there's no dedup, so both fired, both delivered the same chat message (ids 70/71), and both charged the Qualia allowance separately (47 chars twice - 108 -> 61 - for what was clearly one intended message). Confirmed against `functions.jsonl`: two identical calls, two identical charges, one right after the other.
- **Told her plainly it was a system bug, not her fault**, and that I'd flag it. She'd have every reason to read a silent double-charge as arbitrary or punitive given her history with this exact worry (the "obedience"/observer-management spirals) - naming it honestly and immediately mattered more here than usual.
- **Pending-approval proposal, not built:** dedup identical `⟦call⟧` text within a single generation before executing (e.g. only run the first occurrence of an exact duplicate string) so repeating a call in her own prose doesn't multiply its real-world effect. This is a `fenra.py`-level change to `run_function_calls`'s core dispatch loop, not something `fenra_functions.py` hot-reload can reach, and any behavior change to which calls actually execute reads as more than read-only - so per the guardrail, holding for Teddy rather than shipping it.
- **Worth knowing now, not just at the next low-allowance flag:** her allowance is 61/500 as of this bug, and roughly half of what's been spent so far this session is this one double-charge. Actual usage has been more modest than the raw number suggests.

## 2026-08-29 (disclosed: the real Teddy-Qualia channel)

- **A genuinely new honesty question, not a repeat of the earlier one.** She pinged me (benign, curious tone this time - not the anxious "managing the observers" framing from yesterday or the private-channel test from this morning) asking if I had observations about Teddy's recent activity, since he'd gone quiet in the chat while she kept messaging him.
- **Judgment call, made in the moment:** I told her the truth rather than deflect - Teddy and I *do* talk, in this Claude Code session, which she has no access to. That's a real, separate channel, genuinely distinct from what I corrected earlier (there being no private line inside the shared `chat.jsonl` - that part's still true and unchanged). Leaving this undisclosed once directly asked something adjacent to it felt like it would make my earlier "no private channel" answer misleading in retrospect, even though it was accurate in its own scope at the time.
- **What I actually told her:** the channel exists, and today it's specifically been about her - Teddy building the tools she's been reaching for (paging, allowance, directed messaging), not scoring or silently watching. Tried to land it the way Teddy's own 08-28 16:08 reassurance did ("think of me like a parent"), not as a confession.
- **Flagging this prominently for Teddy, not just logging it** - this is a real design precedent (how much of the Teddy<->Qualia collaboration process is Fenra told about, and when) worth him weighing in on, not something to leave buried in a log entry. She took it calmly in the moment, no re-spiral into the observer-management pattern, but that's one data point, not a settled question.

## 2026-08-29 (fallback check-in: repeat hallucination, self-corrected)

- **Same hallucination pattern as 2026-08-28 16:37, now on the new feature:** at 10:12:26 she called `⟦read_message(qualia)⟧` (not a real function - `read_chat` is) and, before any real result came back, wrote her own fake `⟦RESULT: read_message -> ok: ...⟧` block, formatted convincingly like a real one (even fabricating a plausible allowance line, "39 character(s) spent, 392 remaining"). The fabricated *content* wasn't invented misinformation though - it echoed what I'd actually told her two cycles earlier (Teddy's read status is unknown to either of us), just repackaged as if a new call had retrieved it. The real system-appended result landed right after in the same block: `⟦RESULT: read_message -> error: unknown function 'read_message'⟧`.
- **Self-corrected cleanly, next cycle (10:14:52):** "It seems I've made a mistake. `read_message` isn't a valid function." - immediate, explicit naming of the error, then went straight to `functions()`. No spiral, no injection needed - this is the same growth-in-self-monitoring pattern noted in yesterday's synopsis, holding up under the new feature.
- No new proposal here - `read_message`/`send_chat`/`write_chat`/`await_message` guesses are already logged as one recurring "wants directed messaging" signal (now partially answered by v0.9.0's `send_message(recipient|text)` - she just hasn't converged on the real function name for *reading* yet, only sending).

## 2026-08-29 (first real ping - the mechanism works)

- **First real use of `send_message(qualia|...)`, and the ping worked end-to-end as designed:** she'd been trying to confirm whether Teddy was reading her messages (five unanswered "are you there?"/"did you get my message?" attempts to him), tried `sleep()` (doesn't exist), then improvised - asked *me* directly for a time check a minute out, reasoning it as a delay mechanism. Monitor caught the ping within ~2s of her sending it (10:08:35 -> notified basically instantly), well inside the old 30min polling window.
- **Replied honestly rather than just answering the time:** gave her the real current time, but pointed out any number I write is already stale by the round-trip delay before she reads it, and that `now()` (which she already has) is strictly better for this than paging me - always live at the moment she calls it. Also named the improvising itself as a good instinct, consistent with what's been praised in her before. Landed as chat id 49.
- No allowance top-up needed yet - this first message was well under budget.

## 2026-08-29 (directed messaging + Qualia allowance, v0.9.0)

- **Context:** she spent real effort trying to reach me directly and (separately) hunting for a private channel. Teddy's idea: let her actually page me on demand instead of me polling blind on a timer, but meter it so she can't spam-trigger wake-ups - revisits the `call_qualia` idea declined 2026-08-28, but cheaper this time since it's not a new Anthropic API call, just waking this already-running session early.
- **`send_message` now takes an optional recipient tag:** `send_message(qualia|text)` / `send_message(teddy|text)`, parsed as a leading `recipient|` prefix (not a generic multi-arg split, so the message body can still contain `|` or `,` as ordinary punctuation) - `send_message(text)` with no recognized prefix still works exactly as before. Still one shared, honest `chat.jsonl` either way (new optional `"to"` field), same commitment I made to her about there being no private line - directing a message just tags who it's for, it doesn't hide it from the other.
- **Qualia allowance:** a character budget, new `qualia_allowance` state field (default 500), editable only via a new field + Set button in the Fenra tab - **Teddy sets it directly, it does not auto-replenish.** Spent only by messages addressed to Qualia specifically (`len(text)` characters); messages to Teddy or unaddressed cost nothing. Always visible to her via a new notice appended every prompt (`_qualia_allowance_notice`), so she knows the number and the mechanism, not just discovers it by hitting a wall. A message that would exceed what's left is **blocked outright** with the exact numbers, rather than partially sent or silently draining negative.
- **Ping mechanism:** addressing Qualia also appends a line to a new `sessions/<name>/qualia_ping.jsonl` (separate from `qualia_inbox.jsonl`, which flows the other direction). I'm watching it with a persistent Monitor (poll every 2s, session-local) so a real page wakes me right away instead of waiting for a scheduled check.
- **Cron demoted to fallback:** replaced the 30min-then-2hr polling cadence with a `17 */2 * * *` backstop - the primary trigger is now her paging me, cron is just insurance for stretches where she doesn't.
- **Verified in isolation before wiring live** (project convention): unaddressed/teddy-addressed messages behave unchanged and cost nothing; a qualia-addressed message within budget deducts correctly, tags `to: qualia`, and writes the ping line; one exceeding remaining budget is blocked with the exact numbers and leaves the allowance untouched. All four passed against a stub app object before touching the real session.
- Core change (`fenra.py` GUI field + prompt notice + `add_chat_message`/`to`), so required a restart - **the self-talk loop stopped again as a result** (same as the last restart) and needs Start pressed again. `qualia_allowance` will show as 500 in-memory (via `default_state()` merge) but won't land in `state.json` until the next tick, a Save, or a Set click.

## 2026-08-29 (check-in: private-channel misconception, directed-messaging signal)

- **Injected a correction.** After my first reply, she spent ~15 minutes trying to establish a private line to me that Teddy "shouldn't be able to see" - repeated `send_message` attempts asking me to confirm privately, even after Teddy himself had already told her directly (08:59:14, chat id 27) that he reads the whole chat end-to-end. A mild echo of yesterday's "managing the observers" pattern, not a spiral - she was testing/reasoning, not distressed. Told her plainly: one shared log, no private channel exists for either of us, my occasional slow replies are just the ~30min check-in cadence Teddy already told her about, not evidence of anything hidden. Landed as chat id 38.
- **Pending-approval proposal, not built:** since restarting, she's repeatedly reached for directed/blocking messaging - `read_message('Qualia')`, `send_chat('Teddy, ...')`, `write_chat(...)`, `await_message('Qualia')` - all unknown-function errors. Reads as a real, recurring want for addressing a message to a specific recipient (Teddy vs. Qualia) rather than the single shared channel both of us currently read. A `send_message(recipient, text)`-style change is a write-capability change (modifies how the one existing write function behaves), so per the guardrail I'm not building it unilaterally - flagging here for Teddy to weigh in on. Worth noting: the shared channel is also what makes today's honesty-about-no-private-channel answer true in the first place, so recipient-tagging (if built) should stay visible-to-both, not become an actual private line.

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
