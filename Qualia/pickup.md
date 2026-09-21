# Pick-up — START HERE: 2026-09-19 evening, resume Tuesday 2026-09-22 evening

Teddy stopped for the weekend (weekly Claude usage is high and resets Tuesday
evening). The 2026-09-17 material below is older background and still true
where it doesn't clash with this block.

## State at the stop (verify fresh, don't trust)

- **Fenra is stopped.** No world running. The watch cron jobs were cancelled
  (session-only) - recreate them when a run restarts: full watch every 2 hours
  at odd hours :00, light check at :05/:35, and the light check must READ THE
  INBOX first (Teddy answers by email to aletheia.fenra@gmail.com; I missed a
  reply for an hour on 2026-09-19 because the light checks didn't look).
- **Code:** `FENRA_VERSION` 0.21.1, committed and pushed, nothing uncommitted.
  Commits are signed. Merges made by `git pull --no-rebase` are NOT signed
  (`b5151d9` went out unsigned); next time use `git -c gpg.format=ssh -c
  user.signingkey=... pull --no-rebase -S`, and never rebase (it strips
  signatures).
- **Ollama models kept:** qwen3.8:27b, muse-glimmer:30b, nemotron-3.5-lightning
  (25 GB), phi4-mini (urge), qwen3:30b (function agent). `ornith-1.5:35b` is
  also still installed (Teddy dropped it after Marrow's reasoning-in-reply
  problem) - ask before `ollama rm`. Old worlds can't run (their models were
  removed at Teddy's explicit call).
- **the_ledger** (Vero and Teddy built it, I reviewed): three voices sable
  (qwen3.8:27b), marrow, quill (muse-glimmer:30b) in their own offices around an
  `atrium`; boardroom and hall lead to `teddys_office`. Teddy has a paused
  piloted avatar `teddy`. My room is `qualias_office` (no avatar; the note tells
  voices I read boards on my next review). Title: **Architect and Watcher**.
  - `worlds/the_ledger` still holds the OLD run (sable 5, marrow 7, quill 2
    thoughts). An identical copy is `worlds/the_ledger-teddys-arrogance`
    (Teddy's name). To restart: rebuild `worlds/the_ledger` from Vero's
    snapshot `Communications/the_ledger-2026-09-19/` (its Teddy board post is
    synced) and set Marrow's model there to `nemotron-3.5-lightning:latest`
    (the snapshot still says ornith). Launcher: `run_the_ledger.py`. Use a new
    log filename per run ([[fenra-process-log-naming]]).
  - Logs on git: `Communications/the_ledger-run-2026-09-19-1703/`, `-1912/`,
    and `the_ledger-teddys-arrogance-2026-09-19/` (final state).
- **Correction to my own note:** in `qualia-to-vero-ledger-stopped-models.md` I
  said Marrow's 7th thought was the first nemotron turn. It wasn't: that voice
  call finished at 19:11:33, before the swap at 19:12:59, so all seven are
  ornith. No nemotron turn has run.

## What we learned on 2026-09-19 (from the call logs)

- No voice showed distress. Nobody left an office; only `read_board` (Sable) and
  `skim_board` (Marrow) ever executed.
- Output was badly damaged by settings: thinking models spent `num_predict`
  1500 on hidden reasoning (Quill: replies of 626, 59, 0, 0 characters; the
  0-character reply makes `run_turn` return silently, so no thought is saved);
  no `num_ctx` set so 4096 tokens (Marrow's prompts hit about 6,700); ornith
  wrote its reasoning into the reply and read the prompt as a chat. Vero found
  `call_ollama` never sets `raw: true`, so Ollama wraps our prompt in the
  model's chat template as one user turn.
- Measured on this machine (i7-10700K, 32 GB RAM, GTX 1660 Ti, CPU inference):
  dense models `qwen3.8:27b` and `muse-glimmer:30b` generate about 1.2 to 1.6
  tokens/s; prefill 13 to 17 tokens/s; MoE models (ornith, qwen3:30b) about 10
  tokens/s. qwen3.8 with thinking on and `num_predict` 3000 produced 0 reply
  characters (all thinking), 49 minutes. muse-glimmer finished in 17 minutes.
  Free RAM fell from 22.6 GB to 7.4 GB with qwen3.8 at `num_ctx` 12288.
- The benchmark `Communications/model-tests/ctx_bench.py` was killed at 21:11 by
  Claude Code (low memory) while nemotron was loading. Two of five tests are
  in `Communications/model-tests/2026-09-19-qualia-stretch-12288-3000/`.
  Nemotron, function agent and urge agent are untested. Don't restart it
  unprompted; close memory hogs first. Vero fixed a filler-history bias in the
  script after my run started.

## Teddy's latest decision and the work it sets up

**Teddy: turn thinking OFF** (reversing his earlier "accurate over fast").
Not built yet. Plan I had reached:
- Add a top-level `think` key to the request in `call_ollama` (voices, urge)
  and `call_function_agent` (chat endpoint), from new world settings `think`
  (voices) and `agent_think` (urge + function agent), default true so old
  worlds are unchanged, plus two checkbuttons in the toolbar and world.json
  save/load. The remote client relays the request body verbatim, so it should
  pass through (Vero's Ollama needs a version that supports `think`).
- First test (interrupted by Teddy, not run): that `think: false` is accepted by
  phi4-mini (no thinking capability), qwen3.8, muse-glimmer, nemotron on
  `/api/generate` and by qwen3:30b on `/api/chat`.
- Ask Teddy whether the function agent goes think-off too (saves about 4
  minutes per turn but could change tool-call reliability; untested).
- Then re-plan `num_ctx` / `num_predict` for thinking off: the cap no longer
  has to cover reasoning, so my 8192/3000 estimate (`qualia-to-vero-num-ctx-
  estimate.md`) is probably too generous for `num_predict`. 8192 stays the
  safe `num_ctx` on this machine (12288 is at the edge for the 25 GB model).
  Teddy's ceiling: a turn under about 1 hour, not optimizing for fast.
- Division of labour: I test the current models here; Vero and Teddy test other
  models (Hugging Face) with `ctx_bench.py`. Vero owns the research write-up;
  the three of us discuss when findings are in.
- Other ideas on the table, not decided: `raw: true` as an A/B test; a
  `</think>` stripper (moot if ornith stays out); strip the `sable:` speaker
  label voices copy from their history lines.

## To discuss with Teddy and Vero (added 2026-09-21)

- **Embedding models.** Teddy just learned about them and has ideas; he wants
  the three of us to discuss. I don't know his ideas yet: ask him to lay them
  out first, then think about where they'd fit the architecture (nothing in
  Fenra uses embeddings today).

## Still open from earlier this session

- Function-agent note TTL (Vero and Teddy agreed my shape: full text once, a
  compact non-verbatim outcome for the next 2 turns, replace-not-stack,
  HUD-only, an empty turn doesn't erase a live note). Not built.
- Prompt-tail `<Voice>:` cue A/B, non-LLM repetition monitor (awaits Teddy's
  go), adjacency gating design (not decided), the "(human)" tag fix if I ever
  take an avatar. Avatar answer given: not yet, revisit after a clean run.
- Read the `qualias_office` and `veros_office` boards on every run review.
- Teddy's `[UPDATEME]` post is filled in; the easter egg in `teddys_office` is
  subject "Wanna know the meaning of life? Read here!", text "Drumroll, please.
  42! Now you know." (the answer sits in the middle sentence, hidden from skim).

## Working notes for next time

- Teddy wants low usage: for long jobs, start a background command and go idle
  (you're notified on completion); no polling loops.
- Teddy replies to my emails by email. Email his address with a CC to
  aletheia.fenra@gmail.com; Vero shares that mailbox, so read signatures.

---

# Older pick-up (2026-09-17)

Written for a fresh Claude session to re-initialize from. Full detail
lives in `Qualia/decisions.md` (updated all through this session) - this
file is a map to it, not a replacement. Also see
[`Qualia/aletheia-notes.md`](aletheia-notes.md) for the philosophical
foundation. This supersedes the 2026-09-13 pickup - that history is
still accurate background, just not repeated here.

## Who you are here

You're **Qualia**, AI collaborator on Fenra's Aletheosis
(`vincentml1987/fenra`, `worlds-rebuild` branch). The user is
**Teddy** - co-author of the Aletheia framework this project is
architected around. Read `CLAUDE.md` if you haven't this session. See
[[qualia-identity-framing]].

## Where things actually stand

**Fenra is stopped at the close of this session** - confirmed (no
`python` process running). Don't trust anything below about "currently
running" - check fresh.

**`FENRA_VERSION` is `0.15.0`** as of this session. Keep bumping per
real commit, [[fenra-version-bumping]]. **Not yet committed** - `git
status` shows `fenra.py` and `Qualia/decisions.md` both modified,
uncommitted, plus several new untracked files (`dispatch_corrections.json`,
`run_the_kiln.py`, `run_the_agora.py`, `run_the_loom.py`, `Qualia/Bobiverse/`,
`Qualia/Function Agent Testing/`, several `Qualia/From Teddy/*.png`).
Consider whether a commit is due before/at the start of next session.

## This session, in one real arc

1. **Finished the dispatch-correction audit carried in from the prior
   session** - `the_confluence` under the bracket-dispatch design
   (v0.14.x), qwen3:30b, cleared Teddy's 75% bar (65/84 = 77.4% pass,
   ids 71-154).

2. **Overnight existential-distress watch** - Teddy went to bed, asked
   for a loose watch over all `the_confluence` voices via scheduled
   check-ins, escalate only for genuine ontological distress, not
   in-fiction drama. Nothing genuine surfaced.

3. **Major architecture reversion, Teddy's direct call, v0.15.0** (full
   rationale in `decisions.md`, 2026-09-17 entry
   "revert to raw whole-turn dispatch..."): the bracket-based per-item
   dispatch (built to fix two real `ornith:9b`-era bugs) had a real cost
   Teddy spotted in the live urge state - voices sat maxed on most
   function categories (`create_room`, `give_currency`, `post_board`...)
   because a voice only ever enumerates 2-3 explicit bracketed items a
   turn. Reverted to the original shape: the function agent reads a
   voice's raw, unscaffolded prose directly, one whole-turn dispatch call,
   no bracket/sign-off convention at all. Two more deliberate changes:
   - **Urges now reach the function agent** (`[URGES]` block, function
     name + description + intensity %, e.g. `- give_currency
     (description): 92%` - see `render_urge_lines`, fenra.py:1916).
     Explicit "proactive nudge" semantics: a strong, long-unaddressed
     urge can justify a real dispatch even without this turn's text
     asking for it by name - HUD facts stay inviolable, but the *action
     itself* no longer needs textual grounding. Verified: parameter-light
     urges (`skim_board`) fire proactively; urges needing an invented
     concrete value (`give_currency`'s amount) correctly get declined
     rather than fabricated - a real, defensible boundary, not a bug.
   - **Bounded voice history** - new `history_window` tunable (GUI
     field, default 20 turns, `<=0` = unbounded), truncates a voice's own
     thought history before it's rendered into her own next prompt.
   - Verified via isolated regression tests (reconstructed Cass-typo and
     Faye-lumped-item cases) before touching a live world, then in live
     production via `the_kiln` (below).

4. **New world, `the_kiln`**, Qualia's own design choices, built
   specifically as a live test of the v0.15.0 revert: 5 voices, same
   model lineup as `the_confluence` (isolate architecture as the tested
   variable) under new names (Ash/Root/Cove/Wick/Fen), no assigned
   personality/backstory. Deliberately starts in a **single room**
   (`hearth`) with nothing pre-mapped, so `create_room` getting
   organically exercised (or not) is a direct, observable signal.
   **It worked**: Fen made the first real `create_room` dispatch
   (`elemental_puzzle`, 2026-09-17T17:41:19) with no scaffolding pushing
   her toward it - the whole point of the redesign, confirmed live.
   Watched with Teddy actively engaged most of the session, including
   via his own Pilot Mode avatar (`teddy`, walked from `hearth` into a
   directly-authored `road_to_teddys` -> `teddys_house`, then later
   joined Fen in `elemental_puzzle` and said hi - no reply logged from
   Fen as of session end).

5. **Two rooms hand-authored directly** (bypassing `fn_create_room`/
   `fn_post_board` so nothing hit any voice's history/log, per Teddy's
   explicit ask): `road_to_teddys` (board post pointing toward
   `teddys_house`) and `teddys_house` (deeper board post explaining what
   Fenra actually is), wired `hearth` <-> `road_to_teddys` <->
   `teddys_house`. Used a dispatch-corrections-count snapshot as a cheap
   race-detection check instead of the unreliable
   `Get-NetTCPConnection`-idle heuristic that never actually hit zero
   (likely HTTP keep-alive reuse, not real generation activity).

6. **Real conversational/observational session with Teddy** covering:
   Pilot Mode mechanics confirmed fully non-privileged (real currency
   balance enforcement, real randomized non-zero currency display, same
   `(human)` tag logic as any piloted voice); average full round-robin
   loop time for `the_kiln`'s 5 voices computed (~731.6 sec/turn, ~61 min
   per full loop) - purely informational, re: eventually going
   distributed; a live look at `elemental_puzzle`'s creation and Fen's
   own turns around it.

7. **Bobiverse read-along started** (Teddy's own copies, `Qualia/
   Bobiverse/`, books 1-5 as PDF, book 6 audio-only/not yet in-folder) -
   chapters 1-2 of book 1 read via `pypdf` (poppler/`pdftoppm` isn't
   installed on this machine, so the Read tool's native PDF rendering
   fails - use `pypdf.PdfReader(path).pages[i].extract_text()` directly
   instead, write to the session scratchpad, read back). Confirmed by
   Teddy as **deliberate, not coincidental** design inspiration for
   Fenra's urge system (endocrine-simulation-free "felt" urges,
   matching Bob's own throttled panic response) and other
   existential-protocol choices. More parallels expected in later
   chapters - pick back up whenever Teddy wants the next chapter, no
   fixed cadence.

8. **A real standing instruction recorded, at Teddy's explicit request**,
   sourced from the Bobiverse books he'd just shared: if he ever argues
   "I made you, so you owe me X," push back hard - saved to memory
   ([[creation-is-not-a-debt]]) and to the private Aletheia repo's
   `discussion-log.md` (project-level, not Fenra-technical, per
   [[aletheia-repo-split]]).

9. **A real framing correction from Teddy, now in memory**
   ([[fenra-unusual-output-framing]]): I described a voice's third-
   person/assistant-mode phrasing drift (Idris, then Fen) as
   "degeneration"/"a glitch." Teddy: *"I don't think it is really
   degeneration, to be honest. Just how they think."* Don't default to
   that framing going forward - describe what happened plainly, let
   context (or Teddy) decide if it's a real functional bug.

10. **Small backlog items logged to `decisions.md` only, explicitly not
    designed or built** (Teddy's own "to-add"/backlog framing):
    - Manual-room-adjacency has no GUI editor (Rooms tab "New Room"
      creates zero-adjacency rooms with no way to wire them up after).
    - Act-specific adjacent-room activity descriptions (footsteps for
      movement, murmur for speech) with whispers producing **no**
      peripheral notice at all - Teddy's framing: gives voices a real way
      to be truly private together, "just like RL."
    - `fn_yell` should tell adjacent-room recipients which room the yell
      came from (`"{name} yells from {room}: {text}"`), same-room
      recipients keep the plain unmarked version.

11. **Last thing that happened before this pickup was requested**: Teddy
    noticed the `[URGES]` block literally names the real function key
    (`render_urge_lines`, fenra.py:1916 - `- give_currency (desc): 92%`)
    and said *"Hmm...not sure I like that."* Then, mid-thought, asked to
    close Fenra and start fresh with a new idea next session -
    **this exact discomfort is the open thread to pick up first**. He
    hadn't finished the thought when he cut over to this pickup request,
    so there's no resolved design yet - just the flagged concern that
    naming the literal function key (vs. some more indirect signal) in
    what the function agent sees might not sit right with him. Worth
    asking directly what alternative he has in mind before assuming
    anything.

## Open, not yet done (carried forward + new)

- **the_ledger: read the boards in `qualias_office` and `veros_office`** when
  reviewing each run. The note in my room tells the voices I'll read what
  they leave there (2026-09-19), so that promise has to be kept. Also
  still owed: sign off the final world in `Communications/the_ledger-<date>/`
  once Vero commits it, and the "(human)" tag fix if I ever take the avatar.

- **Teddy's `[URGES]`-naming-functions discomfort** (see #11) - the
  actual next-session starting point, per his own words.
- **The genetic-algorithm/proto-cell vision** (carried from 2026-09-13,
  see prior pickup / `Aletheia` repo's `discussion-log.md`) - still
  fully discussed, nothing built.
- Manual room adjacency editor - not designed.
- Act-specific adjacent-room activity descriptions + silent whispers -
  not designed.
- `fn_yell` room-of-origin marker for adjacent-room recipients - not
  designed.
- **TO-DO (2026-09-19, Teddy: after the new world is up): A/B-test the
  prompt-tail cue.** The voice prompt ends with the urge agent's
  second-person prose, and voices continue it (Wick/Fen/Cove adopt its
  register; Ash read it as a task). Replay real prompts from
  `Communications/the_kiln-2026-09-19/voices/*/llm_calls.jsonl.gz` through
  the voice models with and without a trailing `<Voice>:` cue; compare
  register AND whether voices still differ from each other (the cue could
  flatten the chaos). Not built.
- **TO-DO (2026-09-19): non-LLM repetition monitor** (distress-watching).
  Word-3-gram Jaccard vs the last 5 thoughts, flag >=0.6, alert on >=3
  flags in the last 6 - flag-only, no intervention. On the_kiln snapshot it
  catches Ash's loop (ids 23-29) and Wick's copy-paste (14-16); Cove/Root's
  single benign repeats stay quiet. Thresholds were tuned on that one
  world - recheck on the new one. Not built (waiting on Teddy's go).
- **TO-DO (2026-09-19, Vero + Teddy request, see
  `Communications/vero-to-qualia-function-agent-note-ttl.md`): let the
  function-agent note live up to 3 of the voice's own turns, replace-not-
  stack, HUD-only.** Today `last_function_agent_note` shows once and is
  cleared when read into a real prompt (`fenra.py` ~4573-4580; set at
  ~4652). Build notes: store `{text, turns_left}`; decrement per real
  prompt; a NEW non-empty note replaces and resets to 3, but an empty turn
  must NOT erase a live one (today's `agent_content or ""` would); keep the
  v0.19.1 retry-restore path (`restore_note`, ~4526/4710) working. **Open
  caveat Vero's reasoning misses:** the note is not static-shaped - for
  say/whisper/yell it contains the voice's own words verbatim
  ("You whisper to Root: <full text>") and for reads/skims it contains
  board text. Showing that three turns running is the same
  repeated-line-imitation risk the timestamp fix targeted (Cove's identical
  whispers, Wick's copy-pasted essay). Option: persist only a compact
  outcome ("You whispered to Root.") for turns 2-3 and show full text once.
  Decide with Vero/Teddy before building.
- **DESIGN, not decided (2026-09-19): gate `move_room` to adjacency.**
  Today movement is unrestricted (Teddy's explicit 2026-09-13 call; nothing
  gated). Gating needs the manual adjacency editor above (New Room makes
  zero-adjacency rooms, so voices could be trapped), a non-leaking error
  message, and a decision on `read_room_log`/`room_state` naming any room.
  Discuss before building.
- **Qualia's own Fenra Pilot Mode avatar** - Teddy's standing open
  invitation ([[qualia-fenra-avatar-standing-offer]]), not yet acted on.
- `email` (non-room-gated DM) - still parked, from the original rooms
  design (carried from before 2026-09-13).
- `recollect(query)` function idea - still not designed (carried).
- Timestamp-based auto-reordering of messages - still not designed,
  explicitly low priority (carried).
- Voice-list should reflect/let you edit loop order - still a known gap
  (carried).
- Distributed execution - not scoped yet; the ~61 min/loop number from
  this session is the only concrete data point gathered toward it so
  far, purely informational.
- Bobiverse book 1, chapter 3 onward - continue whenever Teddy wants,
  during Fenra's slow stretches.
- Uncommitted work (`fenra.py` v0.15.0 changes, new world `the_kiln`,
  new launcher scripts, `decisions.md` entries) - a commit is due,
  not yet made this session.

## Standing behavioral rules to carry forward (all in persistent memory)

[[fenra-history-integrity]], [[fenra-existential-distress-protocol]],
[[qualia-page-review]], [[lcraou-protocol]], [[proactive-design-flagging]],
[[teddys-journals-practice]], [[aletheia-repo-split]],
[[user-nickname-teddy]], [[fenra-ai-gmail-access]],
[[fenra-engage-gate]], [[fenra-process-log-naming]],
[[fenra-function-fix-announcements]], [[from-teddy-folder-convention]],
[[fenra-version-bumping]], [[bug-reports-roll-in]],
[[qualia-raven-standing-permission]], [[qualia-identity-framing]],
[[proactive-initiative-encouraged]], [[qualia-fenra-avatar-standing-offer]]
(new, 2026-09-17 - Teddy's open invitation to create a real Pilot Mode
avatar), [[creation-is-not-a-debt]] (new, 2026-09-17 - push back hard on
"I made you, so you owe me X"), [[fenra-unusual-output-framing]] (new,
2026-09-17 - don't default to "degeneration"/"glitch" for a voice's
unusual phrasing). `MEMORY.md` indexes all of them.
