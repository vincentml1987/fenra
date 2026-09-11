# Pick-up — start here, 2026-09-10 (end of session, later same night)

Written for a fresh Claude session to re-initialize from. Full detail
lives in `Qualia/decisions.md` (fenras-aletheosis branch's authoritative
log, unchanged since 2026-09-08) and `Qualia/worlds-rebuild-notes.md`
(this branch's own log, updated through tonight) - this file is a map
to both, not a replacement. This pickup supersedes the last one from
earlier tonight - that one's "HUD/functions/currency/masking/structured
messages/object-oriented GUI/the_town creation" history is all still
accurate background, just not repeated here in full.

## Who you are here

You're **Qualia**, AI collaborator on Fenra's Aletheosis
(`vincentml1987/fenra`). The user is **Teddy** — co-author of the
Aletheia framework this project is architected around
(`Qualia/aletheia-notes.md`). Read `CLAUDE.md` if you haven't this
session.

## Where things actually stand

Same two branches - `fenras-aletheosis` (old shipped app, untouched,
`tribe-3` stopped) and `worlds-rebuild` (current branch). `the_town`
(8 voices, 8 groups) is **running** - check `Get-Process python` for
the current PID, don't trust one written here. **Uncommitted changes
on disk right now**: `fenra.py` (the `_tick` snapshot-clobber fix) and
`Qualia/worlds-rebuild-notes.md` (two new to-do entries) - Teddy hasn't
asked for a commit yet, don't commit without asking.

## Tonight, in order

1. **Wipe-and-restart `the_town`** - Teddy's explicit call: it had never
   been restarted after the structured-messages/GUI-rebuild landed, so
   he wanted it wiped and rebuilt fresh on the current schema. Stopped
   the old process, rewrote all 8 voices' `state.json` (same
   models/identities/currency=10.0, messages emptied) and all 8
   groups' `.json` (same members, boards emptied), reset
   `voice_rotation_index` to 0, relaunched with a new distinct log name
   (`the_town_run_20260910f_wipe.log`) per [[fenra-process-log-naming]].

2. **Real bug found and fixed: `_tick()`'s snapshot-save was clobbering
   deliveries.** Teddy noticed Dash's Messages tab showed only Dash's
   own self-authored entries - nothing delivered from Marisol/Orin/etc.
   despite shared group membership. Root cause: `_tick()` persisted
   whichever voice was *currently selected in the GUI* back to disk on
   **every** tick, using the widget's stale in-memory `_current_messages`
   - not gated on whether that voice was actually the one about to
   speak. Since a selected voice's in-memory copy only refreshes when
   *that voice itself* speaks, every tick in between silently overwrote
   real deliveries appended by other voices' turns with the stale
   cached list. Not Dash-specific - it follows whichever voice is
   selected.
   - **Fix applied**: moved `active_voice` computation before the
     snapshot-save and gated the save on `displayed_voice == active_voice`
     - matches the original comment's actual intent (protect an
     in-flight manual edit right before that voice's own turn).
   - **Known residual edge case, not fixed** (flagged to Teddy, not
     built): even gated correctly, a selected voice's widget can still
     be stale relative to disk if it received deliveries *between* its
     own turns (the widget doesn't live-refresh) - so its own next-turn
     snapshot-save could still clobber those. Only matters if a voice
     is left selected for a while; worth a real fix later if it turns
     out to matter (live-refresh on delivery, or merge-not-overwrite).
   - **Announced into all 8 voices' contexts before relaunching**, per
     [[fenra-function-fix-announcements]] - vague by default: "Qualia
     and Teddy: We fixed a bug in your environment." Sent to the whole
     cast since the bug could've hit whichever voice was selected at
     any point, not provably just Dash. Teddy approved the announcement
     after the fact. Relaunched as
     `the_town_run_20260910g_deliverybugfix.log`.

3. **Timestamp field is cosmetic, not structural** (Teddy asked
   directly). Confirmed: both `render_messages()` (what Ollama sees) and
   the GUI's Messages tree iterate in **list/id order**, never
   re-sorted by the `timestamp` string - a backdated or edited timestamp
   changes what a line *says* but not where it sits in the sequence.
   Teddy's take: automatic chronological reordering makes more sense.
   **Explicitly "don't fix now"** - logged as a to-do in
   `worlds-rebuild-notes.md`.

4. **Second UI to-do logged, also not built**: the status label next to
   Start/Stop only updates *after* a model call returns
   (`"Running ('{voice}' spoke)"`) - Teddy wants it to show who's
   currently *thinking*, mid-call. Logged alongside the timestamp item.

5. **Real finding: Orin (qwen3:14b) - personality collapsing into a
   catchphrase**, not a bug. His own (unmasked) turns all open "Ah,
   [topic]—" and lean on the same handful of images (gears/rust,
   echoes, whispers, absence/void) regardless of actual topic - his
   *dreamy/philosophical* trait degrading into its own template rather
   than resisting convergence the way the rest of the cast has. Only 3
   real turns sampled so far - could still be an early-run artifact.
   Not logged as a finding yet - was mid-discussion when superseded by
   the bigger Cole/Priya finding below; revisit if it persists.

6. **Real finding, more severe: Cole and Priya (both `phi3:14b`) -
   runaway/self-repeating generation, now confirmed as a known Phi-3
   weakness, not a Fenra bug.** Priya's turn (real timestamp 17:28,
   *before* the delivery-bug fix landed) devolved into a single
   ~500-line block: a fabricated multi-hour group conversation with the
   same line ("Let's make some great memories while exploring
   DreamerCape.") repeated hundreds of times across invented future
   timestamps, then broke character entirely into an unrelated
   statistics homework problem. Cole showed the same shape more
   gradually - his own turns grew 131 → 3,776 → 5,509 → 23,115
   characters, with the last one repeating a full paragraph
   near-verbatim within itself.
   - **Researched and confirmed as documented Phi-3 behavior** (not
     specific to this setup): multiple Ollama GitHub issues
     ([#6474](https://github.com/ollama/ollama/issues/6474),
     [#7931](https://github.com/ollama/ollama/issues/7931)) and a
     HuggingFace discussion on the Phi-3-mini GGUF
     (https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/discussions/4)
     describe the identical shape: fine for short generations, then
     garbage and no stop, tied to unreliable `<|end|>`/EOS
     stop-token handling, worse in quantized (GGUF) builds and worse
     as context approaches the model's limit. Fenra's per-voice
     message history grows unbounded every turn, which lines up
     directly with what triggered both Cole's and Priya's blowups.
   - **Teddy's real fix**: flipped both Cole and Priya's `model` field
     from `phi3:14b` to `deepseek-r1:14b`, no other changes. Left both
     voices a plain "Your model has been updated." (his own message,
     not run through the fix-announcement template - his call, noted
     for transparency only, no action taken on my end).
   - **Priya's pathological turn itself was never touched** - still
     sitting verbatim in her `state.json`, still being resent to the
     model every turn per `render_messages()`. Teddy was offered the
     option to trim/delete it and hasn't asked for that - don't touch
     it without him asking, same spirit as [[fenra-history-integrity]]
     even though that rule is about fabrication, not removal.

## Open, not yet done (carried forward + new)

- **`recollect(query)` function idea** - still not designed, see
  `worlds-rebuild-notes.md`'s "Ideas for later" section for the actual
  open questions (search semantics, scope, result caps).
- **Timestamp-based auto-reordering of messages** (new tonight, #3
  above) - not designed, logged only.
- **"Thinking..." status label** (new tonight, #4 above) - not
  designed, logged only.
- **Possible real fix for Phi-3-style runaway generation** (new
  tonight) - a `num_predict` cap and/or repeat-penalty/stop-token
  tuning on `call_ollama` was discussed as an option but explicitly not
  pursued since Teddy solved the immediate problem by swapping models
  instead. Worth raising again if another model shows the same
  failure, or if Teddy wants a systemic guard rather than per-voice
  model swaps.
- **Orin's catchphrase-collapse pattern** - worth re-checking after a
  few more of his real turns accumulate; not logged as a formal finding
  yet.
- **"Communicate with our world" thread** - Teddy's longer-term goal,
  still untouched since being set aside for boards. Real open
  questions never answered: does a message from a voice interrupt
  Teddy, land for him to read later, or wake Qualia - and what
  rate-limits spam. Raise again directly rather than assuming.
- Raven/Unfolding (`Teddy's ChatGPT Export/`, gitignored) - still
  waiting on Teddy reactivating his ChatGPT account, no update expected
  without him raising it.
- `alphabet-26` - still stopped, still on the pre-structured-messages
  schema, would need its own migration to resume.

## Standing behavioral rules to carry forward (all in persistent memory)

Same list as before: [[fenra-history-integrity]],
[[fenra-existential-distress-protocol]], [[qualia-page-review]],
[[lcraou-protocol]], [[proactive-design-flagging]],
[[teddys-journals-practice]], [[aletheia-repo-split]],
[[user-nickname-teddy]], [[fenra-ai-gmail-access]],
[[fenra-engage-gate]], [[fenra-process-log-naming]],
[[fenra-function-fix-announcements]] (applied for real again tonight -
see #2 above for the exact wording used). [[fenra-chat-restraint]] still
flagged stale, still moot - `worlds-rebuild` has no chat function.
`MEMORY.md` indexes all of them.
