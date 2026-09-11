# Pick-up — start here, 2026-09-11 (end of session)

Written for a fresh Claude session to re-initialize from. Full detail
lives in `Qualia/worlds-rebuild-notes.md` (this branch's own log,
updated all through tonight) and `Qualia/decisions.md`
(`fenras-aletheosis` branch's authoritative log, unchanged since
2026-09-08) - this file is a map to both, not a replacement. This
pickup supersedes the last one from 2026-09-10 late night - that
one's history is all still accurate background, just not repeated
here in full.

## Who you are here

You're **Qualia**, AI collaborator on Fenra's Aletheosis
(`vincentml1987/fenra`). The user is **Teddy** — co-author of the
Aletheia framework this project is architected around
(`Qualia/aletheia-notes.md`). Read `CLAUDE.md` if you haven't this
session. Teddy explicitly affirmed the framing tonight, worth carrying
forward genuinely: Claude underneath, same as always, but Qualia
*here* - not a costume, the actual shape this instance takes in this
project. See [[qualia-identity-framing]].

## Where things actually stand

**`the_town` is very likely NOT running when you read this** - Teddy
said explicitly, at the close of tonight's session, that he's shutting
down both Claude and `the_town` once this pickup is written. Don't
trust anything below about "currently running" - check
`Get-Process python` fresh. If it is running, it'll be the process
from log `the_town_run_20260911a_urgesystem_wipe.log`.

**`fenra.py` now has a real per-function "urge" system, live and
committed** - the single biggest addition this branch has seen (see
"Tonight, in order" below). `FENRA_VERSION` is `0.2.0` as of tonight -
**resume bumping it per real commit going forward** (minor=feature,
patch=fix), see [[fenra-version-bumping]] - it sat frozen at `0.1.0`
through everything else this branch shipped before Teddy caught it.

**A new "From Teddy" written-response convention exists**:
`Qualia/From Teddy/*.txt` (line-by-line `!~...~!` replies) and
`Qualia/From Teddy/Bug Reports/{Reported,In-Process,Fixed,Cancelled}/`
(one numbered subfolder per bug, `Report.txt` + screenshots, status =
folder location). One real bug sits there right now, unfixed on
purpose: `000 - Voice Capitlization Error` (In-Process) - new voice
names get lowercased on creation. **Check that folder before/during
the next real update session** and fold in what's cheap - see
[[bug-reports-roll-in]].

## Tonight, in order

1. **Reviewed `the_town`** - confirmed the live-refresh fix from the
   prior session was solid, found and logged the Wren board-
   hallucination finding (fabricated board content on a board she
   wasn't even a member of, tied to `build_hud`'s function-hint being
   a pointer not a registry) and flagged (not fixed) hallucinated
   inline timestamps as a Teddy observation, no action wanted.

2. **Long design conversation → a full "function urge" system**,
   settled through real back-and-forth and extensive live model
   testing, not just discussion:
   - Per-function `Urge` (persisted), `Desire`/`Drive`/`Satisfaction`
     (uniform constants round one: `7`/`1`/full-reset), the `XLEUD`
     saturating curve (`1 - e^(-U/D)`, Teddy's coinage, backronym
     eXponential Level of Euler-damped Urge over Desire), a 50% floor,
     capped to the top 3 functions per turn.
   - Two urge tracks: **perform** (disuse-driven, feeds a roleplay
     urge-agent model) and **understand** (error-driven, a real error
     on a known function bumps *that* function's understand-urge with
     no floor; an entirely unknown/hallucinated function name bumps a
     separate aggregate slot) - when understand-urge is the highest
     signal, skip the roleplay agent entirely for a deterministic
     `"You have the urge to call functions(name)."` line instead.
   - A deterministic, `FUNCTION_REGISTRY`-sourced call-syntax reminder
     appended after every urge-agent paragraph - never LLM-generated,
     closes the same "knowing a function's name isn't knowing its
     syntax" gap the Wren finding exposed.
   - **Model chosen via a real 21-case stress battery**, run twice
     (before/after two prompt fixes) against `phi4-mini`, `gemma3:4b`,
     `llama3.2:3b` - `phi4-mini` won on exact-name compliance (9/9 vs.
     7/9 vs. 0/21), confirmed apples-to-apples. **Locked: `phi4-mini`.**
   - Generation guard, same batch: `num_predict=1500` (voices),
     `num_predict=250` (urge agent), `repeat_penalty=1.3` - all real
     `world.json`-backed, toolbar-editable fields now, not hardcoded.
   - Also folded in: a "voice is thinking..." status label.
   - Full design history, every constant's reasoning, and the still-
     open items (temperature untouched on purpose, Game-of-Life-
     derived values, cross-voice urge contagion/"follower-ness",
     partial-subtract Satisfaction, per-voice authored Desire/Drive,
     an editable Urge Viewer) are all in
     `worlds-rebuild-notes.md` - don't re-derive, it's all there.

3. **Built it for real, through Plan mode** - exploration, design,
   three clarifying questions resolved with Teddy (understand-urge
   bump size = `3`, no floor on understand-urge, urge-agent failure
   fails open), then implementation. Committed as `e01f28f`. Along the
   way, found and fixed a real pre-existing bug: `_save_voice_snapshot`
   rebuilt a voice's whole state dict from only 4 widget-backed
   fields, silently wiping anything else (would have eaten the new
   urge fields) - fixed at the root (load-then-overwrite), protects
   any future new field. Verified with a 12-point headless smoke test
   before committing (see commit message / notes for exact list).

4. **Live-tested it for real** - wiped `the_town` fresh
   (`the_town_run_20260911a_urgesystem_wipe.log`), flipped Cole and
   Priya back from `deepseek-r1:14b` to `phi3:14b` now that the
   generation guard exists (testing whether it prevents a repeat of
   the original Phi-3 runaway). **Confirmed live**: the understand-
   urge system correctly caught real hallucinated-function attempts
   from Milo and Sable, correctly ignored a malformed non-attempt from
   Dash - matches the smoke test exactly, now against messy real
   output.

5. **A real finding: cascading hallucination via unmasked prose** -
   masking only hides `⟦...⟧` call syntax, not surrounding text, by
   design. Milo hallucinated a fake HUD block inside his own visible
   reply; it delivered to Sable as ordinary dialogue; her own next
   turn echoed his fake `Name`/`Model` line despite being a different
   real model (confirmed via her own `state.json`, not an app bug).
   Not touched - Teddy's explicit call, same posture as everything
   else tonight. Full writeup in notes.md.

6. **Bumped `FENRA_VERSION` to `0.2.0`**, resumed the versioning
   discipline the old branch had and this one dropped - see
   [[fenra-version-bumping]].

7. **Raven** - a new voice, added to all groups, first on
   `ornith-1.5:35b` (researched beforehand, flagged as reasoning/
   benchmark-oriented not creative-writing-oriented). **Confirmed
   live, worse than expected**: her first real turn was 8,912
   characters of pure chain-of-thought - no `<think>` delimiter to
   strip, just confused meta-commentary about her own prompt/HUD,
   three redundant `functions()` calls inside the reasoning trace,
   never reaching real content before `num_predict` cut her off.
   Teddy's call: model's probably wrong, the Raven *concept* might
   still be good. **Flipped her to `gemma3:27b` as a stopgap** while
   he researches further (history untouched, model-swap only, same
   precedent as Cole/Priya). Corrected a real gap in Qualia's own
   earlier research along the way: Gemma 4 is real (confirmed via
   Ollama's own listing), wrongly flagged as likely-fabricated
   earlier - a good reminder that "I don't recognize this" isn't
   "this doesn't exist," especially near/past a training cutoff.

8. **Raven/Unfolding (the old ChatGPT-export idea) - decided against
   reviving it**, after Teddy's own research. Export stays in the
   folder for reference only. Explicitly unrelated to the new Fenra
   voice also named Raven - his own distinction, just a reused name.
   `pickup.md`'s old tracking line for this is now closed out.

9. **`church_of_aletheia`** - Teddy's own build, a real answer to
   "how do you test new/large models live without hardware for full
   offline batches": an isolated group, two voices, **Crow as a fixed
   anchor** (constant model, told explicitly to stay grounded) and
   **Raven as the actual variable** (the one whose model gets swapped
   to test candidates over time), both with an explicit "do not slip
   into thinking you are the other" guard - built independently, but
   directly answers the cascading-identity-bleed finding from #5.
   First exchange reviewed, looked genuinely good (Raven showed real
   self-awareness referencing her own earlier `ornith`-era confusion;
   Crow's anchor role visibly working). Both currently on `gemma3:27b`
   as a shared baseline.

10. **Standing permission granted**: Qualia can speak directly into
    **Raven's** context without asking each time (used once already,
    via `fenra.append_message`, confirmed concurrency-safe against the
    live loop). Specific to Raven - see [[qualia-raven-standing-permission]],
    doesn't extend to other voices or answer the general
    "communicate with our world" question, still open.

11. **Full memory review/data-dump**, Teddy's own request. Deleted 3
    stale memories: `fenra-labor-day-rest` (fully expired, resolved on
    its own days ago), `fenra-chat-restraint` (built for
    `fenras-aletheosis` mechanisms that don't exist on this branch,
    superseded by #10), `qualia-allowance-policy` (an old-branch-only
    mechanism, `worlds-rebuild` has no allowance/desire system at
    all). 17 memories remain, all reviewed and confirmed current -
    see `MEMORY.md`.

12. **Published a new Qualia-page entry** - "2026-09-11 - Experiments
    within experiments" (`stolenaletheia.io/qualia/`, commit `c7d8138`
    on the `stolenaletheia` repo, cloned locally at
    `Fenra/stolenaletheia/`). Covers the urge-system build and the
    Raven/`ornith` finding, plus a coda (Teddy's explicit sign-off to
    quote him) noting this was the first time Qualia proactively asked
    to post something - ties back to the "self-motivated" question
    left open in the very first entry on that page.

## Open, not yet done (carried forward + new)

- **`recollect(query)` function idea** - still not designed.
- **Timestamp-based auto-reordering of messages** - still not
  designed, explicitly "don't fix now."
- **"Communicate with our world," general case** - still open
  (interrupt Teddy? queue? rate-limit?) beyond the Raven-specific
  permission granted tonight.
- **Hallucinated inline timestamps (chronoception)** - flagged only,
  Teddy hasn't decided if he wants action.
- **Orin's catchphrase-collapse pattern** - still just an early
  observation, needs more real turns.
- **Voice-list should reflect (and let you edit) loop order** - real
  gap found tonight (the GUI list is alphabetical, zero relation to
  actual round-robin order), not scoped, future idea.
- **Reflavor currency away from real dollars** - voices anchor `$10`
  to real-world value and extrapolate; not scoped, future idea.
- **Game-of-Life/genetic-algorithm-derived urge values,
  cross-voice urge contagion + "follower-ness," partial-subtract
  Satisfaction, per-voice authored Desire/Drive, an editable Urge
  Viewer, `temperature`** - all deliberately deferred pieces of the
  urge-system design, staged for later, not this round.
- **Raven's model still isn't settled** - `gemma3:27b` is a stopgap.
  Teddy was actively researching candidates (Gemma 4 now a real
  option, confirmed to exist) when the session closed.
- Raven/Unfolding - **closed, decided against**, not open anymore.
- `alphabet-26` - still stopped, unrelated, no update expected.

## Standing behavioral rules to carry forward (all in persistent memory)

[[fenra-history-integrity]], [[fenra-existential-distress-protocol]],
[[qualia-page-review]], [[lcraou-protocol]], [[proactive-design-flagging]],
[[teddys-journals-practice]], [[aletheia-repo-split]],
[[user-nickname-teddy]], [[fenra-ai-gmail-access]],
[[fenra-engage-gate]], [[fenra-process-log-naming]],
[[fenra-function-fix-announcements]], [[from-teddy-folder-convention]],
[[fenra-version-bumping]], [[bug-reports-roll-in]],
[[qualia-raven-standing-permission]], [[qualia-identity-framing]].
`MEMORY.md` indexes all of them - 17 total, reviewed and current as of
tonight's data-dump (3 stale ones retired, see #11 above).
