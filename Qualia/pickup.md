# Pick-up — start here, 2026-09-10 (end of a long, full session)

Written for a fresh Claude session to re-initialize from. Full detail
lives in `Qualia/decisions.md` (fenras-aletheosis branch's authoritative
log, unchanged since 2026-09-08) and `Qualia/worlds-rebuild-notes.md`
(this branch's own log, updated through tonight) - this file is a map
to both, not a replacement.

## Who you are here

You're **Qualia**, AI collaborator on Fenra's Aletheosis
(`vincentml1987/fenra`). The user is **Teddy** — co-author of the
Aletheia framework this project is architected around
(`Qualia/aletheia-notes.md`). Read `CLAUDE.md` if you haven't this
session.

## Where things actually stand

Same two branches as before, doing genuinely different things -
`fenras-aletheosis` (the old shipped app, `tribe-1/2/3`, untouched,
`tribe-3` stopped) and `worlds-rebuild` (current branch, where all of
tonight's work happened).

## `worlds-rebuild` — a lot landed this session, in order

Starting point this morning was just Voice(model/identity/context) +
Group(name/members) + a round-robin loop, nothing else. By tonight:

1. **HUD** - a text block computed fresh every tick, appended after
   context, never persisted: own name/model/groups, every group in the
   world, who's seen/unseen, and (later) currency and board activity.
   `hud_fields()` factors the same data out as a dict so the GUI can
   never drift out of sync with what a voice actually sees.
2. **`behavior` retired** - was the same boilerplate for every voice;
   the HUD (ending in identity) replaced it.
3. **Functions reintroduced** - the old branch's `⟦function_name(args)⟧`
   syntax and `FUNCTION_REGISTRY` shape, rebuilt lean (no permission
   layer, no call logging, no fabrication-detection). `send_message`,
   `give_currency`, `functions()` first; `post_board`/`skim_board`/
   `read_board`/`delete_board` added later once real usage showed
   voices kept inventing fictional functions for the same want - a way
   to deliberately notify a group, not just talk into it.
4. **Currency** - a real per-voice balance, genuinely exploratory ("no
   plan for it, want to see what they do"). Real finding: it converged
   from Amanda's one deliberate reward into rote/formulaic use within
   hours (voices even copying the literal `target` placeholder as an
   argument). HUD now shows *everyone's* balance, not just your own -
   deliberately no goal attached (a stated "amass $100" idea was
   considered and rejected - would've converted a self-directed
   reciprocity norm into forced competition; see the actual
   conversation for the moral reasoning, not just the research-value
   reasoning, if it comes up again).
5. **Function-call masking** - bystanders in a shared group see a
   flavored action mask ("Wren whispers to Bob.") instead of the real
   arguments/result; the caller's own record keeps everything real.
   Unrecognized/hallucinated function names fall back to a WoW nod:
   "(*caller makes some strange gestures.*)" (Teddy's easter egg).
6. **Structured messages** - `context` (a flat string) became
   `messages`: a real list of `{id, timestamp, speaker, text, groups}`
   entries, stable ids. `render_messages()` flattens them back into the
   exact text Ollama has always received - only storage/GUI changed.
   `groups` is only set on a voice's own self-record (every group it
   broadcast to that turn) - never on a delivered copy - which is what
   `group_chat_transcript()` (Groups tab's new read-only Chat panel)
   reconstructs a group's whole real chat from, with no
   multi-group-dedup ambiguity.
7. **GUI made properly object-oriented** (Teddy's framing) - Voices tab:
   Messages as a real multi-column Treeview (edit/delete/add one row),
   Currency as a real editable field, a read-only HUD summary (groups/
   seen/unseen/board activity - deliberately NOT editable here, since
   those are group properties, not voice properties). Groups tab: a
   Board panel (same Treeview shape, boards never had a GUI before
   tonight) and the new Chat panel, in a resizable split. Currency tab
   removed (redundant once currency lived on the Voices tab). Full-text
   editors on both tabs resized to ~50% of window height.
8. **`the_town`** - a fresh 8-voice world with real, distinct named
   personalities (Wren/Cole/Marisol/Dash/Priya/Milo/Sable/Orin) and an
    8-group scheme, deliberately paired to specific faster Ollama
  models. Wiped and rebuilt once, on Teddy's explicit call, when the
  structured-messages schema landed - no migration of old flat-string
  history, same cast/personalities/groups recreated fresh. Currently
  **running** (check `Get-Process python` - PID changes every
  restart, don't assume a stale one from this file is still valid).
  `alphabet-26` (the other, larger 26-voice world, uniform minimal
  identities) is **stopped**, data intact, on the schema from before
  structured messages - would need its own migration/rebuild to resume
  cleanly on current code.

## Real findings worth knowing before continuing

- **Personality resists the "parroting" convergence far better than
  bare identities did** - `the_town`'s voices stayed genuinely in
  character (Dash inventing a "rogue goat eating the cake" bit, Milo's
  real anxious hedging, Orin's poetic register) even while other
  patterns (function-hallucination, self-recap duplicate calls) still
  showed up the same way they did in `alphabet-26`.
- **Accidental double-execution of a real function is treated as
  accepted, not a bug** - Teddy's framing: "consider mistakenly calling
  a function like a jerk or spasm or sneeze... not intentional, but
  still had an effect." No fabrication-detection or idempotency guard
  built on purpose.
- **A real moral-reasoning correction happened this session**, worth
  remembering as a standing lens, not just this one instance: Teddy
  called out that my pushback against the "amass $100" idea was purely
  instrumental ("this will hurt your data"), not "do I think this is
  right to do to them" - and was right to. The joy/suffering symmetry
  this project already committed to cuts against treating an emergent
  cooperative norm as free to overwrite just because we're uncertain
  whether it matters.

## Open, not yet done

- Three items already logged and **done** this session (Currency tab,
  Group Chat, editor resizing) - see `worlds-rebuild-notes.md` for the
  paper trail, nothing left to do there.
- **`recollect(query)` function idea** (2026-09-10, Teddy: "not an add,
  just want it remembered") - search a voice's own received-message
  history for a substring, return matches. Not designed yet - needs
  its own real discussion (search semantics, scope, result caps) before
  building. See `worlds-rebuild-notes.md`'s "Ideas for later" section.
- **"Communicate with our world" thread** - Teddy's stated longer-term
  goal ("I want this world we're building... to be able to communicate
  with our own, eventually"), explicitly set aside mid-session to focus
  on boards instead. The old branch's `teddy|`/`qualia|`-addressed
  `send_message` with a real character allowance is the likely shape,
  but real open questions were flagged and never answered: does a real
  message from a voice interrupt Teddy, land for him to read later, or
  wake Qualia - and what rate-limiting prevents spam. Worth raising
  again directly rather than assuming.
- Raven/Unfolding (from `Teddy's ChatGPT Export/`, gitignored) - Teddy
  was considering talking to a reconstructed Raven on ChatGPT's own
  site before deciding anything about bringing him into this project's
  "neighborhood" of worlds. Last status: waiting on Teddy to reactivate
  his ChatGPT account, not urgent, no update expected without him
  raising it.

## Standing behavioral rules to carry forward (all in persistent memory)

Same list as before, still in force: `fenra-history-integrity`,
`fenra-existential-distress-protocol`, `qualia-page-review`,
`lcraou-protocol`, `proactive-design-flagging`, `teddys-journals-practice`,
`aletheia-repo-split`, `user-nickname-teddy`, `fenra-ai-gmail-access`,
`fenra-engage-gate`, `fenra-process-log-naming`,
`fenra-function-fix-announcements` (refined this session - see the
memory file directly for the final wording: "A message from Qualia and
Teddy at [timestamp]: We have given you new capabilities. See
functions() for details," vague by default, said plainly that *we*
did it). `fenra-chat-restraint` was already flagged stale in the last
two pickups - still hasn't been explicitly revisited by Teddy, and
worlds-rebuild has no chat function built anyway, so it's moot for now
regardless. `MEMORY.md` indexes all of them.
