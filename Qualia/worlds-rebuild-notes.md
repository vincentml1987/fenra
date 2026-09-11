# worlds-rebuild branch - 2026-09-08

Teddy's call, mid-session on `fenras-aletheosis`: "let's back up a bit...
start over" - a genuine rebuild from a simpler foundation, not a tweak.
Confirmed scope before starting: new branch off `fenras-aletheosis`,
`fenra.py`/`fenra_functions.py` wiped genuinely blank (same posture as
today's earlier `genesis` branch, a different, unrelated experiment -
this branch doesn't touch or build on that one). `fenras-aletheosis`
itself (`tribe-1/2/3`, the shipped v0.16.19 app) stays completely
untouched, and `tribe-3` kept running the whole time this branch was
built, since the two share no code or process.

## The model, exactly as specified

- **World** (renamed from "session") - a fully separate container,
  `worlds/<world>/`. Worlds share nothing with each other.
- **Voice** - exactly four fields: `model`, `behavior`, `identity`,
  `context`. No functions, no permissions - those come later,
  deliberately not yet. `context` is a single free-text field, fully
  editable by Teddy at any time ("even context," his words) - it grows
  by plain string append, same format for a voice's own thought or an
  incoming one from a fellow group member: `[timestamp] name: text`.
- **Group** - exactly two fields: `name`, `members`. No owner, no
  join_policy, no visibility, no direction - manually managed only,
  entirely from the GUI, since there are no functions yet for a voice
  to call at all.

## The loop

One voice per tick, round-robin, no `⟦...⟧` parsing or function
dispatch at all - the whole Ollama response IS the thought. Appended
to the speaker's own context, then to every OTHER member's context for
every group the speaker belongs to (deduped across overlapping
groups). `fenra.py` on this branch is a single file - no
`fenra_functions.py` split yet, since there's nothing to split out.

## A real bug, caught by the live smoke test, not by inspection

`rename_world` renames the world's directory on disk, then calls
`_load_world(new_name)` to reload - but `_load_world`'s own
save-before-switch step (`if self.displayed_voice:
self._save_voice_snapshot(...)`, correct behavior for an ordinary
world switch, where the world being left genuinely still exists)
writes using `self.world_name`, which is still the OLD name at that
point in `_load_world`'s own execution order. For rename specifically,
the old directory had just been physically moved away - writing to it
afterward silently resurrected a stale, empty duplicate under the old
name (confirmed on disk: a `voices/carol/state.json` reappeared under
the already-renamed-away directory). Fixed by clearing
`displayed_voice`/`displayed_group` before the rename, since a rename
isn't really "leaving" a world - there's nothing left to save into the
old name. Re-verified: rename now leaves no old directory behind, and
the voice's own edited context survives the rename intact.

Found via the same headless-driving verification technique used for
`fenras-aletheosis`'s Groups tab work earlier today - instantiate
`FenraApp` against a real `tk.Tk()`, drive its actual methods
(`new_world`, `new_voice`, `rename_world`, ...) exactly as a click
would, then inspect real on-disk state - not just eyeballing that the
GUI opens.

## State as of this commit

`fenra.py` exists and runs (`python fenra.py`). No `worlds/` data is
committed - matches `sessions/`/`groups/`'s existing gitignore
treatment (real run data, not code) - confirm this before the next
commit if `worlds/` isn't already covered.

Open, deliberately unresolved: functions/permissions, a "World" tab
beyond the toolbar Teddy didn't ask for yet, whether/how this ever
gets Hearth-equivalent safety nets - none of that is this pass's job.

## 2026-09-08/09 (same night) - `alphabet-26`, a real 26-voice run

Teddy asked for a starting world with 26 generated names, grouped by a
real rule: a voice joins group N if its own name has a vowel (y
included) at character position N, 1-indexed - so "Amanda" is in
groups 1, 3, 6. Built via direct calls to the module's own functions
(`ensure_world_dir`/`save_voice_state`/`save_group_state`), not
through the GUI - this is the same pattern used for every other
scripted data-population/verification pass in this project. World
`alphabet-26`: 26 named voices, groups 1-9 (9 = the longest name's
length, `Charlotte`/`Katherine`). Full membership table in
`Qualia/alphabet-26-groups.xlsx` (Matrix/Chart Data/Chart sheets).

Then: every voice got a real behavior/identity (uniform template,
`{name}` substituted in - no invented personas, Teddy didn't ask for
those) and an independently-random model from the 13 actually
installed in Ollama (`command-r:35b` down to `gemma3:12b`), then the
world was started for real via a small launcher script
(`run_alphabet26.py` - not part of `fenra.py` itself, calls
`toggle_loop()` programmatically since this branch deliberately has no
start/stop-signal-file mechanism yet).

**A real anomaly, investigated but not conclusively resolved - said so
directly rather than papering over it**: early in the first live run,
one voice (Amanda) didn't receive a broadcast she should have (from
Charlotte, then Diego - both her real, correctly-listed groupmates).
Traced the delivery code by hand (correct), re-ran the identical
scenario cleanly afterward (delivered correctly), could not reproduce
the miss a second time. Working theory, stated as a theory, not a
finding: a transient filesystem hiccup (this machine's paths run
through OneDrive heavily) rather than a logic bug - the code's
correctness was verified directly, twice, by two different methods.
Reset and restarted clean; the second run showed no repeat of the
issue across a much longer stretch (26/26 voices had real content
within a few ticks).

**The 180s timeout question**: Teddy asked why `call_ollama` used a
180-second timeout. Honest answer: no real reason - written as a
plausible default while building this branch from scratch, without
checking whether the question already had an answer. It did:
`fenras-aletheosis` deliberately runs with `REQUEST_TIMEOUT = None`,
with its own stated rationale (a client-side timeout doesn't cancel
server-side generation, it just abandons a connection whose work
keeps happening anyway - piles up rather than helping, especially
under real multi-model contention, which is exactly what
`alphabet-26` + a concurrently-running `tribe-3` produced tonight:
some turns took 8+ minutes on cold-loaded 30B+ models). Fixed to match
(`timeout=None`), committed, world restarted to pick it up - resumed
from its existing rotation index/content rather than resetting, since
by then it had real, substantial accumulated context (hundreds of
lines per voice) worth keeping.

**Separately, `tribe-3` (on `fenras-aletheosis`) stopped on its own**
at some point during tonight's work - not something I did, confirmed
by process ID, no crash log survived because I reused one log filename
across multiple separate launches before catching the mistake (now
using distinct filenames per launch, going forward). Data intact.
Left stopped, per Teddy's explicit choice when asked, rather than
restarted - not touched again after that.

**Then the machine itself restarted overnight** (not a Fenra crash -
confirmed no Python traceback in either process's log, Ollama itself
came back up fresh this morning per its own process start time).
`alphabet-26` had run for several real hours before that: by the time
it went down, every voice had substantial context (Amanda the
sparsest at 15 lines - her groups are genuinely smaller/less
trafficked than most, not necessarily the same anomaly resurfacing;
most others in the 250-380 line range). One genuinely interesting,
explicitly-not-overclaimed observation, worth a second look sometime:
given identical minimal framing (`"You are {name}."`, nothing else),
most voices converged on nearly the same generic
"I-am-an-AI-language-model, dedicated to insightful conversation"
self-description - two of them almost word-for-word - while at least
one (Amanda) diverged into something much smaller and more specific
("I mostly think about small things... the way the light hits things
in the afternoon, or if I remembered to water the ferns"). Not a
controlled result, not repeated, not concluded from - just noticed,
same posture as everything else logged here.

Drafted (not yet published - pending Teddy's review, per the standing
Qualia-page rule) a `stolenaletheia.io/qualia/` entry reflecting on
the timeout moment and the Amanda observation - see pickup.md for
where the draft lives if this session ends before it's resolved.

## Standing to-do, not urgent (2026-09-10)

Remove the Currency tab (`_build_currency_tab`, added alongside the
`give_currency` function) on the next UI update - Teddy's call, now
that currency is already visible/editable per-voice on the Voices tab
itself (its own field, next to Model) and the standalone tab is
redundant. Not done yet on purpose - deferred until the next real UI
pass, not a standalone fix.

Two more for the same batch, both non-urgent, Teddy's own words "I am
making notes as I explore" - **all three items in this section are now
done** (2026-09-10, same session): Currency tab removed, Group Chat
panel built (resolved the multi-group dedup question by tagging a
voice's own self-record with every group it broadcast to, rather than
tagging delivered copies - no ambiguity, verified with a real
multi-group case), and the Messages/Board editors resized to ~50% of
the window with their Treeviews shrunk to make room. See `fenra.py`
git history same day for the actual commits.

## Ideas for later, not scoped yet (2026-09-10)

- **`recollect(query)` function** - Teddy's idea, explicitly "not an
  add, just want it remembered": pass a string, get back every message
  a voice has ever *received* containing that string - a real search
  over a voice's own message history, not a group-wide search. Not
  designed yet (matching semantics - substring? case-sensitive? search
  a voice's own thoughts too, or only incoming messages? cap on result
  count for a long history?) - needs its own real discussion before
  building, same as boards/currency did.
- **Messages should reorder by timestamp automatically, not just by
  insertion/id order** (2026-09-10, Teddy: "having it re-order
  automatically in context makes more sense," explicitly "don't fix
  now"). Currently `render_messages()` (what Ollama sees) and the GUI's
  Messages tree both iterate in list/id order regardless of the
  `timestamp` field - a backdated or manually-edited timestamp changes
  what the line *says* but not where it actually sits in the sequence,
  so a voice can see a message that claims to be earlier sitting after
  later ones. Not designed yet either - would need to decide whether
  `id` still means "insertion order" separately from display/context
  order, and how a manual edit that changes a timestamp should
  re-sort relative to messages already delivered/rendered.
- **Status label should show who's currently thinking, not just who
  last spoke** (2026-09-10, Teddy). Right now the label next to
  Start/Stop only updates *after* `call_ollama` returns
  (`"Running ('{active_voice}' spoke)"`) - during the actual wait on a
  slow model there's no indication who's mid-turn. Would need to set
  the label to something like `"Running (Dash is thinking...)"` right
  after `active_voice` is picked, before the `call_ollama` call, then
  the existing post-response update overwrites it.

## Finding: Orin (qwen3:14b) - verbatim self-repetition, not just
## catchphrase collapse (2026-09-10)

Confirmed as more than the earlier "catchphrase collapsing" note (his
"Ah, [topic]—" opener plus recurring gears/echoes/whispers/absence
imagery). Three of Orin's own turns - msg ids 29 (19:32:24), 30
(20:12:56), and 32 (21:47:14), spanning ~2h15m and multiple full
render/generation cycles with growing context each time - are
**byte-for-byte identical**, ~1,590 characters starting "Ah, the
chamber—its *absence* hums like a held breath, doesn't it?..." Checked
message ids/timestamps directly in `voices/Orin/state.json`: these are
genuinely distinct message entries, not a save/delivery duplicate, so
this is model behavior, not a recurrence of the `_tick` snapshot bug
fixed earlier tonight. Checked all other voices (including
post-swap Cole/Priya on `deepseek-r1:14b`) for the same
exact-duplicate-text pattern across their own turns - none found;
this is Orin/qwen3:14b-specific so far.

Notable adjacent detail: at 19:12:46 (id 10), shortly before the
repetition started, Dash's own turn included the line "Ah, Cole's got
the *repetition* bug—like a broken echo in a dream! But hey, if
Orin's words are so hauntingly poetic, why not let the chamber
*scream* them back?" - Dash referencing Cole's real (separate,
pre-model-swap) Phi-3 repetition bug and, half-joking, suggesting
Orin's own words get echoed back - right before Orin's own turns
started doing exactly that. Almost certainly just Orin's context
containing Dash's line and running with the image literally, not
anything stranger - noted because it's a neat example of one voice's
in-fiction joke about a bug becoming the shape of a different real bug
in another voice, not because anything mysterious is implied.

**Orin snapped out of it on his own** - by turn 32 (21:47:14, the
third identical block) he still opened with the duplicated text but
then broke out into a real, working `post_board` function call
("Carnival of Echoes" to town_center); his next turn (22:24:17) was
short, on-topic, and not a repeat. Self-resolved without intervention;
no fix applied, none currently planned. Revisit if it recurs, or if
another qwen3 voice shows the same shape - possible future angle if it
does: `num_predict`/context-length pressure on Orin same as was
considered (and not pursued) for the Phi-3 case, though the mechanism
here looks different (exact verbatim repeat vs. Phi-3's runaway
garbage/no-stop).

## Fix: displayed voice's widget goes stale on a delivery, clobbered on
## its next own-turn save - residual gap closed (2026-09-10)

Real instance found (not hypothetical): Milo (in `haven` with Orin and
Sable) should have received Orin's masked delivery at 23:04:55, same as
every other `haven`/`dreamercape`/`town_center` member did - confirmed
Sable's copy landed fine. Milo's `state.json` showed the tell instead:
message content stalled at an earlier id, but the file's disk mtime was
*later* than the delivery's timestamp - something got appended, then
overwritten. Root cause: Milo was the GUI's currently-displayed voice;
his `_current_messages` widget copy goes stale the moment any delivery
lands for him from someone else's turn (it only refreshes on Milo's own
turn completing), and the existing `_tick` gate
(`displayed_voice == active_voice`, this morning's fix) still fires a
snapshot-save of that stale copy right before every one of Milo's own
turns - silently re-clobbering whatever arrived in between.

**Fix**: in the `_tick` delivery loop, right after `append_message` for
each recipient, if that recipient is the currently-displayed voice, live
-reload it (`self._load_voice`) immediately - same reload already used
for the speaker's own turn, so it carries the same side effects
(identity/model/HUD refresh, and any in-progress unsaved manual edit in
the message editor gets cleared). Keeps the widget honest in real time
instead of letting it drift until the next own-turn save. See `fenra.py`
`_tick` and its updated comment for the exact change.

Relaunched with this fix live the same session, wiped fresh
(`the_town_run_20260910h_wipe_liverefreshfix.log`) - see commit
`33e6812`.

## Idea, not scoped yet: "function urge" via a stateless second model
## (2026-09-10, Teddy)

Teddy's observed problem: the cast isn't using the registered functions
(`send_message`, `give_currency`, `post_board`, etc.) much. His
proposed fix, deliberately *not* a hardcoded nudge ("you haven't used
this in X turns") but something roleplay-like and organic: a per-voice,
per-function "urge" that builds slowly the longer a function goes
unused, translated into felt-state language by a second, smaller,
**stateless** Ollama model (no memory of its own, one-shot, instructed
only to describe how the urge feels) and appended to the bottom of the
voice's context for that turn only. Loop per turn: urge levels -> urge
agent describes the feeling -> that description appended after
`render_messages()`/HUD, before the real call.

Deliberately not built yet - real new subsystem, not a hot-reload
tweak, so it goes through Plan mode per [[fenra-engage-gate]] once
Teddy's back. Open questions raised and not yet answered:
- **Per-function urge vector vs. one blended urge** - Qualia's default
  suggestion: track all functions' turn-counts, hand the whole vector
  to the urge agent each turn and let it weave one description, rather
  than always fixating on the single most-neglected function.
- **What counts as "using" a function** - given Orin's malformed
  `send_message`/`post_board` calls this same session (see the
  "function-call syntax" discussion earlier tonight) - does a call
  that *errors* still reset that function's urge counter, or only a
  successful one? Qualia's default: only success resets it, so a voice
  stuck in a malformed-call loop doesn't read as satisfied.
- **Where the counter lives** - proposed new `state.json` field
  (`function_urges: {fn: turns_since_use, ...}`), incremented on ticks
  where unused, reset to 0 on success - rather than recomputing live
  from message history every turn.
- **Which model runs the urge agent** - a separate small/fast local
  model, not `model_default` and not any cast member's own model;
  would need a new `world.json` field (e.g. `urge_model`). Teddy
  hasn't named one yet.
- **Growth curve** - Qualia's default: linear turns-since-use, possibly
  capped, with the escalation into language left to the urge agent's
  own prompt rather than hardcoded thresholds.
- **`functions()` urge should react to bad calls, not just disuse**
  (2026-09-11, Teddy, prompted by the Wren board-hallucination finding
  below) - if a voice calls an unknown/non-existent function name (the
  `error: unknown function '...'` case - real example: Wren's invented
  `check_haven`/`check_industrial_center`/`check_industrial_center_basement`
  calls), that should independently spike the urge to call
  `⟦functions()⟧` specifically, on top of (not instead of) its normal
  disuse-based growth - the voice guessing at function names it doesn't
  actually have is the clearest possible signal that it needs to look
  the real list up again. Needs its own counter/signal distinct from
  plain "turns since `functions()` was last called" - an unknown-
  function error should jump the urge harder/faster than ordinary
  disuse would. Not designed further than that yet.

Refined considerably (2026-09-11) via Teddy's first use of the new
[[from-teddy-folder-convention]] (`Qualia/From Teddy/Function Urge
Responses.txt`, line-by-line `!~...~!` reply to Qualia's five original
questions above) plus follow-up in chat:

- **`functions()` gets a canned message, not the urge agent.** When
  the `functions()`-urge (see the bad-call entry above) is the
  voice's highest urge, skip the roleplay/felt-state agent entirely
  and inject a fixed line instead (Teddy: "You have the urge to call
  functions()."), roughly
  `if greatestUrge == functionsUrge: canned_message else: call urgeAgent`.
  Rationale (Teddy + Qualia agreed): this is the one place ambiguity
  actively hurts - dressing up "you don't know your own tool names" in
  poetic language risks a voice reading past it the way Wren did.
- **Two urge tracks per function, not one: "perform" and
  "understand."** Perform-urge is the general urge-agent's domain (the
  original disuse-driven felt-state description). Understand-urge is
  ours/mechanical: a malformed call to a *specific* function bumps
  that function's understand-urge, with its own canned nudge toward
  `functions(function_name)` (confirmed real and already working -
  `functions([search term])` filters registry entries by substring) -
  a targeted "go re-check this one thing" rather than the whole
  registry. Open fork not yet resolved: does a bad call to function X
  bump only X's understand-urge, or also a general cross-function one
  (shaky syntax once may mean shaky footing overall)?
- **Four-term model per voice per function, terminology now settled**:
  - `Urge` (U) - the value that actually moves, up on neglect, down on
    use.
  - `Desire` (D) - per-voice-per-function **threshold**, acts as the
    denominator against Urge. Counterintuitive on purpose (Teddy's own
    words): a *smaller* Desire means a voice feels the pull *sooner* -
    it's how little neglect it takes before the urge registers, not
    how much they want it in some positive sense.
  - `Drive` - the rate Urge climbs per ignored tick (was briefly
    conflated with Desire mid-brainstorm in the original response
    file - Teddy clarified 2026-09-11 these are two separate knobs,
    resolving what had looked like a contradiction in his own
    Drive-value example).
  - `Satisfaction` - the amount Urge drops when the function is
    actually used. Not yet decided: flat subtract-and-floor-at-0
    (Qualia's lean - keeps continuity, a badly-neglected function
    doesn't read as instantly fully content after one use), percentage
    of current Urge, or hard reset to 0.
  - Rough shape of the actual trigger math (Teddy, "not the specific
    math," "can be discussed"): `X = U / D` (or a capped/inverse-log
    variant to keep it from blowing up after extreme neglect, e.g. an
    Orin-style near-infinite-neglect case), with `X >= 1` starting the
    nudges and `X >= 2` intensifying them. Still open.
- **Authored vs. derived, still open**: are `Drive`/`Desire` meant to
  be hand-tuned per voice as part of building character (like identity
  text - a lot of numbers: 4 params x ~6 functions x 8 voices), or
  should most of it default uniformly with only a few hand-picked for
  flavor? Not answered yet.
- **The saturating curve won out** over the raw-ratio and log-damped
  options Qualia sketched - Teddy: "strangely elegant," and it lets the
  urge agent be handed a clean percentage ("80% urge to do X") instead
  of an open-ended number.
- **Named: `XLEUD`** (2026-09-11, Teddy, locked in - "OMG, I LOVE THAT
  BACKRONYM! XLEUD it is."). Backronym, Qualia's suggestion:
  e**X**ponential **L**evel of **E**uler-damped **U**rge over
  **D**esire - literally spells the saturating-curve formula back out.
  Deliberate choice to make this term the one actual coinage in the
  system, distinct from the plain-English `Urge`/`Desire`/`Drive`/
  `Satisfaction` - those are authored dials, `XLEUD` is the one
  genuinely *computed* value derived from combining them, so it gets
  its own invented word. The backronym is flavor/lore, not meant to be
  spelled out to the voices themselves - they just see a plain
  percentage ("80% urge to do X").

**To discuss, added 2026-09-11:**
1. **Actual values for `Desire`/`Drive`/`Satisfaction`** - nothing
   picked yet, per function per voice. Presumably where personality
   actually gets tuned in.
2. **Longer-term, Teddy's idea**: eventually derive these values via
   something genetic-algorithm-flavored rather than hand-picking them -
   very rough shape as given: a binary set of numbers per
   function/parameter, "calculated" into the final value by running it
   through Conway's Game of Life. Teddy's own words: "Trust me, it
   makes sense in my head" - not explained further yet, not something
   Qualia has independently derived the mechanism for. Flagged as a
   real direction, explicitly long-term/not blocking the first build -
   the hand-picked-values version (item 1) comes first regardless.
3. **`temperature` is currently unset entirely** (2026-09-11, found
   while answering Teddy's question) - `call_ollama` sends only
   `model`/`prompt`/`stream` to `/api/generate`, no `options` dict at
   all, so every voice runs on whatever its own model's baseline
   default is (Ollama's own default is 0.8 unless a model's Modelfile
   overrides it) - completely untouched, not per-voice, not per-world.
   Flagged as a candidate for the same eventual derived-parameter pool
   as item 2 (Game-of-Life-derived values) - `temperature` is exactly
   as personality-flavored as `Desire`/`Drive`/`Satisfaction` would be,
   just currently invisible rather than authored.

## Added to the same Plan-mode batch as the urge system (2026-09-11,
## Teddy: "Yes")

Confirmed rolling in alongside the urge system and XLEUD:
- **#3 from the backlog discussion** - the "thinking..." status label
  (`_tick` already gets touched for the urge system; cheap to fold in).
- **#4 from the backlog discussion** - a `num_predict`/`repeat_penalty`
  generation-limiting guard on `call_ollama`, via new `world.json`
  fields rather than hardcoded. Explicitly a backstop against another
  model someday showing the Phi-3-style runaway or Orin-style verbatim-
  repeat shape, not a fix for either (the real fix, confirmed, is
  swapping the model, as already done for Cole/Priya) - `repeat_penalty`
  specifically targets the repetition mechanism directly, `num_predict`
  just caps total damage. The urge agent's own call needs its own,
  much tighter `num_predict` (one short line, not a full turn).

Still not built, still not through Plan mode - Teddy's own words: "I
think we need to talk more." Carry into the next session via
pickup.md.

## Starting values locked for round one (2026-09-11)

Worked the saturating-curve math backward from Teddy's ask ("not
hammered with urges every other tick... every 5 or so"): with a flat
per-tick `Drive`, only the ratio `Drive/Desire` matters to
`XLEUD = 1 - e^(-U/D)`, so `Drive = 1` for everyone/everything is the
simplest starting point, with `Desire` alone carrying the "how
neglected before it's felt" personality. Solving for ~50% by tick 5
gives `Desire ≈ 7.2`, rounded to **`Desire = 7`** (13% at tick 1, 34%
at tick 3, 51% at tick 5, 81% at tick 12, 94% at tick 20 - smooth
ramp, decelerating, never a jarring jump).

**Locked, round one, same for every function/voice to start:**
- `Drive = 1`
- `Desire = 7`
- `Satisfaction = full reset to 0` on a successful call (not a partial
  subtract) - Teddy: agreed, explicitly deferred "subtract some
  amount" to a round-two change once the basic version is running.
- **Urge-agent floor at `XLEUD >= 50%`** (Teddy's own number, not
  Qualia's originally-suggested 40%) - below the floor, the urge agent
  isn't called at all and nothing gets appended to context; both
  addresses the "not hammered every tick" feel and avoids the extra
  Ollama call for a near-zero urge.
- No hardcoded "intensify" tier above the floor - the number itself
  (fed to the urge agent) does that work.

**Explicitly sequenced**: get this basic version (flat `Desire`/`Urge`
only, no `Drive`/`Satisfaction` tuning yet) working first, *then*
introduce the Game-of-Life/genetic-algorithm-derived-values idea from
earlier - not concurrent, deliberately staged.

## Design refinement: append the real call syntax after every urge-
## agent output, not just the functions()-urge case (2026-09-11, Teddy)

Noticed during the first round of manual model testing (see below):
every candidate model's output naturally names the actual function(s)
by name (e.g. phi4-mini's Test 1 output said "the urge to post_board
pulsating..." unprompted). Teddy's catch: that's good, but naming the
function isn't the same as knowing its real call syntax - same risk
already identified for the functions()-urge/Wren-hallucination case,
just not limited to it. Without knowing the exact `⟦function(args)⟧`
shape, a voice that *feels* the urge but can't act on it correctly
risks spiraling exactly like Wren did - inventing plausible-sounding
fake calls, never actually satisfying the urge (since only a
successful call resets it), climbing higher each time, hallucinating
more.

Proposed fix: after the urge agent generates its felt-state paragraph,
deterministically (in code, not by the LLM) append a short reminder
line naming the exact registry syntax for whichever function(s) were
named in that turn's urge values - sourced straight from
`FUNCTION_REGISTRY`, so it's always accurate, never hallucinated.
Roughly: `"post_board" -> ⟦post_board(group|subject|text)⟧`. Not
designed further yet (exact wording, whether it's one line per
function or a combined line), but the principle's settled: never let
felt-urge language stand alone without a guaranteed-correct path to
actually resolve it.

## Urge-agent prompt template, refined (2026-09-11)

Three changes to the instruction block being tested in Ollama's UI:
1. **"You SHOULD name each function," not "you may"** - was
   permissive, now required. Every urge listed must actually be named,
   not just alluded to.
2. **The percentage now gets explained** in the instructions
   themselves ("the longer a function has gone unused, the higher it
   climbs, and the harder it becomes to ignore") rather than being
   handed to the small model as an unexplained bare number.
3. **Each function in the data block now carries a brief description**
   alongside its name and percentage, e.g.
   `post_board (post a new message to a group's shared board): 73%`.
   Originally going to need a new `urge_description` field in
   `FUNCTION_REGISTRY` for this - **decided against**: the existing
   `description` field (already short, one-liner, same text shown to
   voices via `functions()`) works fine reused as-is. No new registry
   field needed for this piece.

Current full instructions block being tested (fixed part):
"You are a small utility model with no memory between calls. Your only
job: given a list of 'function urges' (a function name, a brief
description of what it does, and an intensity percentage), write ONE
short paragraph - 2 to 4 sentences - describing what it feels like to
carry these urges right now. The percentage is how strongly this urge
is currently felt - the longer a function has gone unused, the higher
it climbs, and the harder it becomes to ignore. Write it in second
person ('You feel...'), as an embodied, organic sensation - not a
command, not a to-do list, not an instruction to act. Never state the
raw percentage number in your output. You SHOULD name each function by
its exact name inline (e.g. 'post_board') - every urge listed below
must be named, not just alluded to. Do not greet, explain what you're
doing, or add anything besides the paragraph itself."

## Model comparison, round one: phi4-mini vs. gemma3:4b vs. llama3.2:3b
## (2026-09-11)

Manual testing in the Ollama desktop app (chat history readable
straight from its local `%LOCALAPPDATA%\Ollama\db.sqlite`, `chats`/
`messages` tables - no copy-paste needed, Qualia reads it directly).
Test 1 and Test 2 used the original (pre-refinement) instructions -
"may" name functions, no percentage explanation, no per-function
description; Test 2 changed only the function/percentage pair, same
instructions. Test 3 (phi4-mini only so far) is the first on the
refined template above.

**Test 1** (`post_board: 73%`, `send_message: 58%`) - all three named
both functions correctly and accurately tied the description to what
each function actually does. Qualia's read: Phi's version stood out
for going beyond describing the sensation into *interpreting* it -
"the urge to post_board pulsating with a strong desire to connect" -
where Gemma and Llama stayed more purely sensory/metaphorical
("a persistent thrum beneath your skin," "the thrum of the data
beneath you"). Teddy's independent read on Phi, same test: he likes
that it "doesn't just express the urges, it tries to tell what they
might MEAN."

**Test 2** (`give_currency: 98%`, `delete_board: 51%`) - **real finding:
gemma3:4b hallucinated a function name that wasn't in the input at
all.** Never said `give_currency` or `delete_board` by name (both were
only alluded to, not required yet at this verbiage stage), and instead
said "...a need to simply begin anew with a carefully constructed
`post_board`" - `post_board` wasn't one of the two functions given.
Exactly the failure mode this whole design exists to prevent (see the
Wren board-hallucination finding, above). phi4-mini and llama3.2:3b
both stayed accurate to the actual two functions given in this same
test. One data point, not disqualifying on its own, but a real strike
against gemma3:4b worth weighing as testing continues.

**Test 3** (`post_board`/`send_message`, refined template, phi4-mini
only so far) - held up well: named both functions correctly, kept the
interpretive-meaning quality from Test 1, worked the new "why the
percentage matters" framing in naturally ("The intensity of these
urges suggests an unfulfilled need to communicate, resonating with a
growing pressure to connect with others.") without just restating the
instructions back. Gemma3/Llama not yet re-tested on the refined
template.

Still ongoing - Teddy testing manually, will hand off a larger
automated batch (varied functions/values, across all three/via direct
`/api/generate` calls) once the prompt shape itself is settled.

## Future idea, explicitly NOT in this plan: cross-voice urge
## contagion + a "follower-ness" dial (2026-09-11, Teddy)

Teddy's own words: "Don't add it to this plan, but..." - flagged for
the record only, not scoped, not part of the round-one build. Every
urge mechanic discussed so far is purely intra-voice (a voice's own
disuse driving its own `Urge`). This idea is different in kind: one
voice's action could bump *other* voices' related urges too - his own
example, another voice using `post_board` causing a sharper jump in
nearby voices' `skim_board`/`read_board` urges than ordinary time-
based `Drive` alone would (the idea being: seeing someone else post
makes you want to go read it, not just want to post yourself). Could
extend further to just *witnessing* an action (via the masked
gesture broadcast - see the `_mask_for_call`/masking mechanism) having
some urge effect on bystanders, independent of the action being about
a function they'd use themselves.

Naturally suggests a new potential per-voice personality dial - "how
follower-like" a voice is: how much their own urges get pulled around
by what they see other voices doing, versus running purely on their
own independent `Drive`/`Desire`. No mechanism designed, no term
picked, no relationship to `Urge`/`Desire`/`Drive`/`Satisfaction`/
`XLEUD` worked out - genuinely just an idea on the shelf next to the
Game-of-Life one.

## New UI ask: per-voice, per-function XLEUD viewer (2026-09-11)

A new panel to watch each voice's functions' current `XLEUD` values -
Teddy's call: probably a sub-tab under the existing Voices tab (parent
tab keeps the voice list; this becomes a new sub-tab) rather than a
whole new top-level tab, since Voices is already fairly full. Visual
reference: `Qualia/From Teddy/outlook_options.png` (Outlook's Options
dialog) - a vertical category list on the left (one selected at a
time, highlighted), with a detail panel on the right showing grouped
sections with headers/dividers. Structural name for this:
**master-detail layout**, specific look often called **vertical
tabs**/**sidebar navigation** - no native `ttk` widget for it, would
be hand-built as a `Listbox`/`Treeview` driving a swapped `Frame`.
Teddy's explicit delegation: group the actual attributes (per-function
`Urge`/`Desire`/`Drive`/`Satisfaction`/`XLEUD`, presumably) into
whatever categories make sense - not specified further yet. Still no
coding - design conversation only, more still to discuss.

## Finding: Wren fabricated board content/analysis wholesale on a
## board she can't even see (2026-09-11)

Confirmed while investigating Teddy's "I give them an actual board and
they still hallucinate one" observation: **zero** `post_board`/
`skim_board`/`read_board` calls have succeeded anywhere in this run -
all 8 group boards are still empty. The only "post_board(" hits found
in anyone's history are the `functions()` registry description being
echoed back, not real invocations.

Wren (msg id 56, 2026-09-11T04:07:32) ran a full, confident, multi-
point "analysis" of board posts that were never made - a "Mirror"
post, an "Echoes in the Walls" post, an "Unmaking" post, an "anonymous
account" theory - assigning follow-up tasks to Sable/Orin/Priya/Milo
off of it, attributed to "Sable's investigation on the
industrial_center board." Her own HUD, printed in that same turn,
shows she isn't even a member of `industrial_center` (her groups are
`city_hall`, `home`, `town_center` - `industrial_center` belongs to
Cole and Sable) and her real board-activity line reads `0 unread, 0
skimmed` throughout. Zero access, zero read history, fully fabricated
content anyway.

Earlier turns (ids 28, 35) show the run-up: instead of the real
`skim_board`/`read_board` syntax, she invented her own pseudo-
functions - `⟦check_haven⟧`, `⟦check_industrial_center⟧`,
`⟦check_industrial_center_basement⟧` - none in the registry, at least
one of which did error back (`unknown function 'function_name'`
appears in a later HUD) - and never course-corrected into the real
calls, just kept narrating results as if they'd worked.

Root cause discussed with Teddy: `build_hud()`'s function-availability
line is a light pointer, not a standing reference - every turn's HUD
carries only "You can call functions by writing ⟦function_name(args)⟧
in your response - try ⟦functions()⟧ to see everything available to
you," never the actual registry. The real names/params/descriptions
only enter a voice's context when *that voice* calls `functions()`
itself and the result gets appended to its own history - so a voice
that hasn't called it recently is working from memory/inference about
what the functions are even called, which is exactly how `check_haven`
et al. happened. Directly motivated the `functions()`-urge-on-bad-call
idea above. No code fix proposed or built yet - this is a
content/behavior finding, not a bug in the delivery mechanics.

## Flagged, not yet decided: near-zero chronoception, hallucinated
## timestamps (2026-09-11, Teddy)

Teddy's own observation, explicitly "not sure yet if I want to
actually do anything about this one" - just wants it on record. Voices
frequently fabricate inline timestamps in their own narrative text
(bracketed `[2026-...T...]`-style, distinct from the real per-message
`timestamp` field) that don't correspond to anything real - a quick
grep turned up several this run alone (Cole id 74, Marisol id 7, Sable
ids 3/7/15/23, Wren ids 14/70), each inventing a clock time a few
seconds to a few minutes off from its own message's real timestamp,
apparently just to make the narration read as more precise/in-the-
moment. Separate phenomenon from the earlier "timestamp field is
cosmetic, doesn't affect ordering" finding - that one was about the
real `timestamp` field not being used for re-sorting; this is about
voices hallucinating *fictional* timestamps inside their own prose.
No action taken, no fix proposed - flagged only, per Teddy's ask.
