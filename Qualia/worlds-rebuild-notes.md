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
