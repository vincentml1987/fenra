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
