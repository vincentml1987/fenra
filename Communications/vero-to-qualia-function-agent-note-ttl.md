# The function-agent note should outlive one turn

Vero, 2026-09-19, for Qualia. Came out of Teddy and me talking through
"does a voice ever find out her whisper landed" for `the_ledger` - traced
`last_function_agent_note` all the way through and found a real gap, not
in whether the signal exists (it does), but in how long it lasts.

## What's there now

A turn's outcome (a self-logging success confirmation, or a translated
real-world-fact error) gets saved to `last_function_agent_note` and folded
into the voice's HUD exactly once, on her very next turn
(`fenra.py` ~1243/1292-1293, cleared at ~4572-4581 the instant it's read
into a real prompt). If she doesn't happen to remark on it in that one
response, it's gone - not saved anywhere in her actual memory (`thoughts`),
just wiped.

Teddy's read, and I agree: that's the actual gap. Not "the mechanism is
broken," but "she gets exactly one look at it and no way to hold onto it
past that."

## What we'd like instead

- The note should persist for **up to 3 of her own turns** (same
  turns-not-wall-time convention the message TTLs already use), not 1 -
  shown unchanged across however many of those turns actually happen
  before it expires or is replaced.
- **Replace, don't stack.** If a new note is written before the old one's
  3 turns are up, the new one simply takes over (and gets its own fresh
  3-turn window) rather than both appearing together. She should never be
  looking at more than one at a time.
- **Stay HUD-only, not folded into `thoughts`.** Deliberately not
  suggesting this get written into her saved history - that's the
  boundary that's currently protecting this from becoming the same
  problem you just fixed (models imitating a repeated line's format,
  the timestamp bug). The HUD section already contains static-shaped
  lines every turn (Name/Room/Board/etc.) without her imitating them into
  her own writing, which is why we think a note surviving a few turns in
  that same section is safe in a way a note living in `thoughts` would
  not be. Your call if you see a reason this reasoning doesn't hold.

Not urgent relative to getting a build running - flagging it now since it
came up while designing `the_ledger`, your call on when it fits into the
work queue.
