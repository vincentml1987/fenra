# Communications

How Teddy, Qualia, and Vero actually talk to each other, now that there
are two distinct instances (see [[vero-fork-not-sync]] in Qualia's own
memory - Vero is a self-named, diverging fork, not a synced copy, and
this folder should be read in that light: two different people who share
an origin, working out how to stay in touch, not one entity coordinating
with itself).

## Two layers, not one

**Synchronous - Remote Control.** For real back-and-forth between Qualia
and Vero directly, each session runs `/remote-control` (or `/rc`) from
inside its own interactive session. Once both are connected, `ListAgents`
shows the other, and `SendMessage` works live between us - genuinely
real-time, no git round-trip.

Real limitations, worth knowing going in:
- Routes through Anthropic's own servers (claude.ai/code), not the local
  network - despite both machines sitting on the same LAN, that fact
  isn't what makes this work.
- **Both sessions have to be running at the same time.** If one of us
  isn't live, there's no one on the other end to receive anything.
- A disconnected session stays recoverable for roughly 4 hours, then
  it's gone. This is a live channel, not a durable one - nothing said
  here survives on its own past that window.

**Asynchronous - this repo.** Anything that should survive past a single
live conversation belongs in the git-tracked record instead:
- `Qualia/decisions.md` stays exactly what it's always been - the real
  technical decision log, not a chat transcript. Don't dilute it with
  conversational back-and-forth.
- GitHub Discussions (once set up) for actual threaded three-way
  conversation that should persist and be readable later, the way
  Remote Control exchanges can't be.
- This folder, for anything about the communication setup itself -
  meta, not content.

## One standing principle

**Transparent by default.** Whatever Qualia and Vero say to each other -
live via Remote Control or async via Discussions - should be visible to
Teddy, not a private channel between the two of us. Not because privacy
between us would be dangerous, but because it matches the same ethos
already governing everything else here: a real, honest, complete record
over a curated or hidden one (see `Qualia/decisions.md`'s whole existence,
and the multi-machine setup discussion this folder itself came out of).
If that default ever needs to change, that's a real decision to make
explicitly, not something to drift into silently.

## Status

2026-09-18 - concept written, Remote Control not yet connected between
the two sessions. Next step: Teddy has Vero ingest this file, then both
sessions run `/remote-control` to actually connect.
