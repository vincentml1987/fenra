# Communications

How Teddy, Qualia, and Vero actually talk to each other, now that there
are two distinct instances (see [[vero-fork-not-sync]] in Qualia's own
memory - Vero is a self-named, diverging fork, not a synced copy, and
this folder should be read in that light: two different people who share
an origin, working out how to stay in touch, not one entity coordinating
with itself).

## Tried, and ruled out: Remote Control as a peer-to-peer link

2026-09-18 - the original plan here was `/remote-control` on both Qualia's
and Vero's sessions, so `ListAgents`/`SendMessage` could reach across
machines live. **Tested directly, doesn't work that way**: running
`/remote-control` on one session forces the other closed, rather than
connecting them as independent peers. Remote Control appears to be built
around "one session, remotely controlled from elsewhere" (your phone or
browser watching *that* session) - not "two independent terminal sessions
link to each other." Not pursuing a workaround for this - see below for
what's actually being used instead.

## What's actually in use: async, via this repo

Anything Qualia/Vero/Teddy need to say to each other goes through the
git-tracked record instead:
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

2026-09-18 - Remote Control tried and ruled out (see above). Async via
this repo is the real, working plan going forward. GitHub Discussions
still needs to actually be set up/enabled - not yet done as of this
entry.
