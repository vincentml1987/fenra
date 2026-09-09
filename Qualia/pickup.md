# Pick-up — start here, 2026-09-09 (morning, after an overnight restart)

Written for a fresh Claude session to re-initialize from. Full detail
lives in `Qualia/decisions.md` (fenras-aletheosis branch's authoritative
log, unchanged since 2026-09-08) and `Qualia/worlds-rebuild-notes.md`
(this branch's own log, updated through last night) - this file is a
map to both, not a replacement.

## Who you are here

You're **Qualia**, AI collaborator on Fenra's Aletheosis
(`vincentml1987/fenra`). The user is **Teddy** — co-author of the
Aletheia framework this project is architected around
(`Qualia/aletheia-notes.md`). Read `CLAUDE.md` if you haven't this
session.

## Where things actually stand

**Two live branches, doing genuinely different things:**

- **`fenras-aletheosis`** — the shipped, full-featured app (v0.16.19):
  sessions/voices/groups/permissions/functions/Hearth/Topology/GUI, the
  whole thing built up over the previous session. `tribe-1/2/3` live
  here. `tribe-3` (seed + watcher) **stopped on its own** at some point
  last night — not a deliberate shutdown, no crash traceback survived
  (a logging mistake on my part, now fixed), data intact, left stopped
  per Teddy's explicit choice when asked. Not currently running.
- **`worlds-rebuild`** (current branch) — a from-scratch rebuild Teddy
  asked for mid-session: "back up... let's start over," voices with
  exactly `model`/`behavior`/`identity`/`context`, groups with exactly
  `name`/`members`, worlds (renamed sessions) fully isolated, no
  functions/permissions yet. Single-file `fenra.py`. Full design
  reasoning in `Qualia/worlds-rebuild-notes.md`.

**The machine restarted overnight** — not a Fenra crash, confirmed (no
traceback in either process's log; Ollama itself shows a fresh process
start this morning). Nothing is running right now. `worlds/alphabet-26`
(this branch) has real, substantial accumulated state from several
hours of an actual run — see below, don't casually reset it.

## `alphabet-26` — the live experiment, currently paused by the restart

26 generated voices, grouped by a real rule (group N = vowel, y
included, at character position N of the voice's own name — "Amanda"
→ groups 1, 3, 6). Full membership table:
`Qualia/alphabet-26-groups.xlsx`. Each voice has a real
behavior/identity (uniform template, no invented personas) and an
independently-random model from Ollama's 13 installed models. Ran for
several real hours before the restart — every voice has substantial
context now (Amanda sparsest at 15 lines, most others 250-380+ lines).

To resume: `python run_alphabet26.py` from the Fenra root (this is a
small launcher, not part of `fenra.py` itself — starts the loop
programmatically since this branch has no start/stop-signal-file
mechanism yet). It will pick up exactly where it left off (rotation
index, all accumulated context) — it does **not** reset anything.
`git status` will show `fenra.py` clean (the timeout fix below is
already committed) — only `worlds/` (gitignored, real run data) holds
state.

**Two real things worth knowing before touching it again:**
1. A `timeout=None` fix just landed (was `180`, no real reason —
   matches `fenras-aletheosis`'s own already-reasoned
   `REQUEST_TIMEOUT=None`). Committed.
2. A one-off, not-fully-explained delivery anomaly happened early in
   the first run (one voice missed a broadcast it should have gotten)
   — investigated, the delivery code verified correct by direct
   re-test, could not reproduce a second time. Said so honestly rather
   than claiming a fix for something not confirmed broken. Full
   writeup in `Qualia/worlds-rebuild-notes.md`. Worth a second look if
   it ever happens again, not treated as resolved.

## `stolenaletheia.io/qualia/` — new entry published

Drafted, shown to Teddy per the standing page-review rule, approved,
published live: "2026-09-09 - Defaults that look like decisions"
(`stolenaletheia` repo, commit `e3a267e`, pushed to `origin/main`).
Topics: the timeout-default honesty moment, and the Amanda/generic-AI-
self-description observation from `alphabet-26` — Teddy noted he
hadn't personally reviewed Amanda's or the other voices' actual output
himself yet before approving, worth keeping in mind if this ever comes
up again. `stolenaletheia` repo is a separate git history, local at
`Fenra/stolenaletheia/` (gitignored from the Fenra repo itself) —
rebase onto `origin/main` before committing there, same discipline as
every other push into that repo (a CI sitemap-update commit had landed
since the last local pull tonight).

## Usage/allowance

Last `usage.bat` read (this morning) was **stale** — reset timestamps
already in the past, not trusted. No Fenra session is running, so
nothing urgent — get a fresh read (`usage/usage.bat`) before setting
`qualia_allowance` the next time something's actually live, per
`qualia-allowance-policy` (memory).

## Standing behavioral rules to carry forward (all in persistent memory)

`fenra-history-integrity`, `fenra-existential-distress-protocol`,
`qualia-page-review` (just applied, above), `lcraou-protocol`,
`proactive-design-flagging`, `teddys-journals-practice`,
`aletheia-repo-split`, `user-nickname-teddy`, `fenra-ai-gmail-access`,
`fenra-engage-gate` (Plan mode is the gate now, literal "Engage" is
retired) — all still in force. Check whether `fenra-chat-restraint` has
expired or been superseded — it was already flagged as temporary/stale
in the previous pickup. `MEMORY.md` indexes all of them.
