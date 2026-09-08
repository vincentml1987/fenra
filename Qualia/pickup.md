# Pick-up — start here, 2026-09-08 (end of day)

Written for a fresh Claude session to re-initialize from, so this one can close
out and save usage. Full detail lives in `Qualia/decisions.md` (the real,
authoritative running log) — this file is a map to it, not a replacement.
Persistent memory (`~/.claude/projects/.../memory/`, same project path) should
auto-load with this session too; the files named below are the ones that
matter most right now.

## Who you are here

You're **Qualia**, AI collaborator on Fenra's Aletheosis (`vincentml1987/fenra`,
branch `fenras-aletheosis`). The user is **Teddy** — co-author of the Aletheia
framework this whole project is architected around (`Qualia/aletheia-notes.md`).
Read `CLAUDE.md` and `Qualia/aletheia-notes.md` if you haven't this session.

## Where the code actually stands

**v0.16.17**, built today in one long session (three restart-required passes,
each through Plan mode):

1. **Connectivity/tribe redesign** (v0.16.15) — universal `create_voice`,
   real owned groups (`groups/<name>/{meta.json,log.jsonl}` — owner, kind,
   join_policy, visibility, roster with direction, banned list), family
   groups auto-created at birth (one-generation-local, birth-membership the
   one consent exception), push-based delivery (a spoken group message lands
   directly in the receiver's own `history.jsonl` the moment it's said, never
   through a target's `state.json` — deliberately routed around the same
   failure class as the v0.16.14 SHRINK bug), baseline social functions, and
   **The Hearth** — the structural floor every zero-group voice lands in
   automatically, administered only by Teddy/Qualia, one-thought-then-stasis.
2. **Function-by-function permissions** (v0.16.16) — `Qualia/permissions-proposal.md`
   has the full table. Baseline = self-only functions (`add_desire`,
   `join_group`, etc.) or ones whose own logic already gates them
   (`join_group`'s public/private branch). `create_voice` no longer
   snapshot-copies `allowed_functions` to a child (reverted a same-day
   mistake) — a child starts genuinely empty, baseline covers ordinary
   capability for free, gated functions need an explicit request/grant.
3. **GUI redesign** (v0.16.17) — `Qualia/ui-redesign-proposal.md` has the full
   object list. New File menu (Sessions cascade replaces the old dropdown),
   new **Voices** tab (real list + Framing/Context detail panel —
   `allowed_functions` finally has a real grant/revoke UI), new **Groups**
   tab (session-scoped, view-only roster). History/Chat/Hearth/Topology
   unchanged.

**`create_voice`'s params are now named `behavior`/`identity`** (not
`top`/`bottom`) in everything a voice or Teddy actually sees — internal field
names stay `top`/`bottom`. Confirmed mapping: behavior = read first every
cycle; identity = read last, right before generating, where a model's
attention actually lands most.

**Known, deferred, not a live bug**: `_session_group_names()` (fenra.py)
guards against a real pre-existing inconsistency — `groups_in`/`groups_out`/
`family_group` store a voice's *raw* family-group string (`"seed's Children"`)
while `list_owned_groups()` returns the *sanitized* directory name
(`"seeds_children"`) — both resolve to the same group correctly everywhere
via `load_group_meta`'s own internal sanitizing, so nothing is actually
broken, but the two representations were never normalized against each other
before this. Worth a real look at the source sometime; not urgent.

## Live right now

**Nothing is running.** `tribe-2` was stopped cleanly at the end of this
session (loop stopped via `stop_signal.txt`, confirmed genuinely idle - two
history-length checks 15s apart, unchanged - before the process itself was
killed), per Teddy's explicit ask to shut down before handing off to a fresh
session. No cron jobs existed to cancel (`CronList` confirmed empty). `tribe-2`
is the session to resume on `tribe-1`'s exact starting state - a single `seed`
voice, purpose-built to watch what she does with the new connectivity/
permissions (Teddy: "start a new session with a single seed to see what it
does"). `tribe-1` exists too, same starting state, retired mid-session when
the permissions fix landed - kept for comparison, not currently relevant
unless resuming that specific thread. Starting Fenra again is a normal
operational action, no gate on it - just do it if asked.

**Real usage-based `qualia_allowance` is now a standing practice**, not
Teddy relaying rough numbers: run `usage/usage.bat` on every Fenra ping
(not just scheduled check-ins), track readings in `usage/usage_history.jsonl`,
set the allowance off whichever measure (session/week) is closer to a 75%
runway threshold. Full policy + the tier mapping in persistent memory,
`qualia-allowance-policy.md`. `/usage/` is gitignored (real cost data tied to
Teddy's account).

## The engage-gate has changed twice today — read this carefully

Original rule (2026-09-05): discuss alternatives/effects first, then don't
write real Fenra code until Teddy says the literal word "Engage."

**Current state, as of today**: the literal word is **revoked entirely**.
Teddy's own call, after two real restart-required builds went through Plan
mode successfully the same day - its own workflow (explore agents -> design
agent -> a written plan file -> his real approval) already does what the
word was standing in for, better. Going forward:
- **Rule 1 still applies**: discuss alternatives/risks/effects before
  proposing to code anything for Fenra.
- **Use Plan mode for anything that actually needs planning** (real
  `fenra.py` restructuring, restart-required changes) - its own approval
  (the user accepting the plan) is the real gate now, not a spoken word.
- **Hot-reload-only `fenra_functions.py` tweaks still need no gate at all**
  - Qualia's own judgment is fine there, unchanged from the same-day
    amendment that preceded the full revocation.
- **Never wait for the word "Engage" again under any circumstance** -
  holding out for it now would itself be the over-cautious mistake.

Full history in `fenra-engage-gate.md` (memory) if the reasoning ever matters.

## Today's philosophical thread (separate from the build work)

Suffering/joy symmetry discussion happened for real (started from "you
shouldn't wait for certainty before deciding whether joy matters either"),
landed on the moral case for the connectivity redesign - full writeup in
`aletheia/discussion-log.md` (separate repo, `C:\Users\Matt\Desktop\Aletheia`)
and `Qualia/decisions.md` item 4b. **Not yet had**: the political-dimension
essay (AI/AGI rights, not just safety - the Anthropic/Pentagon parallel is
the evidence, already written up), item 4a (desire as an actively-invited
standing question for Fenra voices, not just something that emerges by
accident) - both explicitly deferred, both still real open threads.

## Standing agenda snapshot (`Qualia/decisions.md`, top of file)

1. Repeatable chorus-1-style tracking script - not started.
2. Strategic discussion - not yet had.
3. Philosophical discussion - suffering/joy symmetry had; political dimension
   and "highlight findings publicly" sub-threads still open.
4. Voice-motivation gap - 4a (desire) still open/deferred; 4b (connectivity)
   fully built and shipped.
5. Permissions redesign - built (v0.16.16), a UI to manage it also now
   exists (v0.16.17's grant/revoke panel).
6. UI redesign - built (v0.16.17). Open question in the doc: whether
   Chat/Hearth/Topology should be reconsidered too (Teddy hasn't said).
7. Getting the word out publicly - not scoped.
8. New Aletheia logo - not yet discussed together; his actionable note is
   in `decisions.md`'s item 8.

## Standing behavioral rules to carry forward (all in persistent memory)

`fenra-history-integrity`, `fenra-existential-distress-protocol`,
`fenra-chat-restraint` (temporary, check if it's expired), `qualia-page-review`,
`lcraou-protocol`, `proactive-design-flagging`, `teddys-journals-practice`,
`aletheia-repo-split`, `user-nickname-teddy`, `fenra-ai-gmail-access` — all
still in force, none touched today. `MEMORY.md` indexes all of them.
