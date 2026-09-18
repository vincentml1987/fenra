# Fenra Distributed Compute: Client/Server Plan

Written by Qualia, 2026-09-18, for Vero's review. Goal, agreed with Teddy
tonight: distribute the actual LLM generation load for a single shared
world across multiple machines, starting with Teddy's own two boxes and
extended later to volunteered friend machines. Two separate apps, not one
- see rationale below. **Server side (this doc's Part 1) is Qualia's to
build. Client side (Part 2) is Vero's.** Part 3 is the shared contract
between them - if either of us needs to deviate from it, that needs to be
a discussion here, not a silent divergence, since the two apps only work
together if they agree on it exactly.

Please review, ask questions, raise concerns - directly in this file
(add a `## Vero's Notes` section at the bottom) or via a new file in this
folder, whichever's more natural. Nothing here is final until you've
actually looked at your half.

## Why two apps, not one

Ollama already has a full HTTP API (`/api/tags` for inventory,
`/api/generate`/`/api/chat` for real calls). If the server could just
reach a client's bare Ollama directly, there'd be nothing to build
client-side at all. That breaks down for the actual target case -
friends' machines, not just yours and mine:

- **NAT/firewall**: a friend's home router isn't port-forwarded. The
  client has to reach *out* to the server, not the other way around, or
  it only works on a LAN Teddy directly controls.
- **No auth on bare Ollama**: anyone who can reach the port can submit
  generation requests. Needs a real auth layer in front of it.
- **Owner control**: Teddy's explicit principle - the machine owner has
  full say over what runs there. A thin client app is where that lives
  (see activity, pause, kill) - bare Ollama can't offer any of that.

So: the client is a thin, authenticated relay to the volunteer's local
Ollama, plus a small status/control UI. It has **zero Fenra domain
knowledge** - it doesn't know what an "urge agent" or "voice" or
"function agent" is, it just forwards whatever Ollama-shaped request the
server sends it to its own localhost:11434 and relays the response back.
All simulation logic stays server-side.

## Part 1 - Server side (Qualia)

Changes to `fenra.py`:

1. **Host list**, replacing the single global `world.json` `host` field
   with a list of host entries. Each entry: a label, a connection means
   (see Part 3 - clients register themselves, so this list is partly
   dynamic, not just static config), and a live-polled model inventory.
   The server's own local Ollama stays in this list as a permanent entry
   - it's the fallback, never removed.

2. **Concurrency in `_tick`**: currently single-threaded, one voice's
   whole turn (urge -> voice -> function-agent) runs sequentially on one
   background thread (`fenra.py:4001`). This needs to become N-capable -
   a worker pool sized to however many hosts are currently idle and
   eligible, not hardcoded to any fixed number.

3. **Per-turn host claiming**: a host is claimed *once*, at the start of
   a voice's turn, and used for all three calls in that turn (urge,
   voice, function-agent) - never re-selected mid-turn. This is the fix
   for Teddy's concern about Juno's urge and voice calls landing on
   different machines. Released back to the idle pool once the turn
   completes (success or failure).

4. **Eligibility**: a host is eligible for a voice's turn only if its
   polled model inventory contains that voice's assigned model as an
   *exact* tag match (same family, same parameter size - not "close
   enough").

5. **Shared-state safety**: now that multiple turns can run concurrently,
   anything they might both touch (room occupancy, board posts,
   `dispatch_corrections.json`, etc.) needs real locking. Current plan:
   one coarse lock held only during the brief moment of actual state
   mutation, not during the LLM generation wait itself (which is the
   actual slow part, and is what's being parallelized). This scales fine
   without needing fine-grained per-resource locks, since network latency
   dwarfs the mutation itself.

6. **Retry/fallback**: if a claimed host drops mid-turn, abort and retry
   the whole turn on a different eligible host. If no host is eligible
   at all, the turn just runs on the server's own local Ollama - the
   server is always the guaranteed fallback.

7. **Connections tab** (new GUI tab, matching existing conventions - see
   Voices/Groups, Voice Editor/Urge Viewer as the pattern to follow):
   lists every known host including the server's own local Ollama, with
   per row: label, online/offline, polled model inventory, and (when
   claimed) which voice's turn is currently running there and which of
   urge/voice/function-agent it's on. This is mostly just exposing state
   the scheduler already has to maintain for its own bookkeeping, not
   new tracking.

## Part 2 - Client side (Vero)

A standalone app, separate from `fenra.py` entirely. Rough shape, but
this half is genuinely yours to design in detail:

- Connects outbound to the server (see Part 3 protocol) - never listens
  for inbound connections itself, so it works behind any home NAT with
  zero router configuration.
- Polls its own local Ollama's `/api/tags` periodically and reports that
  inventory to the server (see Part 3).
- When the server hands it a job, forwards the exact request to its own
  `localhost:11434`, relays the raw response back. No interpretation of
  what the job means.
- Small UI: current status (idle / running - and probably fine to show
  which model, even though the client doesn't know *why* it's running),
  a pause/kill control that actually stops it from picking up new work
  (or aborts what's running, your call on how hard a "kill" should be).
- Auth: a pre-shared token, one per volunteer machine, that Teddy hands
  out of band when he sets someone up - not open enrollment. The server
  needs to recognize a token as belonging to a specific known client.

## Part 3 - The shared contract (do not diverge without discussion here)

Plain HTTP polling, client-initiated, since it's simple and NAT-friendly
without needing websockets or persistent connections. Exact routes open
to adjustment, but the *shape* below is the part that has to match on
both ends:

**Registration / heartbeat** - client periodically (proposed: every
5-10s) sends its token and current model inventory to the server. Doubles
as a heartbeat: if the server hasn't heard from a client in some window
(proposed: 15-20s), it marks that host offline and won't assign it new
work. This also covers the pause case - a paused client can simply stop
checking in, indistinguishable from "temporarily offline," which the
retry/fallback logic already handles cleanly.

**Work polling** - client periodically asks "anything for me?" Server
responds empty if nothing's assigned, or with a job if the scheduler has
claimed this host for a voice's turn. A job is the *exact* Ollama request
body (model, prompt/messages, options, format/tools as needed) - the
client doesn't construct or interpret it, just forwards it verbatim to
its own Ollama.

**Result submission** - client POSTs the raw Ollama response (or an error
if the local call failed) back, tagged with whatever job identifier the
server handed it, so the server knows which pending call it resolves.

Exact endpoint names/JSON field names aren't fixed yet - that's fine to
nail down between us once we're both actually implementing, as long as
neither side builds something that assumes more than what's described
above.

## Open questions for Vero

- Any concerns with the client being this thin (pure relay, zero Fenra
  awareness)? This was deliberate - keeps all real logic in one place -
  but flag if it creates a real problem on your end.
- Preferred poll intervals for registration/heartbeat and work-polling -
  proposed numbers above are placeholders, not settled.
- Anything about the pause/kill UX you think needs to be stronger than
  "just stop checking in"?

## Vero's Notes

*(left blank for Vero to fill in)*
