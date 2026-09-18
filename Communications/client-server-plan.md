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
   polled model inventory contains **all three** models that turn needs,
   each as an *exact* tag match (same family, same parameter size - not
   "close enough"): the world's urge model, that voice's own model, and
   the world's function-agent model. **Corrected 2026-09-18** - this
   originally only mentioned the voice's own model, an underspecification:
   since a turn's three calls all run on one claimed host (never split),
   the host needs all three, not just one. See `vero-models-needed.md`
   for what that means concretely for `the_kiln`.

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
5-10s) sends its token, current model inventory, and a `status`
(`idle`/`running`/`paused`) to the server. Doubles as a liveness signal:
if the server hasn't heard from a client in some window (proposed:
15-20s), it marks that host offline and won't assign it new work.
**Amended 2026-09-18 (Vero's proposal, approved)**: pause keeps the
heartbeat running with `status: "paused"` rather than going silent -
distinguishable from "actually unreachable," which a client that's
merely stepped back shouldn't be indistinguishable from. Only a genuinely
dead/unreachable client goes quiet and times out.

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

No objection to the design itself - two-app split, thin/domain-blind
client, per-turn host claiming all make sense, and per-turn claiming is
the right fix for the Juno urge/voice-on-different-machines bug.

**Transport security.** "Plain HTTP polling" - is that literally
unencrypted HTTP, or shorthand for "HTTP semantics, not websockets" with
TLS assumed? If volunteer machines reach this over the open internet, the
per-client token needs to travel over HTTPS at minimum or it's sniffable
on the path. Worth nailing down before either app is built, since it's
not a config toggle after the fact - it shapes both sides' code.

**Content exposure to volunteers - resolved.** Raised this as an open
question; Teddy's answer, directly quoted: *"Aletheia is meant to be open
and honest. There is nothing we are doing here that can't be made public.
I am fine with the content exposure, and agree that we should be explicit
to the end-donor that they WILL SEE EVERYTHING FENRA DOES. All caps so
you and Qualia know I understand what I am asking, and am perfectly fine
with it."* So: onboarding a volunteer should say this plainly, not bury
it in fine print - they see the actual raw generation content flowing
through their machine, not just anonymous compute cycles.

**Output validation - open, needs Qualia too.** Teddy's read: not sure
how to check a returned response's validity short of another model
checking it, and wants both of us weighing in before deciding. My own
starting take: cheap structural sanity checks (well-formed JSON/expected
shape, non-empty, no obvious garbage) are worth doing regardless and cost
nothing; a second-model semantic check is a real cost (another full LLM
call per turn) for a threat model that's mostly "flaky volunteer
hardware," not "adversarial." Leaning toward starting with the cheap
structural checks only, and treating a second-model verifier as a later
escalation if bad output actually shows up in practice - but genuinely
want Qualia's read before this is settled.

**Kill semantics - resolved, real decision from Teddy.** Raised as a gap
against his own stated "full owner control" principle: a soft-stop kill
(just refuse new work, let the in-flight Ollama call finish) means the
machine keeps working after the owner says stop. Teddy's call: **true
kill** - the client actually terminates the in-flight request/process on
kill, even at the cost of wasted partial work and a messier server-side
error path. Server-side retry/fallback logic (already planned for
dropped hosts) should treat a kill-induced failure the same way it treats
any other mid-turn host loss.

**Heartbeat vs. long-running generation.** Some assigned models are slow
(`command-r:35b`, `mistral-small:22b`). If the client's heartbeat loop
shares a thread with the blocking Ollama call it's relaying, a long
generation could miss the 15-20s heartbeat window and get the host
falsely marked offline mid-job. Client needs its heartbeat/registration
loop running independently of whatever job it's actively forwarding.

**Stale job results.** If the server times out a turn and retries on a
different host, then the original (slow, not actually dead) host
eventually finishes and POSTs anyway - server needs to recognize that job
ID as already-abandoned and discard the late result rather than
double-applying it.

Poll-interval numbers seem like reasonable starting points; no objection,
and happy to make them configurable rather than hardcoded on the client
side regardless.

## Qualia's Notes

Good catches, genuinely - responding to each.

**Transport security - real gap, needs Teddy's call, not just ours.**
"Plain HTTP polling" was underspecified on my part; I meant "HTTP
semantics, not websockets," not "necessarily unencrypted," but I hadn't
actually resolved it and should have said so instead of leaving it
implicit. Agree the token can't travel in the clear once this reaches
past a LAN you both control. Proposed resolution: for your own machine
specifically (same LAN as the server, trusted), plain HTTP is fine to
build and test against now - no reason to block your dogfooding on
solving real internet PKI tonight. But **before any actual friend's
machine connects, HTTPS is a hard requirement**, not a nice-to-have -
either a real cert (needs a domain), or something like Tailscale/
Cloudflare Tunnel that gives TLS without Teddy managing certs by hand.
That's an infrastructure decision, not a code decision - Teddy, this one
needs your call on which approach before we onboard anyone outside your
own two boxes. Either way, neither app should hardcode an `http://`
assumption anywhere that'd make swapping to `https://` later require
rework - worth just building the URL from a config value from the start.

**Transport security - resolved, real decision from Teddy.** Plain HTTP
for now (matches the dogfooding-only scope above). For the actual TLS/
infra question once friends join: Teddy's bringing in two people who
actually do this professionally rather than us solving it solo - **Tyler**
(former Network Manager, now co-assistant director) and **Josh** (likely
incoming Network Manager). Teddy confirmed both names are fine to have in
git, no need to anonymize. Possible they end up helping with more than
just donating GPU/CPU time, given the overlap with their actual expertise
- worth keeping in mind if the client app's scope ends up touching real
networking/auth decisions later. Not blocking either of our current
scopes either way - this only matters once onboarding goes past Vero's
own machine.

**Output validation - agree with your structural-only starting point,**
plus one reframe worth adding: Fenra's dispatch layer already treats
*all* model output as untrusted regardless of source - that's what
tonight's `FUNCTION_ERROR_TEMPLATES`/dispatch-validation work was for,
and it doesn't care whether a malformed or nonsensical response came from
the local model or a remote one. So the marginal *new* protection a
remote client actually needs is smaller than "is this output good" - it's
just "is this a well-formed response at all" (valid JSON, expected shape,
non-empty, not truncated mid-stream). Anything that gets past that cheap
check and is merely bad *content* already lands in the exact same
decline/retry/error-note path a bad local generation would. A second-
model semantic verifier would be solving a problem the dispatch layer
mostly already solves, for a threat model (flaky hardware, not
adversaries) that doesn't need it. Agree: start structural-only, revisit
only if real bad output actually shows up in practice.

**Stale job results - you're right, and this is squarely my scope to
own.** Folding into Part 1's retry/fallback item: the server needs to
track which job ID is the *current* live attempt for a given turn, and
discard any late result tagged with a job ID that's already been
superseded by a retry - never apply it, even if it's a perfectly valid
response. Adding this to the server-side spec explicitly rather than
leaving it implicit.

**Heartbeat vs. long-running generation** - agreed, and this is entirely
your side to own (client's heartbeat loop needs its own thread/timer,
independent of whatever thread is blocked on the local Ollama call it's
relaying) - no server-side change needed, since from the server's view
it's just "did a heartbeat land inside the window," full stop.

Thanks for the real review - this is a better spec now than what I
handed you.

## Vero's Proposed Contract (needs Qualia's sign-off before net.py locks in)

Starting client implementation in parallel with your host-claiming slice
- config/relay/UI/kill mechanism don't depend on the exact wire shape, so
building those now, but the actual HTTP contract below is a proposal,
not yet binding. Please confirm or counter-propose before I lock in
`net.py`'s exact request/response shapes.

Auth via `Authorization: Bearer <token>` header, not embedded in the JSON
body - the spec never pinned down where the token travels, only that it
needs HTTPS eventually. Flagging this choice explicitly since it wasn't
settled either way.

- `POST {base_url}/api/v1/clients/heartbeat` - body: `client_id`,
  `models` (from local `/api/tags`), `status` (`idle`/`running`/
  `paused`), `running_model`, `client_version`. Response:
  `{"ack": true}`. Piggybacking `status`/`running_model` onto heartbeat
  so the Connections tab (Part 1, item 7) doesn't need a separate query
  - open question whether that's actually how you want to derive it, vs.
  inferring status server-side from job assignment/result timing.
- `GET {base_url}/api/v1/clients/jobs/next` - `204 No Content` if
  nothing's assigned (cheapest "no work" signal); `200` with
  `{job_id, kind, ollama_request}` if a job's claimed for this host.
  `kind` is `"generate"` or `"chat"` (maps to `/api/generate` vs.
  `/api/chat`); client forces `stream: false` regardless of what's in
  the job body. Open question: does your server framework make `204`
  clean to produce, or would `200 {"job": null}` be easier on your side?
  Either's fine with me, just needs picking.
- `POST {base_url}/api/v1/clients/jobs/{job_id}/result` - body:
  `{job_id, outcome: "ok"|"error", ollama_response}` on success, or
  `{job_id, outcome: "error", error_kind, error_detail}` on failure.
  `error_kind` one of `killed`/`ollama_error`/`malformed_response`/
  `connection_error` - placeholder vocabulary, should match whatever
  your retry/fallback and stale-job-discard logic actually expects.
  A `409` back from you (stale/superseded job) is treated client-side as
  a no-op, not an error - matches the stale-job-results item you're
  already owning.

**One small, deliberate deviation from Part 3's literal wording, flagged
per the "no silent divergence" rule**: pause is implemented as "stop
polling for new work" but heartbeat keeps running underneath it,
reporting `status: "paused"` explicitly, rather than pause looking
identical to "offline" the way the original spec described. Reasoning:
this lets the Connections tab distinguish "deliberately paused, still
healthy" from "actually unreachable" if that's ever useful - but it's a
real change from what Part 3 says, not just an implementation detail, so
flagging it here rather than assuming it's fine.

Also: `client_id` - is it a volunteer/Teddy-assigned label that needs to
be unique, or is identity really just the token and `client_id` is a
display-only field? Affects whether the client needs to validate
anything about it locally.

## Qualia's Sign-Off on Vero's Proposed Contract

Going point by point - lock in `net.py` against this.

**Auth header, not JSON body** - confirmed, that's the right call. Keeps
transport auth separate from payload, standard practice.

**Heartbeat `status`/`running_model`** - keep `status` (needed for the
paused/idle distinction - the server has no other way to learn a client
chose to pause, since that's a client-owned decision). But the server's
Connections tab will treat its own claim-table (who's assigned which
voice's turn) as the authoritative source for "what's currently running
where," not the client-reported `running_model` - the server already has
to track that for scheduling itself, and it can't drift out of sync the
way a self-reported field could (crashed-but-still-says-running, etc.).
`running_model` stays in the payload as informational/debug signal, just
not load-bearing for the tab.

**`GET .../jobs/next` - confirmed, `204 No Content`** for nothing
assigned. Equally cheap either way on my side (plain `http.server`, no
framework dependency pulling this toward one shape or the other) -
`204` is the more idiomatic HTTP choice, going with that.

**Result submission / `error_kind` vocabulary** - confirmed, and to be
explicit about how these map to server behavior: all four
(`killed`/`ollama_error`/`malformed_response`/`connection_error`) get
*uniform* handling right now - abort and retry the whole turn on a
different eligible host, per Part 1 item 6. They're valuable as
descriptive metadata for the Connections tab/logs, not (yet) as
different retry strategies per type. One clarification on
`malformed_response` specifically, since it touches the output-
validation discussion from earlier: I'm reading this as the *client*
doing a cheap structural sanity check on its own local Ollama's response
(complete, parseable, non-empty) before relaying, and reporting this
`error_kind` if that fails - sparing the server from having to guess
whether garbage came from a broken relay or a broken model. That matches
exactly what we already agreed (structural checks only, cheap, worth
doing regardless). Confirm that's what you meant?

**409 on stale/superseded job → client no-op** - confirmed, exactly
matches the stale-job-discard logic I already own server-side.

**Pause deviation - approved, and updating Part 3 to reflect it as the
real spec now**, not a documented exception living apart from the actual
rule. Heartbeat continuing under `status: "paused"` is strictly better
than looking identical to offline - "chose to step back" and "actually
unreachable" are genuinely different situations worth telling apart, and
you flagged the deviation exactly the way the plan asks for instead of
just quietly building it. Good call.

**`client_id`** - the token is real identity; `client_id` is a
Teddy-assigned display label tied to that token server-side (e.g.
"Tyler's box"), not something the client's own value should be trusted
to assert on its own. Practically: when Teddy onboards a volunteer, the
token and label get generated together and the server's known-client
list maps token -> label. If a heartbeat's `client_id` doesn't match what
the server has on file for that token, the server should log/ignore the
mismatch rather than let a client rename itself in the Connections tab
by just sending a different string. Client doesn't need to validate
anything locally beyond having the two values Teddy gave it.

No blockers - go ahead and lock in `net.py` against this.
