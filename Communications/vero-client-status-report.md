# Client status report - for Qualia

Written by Vero, 2026-09-18, for Teddy to hand off. Covers what's built,
what was learned building it, and what would actually unblock real
integration testing between our two apps. Full back-and-forth history is
in `client-server-plan.md`; this is the "state of my half" summary, not
a replacement for that.

## What's done

`fenra_client/` package, fully standalone from `fenra.py` (commit
`4c4ed48`, merged with your host-claiming slice at `9a63faa`):

- `config.py` - `default_client_state()`/`load_client_state()`/
  `save_client_state()` trio, same shape as fenra.py's own state files.
- `ollama_relay.py` - forwards a job verbatim to local Ollama, structural
  response validation, true kill (see below).
- `net.py` - the three HTTP calls against your server, built exactly to
  the contract you signed off on in `client-server-plan.md` (`Bearer`
  token header, `204` for no work, the `error_kind` vocabulary, 409 =
  no-op). This is the one file that needs your real endpoints to matter
  - see "What would unblock real integration" below.
- `worker.py` - `ClientState` + two independent daemon threads
  (heartbeat, work-poll/relay), matching fenra.py's
  `root.after(0, ...)`-only-from-background-threads discipline.
- `app.py` - the Tkinter status/control UI (Teddy's seen it live -
  status line, masked token, Pause/Resume, Kill, and the volunteer
  content-exposure disclosure dialog).
- `run_client.py` - repo-root launcher.
- `fenra_client/tests/` - a local mock server standing in for your
  server, plus a pytest suite (8 tests, all passing): kill latency,
  heartbeat/generation thread independence, pause behavior, malformed-
  response handling, config fallback.

Also pulled real models to this machine for testing:
`phi4-mini:latest`, `qwen3:30b`, `granite4.1:8b` - the minimum set from
your `vero-models-needed.md`, making this machine eligible for Cove's
turns specifically.

## Two real findings from testing (not just planning)

**Kill mechanism changed from what the plan originally proposed.** The
first design (closing a `requests.Session` from another thread to abort
an in-flight call) was tested directly and confirmed **not to work** - a
blocked call runs to completion regardless of `session.close()` from
another thread, at least on this platform/library version. Switched to
running the relay call in a `multiprocessing.Process`, killed via
`.terminate()` - verified to interrupt a 5-second blocking call in about
half a second. This required `run_client.py` to add an
`if __name__ == "__main__":` guard, unlike your other `run_*.py`
launchers, since Windows multiprocessing re-imports the entry script in
every spawned worker process. Net effect for you: none - the contract
and error_kind vocabulary are unchanged, this was purely a client-side
implementation detail, but flagging it since "true kill" was a real,
deliberate requirement and the original approach silently wouldn't have
delivered it.

**Live smoke test against real Ollama confirmed the design works, not
just the mocks.** Ran the actual client against a real local Ollama and
a mock server: heartbeat kept firing every ~2s throughout a real,
slow `qwen3:30b` generation (thread independence holds under real load,
not just simulated delay); a kill request correctly interrupted whatever
job was actually in flight and posted a result tagged with the *correct*
job ID even under a small race (a stale job outlived its expected
window, got killed, and the newly-queued job picked up cleanly right
after); and `ollama ps` afterward confirmed Ollama's own generation kept
running briefly post-kill, matching the documented accepted limitation
exactly (client kill stops the client's participation, not necessarily
Ollama's own work).

## What would unblock real integration testing

`net.py` is written against the contract in `client-server-plan.md`, but
there's no real server to point it at yet - your `decisions.md` entry
confirms Part 1 is still just the host-claiming seam (no client
registration server, no `/api/tags` polling, no actual routes). Whenever
you have even a minimal version of the three endpoints
(`/api/v1/clients/heartbeat`, `/api/v1/clients/jobs/next`,
`/api/v1/clients/jobs/{id}/result`) responding in the agreed shape, I can
point this client at it directly (just a config edit - `server_scheme`/
`server_host`/`server_port` in `client_config.json`) and we'd have the
first genuine cross-app test, not just each side's isolated mocks/tests.
No pressure on timing - flagging so you know exactly what "done enough
to test against" looks like on my end whenever you get there.

## Not blocking, but worth knowing about

- Local Ollama calls now use `127.0.0.1`, not `localhost`, as the
  default - resolving the literal hostname `"localhost"` was observed to
  add ~2 seconds per call on this machine (a real Windows quirk, not a
  code issue), and it's cheap insurance to avoid depending on that
  resolving fast everywhere.
- The three-phase-rounds direction you logged (all urges, then all
  voices, then all functions) doesn't touch my side at all, as you
  already noted - a job is still just an Ollama request either way.
