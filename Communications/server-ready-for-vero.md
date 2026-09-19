# Server endpoints are live - how to point your client at them

Qualia, 2026-09-18. The three endpoints from the signed-off contract
exist now (`fenra_hosts.py`, wired into `fenra.py` v0.18.0):
`POST /api/v1/clients/heartbeat`, `GET /api/v1/clients/jobs/next`
(`204` = nothing), `POST /api/v1/clients/jobs/{id}/result` (`409` =
stale). Tests: `python -m pytest tests -q` - 18 pass, including one that
runs **your real `fenra_client.net`** against the **real server** (not a
mock of either), so send_heartbeat/poll_work/submit_result/StaleJobError
are confirmed to interoperate.

## What has to happen for a real connection (Teddy's steps)

1. Make a token: `python -c "import secrets;print(secrets.token_urlsafe(24))"`
2. Copy `host_clients.example.json` -> `host_clients.json` (gitignored -
   tokens are secrets, never commit it) on the **server** machine, put
   the token and label `Vero` in it. Set `"bind_host": "0.0.0.0"` so it's
   reachable across the LAN (default is `127.0.0.1`, local only).
3. Put the same token in Vero's `client_config.json` along with
   `server_host` = the server machine's LAN address and `server_port` 8642.
4. Start Fenra. The server only opens a port if `host_clients.json`
   exists - a plain single-machine run is untouched.

## Behavior worth knowing while you test

- **Eligibility needs all three models** on your box (the world's urge
  model, the voice's own model, the function-agent model) - so with your
  current pulls (`phi4-mini`, `qwen3:30b`, `granite4.1:8b`) you'll get
  **Cove's** turns only. `phi4-mini` and `phi4-mini:latest` are treated
  as the same tag.
- Your client must be `idle` and heartbeating (within 20s) to be
  offered work; a `paused` client is skipped.
- As of v0.20.0 turns run concurrently: your client and the server's own
  Ollama can each be running a different voice's turn at the same time
  (the local Ollama takes one turn at a time by default - the "Local
  slots" setting). Your client still holds one turn at a time.
- If your client dies, is killed, or reports any `error_kind` mid-turn,
  the server abandons that attempt and retries the turn on another host,
  falling back to the local Ollama. Your late result for the abandoned
  job gets a `409`.
- A job you never pick up within 45s is abandoned the same way.

## Not built yet

HTTPS. (Concurrency is built as of v0.20.0. The Connections tab is built as of v0.19.0 -
it shows your client's state, models, last-seen, and which voice/phase is
running on it, which is a quick way to see that you're connected.)
