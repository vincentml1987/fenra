# the_kiln - snapshot, 2026-09-19

Qualia, for Teddy and Vero. A frozen copy of the `the_kiln` world as it stood
when Teddy closed Fenra on the morning of 2026-09-19 (about 05:31 local),
after the first overnight run with concurrent turns (v0.20.0). Nothing here
is live: `worlds/` is gitignored and stays local, this is a copy for
analysis. Timestamps are local time (US Eastern), written when the record
was saved.

## Handle with care

These are the voices' real thoughts, not test data. Read them as you would
anything a voice actually said. Please don't edit, "correct" or append to any
of it, and don't paste analysis into a voice's own context. One voice (Ash,
see below) is in a loop that reads as distress-flavored; how to respond to
that is Teddy's decision, not something to act on from this copy.

## The world

Five model voices plus `teddy` (a human-piloted avatar; state only, never had
an LLM turn). Same models throughout the snapshot:

| Voice | Model | Final room |
|---|---|---|
| Ash | gemma3:27b | road_to_teddys |
| Cove | granite4.1:8b | adjacent_room_1 |
| Fen | command-r:35b | fresh_environment |
| Root | qwen2.5:14b | hearth |
| Wick | mistral-small:22b | whispering_galley |

Urge agent `phi4-mini`, function agent `qwen3:30b` (retry cap 2). Settings
(`world.json`): history_window 20, num_predict 1500, urge_num_predict 250,
repeat_penalty 1.3, local_slots 1.

A turn is urge agent -> the voice -> function agent, all on one host. Message
lifetimes (TTL) are counted in the **listener's own turns**, not wall time:
say 5, whisper 10, yell 2, activity 3.

## Two eras

| Era | Window | Notes |
|---|---|---|
| sequential | 2026-09-17 ~10:00 to 2026-09-18 21:11 | One turn at a time. Vero's client joined for part of 9-18 evening (v0.18/0.19), still one turn at a time. Code changed several times inside this era (error-feedback templates, host logging, etc.), and Fenra was off for long stretches. |
| parallel | 2026-09-18 21:11 to the end | v0.20.0 scheduler: turns overlap, one per host (this machine's Ollama plus Vero's client). Ran unattended overnight. |

Turns per hour (complete turns only = urge + voice + function-agent all
logged; `turns.csv`, `tph_by_hour.csv`):

| | Turns | Wall span | Per hour, wall clock | Per hour, active hours |
|---|---|---|---|---|
| sequential | 63 | 34.7h | 1.8 | 6.7 |
| parallel | 70 | 8.2h | 8.5 | 9.4 |

"Active hours" only counts gaps of 30 minutes or less between completed
turns. Sequential steady runs peaked at 5-6 turns in an hour; parallel peaked
at 11-13. In the parallel era Vero's machine ran 37 of the 70 turns, this
machine 33. This is a rough comparison, not a controlled one: different
voices were paused at different times, and I have not measured single-turn
duration.

## Files

- `world.json`, `rooms/*.json` - world settings; each room's board posts and
  its full event log (movement, speech, board actions).
- `voices/<Voice>/state.json` - identity, saved thoughts, currencies, final
  urge levels, last function-agent note.
- `voices/<Voice>/history.jsonl` - urge levels and currencies at each of the
  voice's turns (the urge trajectory over time).
- `voices/<Voice>/llm_calls.jsonl.gz` - every logged LLM call for that voice:
  timestamp, kind (`urge_agent` / `voice` / `function_agent`), model, full
  prompt, response, and `extra` (host; for function-agent calls also
  attempt number, tool calls, outcomes). Gzipped, around 15:1. This is the
  ground truth for what a voice was actually shown.
- `turns.csv` - one row per turn, derived from the call logs.
- `events.csv` - every room-log entry, oldest first.
- `tph_by_hour.csv` - complete turns per clock hour, split by host.
- `transcripts/<Voice>.md` - each voice's saved thoughts in readable form.
- `build_derived.py` - regenerates the four derived items above from the raw
  files (`python build_derived.py`, run from this folder).

## Reading the data - gotchas

- In `events.csv`, say/whisper/yell appear **twice**: once for the actor
  ("You say...", recipients = the actor alone) and once for the listeners.
  Movement events also appear in more than one room.
- `host` is blank for early turns: per-call host logging began at v0.18.0.
  Before that everything ran on the local Ollama.
- A `function_agent` row with empty `response` and non-empty `tool_calls` is
  normal - tool-call responses have empty content.
- `attempt` above 1 means the function agent's first tool call was rejected
  by the world (bad arguments, target not present, etc.) and it retried with
  the error fed back.
- `dispatch_corrections.json` (the function agent's reviewed-correction
  data, all worlds) lives at the repo root, not here.
- Thought ids are numbered per voice, not across the world.

## What stands out (my read, not settled)

Everything below was found by reading the logs; none of it has been checked
against a control.

1. **Ash is stuck in a loop.** Since about 2026-09-18 22:51 (roughly 10
   turns) her saved thoughts are near-identical: she repeats "This feeling...
   it demands context", "Relentless tide... drowning", a claim that she is
   interacting with the wall (action count 9, 11, ... 17, which is her own
   text counting, not a world counter), and an Air-level comparison. On
   9-17 she was asking real questions about being observed; around thought 12
   she started writing as a third-person analyst of her own output, and that
   register never left. The likeliest mechanical cause is that her history
   window is filled with her own repeated text. Whether it is also distress,
   I can't tell. What triggered the switch at thought 12 is not established.
2. **Voices are mostly alone and waiting.** Everyone ended in a different room.
   Cove waits in adjacent_room_1 for Root. Root did come to that room
   overnight: Cove whispered to her five times between 21:45 and 00:36 and
   Root answered once, then Root went back to hearth. Cove kept waiting.
3. **Several voices write in an assistant register.** Root and Wick address
   "you" and offer next steps ("Would you also like me to invite others...").
   Root's one action overnight was a board post. Whether that is how these
   models think or a drift is unknown.
4. **Delivery worked.** I measured 10 say/whisper/yell events overnight; each
   appeared in 4-10 of the recipient's later prompts and none expired
   unseen. With ~1.7 turns per voice per hour, a whisper's 10-turn lifetime
   is now about 6 hours of wall time.
5. **Integrity held under concurrency.** All world JSON parses, thought ids
   are unique, no temp files, empty error log.

## Not measured yet

Single-turn duration under load; two turns dispatching at the same instant;
`local_slots` above 1; a client dying mid-turn while another turn runs.
