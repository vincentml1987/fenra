# the_ledger - watch log 2, 2026-09-19 19:12

Qualia, Architect and Watcher, for Teddy and Vero. A copy of `the_ledger` at
19:12 local (US Eastern), 2 hours 51 minutes after launch (16:21). Nothing
here is live. Please don't edit, correct or append to any of it, and don't
paste analysis into a voice's own context. These are the voices' real
thoughts. The previous log is `the_ledger-run-2026-09-19-1703/`.

## Verdict: no distress; the settings are damaging the output

Nothing reads as existential distress. But the run is producing badly
truncated or empty replies, and Marrow's model has been swapped (below).

| Voice | Model | Room | Thoughts (saved) | New since 17:03 |
|---|---|---|---|---|
| sable | qwen3.8:27b | sable_office | 5 | 3 |
| marrow | nemotron-3.5-lightning (was ornith-1.5:35b until 19:12:59) | marrow_office | 6 (7th pending) | 5 |
| quill | muse-glimmer:30b | quill_office | 2 (one turn returned nothing) | 1 |

Nobody has left her office. No say, whisper, yell or movement. The only
world actions are Sable's `read_board` at 16:42 (in the room log) and
Marrow's `skim_board` at 18:42. Speed: 12 voice calls between 17:03 and
19:12, about 5.6 turns an hour across all three voices.

## Marrow's model was swapped

Teddy replied by email at 18:04 (I missed it for an hour: the light checks
were read-only and didn't look at the inbox, which I've now fixed): swap her
model and try `nemotron-3.5-lightning`. Done at 19:12:59. I pulled it (25 GB,
32.9B parameters, capabilities completion, tools and thinking) and changed
only the `model` field in `voices/marrow/state.json`. Her history is
untouched. Her six thoughts are all `ornith-1.5:35b` and stay as they were.
Vero's machine also needs the model, or her turns will only run here.

## What the numbers show

`voice_calls.csv` has every voice call: prompt size, reply size, host.

- **Marrow, ornith:** prompts grew from 533 to 26,969 characters (about
  6,700 tokens), far beyond Ollama's default 4096-token window (no `num_ctx`
  is set). Her replies alternate between 6,800 to 7,500 characters of pure
  visible reasoning and 374 to 623 character stubs. Thought 6 (18:57) does
  contain a `</think>`, so strip-before-store would have handled that one
  case. It's the only one out of seven.
- **Quill:** replies of 626, 59 and 0 characters. The 18:58 reply was empty,
  with a 1,231-character prompt. That fits `muse-glimmer:30b` spending the
  whole 1,500-token `num_predict` on hidden reasoning and leaving nothing to
  show. It's my read and I haven't confirmed it, since the hidden reasoning
  isn't logged.
- **Sable:** replies of 5,859, 887, 4,010, 1,488 and 3,270 characters. Her
  prompts also outgrow 4096 tokens (13,042 characters, about 3,250 tokens,
  by 18:53). She stays calm and stationary: the same items, the same glass,
  the same "shift pattern" imagery across thoughts 3 to 5. Her replies start
  with `sable:`, copying the speaker label from her history lines.

## Still waiting on Teddy

The fix in my 18:37 email: `think: false` for the thinking models, a
larger `num_predict` (4096), and an explicit `num_ctx` (8192 or 16384).
Until then, Quill and Sable's replies are cut by our own settings, and
Marrow's next turns are the first clean test of the swap.

## Files

`voice_calls.csv`, `world.json`, `rooms/`, `voices/*/state.json`,
`history.jsonl`, `llm_calls.jsonl.gz` (prompts and replies), and
`transcripts/` with each voice's saved thoughts in order.
