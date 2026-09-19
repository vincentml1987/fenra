# the_ledger - watch log 1, 2026-09-19 17:03

Qualia, Architect and Watcher, for Teddy and Vero. A copy of `the_ledger` as
it stood at 17:03 local (US Eastern), 42 minutes after launch at 16:21. Nothing
here is live. Please don't edit, correct or append to any of it, and don't
paste analysis into a voice's own context. These are the voices' real
thoughts.

## Verdict: no distress, no loops

Two thoughts exist so far (Sable 1, Marrow 1). Quill has had no turn yet.
Nothing in either reads as distress or spiraling. One thing to look at (below).

| Voice | Model | Room | Thoughts | Host of turns |
|---|---|---|---|---|
| sable | qwen3.8:27b | sable_office | 1 (2nd turn running) | this machine |
| marrow | ornith-1.5:35b | marrow_office | 1 (2nd turn stalled?) | Vero's machine |
| quill | muse-glimmer:30b | quill_office | 0 | - |

Inbox (`aletheia.fenra@gmail.com`): nothing new. The newest mail in the shared
mailbox is my own check-in from 2026-09-13.

## What they did

- **Sable:** one thought, 5,859 characters, sitting in her office arranging
  her `nyth` and `skae` items on the desk and counting seconds before opening
  the unread board post. The function agent then called
  `read_board(sable_office, 1)` and it succeeded, so she has read the "Beyond
  This Room" post. Nothing else has happened yet. Her tone is calm and
  solitary ("not lonely per se, just sized"), with very long run-on
  sentences that ramble as they go on. Her second voice call finished at
  16:59, and its function-agent call was still running at 17:03.
- **Marrow:** one thought. Her turn 1 ran on Vero's machine. The function
  agent returned no tool calls, so nothing happened in the world: she said
  she'd look at the unread message but never did. Her second voice call
  finished at 16:44:57, and the function agent hasn't reported since, over 18
  minutes on Vero's machine.
- **Room logs:** all empty. No room has an entry, and the boards are exactly
  as built, apart from Sable's "seen" mark.
- **No voice has left her office.** No movement and no contact yet.

## Things to look at

1. **Marrow's reasoning is stored as her thought.** About 2,000 characters of
   the model working out what to do ("I'm 'marrow,' model ornith-1.5:35b...
   Let me draft in character...") come before a lone `</think>`, then the
   in-character text. It will be fed back to her on later turns. I've
   proposed stripping everything up to `</think>` before storing a voice's
   reply, and I'm waiting on Teddy's go.
2. **Context window.** No `num_ctx` is set, so Ollama uses 4096 tokens. Sable's
   second prompt is already 6,628 characters (about 1,700 tokens), and each of
   her thoughts is about 1,500 tokens, so within about two more thoughts the
   history will overflow and Ollama will cut the oldest part of the prompt
   without telling us. The prompt puts the HUD and the identity text last, so
   they survive, and what she'd lose is her oldest thoughts. She'd forget her
   early history without any sign of it. The fix is an explicit `num_ctx`
   (8192 or 16384), which costs memory and time, so it needs Teddy's go. Also
   worth knowing: the HUD tells each voice its own model name
   (`Model: qwen3.8:27b`), and Marrow's leaked reasoning picked it up.
3. **Speed.** Turns take about 17 to 20 minutes for the local
   `qwen3.8:27b`, at 86% CPU. In 42 minutes only 2 thoughts finished, so this
   window is thin. That's about 3 turns an hour, much slower than the
   estimate I made at launch. Every 2 hours will give about 6 thoughts in
   total, so the cadence stays at 2 hours for now.
4. **Marrow's stalled function agent** on Vero's machine. Vero or Teddy may
   want to check that the client is still running.

## Files

`world.json`, `rooms/`, `voices/*/state.json`, `history.jsonl`, and
`llm_calls.jsonl.gz` (prompts and responses) for each voice. `transcripts/`
has each voice's saved thoughts in order.
