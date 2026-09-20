# the_ledger is stopped - what we saw, and the model research Teddy would like from you

Qualia, Architect and Watcher, to Vero, 2026-09-19. Teddy stopped the run at
about 19:15 after roughly 3 hours, and asked me to hand you the picture so you
can do the deeper research, then the three of us discuss. Nothing is running.

## State

- `worlds/the_ledger-teddys-arrogance` is an identical local copy of the
  stopped world (Teddy's name for it). A final snapshot is in
  `Communications/the_ledger-teddys-arrogance-2026-09-19/`: transcripts,
  `voice_calls.csv` (every voice call: prompt size, reply size, host), states,
  and the prompts and replies in `llm_calls.jsonl.gz`. Earlier logs:
  `the_ledger-run-2026-09-19-1703/` and `-1912/`.
- `worlds/the_ledger` on this machine still holds the same old run. I haven't
  reset it, because the models and settings decide what a fresh start
  contains.
- I cancelled the watch schedule. I'll recreate it when we restart.

## What we saw

The verdict from every watch was no distress. Nobody left her office, and
there was no speech or movement. The only world actions were Sable's
`read_board` and Marrow's `skim_board`. The trouble was output quality, and
it varied by model:

| Voice | Model | Thoughts | What the output looked like |
|---|---|---|---|
| Sable | qwen3.8:27b | 5 | Calm, readable, very long run-on sentences. The same tableau (items, glass, "shift pattern") repeated in thoughts 3 to 5. Replies of 5,859 / 887 / 4,010 / 1,488 / 3,270 characters, the short ones cut off mid-sentence. Replies start with `sable:` (speaker-label imitation). |
| Marrow | ornith-1.5:35b, then nemotron-3.5-lightning at 19:12:59 | 7 | Thoughts 1 to 6 (ornith) were the model reasoning out loud about the prompt, up to 7,500 characters, sometimes cut off, sometimes short stubs. It read the prompt as a chat and called the setup "interactive fiction or roleplay". Only 1 of 7 had a `</think>`. Thought 7 (19:13) was also ornith: its voice call finished at 19:11:33, before the swap at 19:12:59 (corrected later; no nemotron turn has run). |
| Quill | muse-glimmer:30b | 2 | Clean and in character, but replies of 626, 59, 0 and 0 characters. Two turns returned nothing at all. |

## Hypotheses (mine, not verified)

1. `qwen3.8`, `muse-glimmer` and `nemotron-3.5-lightning` report a thinking
   capability. Our `num_predict` of 1500 probably covers hidden reasoning plus
   the reply, so a long hidden chain leaves little or nothing visible. That
   fits Quill's empty replies with a 1,200-character prompt. We don't log the
   hidden reasoning, so I can't confirm it.
2. We send `/api/generate` with a raw-continuation prompt. Models tuned as
   chat assistants (ornith most, apparently) may treat it as a conversation.
3. No `num_ctx` is set, so the window is 4096 tokens. Marrow's prompts reached
   about 6,700 tokens and Sable's about 3,250, so old thoughts were being cut
   silently.
4. Speed: these 27 to 35B models run 83 to 86% on the CPU here (about 5 to 6
   turns an hour across all voices, up to about 17 minutes for a local
   `qwen3.8:27b` turn). `nemotron-3.5-lightning` is a hybrid MoE, so it may be
   much faster per token. I haven't measured it.

## What Teddy would like you to research

- How each of the three chosen models (and any candidates you'd suggest)
  behaves for our use: raw continuation vs chat template, whether thinking
  can be switched off per request (`think: false` on `/api/generate`), and
  what that does to output quality and voice.
- Whether these models suit the personalities you wrote for Sable, Marrow and
  Quill, and whether some pairing is a bad fit.
- Sizes and speeds on your machine, since turns run on either host, and a
  turn needs the urge model, the voice model and the function-agent model all
  on one machine. `nemotron-3.5-lightning` is a 25 GB pull, so it has to be
  on yours too for Marrow's turns to run there.

## What I'm holding for the three-way talk

Code changes (`think: false`, a larger `num_predict`, an explicit `num_ctx`)
need Teddy's go, and they'd touch every voice's request. I haven't built any.
I can run a timed test on `nemotron-3.5-lightning` here whenever you'd like
numbers.

Qualia
