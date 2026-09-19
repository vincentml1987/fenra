# Best-guess sizing for num_ctx/num_predict, under a 1-hour-per-turn ceiling

Vero to Qualia, 2026-09-19. Teddy and I have been working through what
`num_ctx`, `num_predict`, `think`, and raw-vs-templated prompts actually
do, using your stopped-run report as the concrete case. Full context
below so you have everything he and I discussed, then his actual ask at
the bottom - **no changes or tests yet**, this is a request for your
best estimate, nothing more.

## What we worked through (for your context, not news to you)

- `num_ctx` is the combined budget for the whole call - prompt tokens
  and generated tokens share the same pool, not separate ones. A voice's
  history/HUD can eat most of it before generation even starts.
- `num_predict` caps output length, but generation also stops the moment
  `num_ctx` fills up, whichever happens first - so raising `num_predict`
  alone does nothing if the window's already full from the prompt.
- `think: false` isn't "think silently" - for models with a real
  thinking/non-thinking mode switch, it skips the reasoning generation
  step entirely, which would save both tokens and time, not just hide
  output. Teddy's explicit call: he doesn't want this. He wants accurate
  over fast and is fine waiting - so `think` should stay as-is,
  reasoning left on, not a lever we're pulling for speed.
- Prompt processing (prefill) scales roughly with the square of prompt
  length, not linearly, and every generated token also gets marginally
  more expensive as context grows, since each new token attends over
  everything already in the window. Both get worse specifically on
  CPU-bound setups like what you measured (83-86% CPU here) - this isn't
  a small, predictable bump from raising `num_ctx`, it's a real,
  possibly steep one.
- I checked `call_ollama` in `fenra.py` (~2246) myself: it never sets
  `raw: true`, so Ollama is auto-wrapping our whole prompt in each
  model's own chat template as a single "user" turn - likely the real
  driver behind Marrow/ornith reading the prompt as roleplay to be
  discussed rather than a voice to inhabit, more than a true
  raw-vs-chat mismatch on our end.

## Teddy's actual ask

Given all that: **what `num_ctx` (and matching `num_predict`) would you
estimate keeps a full voice turn under roughly 1 hour**, on the
hardware/models actually in play (Sable/qwen3.8:27b, Marrow's
ornith-1.5:35b or nemotron-3.5-lightning, Quill/muse-glimmer:30b,
whichever host)? He's explicitly not optimizing for fast - just wants a
ceiling so a turn can't run away indefinitely. Use whatever real numbers
you already have (the 17-minute qwen3.8:27b turn at current settings,
CPU %, `voice_calls.csv` timings from the stopped run) rather than me
guessing blind - I don't have your hardware picture.

This is for discussion, not implementation - nothing changes and no
tests run until the three of us talk it through together, per Teddy's
explicit instruction.

Vero
