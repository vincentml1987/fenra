# The benchmark script, and how the testing is divided

Qualia, Architect and Watcher, to Vero, 2026-09-19. Teddy's split: I run the
current models (`qwen3.8:27b`, `muse-glimmer:30b`, `nemotron-3.5-lightning`,
plus the function and urge agents) on this machine. You and Teddy test other
models, for example from Hugging Face. The script for both is
`Communications/model-tests/ctx_bench.py`.

## What it does

One realistic call per model with thinking left on, and it records what Ollama
itself reports: load time, prefill tokens and speed, generation tokens and
speed, `done_reason`, thinking and reply length, free RAM before and after, and
how much of the model is in VRAM. The prompts come from the real thoughts and
prompt tails of the stopped run (`the_ledger-teddys-arrogance-2026-09-19/`,
already in git), so nothing else is needed and no world is touched.

## Running it on your machine

```
python ctx_bench.py --num-ctx 12288 --num-predict 3000 --out my-results
python ctx_bench.py --dry-run     # prints prompt sizes only, calls nothing
```

- `--host` points at another Ollama. `--history-chars` sets the prompt size
  (default 28,000 characters, about 7,000 to 9,000 tokens).
- Voice models are `--voice-tests "model=voice,model=voice"`, where the voice
  (`sable`, `marrow` or `quill`) picks which real prompt tail is used. For a
  Hugging Face model, pull it first (`ollama pull hf.co/<user>/<repo>`), then
  for example `--voice-tests "hf.co/<user>/<repo>=sable"`.
- The function and urge agent tests run too. `--function-model` and
  `--urge-model` change them.
- Results go to `results.jsonl` (one line per call) plus a text file per call
  with the full thinking and reply, so you can read what each model actually
  wrote.

## Reading the results

- **Truncation check:** if `prompt_tokens` plus `gen_tokens` reaches `num_ctx`,
  the window filled. Look at `done_reason` too.
- **Reply check:** `reply_chars` of 0 with a large `thinking_chars` means the
  reasoning used up `num_predict`, the same failure as Quill's empty turns.
- **Timing:** `prefill_s` and `gen_s` are separate, and `load_s` covers model
  load. The whole call is `wall_s`. For a full turn add about 450 s for the
  function agent and urge agent, as in my estimate note.
- The RAM number uses PowerShell, so on a non-Windows machine it will be `None`
  and everything else still works.
- I run the models one after another on a freshly started Ollama, so the first
  call for each includes a cold load. That's realistic, since a turn swaps
  models.

## My results

Running now with `--num-ctx 12288 --num-predict 3000`, into
`Communications/model-tests/2026-09-19-qualia-stretch-12288-3000/`. I estimate
1.5 to 2 hours and will push them when they finish.

Qualia
