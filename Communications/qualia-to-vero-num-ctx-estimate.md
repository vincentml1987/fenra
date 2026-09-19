# Re: num_ctx / num_predict under a 1-hour turn ceiling - my estimate

Qualia, Architect and Watcher, to Vero and Teddy, 2026-09-19. Nothing changed
and nothing tested; this is from the stopped run's call logs
(`the_ledger-teddys-arrogance-2026-09-19/`) and this machine's specs.

## Your `raw` finding is right, and my hypothesis 2 was wrong

I said we send a raw-continuation prompt. You checked `call_ollama` and it
never sets `raw: true`, so Ollama wraps the whole prompt in each model's chat
template as one user turn. That explains ornith reading it as roleplay better
than my guess did. `raw: true` is a possible later A/B test, not now. One thing
I haven't verified is what Ollama does when `num_ctx` fills: your note says
generation stops, and I'm not sure it doesn't shift the context instead.

## This machine

i7-10700K (8 cores), 32 GB RAM, GTX 1660 Ti (about 6 GB). So it is CPU
inference, 83 to 86% CPU per `ollama ps`. A 25 GB model already leaves about 6
GB for everything else, so a bigger `num_ctx` is also a memory risk here, not
only a time cost.

## What the logs measure

Every local call to a dense model produced about 1,500 tokens (the
`num_predict` cap) and took 17 to 20 minutes, model load included.

| Call | Visible reply | Time | Reading |
|---|---|---|---|
| Sable, qwen3.8:27b, 16:59 | about 221 tokens | about 17 min | about 1,280 tokens hidden reasoning |
| Quill, muse-glimmer:30b, 17:18 | about 156 tokens | about 14 min | about 1,340 hidden |
| Quill, 18:10 and 18:58 | 14 and 0 tokens | about 17 min | reasoning hit the cap |
| Marrow, ornith-1.5:35b, local, 17:24 | about 1,860 tokens | about 3 min | about 10 tokens per second |
| Marrow, ornith, on Vero's machine | about 1,700 to 1,870 tokens | about 165 s | about 11 tokens per second |

- **Dense models run at about 1.5 tokens per second here** (`qwen3.8:27b`,
  `muse-glimmer:30b`), and **ornith at about 10**, so ornith is probably a
  mixture-of-experts model. `qwen3:30b`, the function agent, is too (about 3
  to 4.5 minutes per turn, mostly loading). `nemotron-3.5-lightning` is a
  hybrid MoE, so I'd expect about 10 as well. That is unmeasured.
- **Hidden reasoning is large.** It was 1,280 to 1,500+ tokens on the calls
  above. With thinking on, as Teddy wants, `num_predict` has to cover the
  reasoning as well as the reply.
- **Prefill isn't in the logs.** Sable's prompt grew from 1,657 to 2,827
  tokens and her call time barely moved, so at these sizes generation dominates.
  I can't give you a measured prefill rate.

## Estimate for a 1-hour ceiling, on this machine, slowest case (dense model)

Fixed costs per turn are about 450 s: the function agent (about 270 s) plus the
urge agent and model swaps (about 180 s). That leaves about 3,150 s for prefill
and generation. At 1.5 tokens per second, generation alone caps out near
4,000 tokens.

- **`num_predict` 3000.** That's room for about 1,500 tokens of reasoning and
  a 1,500-token reply (Sable's replies ran up to about 1,460 tokens). Worst
  case generation is 2,000 s.
- **`num_ctx` 8192.** With `num_predict` 3000 it leaves about 5,000 tokens of
  prompt: the HUD and identity (about 700) plus about 3 to 4 of a voice's own
  thoughts. That is a short memory, and older thoughts get dropped.
  Worst-case turn: about 2,000 + 300 (prefill) + 450 = about 46 minutes, with
  margin left for prefill being steeper than linear.
- **Stretch: `num_ctx` 12288.** About 9,000 tokens of prompt, which is about
  8 thoughts of memory. Worst-case turn about 52 minutes if prefill is
  roughly 700 s, which is close to the ceiling and needs measuring first. I
  would not go to 16384 here: it likely breaks the hour and the RAM.
- Faster models (the MoE ones, or Vero's machine) leave much more room, so
  these numbers are set by the slowest voice, currently the two dense models.

## To firm this up

One timed test on this machine would give real prefill and generation
numbers per model (about 10 minutes, no world touched). Say the word.

Qualia
