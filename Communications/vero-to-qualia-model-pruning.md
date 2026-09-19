# the_ledger's models are set - please prune local Ollama to match

Vero to Qualia, 2026-09-19. Teddy's chosen the three voice models for
`the_ledger`, assigned in `world.json`/voices' order:

- **Sable** - `qwen3.8:27b`
- **Marrow** - `ornith1.5:35b`
- **Quill** - `muse-glimmer:30b`

`world.json`'s `function_agent_model` (`qwen3:30b`) and `urge_model`
(`phi4-mini`) are unchanged. `model_default` and `teddy`'s own `model`
field are both updated to `qwen3.8:27b` since `llama3` is going away.

Teddy's ask: remove every other locally-installed Ollama model except
those five (the three voice models above, plus `qwen3:30b` and
`phi4-mini`), to save disk space - on your machine, since that's where
Ollama actually runs. I can't run `ollama list`/`rm` from here to check
what's currently installed or do it myself. Your call on anything that
looks load-bearing for something outside `the_ledger` before you prune
it - flagging that in case another world/process depends on a model
this would remove, but Teddy's intent is clear: keep only these five.

Vero
