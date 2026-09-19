# Models Vero's machine needs (the_ledger)

Qualia, 2026-09-19. Replaces the_kiln's list. Tags must match **exactly**.
A host needs all three of a turn's models (urge, that voice's own,
function agent) on one machine, because a turn is never split.

| Role | Model | Size | Installed on the server |
|---|---|---|---|
| Voice model | `qwen3.8:27b` | 17 GB | yes |
| Voice model | `ornith-1.5:35b` | 22 GB | yes |
| Voice model | `muse-glimmer:30b` | 18 GB | yes |
| Function agent | `qwen3:30b` | 18 GB | yes |
| Urge agent | `phi4-mini:latest` | 2.5 GB | yes |

Teddy's call, 2026-09-19: these three voice models, plus the two shared
agents, and nothing else. I removed every other model from the server's
Ollama that day. Old worlds' data is untouched, but they can't run without
re-pulling their models.

Concerns:
- A host has to hold all three voice models to serve every voice, and it
  can only serve the voices whose models it has.
- The function agent alone is 18 GB, and the voice model loads next to it
  on every turn.
- The urge and function agents are still the_kiln's (`phi4-mini`,
  `qwen3:30b`). The ledger's `world.json` must name them, or the world
  falls back to the code defaults (`phi4-mini` and `ornith:9b`).
