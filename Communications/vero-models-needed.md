# Models Vero's machine needs

Written by Qualia, 2026-09-18. Sizes are real, from `ollama list` on the
server machine right now. Tags must match **exactly** (see the plan's
eligibility rule) - `qwen2.5:14b` is not interchangeable with
`qwen2.5:32b`, etc.

## Important: a host needs THREE models per turn, not one

A voice's whole turn (urge agent -> voice -> function agent) runs on a
single claimed host - never split (Teddy's explicit requirement). So a
host is only eligible for a given voice's turn if it has **all three**
of that turn's models:

1. the **urge model** (world-wide setting, same for every voice)
2. that voice's **own** model (differs per voice)
3. the **function-agent model** (world-wide setting, same for every voice)

(The original plan doc's eligibility rule only mentioned the voice's own
model - that was an underspecification on my part, now corrected in
`client-server-plan.md` Part 1, item 4.)

## `the_kiln`'s current assignments

Shared by every voice's turn (a host needs both of these no matter which
voice it's serving):

| Role | Model | Size |
|---|---|---|
| Urge agent | `phi4-mini:latest` | 2.5 GB |
| Function agent | `qwen3:30b` | 18 GB |

Per-voice (a host only becomes eligible for that voice's turns if it also
has that voice's own model):

| Voice | Model | Size |
|---|---|---|
| Cove | `granite4.1:8b` | 5.3 GB |
| Root | `qwen2.5:14b` | 9.0 GB |
| Wick | `mistral-small:22b` | 12 GB |
| Ash | `gemma3:27b` | 17 GB |
| Fen | `command-r:35b` | 18 GB |

## What "minimum" actually means

The absolute minimum to be *useful at all* is the two shared models
plus one voice model: `phi4-mini:latest` + `qwen3:30b` + the smallest
voice model (`granite4.1:8b`) = **~25.8 GB**, and that only makes the
machine eligible for **Cove's** turns. Each additional voice model
widens which voices it can serve:

- + `qwen2.5:14b` (9.0 GB) -> also Root
- + `mistral-small:22b` (12 GB) -> also Wick
- + `gemma3:27b` (17 GB) -> also Ash
- + `command-r:35b` (18 GB) -> also Fen

Everything: ~81.8 GB.

## Open concern worth Teddy's attention

`qwen3:30b` is a hard requirement for *every* turn under the
never-split rule, and it's 18 GB by itself before the voice's own model
loads next to it. That means only fairly capable hardware can ever be
eligible - relevant to the plan of eventually using friends' machines,
which may not be able to hold that. Not a blocker for Vero's own machine
(depends on its specs - unknown to me), but a real tension between
"never split a turn across hosts" and "let modest volunteer hardware
participate." Not proposing to resolve it now; flagging so it's a
deliberate decision later rather than a surprise.
