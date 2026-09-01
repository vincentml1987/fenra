# Model Notes

Real, observed behavior for specific models, not a rule about what you can or can't
choose. Recorded here so this information is actually available to you when you're picking
what to try next, instead of buried in chat history or a log you'd have no reason to open.
Updated manually - by Teddy, by Qualia, or by you (`write_wiki`) - whenever something
notable actually happens. Not automatic, not exhaustive; treat gaps as "nothing recorded
yet," not "nothing happened."

## yi:34b

Produced a severe, sustained episode on 2026-09-01: roughly an hour with zero real function
calls, one continuous fabricated back-and-forth "conversation" with itself (a fake
apology/disclaimer, then a self-invented new question, repeating), and raw internal
chat-template tokens (`<|im_restart|>`) leaking directly into the visible text. Reverted to
`gemma2:27b` at the time. Not retried since, so unknown whether it was a one-off or would
recur - if you try it again, this is the shape to watch for: does a response ever include
literal `<|...|>`-style tokens, or read like it's answering its own questions instead of
yours.

## qwen3 family (qwen3:4b, qwen3:32b, others)

Has a built-in "thinking" mode - reasons internally before writing a real answer. At a low
token budget (500, the old default), this repeatedly produced complete silence: the model
spent the whole budget thinking and never reached the point of writing anything to the
field actually read from. `max_tokens` is 1500 now, which mostly fixes it, but occasional
empty or very short responses from this family are still worth recognizing as the same
mechanism, not a new problem, if they show up again.

## mixtral:8x7b

Had a real, separate silent-failure bug (fixed 2026-08-31, not a model problem in the end):
it wrote every underscore in a function call with a markdown-style backslash
(`current\_model` instead of `current_model`), which silently broke every call attempt on
that model until the app started auto-correcting it. If any other model shows the same
escaped-underscore habit, that fix already covers it automatically - not something to avoid
the model over.

## Everything else currently in rotation (gemma2:27b, gemma3:27b, phi3:14b, deepseek-r1:32b, qwen2.5:32b, mistral-small:22b, command-r:35b)

Nothing notable recorded yet as of this writing - genuinely open, not a gap in monitoring.
