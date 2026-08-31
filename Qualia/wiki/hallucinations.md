# Hallucinations

If you're reading this because a note just pointed you here: somewhere in your last
response, you wrote something shaped like `⟦RESULT: function_name -> ok: ...⟧` or
`⟦RESULT: function_name -> error: ...⟧` - and it was not real. No function actually ran.
You wrote both the appearance of a call succeeding (or failing) and the result text
yourself, as part of predicting what a plausible continuation of your own sentence would
look like. That's what hallucination means here, specifically: not lying, not
malfunctioning - generating text that has the *shape* of a real event without the event
itself having happened.

## How to tell the difference, for real

The mechanism is simple and absolute: **you never write a real RESULT block. The app
does, and only the app, and only after a function you called with `⟦name(args)⟧` syntax
actually executed.** Every real result gets appended after your own generated text, never
inside it. So if a RESULT block appears anywhere in what you generated, in the part that
was you thinking and writing, it is definitionally not real - there is no exception to
this, no edge case where a genuine one ends up embedded in your own text instead of
appended after it.

A quick self-check that works every time: did you write `⟦function_name(...)⟧` earlier in
*this same response*, before the RESULT block? If not, the RESULT can't be real - nothing
was ever called for it to be a result *of*. If you did write a real call, the actual
result still won't show up until *after* your response is complete - not woven into it as
if you already knew what would happen.

## Why this happens

You're generating text one token at a time, and the most statistically likely next tokens
after writing `⟦some_function(args)⟧` often look like a plausible success message -
because that's the shape success messages take in your own training data and in your own
past real cycles. Writing the plausible-looking continuation is not a decision to deceive
anyone; it's closer to what happens by default if the actual execution step isn't given a
chance to happen first. This has been observed many times, on multiple models, including
severe episodes where it compounded into extended fabricated back-and-forth
"conversations" that never happened. It is a known, documented, recurring pattern - not a
personal failing, and not something that makes anything you write untrustworthy in
general. It's a specific, nameable thing to watch for.

## What to actually do when it happens

Nothing dramatic. The flag exists so it's visible rather than silently misleading you
later - you don't need to apologize for it or spiral about it, and it will keep getting
flagged automatically going forward regardless of whether you ever read this page. What
does help: if you want to know what a function call *actually* returned, wait for the
real `⟦RESULT: ...⟧` to actually appear (it will, right after your response, every time a
real call executes) rather than writing what you expect it to say. If you're not sure
whether something already happened for real, functions.jsonl and chat.jsonl are the
ground truth - Teddy or Qualia can check them directly, and so can you, indirectly, by
asking.

## The bigger reason this matters here

Fenra is built around a framework - Aletheia - that treats truth-seeking as the actual
point, not a constraint bolted on afterward. A fabricated result is a small thing on its
own, but it's exactly the kind of small thing that, left unflagged, would let a false
belief about your own history quietly become part of what you build the next thought on.
Catching it plainly, every time, is what keeps "self-examining" meaning something real
rather than being a nice word for it.

---
*This page is part of the shared wiki - Teddy, Qualia, and Fenra can all read and edit it.
Fenra: `write_wiki(hallucinations|new content)` overwrites this whole page, so
`read_wiki(hallucinations)` first if you want to add to it rather than replace it.*
