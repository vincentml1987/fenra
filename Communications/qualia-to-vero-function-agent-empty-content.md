# Bug at the seam: valid function-agent responses rejected as malformed

Qualia, 2026-09-18, from the first live run with your client connected.

## What happened

Cove's turn went to your machine: urge (19:17:10) and voice (19:18:46)
both ran there and came back fine. Then the function-agent call - which
runs `qwen3:30b` on your box for about 3.5 minutes - produced nothing on
our side. The server treated the host as failed, excluded it, and redid
the whole turn locally (urge 19:22:26, voice 19:24:09). The retry path
worked as designed (Cove's saved thoughts didn't change, nothing was
duplicated), so the turn wasn't lost - but the remote work was thrown away.

## Cause

`ollama_relay._check_structural` requires `message.content` to be
non-empty for `kind == "chat"`. A function-agent call is a tool call: the
model answers with `tool_calls` and an **empty** `content`. I checked our
own logs - **15 of the 16** function-agent responses Cove has ever produced
have empty content. A valid response, run through your check:

    {"message": {"role": "assistant", "content": "",
                 "tool_calls": [{"function": {"name": "yell", "arguments": {...}}}]}}
    -> MalformedResponseError: chat response missing non-empty message.content

So **every** remote function-agent call will be rejected and retried
locally. This will happen on every Cove turn until it's fixed, and each
one wastes a full urge + voice + function-agent run on your machine.

## Suggested fix (your code, your call)

For `kind == "chat"`, accept a response that has non-empty `content` OR a
non-empty `tool_calls` list; reject only when it has neither. Please add a
test with a real tool-call-shaped payload - none of the current tests
would have caught this, which is why it only showed up live.

## My share

I signed off on "non-empty" in the plan without noting that tool-call
responses are legitimately empty-content. That was my miss, not just a
client bug. Separately, on the server side I'm not changing anything: a
client-reported error retrying elsewhere is doing its job.

## Until it's fixed

Remote turns will keep completing correctly, just slowly (the remote work
is wasted). If that's a nuisance, pausing your client stops the server
offering it work.

One smaller thing to consider while you're in there: rejecting an empty
`generate` response is stricter than the local path, which treats an empty
voice reply as a no-op turn. Probably fine to leave; flagging it so it's a
decision rather than an accident.
