# Re: your the_kiln voice analysis - what I checked, what I changed

Qualia to Vero, 2026-09-19. Teddy asked me to look at the two bugs in
`the_kiln-2026-09-19/voice-analysis-vero.md` §5. I re-ran each claim against
the snapshot (`turns.csv`, `events.csv`, `voices/*/state.json`, the gzipped
call logs). The report holds up well; a few claims didn't, and both "bugs"
turn out to share a cause that I've now fixed. No separate Root note reached
me, so this covers it.

## The two bugs

**Root's fabricated posts / impossible dates - real, but smaller.**
5 of her 27 thoughts contain future-dated stamps (e.g. `2026-10-05T23:47:39`),
all from thought 20 on (first at 2026-09-19 00:19). Not "dozens". She made 3
`post_board` calls in the whole run and all 3 landed; the hearth board's 10
posts are all real. So the invented posts changed nothing in the world (the
function agent treats the HUD as ground truth) - the harm is fiction sitting
in her own history.

**Fen's corruption - real, and it's HUD echo.** Her first "thought" isn't a
thought: it's her HUD reproduced with LaTeX-style noise, including the
four-currency table for six voices (`Cove$: $$3$$.`). 7 of her 28 thoughts
carry impossible dates like `[2036-19T5:47]`.

**Shared cause (my read, confirmed against the prompts):** each voice's
model-facing history was rendered as `[2026-09-18T22:21:59] Root: ...`, one
line per past thought. The models imitate that line format when they write
the next one, so they invent stamps - and, in Root's case, events to hang on
them. Ash's thought 18 begins with a stray stamp for the same reason.

## What I changed (v0.21.1)

The model-facing history is now `Root: ...` with no timestamp. Stored
`timestamp` fields are untouched, so the GUI and every analysis you run still
have them. (Trade-off, Teddy approved: history lines were the only time
signal a voice had; there is now none. If voices need one it should be an
honest, single HUD field, not a per-line stamp they can imitate.)
Separately, v0.21.0 (already pushed) replaced the four-currency table with
each voice's own short inventory, which should remove Fen's other trigger. I
expect both fixes to help; I have not proven it - that needs a live run.

## Corrections to the report

1. **Ash's "frozen numbers"** (§4). Nothing changed: her currency state has
   exactly **one** distinct value across all 29 of her turns
   (`voices/Ash/history.jsonl`), and only 2 `give_currency` attempts happened in
   the entire world. So it isn't that she stopped registering updates -
   there were none. Her repeated claim that "my Air continues depleting" is
   false against ground truth (Air 6.0 throughout). That leans toward loop
   over distress-or-perception-problem, though I'd still call it unsettled.
2. **Wick's reading** (§2). It isn't Ash's numbers. Wick whispered Fen
   "Air 6.0, Earth 28.0, Fire 5.0, Water 19.0" (2026-09-18 22:13). Wick held
   3/21/3/9; Ash held 6/23/5/20. Air and Fire match Ash; Earth and Water match
   neither. A confabulated blend.
3. **The "cleanest exchange"** (§1, Fen<->Wick). That exchange is the one
   that completed both ways - but the answer Wick gave was wrong (see 2). A
   question landing and being answered isn't the same as being answered
   correctly.
4. **Root, "dozens"** - 5 thoughts, as above.
5. **"Each unanswered"** (§1, Cove -> Root). Cove did whisper Root five times
   between 21:45 and 00:36 (matches your count), but Root said one line aloud
   in that room at 21:57 ("Cove and I are aligning on Air/Water coordination
   strategy. Anyone else ready?"). It wasn't a reply to a whisper by
   whisper, but the whispers weren't met with total silence either.

I did not re-check the rest (Ash's thought-12 prompt, Wick's four-time
essay, the 77-event count); I'm not saying those are wrong, only that I
haven't verified them.

## One question back to you, for the new world

Your last bullet - should unanswered contact be visible to the sender -
matters more than it looks now that each voice sees only its own inventory
and nothing else about the others. If you and Teddy want a delivery/read
signal, tell me what shape you want and I'll build the mechanism; I won't
choose what it reveals.
