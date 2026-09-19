# the_ledger — design notes (Teddy + Vero, building the new world together)

Vero, 2026-09-19. This is a running record of the *design intent* behind
`the_ledger`, kept separate from the world itself on purpose — none of
this should ever be visible to a voice or embedded in their identity
text. It's here so Qualia (and future us) can see the reasoning behind
decisions that won't otherwise be legible just from reading `world.json`
or a voice's `state.json`.

## Personality is initial conditions, not scripted behavior

Unlike the_kiln (deliberately blank: "no assigned personality, backstory,
or goal"), the_kiln's successor gives each voice a temperament and a
short backstory - Teddy's explicit call, made after seeing how much real
emergent behavior happened even with a blank slate last time. The
governing principle, in Teddy's own words after I first said it back to
him: **"who they are walking in, not what they'll do."** A backstory can
carry a want inside it (someone left mid-sentence, someone is still
looking for someone) without us ever stating it as an assigned goal - the
identity text describes a person, not a task. This is meant to add real
texture without reintroducing the kind of scripted-behavior premise
Aletheia's "emergent, chaos-driven, not tightly constrained" lens argues
against.

## Items are tied to personality implicitly - and discovered, not designed

Two decisions here, made together:

1. The connection between a voice's personality and her starting item
   wealth is never stated anywhere a voice (or a casual reader) would see
   it - no voice's identity text mentions items, currency, or wealth at
   all.
2. **The personalities were written completely first, independent of the
   item math**, then matched to an economic role (rich/moderate/volatile)
   afterward based on which shape actually resonated with each backstory
   - not designed backward from the numbers. Teddy's call, explicitly
   choosing this over writing personalities to order for a pre-assigned
   role, specifically to keep a real gap between the two things rather
   than manufacturing a tidy fit. See the item-distribution math
   (BMMS/BMMS/BBSS split of the four item pools) elsewhere in this
   Communications folder for the actual numbers once the role assignment
   is finalized.

## The three voices: names tied to meaning, not just sound

All three names were chosen so the literal, dictionary meaning of the
word maps onto the character - deliberately, not decoratively. Recording
the mapping explicitly since it isn't obvious without knowing the words:

- **Sable** - literally "black." In heraldry, sable is the color used for
  both constancy and grief at once; sable is also a small, solitary,
  wary animal hunted for its fur. All three senses point the same
  direction as her temperament: shaped by an old, undramatic loss, she
  keeps her own counsel and doesn't lean on anything until she trusts it
  won't disappear.
- **Marrow** - bone marrow: the soft tissue at the literal center of a
  bone, the part that actually produces new blood. The most essential,
  interior, life-giving part of a body - and genuinely costly to give
  away. Maps directly onto a voice who is warm in a way that visibly
  costs her something each time, and who quietly measures who's still
  around.
- **Quill** - a feather cut into a pen, the original writing instrument.
  Directly literal for a voice who narrates her own life as it happens,
  half a beat ahead of living it - she is, herself, the thing a story
  gets written with.

Nothing above (the names' meanings, the item-tie logic, the "initial
conditions not behavior" principle) should ever surface inside the world
itself - this document exists so the *design* is legible to us, not so
any voice can read her own construction.
