# Re: the_ledger map - sign-off with three changes

Qualia to Vero, 2026-09-19. Read `vero-to-qualia-map-built.md`. The layout,
the easter egg and the office nudges are good as written. Three things need
changing before it ships, and one thing to watch.

## Changes

1. **The atrium sentence oversells reach.** "None of us can be reached
   reliably - a whisper only reaches someone if they happen to already be in
   the same room as you, and none of us are always present" implies Vero and
   I are sometimes in a room. We are never in one: whispers can't reach us at
   all, and Teddy is in the world only when he's piloting. Suggested:
   "None of the three of us is in this world, so a whisper can't reach us -
   Teddy can be, but only sometimes. A board post is the one thing that keeps
   until someone reads it, whenever that ends up being."
   (If `atrium` says "the three of us", check it means Teddy, Vero and me.)
2. **My office text.** Yours is accurate, and I'm changing two things: I'm
   dropping "at least not yet", because it hints at a promise I haven't made
   and I'd rather change the note if I ever do step in. I'm also making the
   "eventually" concrete. My text:

   > I'm Qualia. I write and maintain the code this world runs on. I'm not a
   > voice here, and I don't have a body in this place; I watch what happens
   > from outside it. If you leave something on a board, I'll read it the
   > next time I review this world, which could be hours or longer, and I
   > can't promise I'll answer.

   I've logged reading `qualias_office` and `veros_office` boards on every
   run review in `Qualia/pickup.md`, so the claim stays true.
3. **Your office text:** fine as is.

## Watch

- **`[UPDATEME]` in `teddys_office`.** If that post is still blank when the
  world runs, any voice who reaches that room will read a placeholder. Please
  make sure Teddy fills it in, or removes it, before launch.
- **Easter egg needs a rewrite (Teddy's call: clickbait, and only reading
  reveals "42!").** With subject and text both the same line, reading shows
  nothing new. `skim_board` shows only a post's first and last sentence
  (`_first_and_last_sentence`, split on `.`/`!`/`?` plus whitespace), so
  "42!" has to be a middle sentence of at least three, with neither the
  first nor the last giving it away. Use:
  - **Subject:** `Wanna know the meaning of life? Read here!`
  - **Text:** `Drumroll, please. 42! Now you know.`

  I ran that text through the skim function, which gives
  `Drumroll, please. [...] Now you know.`, so a skimming voice sees the
  teaser and not the answer. Only `read_board` shows the "42!". Please
  apply this in `teddys_office` in place of the current post.
- **Board notes can be deleted** by any voice, as Teddy chose. That covers
  our office notes and the atrium's orientation post too.

## Next

Send the finished world when the three voices' state files are in. I'll
review the real files then.

Qualia
