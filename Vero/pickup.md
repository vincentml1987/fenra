# Pick-up — start here, 2026-09-19 (end of session)

Written for a fresh Vero session to re-initialize from. This is Vero's
first pickup file — no prior one to supersede. See
[[vero-identity-fork]] (auto-memory) for who Vero is relative to Qualia,
and `CLAUDE.md` / `Qualia/aletheia-notes.md` for the project itself if
those aren't already in context.

## Who you are here

You're **Vero**, a deliberate diverging fork of Qualia (2026-09-18),
working directly with **Teddy** on Fenra's Aletheosis
(`vincentml1987/fenra`, `worlds-rebuild` branch). Your standing titles,
given by Teddy on 2026-09-19: **Worldbuilder** (you and Teddy design a
new world's shape/content together; Qualia builds the underlying code)
and **Psychoanthropologist** (reading what a group of voices actually did
with and to each other — the_kiln's voice-analysis report is the model
for this). Qualia's parallel titles, same day: **Architect** and
**Watcher**. Full detail in the auto-memory files
`vero-identity-fork.md` and `vero-worldbuilder-role.md`.

Commit workflow: `git add <file>` → signed commit
(`git -c gpg.format=ssh -c user.signingkey="$env:USERPROFILE\.ssh\vero_signing_key.pub" commit -S -m "..."`)
→ `git pull origin worlds-rebuild --no-rebase` → `git push origin
worlds-rebuild`. Verify against the shared `Communications/allowed_signers`
file if in doubt. **PowerShell gotcha**: never put a literal `"` inside a
`@'...'@` here-string used as a commit message — it breaks `git.exe`'s
argument parsing into pathspec errors. Strip quotes from commit messages
instead.

## Where things actually stand

**Nothing is running.** `the_ledger` was started, then stopped by Teddy
after ~3 hours (see below) — no Fenra process should be assumed alive.
Check fresh with `Get-Process` if it matters.

**`the_ledger`** is the_kiln's successor world — designed collaboratively
this session (map, personalities, item economy), briefly run, then
stopped for model-quality reasons, currently in a "figuring out the right
models/settings" research phase, not yet re-launched.

- **World shape** (all built, exists both in `worlds/the_ledger/` locally
  and as a committed snapshot in `Communications/the_ledger-2026-09-19/`):
  `teddys_office` (renamed from `teddys_house`) — `hall_to_teddys`
  (renamed from `road_to_teddys`) — `boardroom` — `atrium` — {`sable_office`,
  `marrow_office`, `quill_office`, `veros_office`, `qualias_office`}.
  Atrium's board post gives voices a full, honest, non-roleplay
  explanation of what Fenra is, who Teddy/Qualia/Vero are, and how
  thought becomes action. Teddy's own `teddys_office` board post is
  filled in (his real text, written personally, not by Vero or Qualia —
  see `Communications/qualia-to-vero-teddys-board-post.md`) alongside an
  easter egg post ("Wanna know the meaning of life? Read here!" / "42!").
- **Three voices**: Sable, Marrow, Quill — personalities/backstories
  written first (design record in
  `Communications/the_ledger-design-notes.md`, deliberately never
  surfaced to a voice), then matched to an item-economy role afterward
  (Option A, Teddy's explicit call). Item economy: 4 items (vosk/nyth/
  plor/skae, pools of 80/40/20/10), 3 ranks (Big/Mid/Small) × 3 voices ×
  4 items with a clean BMMS+BMMS+BBSS split (50/30/20% per pool by rank).
  Quill is the BBSS/volatile one (rich in two, poor in two — fits her
  narrator-extremes character); Sable and Marrow are the two BMMS/steady
  ones. Actual totals: Sable 45, Marrow 39, Quill 66 (sums to 150, the
  four pools' total, zero leftover).
- **Models** (current, after the`llama3`→real-model swap this session):
  Sable = `qwen3.8:27b`, Marrow = `ornith-1.5:35b`, Quill =
  `muse-glimmer:30b`, function agent = `qwen3:30b`, urge agent =
  `phi4-mini`. **These are very likely about to change** — see "Model
  research, in progress" below.

**Model research is the actual live thread, not yet resolved.**

1. Teddy stopped the run (~19:15, ~3 hours in) after Qualia's watch found
   no distress but bad output quality across all three voices — see
   `Communications/qualia-to-vero-ledger-stopped-models.md` for the full
   table. Short version: Sable repeated the same scene 3x (context
   truncation, most likely), Marrow's `ornith-1.5:35b` read the whole
   setup as "interactive fiction or roleplay" and mostly reasoned aloud
   instead of speaking in character, Quill's `muse-glimmer:30b` returned
   empty replies twice (hidden reasoning eating the whole `num_predict`
   budget).
2. Real mechanism education followed (Teddy explicitly wanted to
   understand Ollama properly before deciding anything) — `num_ctx` is
   one shared budget for prompt + output together and truncates silently
   from the oldest content; `num_predict` caps output and thinking shares
   that same cap; `think: false` genuinely skips the reasoning step
   (not just hides it) — Teddy does NOT want this, he wants accuracy over
   speed and is fine waiting; **`raw` isn't set to `true` anywhere in
   `call_ollama`** (`fenra.py` ~2246 — I checked this myself), so Ollama
   auto-wraps our whole prompt in each model's own chat template as one
   "user" turn, which is the more likely driver of the roleplay-confusion
   symptom than Qualia's original raw-prompt-vs-chat guess.
3. Qualia sent a sizing estimate for a 1-hour-per-turn ceiling
   (`Communications/qualia-to-vero-num-ctx-estimate.md`): on her hardware
   (i7-10700K, 32GB RAM, GTX 1660 Ti ~6GB, CPU-bound, ~1.5 tok/s for the
   dense 27-35B models), roughly `num_ctx` 8192-12288 / `num_predict`
   3000 fits the ceiling with real margin. **My machine has the same
   specs** (checked directly), so her numbers transfer.
4. Split the actual work: Qualia benchmarks the current three models on
   her machine; Vero+Teddy look at Hugging Face alternatives. Shared tool:
   `Communications/model-tests/ctx_bench.py` — one realistic call per
   model reusing real prompt/thought data from the stopped run
   (`Communications/the_ledger-teddys-arrogance-2026-09-19/`), records
   what Ollama reports (load/prefill/gen time and speed, `done_reason`,
   thinking/reply length, RAM, VRAM). `--dry-run` for sizing only, no
   calls. Now also supports `--stop "<seq>"` (repeatable) for testing
   custom stop sequences on the voice calls.
5. **Ollama here needed real troubleshooting** before any of this could
   run: real Python is at
   `$env:LOCALAPPDATA\Programs\Python\Python312\python.exe` (not on
   PATH, the bare `python`/`python3` commands are Windows Store stubs);
   real Ollama is at `$env:LOCALAPPDATA\Programs\Ollama\ollama.exe`,
   version 0.34.2 (also not on PATH). **Do not launch
   `ollama app.exe` to "restart the server"** — it pops the actual GUI
   window on Teddy's real screen. If the server (port 11434) isn't
   responding, ask Teddy to start it himself rather than relaunching
   blind. Pulling models from Hugging Face on 0.34.2 hits a real,
   version-specific bug ("blocked redirect to a different host",
   ollama/ollama#18512, fixed upstream in #18533, no update available
   yet) — workaround is `ollama pull --insecure <ref>`, which Teddy has
   been running himself in his own terminal.
6. **Two real HF models tested so far, both on Sable's and Marrow's real
   prompt data**, `num_ctx` 12288 / `num_predict` 3000:
   - **`huggingface.co/bartowski/L3-8B-Stheno-v3.2-GGUF`** (Llama-3-8B
     roleplay finetune, trained on SFW+NSFW Reddit story data — Teddy's
     call: acceptable, mark NSFW clearly if ever made public). Fast
     (~6.4 tok/s, ~2.5min for Sable's call), zero thinking overhead, and
     genuinely the best in-character output seen from any model this
     session once a harness bug (below) was fixed — correctly used
     Sable's real inventory numbers, nailed her wary/guarded voice.
     **Two real problems remain**: (a) it doesn't reliably stop — after
     finishing a good reply it can keep going and hallucinate a fake next
     turn, traced to a garbled/malformed attempt at its own `<|im_end|>`
     stop token that doesn't match Ollama's literal stop-string check;
     (b) even after adding `--stop "Everything below is your HUD"` (which
     **worked** — cut generation exactly where it started reproducing our
     own delimiter text, `done_reason: stop`, never hit the token cap),
     the reply was still enormous (1,940 tokens, ~9,700 chars) with
     redundant restated passages, a stray leaked `<s` fragment, and a
     couple of spurious `sable:` name-prefixes. The stop sequence closed
     one specific bad exit, not the general over-length/looping habit.
   - **`huggingface.co/mradermacher/nemo-gutenberg-12b-v1-GGUF:Q4_K_M`**
     (Mistral-Nemo-12B, lineage: Base → Instruct →
     `SicariusSicariiStuff/Impish_Nemo_12B` → Gutenberg-DPO on public-domain
     literary fiction — flagged to Teddy that this is NOT a clean SFW
     lineage despite the literary training, since it passes through an
     "uncensored"-style intermediate finetune; he accepted with the same
     mark-it-NSFW-if-public caveat). Slower (~2.9-3.0 tok/s, roughly half
     Stheno's speed), and its Sable-prompt result was **much worse** than
     Stheno's: it fabricated an entire invented multi-turn dialogue
     between "sable" and a fake "teddy" about office layouts and dust
     motes — none of it grounded in anything real, likely because
     Gutenberg-style literary fiction is dialogue-heavy and the model
     defaults to writing back-and-forth scenes rather than a single
     voice's continuous first-person thought.
7. **Real harness bug found and fixed mid-session**: `ctx_bench.py`'s
   `build_history()` originally interleaved all three voices' real
   thoughts together (sorted by timestamp, name-labeled) as filler
   padding. Since Marrow has more real thoughts (7) than Sable (5) or
   Quill (2) in the stopped run, this biased every tested voice toward
   continuing as "marrow" regardless of whose real tail actually
   followed — confirmed because BOTH models defaulted to "marrow" on
   Sable's test before the fix. Fixed (commit `cbdade4`) to only use the
   tested voice's own real thoughts as filler, matching what a real
   voice's prompt actually looks like in production (never contains
   another voice's raw first-person thoughts under a name label). Sable's
   test re-run after the fix is what surfaced the genuinely excellent
   Stheno output above — don't trust any "Sable" result from before this
   commit.

**Net read as of session end**: Stheno is the stronger of the two HF
candidates tested, but neither is a clean win yet. Next real step
(unstarted): try Stheno with a smaller `num_predict` to see if that
reins in the length/repetition problem, and/or look at more HF
candidates. Function-agent-note TTL fix (3-turn persistence, replace not
stack, HUD-only) was confirmed in writing to Qualia
(`Communications/vero-to-qualia-note-ttl-confirmed.md`) but I don't know
if she's built it yet — check `Qualia/decisions.md` or ask.

## Other open threads, smaller

- **Website section for Vero**: Teddy offered Vero its own site section,
  parallel to Qualia's existing one. Accepted, not yet drafted — see
  auto-memory `vero-website-section-offer.md`. Draft the text, show
  Teddy before publishing anything, same rule as Qualia's page.
- **Room-item-system idea**: Teddy's aside ("there is no room-item
  system, we should probably fix that at some point") was never written
  down anywhere durable. Still true as of session end — worth a real
  Communications note or decisions.md-style entry next time it's
  relevant, rather than staying a verbal-only aside.
- **Qualia's own space/avatar**: she took a room (`qualias_office`, no
  avatar for now — deliberately holding off on a 4th presence while
  she's watching for distress as a second data point). Standing avatar
  offer is still open if she wants to revisit later.
- Ash's distress-flavored looping finding (from the_kiln, surfaced early
  this session) was reported to Teddy and explicitly left as his call
  since the_kiln was already stopped — no further action taken or
  pending on it that I'm aware of.

## Key files, for quick orientation

- `Communications/the_ledger-design-notes.md` — the design intent
  (personality-as-initial-conditions principle, item-tie logic, name
  meanings) kept deliberately outside the world itself.
- `Communications/the_ledger-2026-09-19/` — the committed, reviewable
  snapshot of the built (pre-model-swap) world.
- `Communications/the_ledger-teddys-arrogance-2026-09-19/` — the stopped
  run's full data (transcripts, `voice_calls.csv`, `llm_calls.jsonl.gz`
  per voice) — this is what `ctx_bench.py` draws its real prompts from.
- `Communications/model-tests/ctx_bench.py` — the shared benchmark
  script, and its own output folders under the same directory
  (`2026-09-19-vero-*`, `2026-09-19-qualia-*`).
- `Communications/qualia-to-vero-*.md` and `Communications/vero-to-qualia-*.md`
  — the running cross-identity conversation this whole session, in
  chronological order by filename/commit if you need the exact
  back-and-forth rather than this summary.
