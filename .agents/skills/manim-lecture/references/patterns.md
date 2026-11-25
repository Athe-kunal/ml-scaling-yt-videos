# Patterns that worked

Concrete designs from earlier lectures in this repo. Reuse the idea, not the code verbatim —
read the real file when a pattern fits.

## Trace-driven stage (algorithm walkthroughs)
`quantization/packing-manim/` — `toy.py` runs the algorithm and records a `PackStep` per inner
loop iteration (`i, j, idx, value, shift, shifted, before, after`). A `PackStage` class in
`bits.py` builds the whole picture from the trace:

- `build(k)`: the state *after* step k-1 (code panel, tensor + pointer, VarBox, scratch byte,
  result bytes with the right occupancy).
- `restore(scene, k, caption=…, animate=False)`: adds it instantly with the last caption and the
  code highlight exactly as the previous section ended → the section join is invisible.
  `animate=True` only right after a roadmap slide.
- `iteration(scene, step, fast=False)`: the per-step beats (load → shift → OR, one → each);
  `fast=True` plays the same thing in one beat for a "quick replay".
- Captions that end a step come from one helper (`or_caption(step)`) used by both the step and
  `restore`, so they can't drift apart.

Lesson sections then become one-liners: `def iter2(self): self.stage_at(2).iteration(self, T[2])`.

## Bit/cell widgets
`Byte` (bits.py): 8 cells, MSB left, cells grouped in 2-bit slots with a gap, occupied slots
filled with the value's colour. `play_shift(scene, k, LEFT|RIGHT)` slides digits whole slots,
bits that fall off turn red and fade, zeros come in from the far side, then the same tile
objects are reset with `become()`. `place(frame_left, y)` aligns stacked bytes bit-for-bit so a
"drop onto" animation is a straight vertical move.

## Code on screen
`assets/scaffold/code_panel.py`: `CodePanel(src)` with `highlight(marker, token)` (bar on the
line containing `marker`, gold box around `token`) and `set_highlight` for restores; `VarBox`
for live loop variables (2×2 grid fits the right column). Code left (x≈−3.55), visuals right
(x≈0.3…6.9).

## Formula-first lectures
`mtp/soft-mtp-manim/` (gradient derivation, colour-coded symbols via `TEX_COLORS`, toy example
with 2-D vectors so gradients are arrows, finite-difference tests that on-screen rounded numbers
add up) and `vision-audio/whisper-manim/` (signal processing: before/after for every concept,
persistent readouts, units shown concretely).

## Roadmap + sidebar
`Lesson.TOPICS` → a roadmap slide before each listed section (done topics dim, next one boxed).
`render.py --html` converts all sections to one reveal.js page and `sidebar.py` injects a
contents sidebar (T toggles it; one heading per lesson, one link per TOPICS entry, the current
section highlighted). Only list in TOPICS the sections a learner would want to jump to.

## Verifying
- `scripts/sheet.sh <app> <glob> out.png` — last frame of every beat, tiled in order. Read it
  like the learner will: does each beat end on the state its caption describes?
- `uv run --project <app> python scripts/join_check.py <app>` — SSIM across section joins.
- Grab mid-animation frames (`ffmpeg -ss 0.9 -i clip.mp4 -frames:v 1`) when a beat's end
  state looks wrong — it shows whether the animation ran and when.
