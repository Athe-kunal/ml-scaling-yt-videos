# ManimGL 1.7 + manim-slides gotchas

Each of these cost real debugging time on an earlier lecture (whisper-manim, soft-mtp-manim,
packing-manim). Read before writing lesson code; re-read when something renders oddly.

## Slides / beats

- **Never call `self.wait()` inside a lesson.** manim-slides cuts slides by counting `play()`
  calls; `wait()` frames go into the video but aren't counted, so every later slide boundary
  drifts one animation early (a beat ends showing the *previous* state). When you need the final
  state rendered (e.g. after resetting mobjects post-animation), play a 0.1 s
  `self.play(Animation(mob), run_time=0.1)` instead.
- `beat()` and `clear_all()` are lazy (they fire when the next animation starts) so section scenes
  join without empty keypresses. Don't `self.add(...)` a stage right after a pending `clear_all()`
  — the add happens before the fade and the fade then removes it. Bring things in with
  `self.play(FadeIn(...))`, which flushes first.
- Don't name a section `setup` (shadows `Scene.setup`, the lesson dies with
  "no attribute _pending_beat"). Avoid other Scene method names too (`construct`, `play`, `wait`, `add`, `remove`).
- `skip_reversing = True` in Lesson: ← replays the previous beat forward (reversed clips only serve V).

## Mobject identity (the one that bites hardest)

- **`scene.remove(x)` where `x` is inside an on-screen group splits that group**: ManimGL
  replaces the parent in `scene.mobjects` with its other children. Anything you add to the
  parent afterwards is never drawn. Symptom: half-transformed glyphs, ghost digits, a widget
  that stops updating. Fix: never swap a child object for a new one — keep the same objects and
  reset them in place with `child.become(fresh)`.
- `Transform(a, b)` / `a.animate...` add `a` to the scene top level if it isn't already in the
  scene's family. Same root cause as above when objects get swapped.
- Mobject subclasses must not define a method called `shift` (or `scale`, `move_to`, …):
  `move_to` calls `self.shift(vector)` internally. Name it `play_shift`, `slide`, etc.
- A `Mobject` you create next to a group (e.g. a pointer triangle) is not moved when the group
  moves — reposition it after layout, or make it a submobject.
- Two animations on the same mobject in one `play()` (e.g. `FadeIn(vars)` + `vars.update_anim()`)
  → "could not broadcast input array" errors. Split into two plays.

## Text / LaTeX

- `Text(color=...)` is ignored → always `.set_color()` (style.T / tag / mono do this).
- `get_fill_opacity()` returns a numpy scalar; `set_fill(opacity=x)` with it can raise
  "len() of unsized object" → wrap in `float()`.
- `set_opacity(o)` on a group also raises the fill of empty (fill 0) cells to o → grey blocks.
  Dim by scaling each part: `sq.set_fill(opacity=float(sq.get_fill_opacity()) * o)`,
  `frame.set_stroke(opacity=o)`, `text.set_opacity(o)`.
- LaTeX: `\|` breaks (tipa) → `\Vert`; no `\!` before `\Big(` / `\left(`; `\div` renders as ∇·;
  indexing Tex by substring needs `isolate=`.
- Menlo glyph width ≈ 0.134 frame units per char at font_size 22 (0.146 at 24). Half the frame
  (~6.7 units) fits ~48 chars at size 22 → wrap code lines, 2-space indent.
- `Text("")` is fine as an empty placeholder.

## Live mobjects

- FadeIn on `always_redraw` / DecimalNumber-with-updater crashes (point counts change) → `self.fade_in_live(...)`.
- Call `clear_updaters()` before fading `always_redraw` mobjects (fade_all does this).
- Round live DecimalNumbers (−0.00 flicker changes glyph count).
- Never build per-point `axes.c2p` lists or rebuild Text/DashedLines every frame — vectorise, and rebuild only when a value changes.

## Arrows

- ManimGL `Arrow` is a filled shape: `stroke_color` is ignored → `style.arrow()` (fill_color) / `dashed_arrow()`.

## Environment

- Own uv venv per app; never let it join a root uv workspace (`uv init` silently adds one — revert).
- `setuptools<81` (manimgl 1.7.2 imports pkg_resources; the warning is harmless).
- Needs `dvisvgm` (brew). Installing it once broke ffmpeg via x265 → `brew upgrade ffmpeg`.
- Plain `manimgl file.py Scene` opens a window and hangs in automation — always go through
  `render.py` / `manim-slides render --GL`.
- Parallel renders need a per-process TMPDIR (Text SVG temp files collide) — render.py does this.
- Scripts that import the app (join_check.py) must run in the app's venv: `uv run --project <app> python …`.
