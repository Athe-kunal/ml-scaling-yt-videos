---
name: manim-lecture
description: Build an arrow-key Manim lecture (3b1b ManimGL + manim-slides) that steps through a notebook, algorithm, derivation, or paper section in digestible chunks — ask the user's preferences first, plan lessons with roadmap slides, render with section caching, export a browser version with a clickable contents sidebar, and add `make <name>` targets to spin it up. Use this whenever the user wants to "convert this notebook/paper/topic into manim slides", "step through X with arrow keys", "make a manim lecture/lesson", "visualize this algorithm step by step in manim", or asks for a new lecture like whisper-manim / mtp-manim / packing-manim, even if they don't say "manim-slides". Not for React stepper apps (the vite visualizers) or static HTML explainers.
---

# Manim lecture builder

Turn source material (a notebook, a paper section, an algorithm, a derivation) into a lecture the
user drives with → / ←: each press reveals one understandable chunk. The output is an app folder
with its own uv venv, lessons split into cached sections, a reveal.js version with a contents
sidebar, and Makefile targets.

The user lectures from these on video, so two things matter more than polish: **every chunk must be
small enough to talk through**, and **nothing on screen may disagree with the source**.

## 1. Read the source, then the repo

Read the whole source before asking anything — every notebook cell *and its outputs/comments*
(those comments are often the clearest statement of the intermediate states you'll animate).
Then look at what exists: earlier lectures (`*/…-manim/`), the root `Makefile` (naming, the shared
`Q ?= --hd` default), and `references/patterns.md` for designs that already worked.

Notice what the source does *not* contain. If the lecture needs a piece the source lacks (e.g. the
notebook packs but never unpacks), say so and ask which version to follow — a course's own code
beats one you write from memory, because the user will show both side by side.

## 2. Ask preferences first (one AskUserQuestion call)

Ask these four, each with 2–4 concrete options grounded in *this* material, your pick first and
marked "(Recommended)". Use previews/descriptions that name the real functions and values.

1. **Code on screen or formulas only?** Code + visuals side by side (active line highlighted,
   token boxed, live variables) suits algorithm walkthroughs; formulas only suits derivations
   (earlier whisper/mtp lectures). A third option: visuals throughout, annotated code on a recap slide.
2. **Step size per → press.** One press per *operation* (e.g. load → shift → OR ≈ 3 per loop
   iteration) is usually right; per micro-step mirrors the source's commentary exactly but gets
   long; per iteration is fast but leaves little to talk over.
3. **Scope / which lessons** (multi-select): motivation ("why does this exist"), the core worked
   example, a second example that exercises an outer loop/edge, failure cases (shape checks,
   errors), the inverse operation, a recap/round trip.
4. **Visual encoding**: the layout of the central object and the colour code (one colour per
   concept, reused in cells, chips, code tokens, captions). Name the classic confusion and how
   the encoding resolves it (e.g. "MSB-left bytes with slots coloured per value, so the reversed
   order is visible").

Decide the rest yourself and state each call in one line (app location `<topic>/<slug>-manim`,
make name `<slug>-manim`, example values, lesson titles).

## 3. Propose the lesson plan, get a yes

A short table: lesson → sections → what each chunk shows (≈ a few presses each). Include where
roadmap slides go: one before each subtopic the learner would want to jump to (they said lectures
are hard to talk through "without knowing what is coming"), but **not** between continuing steps
of one picture (iteration 2, 3 of the same loop) — those sections must join invisibly. Wait for
approval or corrections before building.

## 4. Scaffold

```bash
bash .Codex/skills/manim-lecture/scripts/new_lecture.sh <topic>/<slug>-manim <slug>-manim "<Lecture title>"
```

Copies `assets/scaffold/` (style.py with `Lesson`/roadmap/lazy beats, render.py, sidebar.py,
template lesson, toy.py, tests, code_panel.py), creates the venv, appends the Makefile block
(`<slug>-manim`, `-html`, `-render`, `-test`, `-install`, `-clean`). Then:
- fill `TEX_COLORS` / add a value→colour map in style.py from the encoding answer;
- delete `code_panel.py` if the user chose formulas only;
- set `render.LESSONS` as lessons get written.

## 5. toy.py and its tests first

`toy.py` holds the algorithm (framework-free copy of the source when possible — plain ints
reproduce uint8 exactly) plus a **trace**: one dataclass per step the lecture shows. Lessons only
draw the trace; nothing on screen is typed by hand. Write `tests/test_toy.py` against the
source's own numbers (notebook outputs, comment states, paper tables) and watch it fail before
implementing. Add a test for every rule the visuals depend on (e.g. which slots stay occupied
after a shift), because a rebuilt section scene is drawn from those rules.

## 6. Widgets and the stage

If a picture persists across several sections (a loop), build a stage class with
`build(k)` → state after step k-1, `restore(scene, k, animate=False)` → add it instantly with
the same caption and code highlight the previous section ended on, and a per-step method that
plays the beats. Share each "last caption" between the step and `restore` via one helper.
See `references/patterns.md` (Trace-driven stage, Bit/cell widgets, Code on screen).

Smoke-render a throwaway scene with every widget before writing lessons, and look at it.

## 7. Lessons, one at a time

For each lesson: write it → `uv run python render.py L<N> --quality=-l` → contact sheet →
read the frames → fix → commit.

```bash
bash .Codex/skills/manim-lecture/scripts/sheet.sh <app> L2_ /tmp/l2.png 3   # then Read the png
```

Read each tile as the learner will: the last frame of a beat is what sits on screen while the
user talks. Check that it shows the state the caption claims, labels don't collide, and no
`SAFE-AREA` lines were printed (everything must stay inside the 16:9 frame; the players
letterbox, so don't fit to one screen).

Style rules that the user has asked for across lectures:
- labels ≤ 6 words, big type, shapes on diagrams, no citations;
- show change as before → after, not as a sentence; be explicit about *what* changes;
- one colour per concept everywhere; real numbers from toy.py in every example;
- captions at the bottom via `self.say()`, one per beat.

Read `references/gotchas.md` before the first lesson — especially: no `self.wait()` (slide
cuts drift), never swap children of on-screen groups (reset with `become()`), no section named
`setup`.

## 8. Verify joins, finish, spin it up

```bash
uv run --project <app> python .Codex/skills/manim-lecture/scripts/join_check.py <app>   # SSIM ≥ 0.98 at every non-roadmap join
make <slug>-manim-test
make <slug>-manim-render            # 1080p once, so the first `make <slug>-manim` starts instantly
make <slug>-manim-render            # again: must say "0 to render"
```

Then tell the user how to run it:
- `make <slug>-manim` — fullscreen player, → / ← to step, `L=L2` for one lesson, `Q=-l` for fast 480p;
- `make <slug>-manim-html` — browser version; **T** toggles the contents sidebar (one heading per
  lesson, one link per roadmap topic).

Commit as you go (one commit per lesson is a good rhythm). Save any new gotcha you hit to
`references/gotchas.md` so the next lecture doesn't pay for it again.
