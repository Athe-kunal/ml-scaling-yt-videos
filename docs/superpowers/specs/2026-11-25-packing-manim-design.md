# Packing 2-bit weights — Manim lecture (design)

## Intent
Turn `quantization/L5_packing_2bit_weights.ipynb` into an arrow-key (manim-slides) lecture the
user steps through to understand the packing algorithm, chunk by chunk. Launched from the root
Makefile like `mtp-manim` / `whisper-manim`.

## Goals / non-goals
- Goal: every bit operation of `pack_weights` visible, one → press per operation (load / shift / OR).
- Goal: code shown side by side with the bits (user choice; unlike the formula-only mtp/whisper lectures).
- Non-goal: refactoring a shared Manim library across apps.
- Anti-goal: numbers on screen that disagree with the notebook's function.

## Approaches
1. **Copy the mtp scaffold into `quantization/packing-manim/`** (picked) — proven pipeline, zero risk to other apps.
2. Shared library extracted from whisper/mtp — refactor not asked for, risks the existing lectures.
3. Add lessons to the mtp app — wrong topic, couples unrelated lectures.

## Decided
- Lessons: L1 Why pack (wasted 6 bits, 4×, shape check + ValueError on 5 values);
  L2 pack `[1,0,3,2]` → 177, reveal slots read `[2|3|0|1]`; L3 `[1,0,3,2,3,3,3,3]` → `[177,255]`,
  byte 0 fast replay, outer loop + `unpacked_idx` carry; L4 unpack via `(p >> 2j) & 0b11`.
- Each lesson opens with a roadmap slide.
- Byte = 8 cells, MSB left; 2-bit slots colored by value (1 blue, 0 grey, 3 gold, 2 pink), same colors in code/tensor.
- `toy.py` holds `pack_weights` (notebook copy) + `unpack_weights`; all on-screen states derived from it;
  pytest checks intermediate states vs the notebook comments.
- Makefile: `packing-manim[-html|-render|-test|-install|-clean]`, `L=` / `Q=` like mtp.

## Left to the builder
Exact layout coordinates, animation timing, code highlight style.
