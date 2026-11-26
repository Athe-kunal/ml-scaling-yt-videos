# Floating-point stepper app (design)

## Intent
A step-by-step React app that teaches IEEE-754 floats, following "Exposing Floating Point"
(ciechanow.ski) and float.exposed (`b0x44c0` = bf16 0x44c0 = 1536). The user records videos
from it next to the quantization lectures.

## Goals / non-goals
- Goal: prev/next (← →) guided steps, each with a live float.exposed-style bit editor.
- Goal: exact numbers (BigInt), with tests against the article's values.
- Non-goal: manim slides; double/fp8; printing (%a) and exact-decimal derivation chapters.
- Anti-goal: hand-typed numbers that disagree with the bit pattern on screen; walls of text.

## Approaches
1. **Copy whisper-app shell (Vite + React 19, theme, StepShell)** (picked): consistent with the other apps.
2. New layout: drifts from house style, for no gain.

## Decided (user answers)
- Scope: binary & sci. notation; encoding; specials/subnormals/spacing; fp32 vs fp16 vs bf16 + rounding.
- Interaction: stepper + live bit editor on every step (click bits; reset to the step's bits).
- Examples: the article's numbers (9.3125, 0.2, −2343.53125, 16777217, 2097151.875,
  −282960.039306640625) + bf16 0x44c0 = 1536 (fp16 reading: 4.75).
- Folder `quantization/floating_point_numbers`; `make float` / `make float-install`, port 5185.

## Left to the builder
Layout details, colours (sign / exponent / significand), animation timing.
