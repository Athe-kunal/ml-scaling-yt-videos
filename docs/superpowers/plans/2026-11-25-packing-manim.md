# Packing 2-bit Weights — Manim Lecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An arrow-key manim-slides lecture (4 lessons) that steps through `pack_weights` / `unpack_weights` bit by bit, launched with `make packing-manim`.

**Architecture:** Copy of the `mtp/soft-mtp-manim` scaffold (Lesson/beat/roadmap, section-cached parallel `render.py`, reveal.js sidebar) into `quantization/packing-manim/`. `toy.py` is the single source of truth: it runs the notebook's algorithm and records a trace of every intermediate state; lessons only draw that trace. `bits.py` holds the two reusable widgets (`Byte`, `CodePanel`).

**Tech Stack:** ManimGL 1.7.2, manim-slides ≥5.7 (pyqt6), numpy, uv, pytest. No torch (plain Python ints reproduce uint8 exactly).

**Spec:** `docs/superpowers/specs/2026-11-25-packing-manim-design.md`

## Global Constraints

- Own uv venv in `quantization/packing-manim/`; do NOT add it to any root uv workspace. Deps identical to `mtp/soft-mtp-manim/pyproject.toml` (`setuptools<81`).
- One → press per operation inside a loop iteration: **load → shift → OR** (pack), **shift → mask** (unpack).
- Byte = 8 cells, MSB (bit 7) on the left; 2-bit slot k = bits (2k+1, 2k).
- Value colors everywhere (cells, code tokens, tensor chips): `0 → DIM`, `1 → BLUE`, `2 → PINK`, `3 → GOLD`. Empty/unused bits = `RULE` outline, no fill.
- Labels ≤ 6 words, big type, everything inside the 16:9 frame (no SAFE-AREA lines in render output).
- Every number on screen comes from `toy.py`; never type a byte state by hand.
- ManimGL gotchas (from the existing apps): `Text(color=)` ignored → use `style.T`/`tag`; ManimGL `Arrow` → `style.arrow()`; `clear_updaters()` before fading live mobjects; don't render with plain `manimgl` (hangs) — use `render.py`.

## Review Focus

1. Slot order reversed on screen (value 0 drawn in the leftmost slot) — test `toy.slots(177) == [1,0,3,2]` (slot 0 = rightmost) and Byte draws slot 0 at the right (Task 2 check).
2. L3 byte 1 must start from `unpacked_idx = 4`, not 0 — trace test asserts `idx` sequence `0..7`.
3. `5 values` shape error must use the notebook's exact rule `n * bits % 8 != 0` — test raises `ValueError` for 5, passes for 4 and 8.
4. Unpack mask must remove the higher slots, not just shift — trace test asserts the post-shift value for j=0 is 177 (not 1) and post-mask is 1.
5. Code highlight line numbers drifting from the code text — `CodePanel` looks lines up by a marker substring, not hard-coded indices (Task 2).

---

### Task 1: Scaffold + `toy.py` (algorithm and trace)

**Files:**
- Create: `quantization/packing-manim/pyproject.toml` (copy of mtp's, `name = "packing-manim"`)
- Create: `quantization/packing-manim/style.py` (copy of mtp's; delete the mtp colour-code block and `TEX_COLORS` entries, keep `TEX_COLORS = {}`; drop `heat/colormap/envelope/stems/vec_rows/wave/graph/c2p` — unused; add `VALUE_COLOR = {0: DIM, 1: BLUE, 2: PINK, 3: GOLD}`)
- Create: `quantization/packing-manim/render.py` (copy; `LESSONS = [("l1_why","L1_Why"),("l2_pack","L2_Pack"),("l3_two_bytes","L3_TwoBytes"),("l4_unpack","L4_Unpack")]`; shared fingerprint files `("style.py","toy.py","bits.py")`)
- Create: `quantization/packing-manim/sidebar.py` (verbatim copy)
- Create: `quantization/packing-manim/toy.py`
- Test: `quantization/packing-manim/tests/test_toy.py`

**Interfaces:**
- Produces (`toy.py`):
  - `pack_weights(values: list[int], bits: int) -> list[int]` — notebook algorithm, same `ValueError` text.
  - `unpack_weights(packed: list[int], bits: int) -> list[int]` — `(p >> bits*j) & (2**bits - 1)`.
  - `@dataclass PackStep: i:int, j:int, idx:int, value:int, shift:int, shifted:int, before:int, after:int`
  - `pack_trace(values, bits=2) -> list[PackStep]`
  - `@dataclass UnpackStep: i:int, j:int, idx:int, shift:int, before:int, shifted:int, masked:int`
  - `unpack_trace(packed, bits=2) -> list[UnpackStep]`
  - `bits8(n:int) -> list[int]` (MSB first, len 8); `slots(n:int, bits=2) -> list[int]` (slot 0 = lowest bits)
  - Constants: `EX1 = [1,0,3,2]`, `EX2 = [1,0,3,2,3,3,3,3]`, `BAD = [1,0,3,2,1]`

- [ ] **Step 1: Write failing tests**

```python
import pytest, toy

def test_pack_matches_notebook():
    assert toy.pack_weights(toy.EX1, 2) == [177]
    assert toy.pack_weights(toy.EX2, 2) == [177, 255]

def test_trace_states_match_notebook_comments():
    t = toy.pack_trace(toy.EX1)
    assert [s.after for s in t] == [0b00000001, 0b00000001, 0b00110001, 0b10110001]
    assert [s.shifted for s in t] == [0b1, 0b0, 0b00110000, 0b10000000]
    assert [s.shift for s in t] == [0, 2, 4, 6]

def test_two_byte_trace_carries_idx():
    t = toy.pack_trace(toy.EX2)
    assert [s.idx for s in t] == list(range(8))
    assert [s.i for s in t] == [0]*4 + [1]*4
    assert t[4].before == 0 and t[-1].after == 255

def test_shape_check():
    with pytest.raises(ValueError):
        toy.pack_weights(toy.BAD, 2)

def test_unpack_roundtrip_and_mask():
    assert toy.unpack_weights([177], 2) == toy.EX1
    assert toy.unpack_weights([177, 255], 2) == toy.EX2
    u = toy.unpack_trace([177])
    assert u[0].shifted == 177 and u[0].masked == 1
    assert [s.shifted for s in u] == [177, 44, 11, 2]

def test_bits_and_slots():
    assert toy.bits8(177) == [1,0,1,1,0,0,0,1]
    assert toy.slots(177) == [1,0,3,2]
```

- [ ] **Step 2:** `cd quantization/packing-manim && uv sync && uv run pytest -q` → FAIL (no `toy`).
- [ ] **Step 3:** Implement `toy.py`; copy the other scaffold files as listed.
- [ ] **Step 4:** `uv run pytest -q` → all pass; `uv run python -c "import style"` → no error.
- [ ] **Step 5:** Commit `feat(packing-manim): scaffold + toy trace`.

### Task 2: Widgets `bits.py` (`Byte`, `CodePanel`, `TensorRow`)

**Files:**
- Create: `quantization/packing-manim/bits.py`
- Create: `quantization/packing-manim/l0_smoke.py` (temporary scene, deleted at end of task)

**Interfaces:**
- Consumes: `style.*`, `toy.bits8`, `toy.slots`.
- Produces:
  - `class Byte(VGroup)`: `Byte(n:int=0, cell=0.62, label:str|None=None, colored=True)`; children `cells[8]` (Square), `digits[8]` (tag "0"/"1"), bit-index row `7…0` above; when `colored`, 2-bit slot k cells filled `VALUE_COLOR[slots(n)[k]]` at 0.35 opacity, only when that slot is "occupied" (`occupied: set[int]` kwarg, default all). Methods: `set_value_anim(n, occupied) -> Animation` (digits + fills cross-fade), `slot_brace(k, text) -> VGroup`, `slot_cells(k) -> VGroup`.
  - `def shift_anim(byte: Byte, k: int) -> list[Animation]` — moves digit glyphs k cells left; glyphs that leave bit 7 fade out red (`BAD`), zeros fade in from the right. One animation for the whole shift (≈1.2 s).
  - `class CodePanel(VGroup)`: `CodePanel(src: str, size=24)` → Menlo lines, Python keywords SOFT/ACCENT; `highlight(marker: str) -> Animation` moves a translucent ACCENT bar to the first line containing `marker`; `line(marker) -> Text`.
  - `class VarBox(VGroup)`: live readout rows `i = 0`, `j = 2`, `idx = 2`, `shift = 2·j = 4`; `update_anim(**vals) -> Animation`.
  - `class TensorRow(VGroup)`: `TensorRow(values)` chips colored by `VALUE_COLOR`, `pointer_anim(idx)` moves a ▼ marker.
  - `PACK_SRC`, `UNPACK_SRC`: code strings (PACK_SRC = notebook's `pack_weights` minus its scratch comments; UNPACK_SRC = mirror with `>>` and `& 3`).

- [ ] **Step 1:** Write `l0_smoke.py` scene `L0_Smoke(Lesson)` with `SECTIONS=["show"]`: draws `CodePanel(PACK_SRC)` left, `Byte(177)` right with `slot_brace(0,"value 0")`, plays `highlight("|=")`, `shift_anim` by 4 on `Byte(3)`, then `beat()`.
- [ ] **Step 2:** Render: `uv run manim-slides render --GL l0_smoke.py L0_Smoke -l` → exit 0, no `SAFE-AREA` lines. Extract last frame (`ffmpeg -sseof -0.1 -i videos/L0_Smoke/*.mp4 -frames:v 1 /tmp/x.png`) and look at it: slot 0 (blue, value 1) on the RIGHT, slot 3 (pink) on the LEFT, highlight on the `|=` line.
- [ ] **Step 3:** Delete `l0_smoke.py`, `videos/L0_Smoke`, `slides/L0_Smoke*`. Commit `feat(packing-manim): byte/code widgets`.

### Task 3: L1 Why pack — `l1_why.py`

**Interfaces:** Consumes `bits.Byte`, `TensorRow`, `toy.EX1`, `toy.BAD`, `toy.pack_weights`.
`class L1_Why(Lesson)`: `NUM=1, TITLE="Why pack?", SUB="2-bit weights in 8-bit boxes"`, `SECTIONS=["title","waste","four_x","shape"]`, `TOPICS={"waste":"6 of 8 bits wasted","four_x":"four weights, one byte","shape":"when packing fails"}`; `split_sections(L1_Why, globals())` at bottom.

Beats:
- `waste`: `TensorRow([1,0,3,2])` → each as its own uint8 `Byte` (4 stacked) with the top 6 bits greyed/crossed → caption "only 2 of 8 bits used".
- `four_x`: 4 bytes slide together into one `Byte(177)` (only slot fills, no algorithm yet) → `4 bytes → 1 byte` / `4× smaller` tag.
- `shape`: `M("n \\cdot b \\,/\\, 8")` with n=4 → 1 ✓, n=8 → 2 ✓, n=5 → 1.25 ✗ → red `ValueError` chip; one beat each.

- [ ] Step 1: write lesson. Step 2: `uv run python render.py L1 --quality=-l` → all `ok`, no SAFE-AREA. Step 3: check last frame of each section video. Step 4: commit.

### Task 4: L2 Pack one byte — `l2_pack.py`

**Interfaces:** Consumes `CodePanel(PACK_SRC)`, `VarBox`, `TensorRow`, `Byte`, `shift_anim`, `toy.pack_trace(EX1)`.
`class L2_Pack(Lesson)`: `NUM=2, TITLE="Pack one byte", SUB="[1, 0, 3, 2] → 177"`, `SECTIONS=["title","setup","iter0","iter1","iter2","iter3","reveal"]`, `TOPICS={"setup":"the loop variables","iter0":"j = 0: no shift","iter1":"j = 1: zero changes nothing","iter2":"j = 2: shift by 4","iter3":"j = 3: shift by 6","reveal":"read it back"}`.

Layout (persistent across iter sections — rebuild identical layout at section start from the trace so section scenes join): code left (x ≈ −3.6), right column top→bottom: `TensorRow` with pointer, `VarBox`, "value" `Byte` (scratch), "packed" `Byte`.

Per iteration `iterJ` (shared method `self.iteration(step)`), 3 beats:
1. **load** — highlight line `uint8tensor[unpacked_idx]`, pointer to idx, scratch byte shows `bits8(value)`; VarBox updates j, idx, shift.
2. **shift** — highlight `<< (bits * j)`, `shift_anim(scratch, step.shift)`; caption "shift left by 2·j".
3. **OR** — highlight `|=`, scratch copy drops onto packed byte, packed `set_value_anim(step.after, occupied=slots 0..j)`; caption e.g. "0000 0001 | 0011 0000". `setup` section: highlights `num_values`, `num_steps`, `zeros`, shows packed=0.
`reveal`: braces over the 4 slots read `2 | 3 | 0 | 1`, arrow showing reversal vs `[1,0,3,2]`; then `= 128+32+16+1 = 177`.

- [ ] Steps: write → `render.py L2 --quality=-l` ok, no SAFE-AREA → inspect last frames of iter2 (packed = 0011 0001) and reveal (177) → commit.

### Task 5: L3 Two bytes — `l3_two_bytes.py`

**Interfaces:** reuses `L2_Pack.iteration` logic — move `iteration(step)` and the layout builder into `bits.py` as `PackStage` (`PackStage(values).build() -> VGroup`, `iteration_beats(scene, step, fast=False)`) during this task and switch L2 to it (re-render L2 to confirm unchanged).
`class L3_TwoBytes(Lesson)`: `NUM=3, TITLE="Two bytes", SUB="[1,0,3,2,3,3,3,3] → [177, 255]"`, `SECTIONS=["title","outer","byte0_fast","byte1","result"]`, `TOPICS={"outer":"the outer loop i","byte0_fast":"byte 0 replay","byte1":"byte 1 from idx 4","result":"two packed bytes"}`.
- `outer`: highlight `for i`; packed tensor shown as 2 `Byte`s, active one framed.
- `byte0_fast`: `fast=True` → each iteration one animation, one beat for the whole byte.
- `byte1`: i=1, idx pointer jumps to 4 (call it out: "idx keeps counting"), full 3-beat iterations (4 × 3 beats).
- `result`: `[177, 255]` with slot readouts.

- [ ] Steps: write → render L2+L3 at `-l` → inspect → `uv run pytest -q` → commit.

### Task 6: L4 Unpack — `l4_unpack.py`

**Interfaces:** `CodePanel(UNPACK_SRC)`, `toy.unpack_trace([177])`, `Byte`, `shift_anim` (add `direction` kwarg: `-1` = right shift, bits fall off bit 0 in red).
`class L4_Unpack(Lesson)`: `NUM=4, TITLE="Unpack", SUB="177 → [1, 0, 3, 2]"`, `SECTIONS=["title","mask","j0","j1","j2","j3","roundtrip"]`, `TOPICS={"mask":"the mask 0b11","j0":"j = 0: just mask","j1":"j = 1: shift 2, mask","j2":"j = 2: shift 4, mask","j3":"j = 3: shift 6, mask","roundtrip":"pack → unpack = identity"}`.
Per j: beat **shift** (`>> 2j`, scratch byte), beat **mask** (`& 3`: mask byte `0000 0011` under it, top 6 bits turn grey/zero, result chip appended to the output `TensorRow`). `mask` section explains why shifting alone isn't enough (j=0 shifted = 177, not 1).

- [ ] Steps: write → render L4 → inspect j0 last frame (output row `[1]`) → commit.

### Task 7: Makefile + full verify

**Files:** Modify `Makefile` (append after mtp block).

```make
# ------------------------------------------------------------ Packing 2-bit weights (ManimGL + manim-slides, ← / →)
#   make packing-manim                   render what changed, then present all 4 lessons
#   make packing-manim L=L2              one lesson (any part of a scene name: L2_Pack, iter2, ...)
#   make packing-manim Q=-l              fast 480p while editing (default: --hd 1080p)
#   make packing-manim-html              same slides in the browser (T = contents sidebar)
.PHONY: packing-manim packing-manim-html packing-manim-install packing-manim-render packing-manim-test packing-manim-clean
PACK_DIR := quantization/packing-manim
PACK_ARGS = $(L) --quality=$(Q)
```
plus `-install` (uv sync), `-render`, bare (`--present`), `-html`, `-test` (pytest), `-clean` (rm slides media videos) — same recipes as mtp with `PACK_*`.

- [ ] `make packing-manim-test` → pass. `make packing-manim-render Q=-l` → every section `ok`, zero SAFE-AREA. `make packing-manim-html Q=-l` → `wrote …/lectures.html`. Run the no-op render again → "0 to render".
- [ ] Add `.gitignore` in `packing-manim` matching mtp's (slides/, media/, videos/) if mtp has one. Commit `feat: packing-manim make targets`.
