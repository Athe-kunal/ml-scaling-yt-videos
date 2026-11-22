# whisper-manim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** ManimGL + manim-slides lecture app (L1, L2, L3, L5) with prev/next stepping, started from the Makefile.
**Architecture:** `dsp.py` (NumPy port of whisper-app/src/dsp.js) is the only source of numbers; `style.py` holds palette/fonts/helpers; one scene file per lesson, each a `manim_slides.Slide` subclass using `self.next_slide()` per beat.
**Tech Stack:** Python 3.11 uv venv inside the app; manimgl, manim-slides[pyqt6], numpy, pytest.
**Spec:** docs/superpowers/specs/2026-11-22-whisper-manim-design.md

## Global Constraints
- SR 16000, N_FFT 400, HOP 160, N_FREQ 201, N_SAMPLES 480000, N_FRAMES 3000, N_MELS 80, tiny D 384, 4+4 layers, 6 heads, 1500 audio ctx.
- Palette: bg #0A0E14, ink #F5F8FB, dim #94A1B2, accent #6EF3DD, pink #FF8AC6, gold #FFC94D, rule #334053.
- Labels ≤ 6 words, no code, no citations; shapes on diagrams.

## Review Focus
- Slaney mel matrix must equal librosa's (row sums / peak positions) — test against analytic Slaney formula.
- STFT frames: 3001 with center reflect pad, last dropped → 3000.
- log-mel range after scale ≈ [−1, 1.x]; max − floor ≤ 8 before scaling.
- Each scene must render under `manim-slides render --GL` without error (smoke render at low quality).
- Make targets work from repo root.

### Task 1: venv + dsp.py + tests
Files: `vision-audio/whisper-manim/{pyproject.toml,dsp.py,tests/test_dsp.py}`
Produces: `melody()->f32[64000]`, `to_int16`, `from_int16`, `pad_or_trim(y, n=N_SAMPLES)`, `hann(n=400)`, `stft_power(y)->[201,3000]`, `hz_to_mel_slaney`, `mel_to_hz_slaney`, `hz_to_mel_htk`, `mel_filters(n_mels=80)->[80,201]`, `log_mel(y)->dict(power, mel, log, floor, scaled)`.
Tests: shapes; hann[0]==0 & periodic; filter peak freqs monotonic; slaney mel(1000)=15; scaled max ≤ (max+4)/4 and floor gap = 8.

### Task 2: style.py + L1_Waveform scene
### Task 3: L2_Fourier scene (probe/winding/DFT/framing/Hann/STFT)
### Task 4: L3_LogMel scene
### Task 5: L5_Encoder scene
Each: render with `manim-slides render --GL <file> <Scene>` → exit 0; spot-check frames.

### Task 6: Makefile targets `whisper-manim-install`, `whisper-manim` (L=all default), commit.
