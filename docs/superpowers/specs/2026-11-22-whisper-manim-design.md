# whisper-manim — Manim lecture slides for audio → Whisper

**Intent.** Turn lectures 1–3 + 5 of the audio course into 3Blue1Brown-style (ManimGL) animations the
lecturer steps through with next/prev, then switches to `whisper-app` for the code walkthrough.

## Approaches
- **ManimGL + manim-slides (picked).** The 3b1b engine; manim-slides records a segment per
  `next_slide()` and plays them with ←/→. Each slide is also an mp4 for editing.
- Manim Community + manim-slides — better docs, but not the 3b1b engine the user wants.
- ManimGL interactive only — no real prev/next.

## Decisions (user)
- Scope: L1 waveform & sampling, L2 Fourier → STFT, L3 mel → log-mel, L5 Whisper encoder (+ brief decoder/training).
- Audio: the same synthetic melody as whisper-app; Whisper tiny numbers (16 kHz, n_fft 400, hop 160, 201 bins, 80 mels, D 384, 4+4 layers).
- Formulas only, no code on screen.
- Separate app at `vision-audio/whisper-manim/`, Makefile targets.

## Content order (follows AudioSignalProcessingForML, with Whisper numbers)
- L1: sound wave (freq/amp/phase) → sampling (t=nT, sr) → sample-rate vs frequency content → Nyquist/aliasing → int16 quantization → stereo→mono, 44.1k→16k resample → pad to 30 s.
- L2: probe-sine "multiply & area" → winding / complex form → DFT (bins k·sr/N, mirror, 201 bins) → framing → leakage → Hann → hop & reflect padding → STFT [201, 3001→3000] → resolution trade-off.
- L3: power |X|² → mel curve (HTK for intuition, Slaney overlay = Whisper's) → 5-step filterbank → [80,201]@[201,3000] → log10, max−8, (x+4)/4.
- L5: conv1 (k3) → conv2 (k3, s2): 3000→1500 → sinusoidal positions → 4 blocks → [1500,384] → decoder cross-attn + next-token loss trains the encoder.

## Constraints / style
Dark bg, whisper-app palette, labels ≤ 6 words, shapes on diagrams, big type, no paper citations,
same numbers persist across consecutive slides. Numbers from a NumPy port of `whisper-app/src/dsp.js`.

## Non-goals
Lessons 4, 6–8; mic input; on-screen code; MFCC/RMS/ZCR.

## Left to the builder
Exact animations per beat, layout, venv/tooling details.
