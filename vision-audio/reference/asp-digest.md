# Audio Signal Processing for ML — digest (musikalkemist / Valerio Velardo)

Source: `reference/AudioSignalProcessingForML/` (shallow clone, `main` = "v2" code; README notes the video-era code lives on the `legacy` branch, so cell code may differ slightly from the videos). No code was run. Shapes below are the outputs saved in the notebooks, or derived from file headers (`afinfo`) where marked *derived*.

Stack (requirements.txt): librosa 0.11.0, numpy 2.2.6, scipy 1.16.1, matplotlib 3.10.6.

Slides: every lesson folder except 08, 09, 14, 16, 18, 20, 22, 23 has a PDF. The text is extractable with `pdftotext`. Formulas are images; the ones below were read from rendered pages.

---

## 0. Audio files (`audio_resources/`)

| file | on disk (ch, sr, fmt, dur) | after `librosa.load` (sr=22050, mono) | used in |
|---|---|---|---|
| debussy.wav | 1ch, 22050, int16, 30.0 s | `(661500,)` | 08, 09, 16, 20, 22, 23 |
| redhot.wav | 1ch, 22050, int16, 30.0 s | (661500,) *derived* | 08, 09, 16, 22, 23 |
| duke.wav | 1ch, 22050, int16, 30.0 s | (661500,) *derived* | 08, 09, 16, 22, 23 |
| voice.wav | 1ch, 22050, int16, 30.0 s | loaded with `duration=15` → (330750,) *derived* | 09 |
| noise.wav | **2ch, 48000**, float32, 26.37 s | `duration=15` in 09; full in 14 → downmixed + resampled | 09, 14 |
| piano_c.wav (C5, ~523 Hz) | **2ch, 44100**, int16, 1.54 s | `(33968,)` | 03, 10, 14 |
| violin_c.wav (C4) | **2ch, 44100**, int16, 2.71 s | `(59772,)` | 03, 14 |
| sax.wav (C4) | 2ch, 44100, int16, 1.31 s | (~28810,) *derived* | 14 |
| scale.wav | 2ch, 44100, **24-bit**, 7.93 s | (~174.9k,) → STFT 342 frames | 16, 18 |
| tremolo.wav | 2ch, 44100, int16, 14.1 s | played only | 03 |

**Mono/stereo and resampling.** No lesson covers resampling directly. Every notebook calls `librosa.load(path)` with the defaults `sr=22050, mono=True`. That call quietly averages stereo to mono and resamples 44.1k/48k down to 22.05k. This is a good "hidden step" to animate: stereo (2, N) at 44100 → mono (N,) at 44100 → resample → (≈N/2,) at 22050.

---

## Lesson 02 — Sound and waveforms (slides only)
Concepts, in the order taught:
1. Sound is the vibration of an object. Air molecules oscillate, the air pressure changes, and this makes a mechanical wave.
2. The waveform carries frequency, intensity and timbre. Sounds can be periodic or aperiodic.
3. Sine parameters: period/frequency, amplitude, phase. Higher frequency → higher pitch. Larger amplitude → louder.
4. Hearing range is about 20 Hz–20 kHz. Pitch is perceived logarithmically: an octave doubles the frequency.
5. MIDI notes: A4 = 440 Hz, A5 = 880 Hz. Pitch-to-frequency map. Cents: 1200 per octave, 100 per semitone. The smallest noticeable difference is 10–25 cents.

Formulas (standard ones the slides show as images):
- $y(t) = A\sin(2\pi f t + \varphi)$, $f = 1/T$
- $F(p) = 2^{\frac{p-69}{12}}\cdot 440$ (MIDI p → Hz); ratio per semitone $2^{1/12}$

## Lesson 03 — Intensity, loudness, timbre (slides + `intensity_and_timbre.ipynb`)
Slides: sound power (W) → intensity (W/m²) → intensity level in dB, $IL = 10\log_{10}(I/I_{TOH})$, where $I_{TOH}=10^{-12}\,\mathrm{W/m^2}$. Intensity doubles every ~3 dB. Loudness is measured in phons (equal-loudness contours). Timbre: ADSR envelope, partials/harmonics/inharmonicity, vibrato (FM), tremolo (AM).
Notebook:
- `violin_c4, _ = librosa.load(...)`, `piano_c5, _ = ...`
- `librosa.amplitude_to_db(np.abs(librosa.stft(signal)))` (stft defaults: n_fft=2048, hop=512, Hann), shown with `specshow(..., y_axis="log")`
- `X = np.fft.fft(violin_c4)`; `X_mag = np.absolute(X)`; `f = np.linspace(0, sr, len(X_mag))`. `len(violin_c4)=59772`.

## Lesson 04 — Understanding audio signals (slides only) ★ sampling / Nyquist / aliasing / quantization
Order:
1. Analog signal (continuous time and amplitude) vs digital signal (discrete values, finite set).
2. ADC = **sampling + quantization** (pulse-code modulation).
3. Sampling period $T$. Locating samples: $t_n = n\cdot T$. Sampling rate $s_r = 1/T$. Picture: low vs high sample rate.
4. Why 44100 Hz? **Nyquist frequency** $f_N = \dfrac{s_r}{2}$. For CD: $f_N = 44100/2 = 22050$ Hz, which is above the ~20 kHz hearing limit.
5. **Aliasing**: a sine above $f_N$, once sampled, looks like a lower-frequency sine. Slide figure: a red fast sine with sample dots that trace a slow black sine.
6. **Quantization**: resolution = number of bits (bit depth). CD = 16 bits → $2^{16}=65536$ levels.
7. Memory for 1 min, mono, 44.1k, 16-bit: $((16\cdot 44100)/1048576/8)\cdot 60 = 5.05\,\text{MB}$.
8. Dynamic range: more resolution → more dynamic range. SQNR: $SQNR(16)\approx 96\,\text{dB}$ (≈6.02 dB per bit).
9. Recording: mic → ADC. Playback: DAC → speaker.

## Lesson 05 — Types of audio features (slides)
Features are grouped by abstraction level, temporal scope (instantaneous ~50 ms / segment / global), music aspect, and signal domain:
- time domain: AE, RMS, ZCR
- frequency domain: BER, spectral centroid, flux
- time-frequency: spectrogram, mel-spectrogram, CQT

The talk ends with traditional ML (hand-made features) vs deep learning (features learned from spectrograms).

## Lesson 06 — How to extract audio features (slides) ★ framing / windowing / overlap
Order:
1. Time-domain pipeline: ADC → **framing** (e.g. frame 1: samples 1…128, frame 2: 64…192, …) → feature per frame → aggregation (mean, median, GMM) → feature value/vector/matrix.
2. Frames: 1 sample at 44.1 kHz = 0.0227 ms, far below the ear's ~10 ms resolution. Frame sizes are powers of 2, typically 256–8192. Example: $512/44100 = 11.6$ ms.
3. Frequency-domain pipeline: framing → **spectral leakage**. A frame that is not a whole number of periods has discontinuous endpoints, and these show up as false high-frequency components after the FT.
4. **Windowing** fixes leakage. The window tapers the frame ends to zero. **Hann**: $w(k) = 0.5\left(1-\cos\frac{2\pi k}{K-1}\right),\ k=1\dots K$. Figure: noise frame × Hann → frame with tapered ends.
5. "Houston, another problem": windowing throws away the samples at the frame edges → **overlapping frames**. Parameters: frame size $K$ and hop length.
6. Then FT → feature computation → aggregation.

## Lesson 07 — Time-domain features (slides), 08 & 09 (notebooks)
Formulas (K = frame size, s(k) = k-th sample, frame t covers samples $tK \dots (t+1)K-1$):
- $AE_t = \max_{k=tK}^{(t+1)K-1} s(k)$. Rough loudness measure, sensitive to outliers.
- $RMS_t = \sqrt{\frac{1}{K}\sum_{k=tK}^{(t+1)K-1} s(k)^2}$
- $ZCR_t = \frac{1}{2}\sum_{k=tK}^{(t+1)K-1}\left|\operatorname{sgn}(s(k)) - \operatorname{sgn}(s(k+1))\right|$. Each crossing adds |−1−(+1)| = 2, halved to 1.

**08 notebook** (`Implementing the amplitude envelope.ipynb`): `debussy, sr = librosa.load(...)` → `(661500,)`. Sample duration `1/sr = 0.000045` s; total duration `661500/22050 = 30.0` s. `waveshow(..., alpha=0.5)`. `FRAME_SIZE=1024, HOP_LENGTH=512`. AE loop: `max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)` → `len=1292`. Time axis: `librosa.frames_to_time(frames, hop_length=512)`.
**09 notebook**: `librosa.feature.rms(y=..., frame_length=1024, hop_length=512)[0]` is checked against a from-scratch `np.sqrt(sum(frame**2)/frame_size)`. `librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=512)[0]` → size 1292. Then voice vs noise (both `duration=15`): noise has a much higher ZCR.

## Lesson 10 — Fourier Transform: the intuition (slides + `Fourier Transform.ipynb`) ★
Slides, in order:
1. Break a complex sound into its frequency components (time domain → FT → frequency domain).
2. Compare the signal with sinusoids of many frequencies. Each frequency gives a magnitude and a phase. High magnitude means the signal is very similar to that sinusoid.
3. Step by step: **choose a frequency → optimise the phase → compute the magnitude**.
   - $\varphi_f = \arg\max_{\varphi\in[0,1)} \int s(t)\sin(2\pi(ft-\varphi))\,dt$
   - $d_f = \max_{\varphi\in[0,1)} \int s(t)\sin(2\pi(ft-\varphi))\,dt$
   - In words: multiply the signal by the sinusoid → take the area → pick the phase with the largest area.
4. Rebuild the signal by adding sinusoids weighted by magnitude, with their phases (inverse FT, additive synthesis). Original signal and FT hold the same information.

Notebook (piano_c.wav, `(33968,)`, sr=22050):
- `ft = sp.fft.fft(signal)`; `magnitude = np.absolute(ft)`; `frequency = np.linspace(0, sr, len(magnitude))`; plot `[:5000]` → peak at ~523 Hz (C5).
- `d = 1/sr = 4.535e-05`; period of 523 Hz is `1/523 = 0.00191` s; 400 samples = 0.0181 s. Zoom with `t = librosa.samples_to_time(range(len(signal)), sr=sr)`, slice `[10000:10400]`.
- Probe sinusoid: `0.5*np.sin(2*np.pi*(f*t - phase))`, with `f=523` and `phase=0` vs `0.2`. Then `0.1*sin(...)` with `phase=0.55` overlaid on the signal, and `fill_between(t, sin*signal)` shades the product area. **This is the key animation.**
- Superposition: `t=np.linspace(0,10,10000)`, sines at 1, 2, 3 Hz summed.

## Lesson 11 — Complex numbers (slides)
$c = a + ib$ (real, imaginary). Polar form: $|c|=\sqrt{a^2+b^2}$, $\gamma=\arctan(b/a)$, $c=|c|(\cos\gamma + i\sin\gamma)$. Euler: $e^{i\gamma}=\cos\gamma+i\sin\gamma$. Euler identity: $e^{i\pi}+1=0$. So $c=|c|e^{i\gamma}$: $|c|$ scales the distance from the origin and $\gamma$ sets the direction.

## Lesson 12 — FT with complex numbers (slides + notebook) ★ magnitude / phase
Slides, in order:
1. Store magnitude and phase as polar coordinates in one complex number: $c_f = \frac{d_f}{\sqrt2} e^{-i2\pi\varphi_f}$.
2. Continuous signal $g:\mathbb{R}\to\mathbb{R}$.
3. Complex FT: $\hat g(f) = \int g(t)\,e^{-i2\pi ft}dt = \int g(t)\cos(-2\pi ft)dt + i\int g(t)\sin(-2\pi ft)dt$.
4. Magnitude: $|\hat g(f)|$. Then $d_f = \sqrt2\,|\hat g(f)|$ and $\varphi_f = -\frac{\gamma_f}{2\pi}$.
5. Inverse FT: $g(t)=\int c_f\, e^{i2\pi ft}df$. Weight each pure tone by its magnitude, add its phase, and sum ("Fourier roundtrip").

Notebook ("winding" visual):
- `create_signal(f, t) = sin(2πft)+sin(2π·2ft)+sin(2π·3ft)`
- `create_pure_tone(f,t) = cos(-2πft) + 1j*sin(-2πft)`
- `mult_signal = pure_tone * signal` is scattered in the complex plane. Its centre of gravity (mean of re/im) and its sum are the FT coefficient.
- Demo: `time=np.linspace(0,1,10000)`, `pure_tone_frequency=1.1`, `signal_frequency=1`, xlim/ylim ±3.

## Lesson 13 — Discrete Fourier Transform (slides) ★ frequency bins
1. Digitalisation: $t_n = nT$, so $x(n)$ replaces $g(t)$.
2. First attempt: $\hat x(f)=\sum_n x(n)e^{-i2\pi fn}$ (a Riemann-sum picture). Problem: continuous frequency and infinite time.
3. **Hack 1 (time)**: the signal is non-zero only in a finite window $x(0),\dots,x(N-1)$.
4. **Hack 2 (frequency)**: compute only $M=N$ frequencies. This keeps the transform invertible and fast.
5. **DFT**: $\hat x(k) = \sum_{n=0}^{N-1} x(n)\,e^{-i2\pi n\frac{k}{N}},\ k=0..N-1$.
6. **Bin → Hz**: $F(k) = \dfrac{k}{NT} = \dfrac{k\,s_r}{N}$.
7. **Redundancy**: $k=N/2 \Rightarrow F(N/2)=s_r/2$ = Nyquist. The spectrum mirrors around it, so only bins $0..N/2$ are kept. Figure: full `np.fft` magnitude, symmetric about the centre.
8. DFT cost is $O(N^2)$. FFT cost is $O(N\log_2 N)$; it reuses redundancy across sinusoids and works best when N is a power of 2.

## Lesson 14 — Extracting the DFT (`Visualising the Power Spectrum.ipynb`)
- Loads violin_c4 (sr returned = 22050), piano_c5, sax_c4, noise.
- `len(violin_c4) = 59772`; `X = np.fft.fft(violin_c4)` → `len(X) = 59772` (N bins = N samples).
- `plot_magnitude_spectrum(signal, sr, title, f_ratio=0.1)`: `f = np.linspace(0, sr, len(X_mag))`; plot the first `int(len*f_ratio)` bins (0–2205 Hz). Shows harmonic peaks for the instruments and a flat spread for noise.

## Lesson 15 — STFT explained (slides) ★★
1. FT problem: **"we know WHAT, we don't know WHEN"**.
2. STFT idea: slide a window, take a DFT per frame. Window size = frame size (the slides note they can differ, with zero-padding).
3. Windowing → overlapping frames with **hop size H**.
4. DFT → STFT:
   $$\hat x(k)=\sum_{n=0}^{N-1}x(n)e^{-i2\pi n\frac{k}{N}}\quad\longrightarrow\quad S(m,k)=\sum_{n=0}^{N-1}x(n+mH)\,w(n)\,e^{-i2\pi n\frac{k}{N}}$$
   ($m$ = frame index, $mH$ = start sample of the current frame; frames m=1, 2, 3 slide along.)
5. Outputs: DFT → spectral vector (N complex coefficients). STFT → complex spectral **matrix** (#bins, #frames):
   - $\#\text{bins} = \frac{\text{framesize}}{2}+1$ (covering 0 … $s_r/2$)
   - $\#\text{frames} = \frac{\text{samples}-\text{framesize}}{\text{hopsize}}+1$
   - **Worked example**: 10k samples, frame 1000, hop 500 → bins = 501, frames = (10000−1000)/500+1 = 19 → **STFT (501, 19)**.
6. Parameters:
   - frame size: 512, 1024, 2048, 4096, 8192
   - hop: 256 … 4096, or ½K, ¼K, ⅛K
   - window function: Hann
7. **Time/frequency trade-off**: a larger frame gives finer frequency resolution but coarser time resolution (and the reverse).
8. Spectrogram = $|S(m,k)|^2$ shown as a heatmap (time × freq × colour = dB).

⚠ For animation: librosa's `stft` uses `center=True` by default. It pads n_fft/2 on each side, so frames = `1 + len // hop` (debussy: 1 + 661500//512 = **1292**), not the slide formula (which gives 1288).

## Lesson 16 — Extracting spectrograms (notebook) ★★
- `scale, sr = librosa.load(scale_file)` (sr = 22050); also debussy, redhot, duke.
- `FRAME_SIZE = 2048`, `HOP_SIZE = 512`
- `S_scale = librosa.stft(scale, n_fft=2048, hop_length=512)` → **`(1025, 342)`**, dtype `complex64` (1025 = 2048/2+1).
- `Y_scale = np.abs(S_scale) ** 2` → `(1025, 342)` `float32` (power spectrogram).
- `specshow(Y, sr=sr, hop_length=512, x_axis="time", y_axis="linear")` → almost all black: a linear amplitude scale hides the detail.
- **Log-amplitude**: `Y_log_scale = librosa.power_to_db(Y_scale)` ($10\log_{10}(Y/\text{ref})$, ref=1, top_db=80).
- **Log-frequency**: same plot with `y_axis="log"`.
- Genres: `power_to_db(np.abs(librosa.stft(x, n_fft=2048, hop_length=512))**2)` for debussy / redhot / duke with `y_axis="log"` (shape (1025, 1292)).

So the order taught is complex STFT → power → dB → log-freq axis.

## Lesson 17 — Mel spectrograms explained (slides) ★★
1. Psychoacoustic test: C2→C4 (65→262 Hz) and G6→A6 (1568→1760 Hz) are both ~200 Hz apart, yet they sound very different. **Humans perceive frequency logarithmically.**
2. An ideal feature has a time-frequency layout, perceptual amplitude (dB), and perceptual frequency (mel).
3. **Mel scale**: a log scale where equal distances have equal perceptual distance. 1000 Hz = 1000 mel.
   - slide: $m = 2595\cdot\log\left(1+\frac{f}{500}\right)$ ← **slide typo**. The standard HTK form is $m = 2595\log_{10}\left(1+\frac{f}{700}\right)$, which matches the inverse on the same slide.
   - inverse: $f = 700\left(10^{m/2595}-1\right)$
4. Recipe: (1) extract STFT, (2) convert amplitude to dB, (3) convert frequencies to mel.
5. Converting to mel: choose #bands (40, 60, 90, 128 — "depends on the problem") → build the filterbank → apply it.
6. **Building the filterbank**:
   1. convert the lowest and highest frequency to mel
   2. make #bands equally spaced points in mel (#bands+2 points are needed for the triangle edges)
   3. convert the points back to Hz
   4. round each to the nearest frequency bin
   5. make triangular filters (rise from point i−1 to a peak at i, fall to i+1)

   Figure: 6 triangles, 1000–8000 Hz, peaks labelled 1263…2577 mel. The triangles get wider as frequency rises.
7. Filterbank shape: **M = (#bands, framesize/2+1)**.
8. Apply: $Y$ = (framesize/2+1, #frames) → **Mel spectrogram = M·Y**, shape (#bands, #frames).
9. Uses: audio classification, mood, genre, instrument.

## Lesson 18 — Extracting mel spectrograms (notebook) ★★
- `scale, sr = librosa.load(scale_file)` (22050)
- `filter_banks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=10)` → **`(10, 1025)`**; shown with `specshow(filter_banks, sr=sr, x_axis="linear")`.
- `mel_spectrogram = librosa.feature.melspectrogram(y=scale, sr=sr, n_fft=2048, hop_length=512, n_mels=10)` → **`(10, 342)`** (power=2 by default, so this is $M\cdot|S|^2$).
- `log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)` → `(10, 342)`; `specshow(..., x_axis="time", y_axis="mel", sr=sr)`.
- Note: librosa's defaults are `htk=False` (Slaney mel: linear below 1 kHz, log above) and `norm="slaney"` (area-normalised triangles), `fmin=0`, `fmax=sr/2`. These are not the HTK formula on the slide.

## Lesson 19 — MFCCs explained (slides) ★
1. Cepstrum terms are anagrams: spectrum→**cepstrum**, frequency→**quefrency**, filtering→**liftering**, harmonic→**rhamonic**. History: seismic echoes (1960s) → speech (1970s) → music (2000s).
2. $C(x(t)) = \mathcal{F}^{-1}\left[\log\left(\mathcal{F}[x(t)]\right)\right]$. Steps: signal → DFT → power spectrum → log → IDFT → cepstrum (peak = 1st rhamonic, i.e. the pitch period).
3. Speech model: glottal pulse (source) convolved with the vocal tract (filter).
   - $x(t)=e(t)*h(t)$ → $X=E\cdot H$ → $\log X = \log E + \log H$
   - The log turns the product into a sum, so the parts can be split.
   - Spectral envelope (formants, slow) is the vocal tract and carries the identity of the sound. Spectral detail (fast) is the glottal pulse.
   - In quefrency, low quefrency = envelope and high = pitch.
4. **MFCC pipeline**: waveform → DFT → log-amplitude spectrum → **mel-scaling** → **DCT** → MFCCs.
5. Why DCT: it is a simplified real-valued FT. It decorrelates the mel bands and compresses the spectrum into fewer dimensions.
6. How many coefficients: traditionally the first **12–13**, which hold the envelope and formants. Add Δ and ΔΔ → **39 per frame**.
7. Pros: captures the large spectral structure and ignores the fine detail; works for speech and music. Cons: not robust to noise, heavily hand-engineered, poor for synthesis.

DCT-II (librosa default, `type=2, norm='ortho'`): $c_n = \sum_{m=0}^{M-1} \log S_m\cos\left[\frac{\pi n}{M}\left(m+\tfrac12\right)\right]$

## Lesson 20 — Extracting MFCCs (notebook)
- `signal, sr = librosa.load(debussy.wav)` → 661500 samples at 22050.
- `mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)` → **`(13, 1292)`**. Implicit defaults: n_fft=2048, hop=512, n_mels=128, power_to_db, DCT-II ortho.
- `delta_mfccs = librosa.feature.delta(mfccs)` → (13, 1292); `delta2 = librosa.feature.delta(mfccs, order=2)`.
- `np.concatenate((mfccs, delta, delta2))` → **`(39, 1292)`**.

## Lessons 21–23 — frequency-domain features (brief; less relevant)
- 21 (slides): BER, spectral centroid, bandwidth.
- 22: `stft(n_fft=2048, hop=512)` → `(1025, 1292)`. The split bin is `floor(split_freq / ((sr/2)/num_bins))`, and `calculate_split_frequency_bin(2000, 22050, 1025) = 185`. ⚠ `band_energy_ratio` passes `len(spectrogram[0])` (= #frames, 1292) as num_bins, which is a bug in the repo. $BER_t = \sum_{n<F} m_t(n)^2 / \sum_{n\ge F} m_t(n)^2$.
- 23: `spectral_centroid` / `spectral_bandwidth(y, sr, n_fft=1024, hop_length=512)[0]` → `(1292,)`.

---

## Scene skeleton (suggested order for animation)
1. **Wave → samples**: continuous sine; sample dots at $t_n=nT$; $s_r=1/T$.
2. **Quantization**: snap dots to $2^b$ levels (b=3 for a visual, then 16 → 65536); SQNR ≈ 6.02·b dB.
3. **Nyquist/aliasing**: sine above $s_r/2$ is sampled and the dots trace a lower alias.
4. **Mono/resample**: stereo (2, N) at 44.1k → mean → (N,) → resample to 22050 (piano_c: 33968 samples).
5. **FT intuition**: piano C5 × probe sine 523 Hz; shade the product; sweep the phase; area → magnitude.
6. **Complex winding**: signal × $e^{-i2\pi ft}$ in the complex plane; centre of gravity = coefficient; magnitude/phase as polar coordinates.
7. **DFT bins**: N samples → N bins; $F(k)=k s_r/N$; mirror at N/2; keep N/2+1.
8. **Framing + Hann + hop**: frames of 2048, hop 512; leakage without the window vs with Hann.
9. **STFT matrix**: columns stacked → (1 + N_fft/2, frames) = (1025, 342) for scale.wav; toy (501, 19).
10. **Spectrogram**: |S|² → power_to_db → log-freq axis.
11. **Mel**: Hz→mel curve; equally spaced mel points → back to Hz → bins → triangles (10 × 1025); M·Y → (10, 342).
12. **Log-mel**: power_to_db.
13. **MFCC**: log-mel column → DCT → keep 13; + Δ, ΔΔ → 39 × 1292.
