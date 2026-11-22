"""Whisper's audio.py in NumPy — the single source of numbers for every scene.

Same constants and the same synthetic melody as whisper-app/src/dsp.js, so the slides
and the code walkthrough show identical values.
"""
import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000
N_FREQ = N_FFT // 2 + 1  # 201
N_MELS = 80
PAD = N_FFT // 2  # center=True reflect padding

# Whisper tiny
D_MODEL, N_HEAD, N_ENC_LAYERS, N_DEC_LAYERS = 384, 6, 4, 4
N_AUDIO_CTX, N_TEXT_CTX, N_VOCAB = 1500, 448, 51865

TAU = 2 * np.pi
NOTES = [(60, 0.0), (64, 0.45), (67, 0.9), (72, 1.35), (71, 1.9), (67, 2.25), (69, 2.7), (64, 3.15)]


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def melody():
    """4 s of plucked notes with 8 harmonics each, at 16 kHz → float32[64000]."""
    n = round(4.0 * SAMPLE_RATE)
    y = np.zeros(n, np.float64)
    length = round(0.8 * SAMPLE_RATE)
    for m, t0 in NOTES:
        f, s0 = midi_hz(m), round(t0 * SAMPLE_RATE)
        L = min(length, n - s0)
        t = np.arange(L) / SAMPLE_RATE
        env = np.minimum(1, t / 0.008) * np.exp(-t * 4.5)
        v = np.zeros(L)
        for h in range(1, 9):
            if f * h < 7800:
                v += np.sin(TAU * f * h * t) / (h * h * 0.7 + 0.3)
        y[s0:s0 + L] += 0.22 * env * v
    return y.astype(np.float32)


def to_int16(y):
    return np.clip(np.round(y * 32768), -32768, 32767).astype(np.int16)


def from_int16(q):
    return (q.astype(np.float32) / 32768).astype(np.float32)


def pad_or_trim(y, length=N_SAMPLES):
    out = np.zeros(length, np.float32)
    k = min(len(y), length)
    out[:k] = y[:k]
    return out


def hann(n=N_FFT):
    """torch.hann_window(n) — periodic: 0.5 − 0.5·cos(2πi/n)."""
    return (0.5 - 0.5 * np.cos(TAU * np.arange(n) / n)).astype(np.float32)


def frames(y):
    """center=True: reflect-pad 200 each side, slide 400 by 160 → [3001, 400]."""
    yp = np.pad(y, PAD, mode="reflect")
    n = 1 + (len(yp) - N_FFT) // HOP_LENGTH
    idx = np.arange(N_FFT)[None, :] + HOP_LENGTH * np.arange(n)[:, None]
    return yp[idx]


def stft(y):
    """torch.stft(y, 400, 160, window=hann, return_complex=True) → complex [201, 3001]."""
    return np.fft.rfft(frames(y) * hann(), axis=1).T


def stft_power(y):
    """stft[..., :-1].abs() ** 2 → [201, 3000]."""
    return np.abs(stft(y)[:, :-1]) ** 2


def bin_hz(k):
    return k * SAMPLE_RATE / N_FFT  # 40 Hz per bin


# ---------------------------------------------------------------- mel scales
F_SP, MIN_LOG_HZ = 200 / 3, 1000.0
MIN_LOG_MEL = MIN_LOG_HZ / F_SP  # 15
LOGSTEP = np.log(6.4) / 27


def hz_to_mel(f):
    """Slaney mel (librosa default, what Whisper uses): linear < 1 kHz, log above."""
    f = np.asarray(f, np.float64)
    return np.where(f >= MIN_LOG_HZ, MIN_LOG_MEL + np.log(np.maximum(f, 1e-9) / MIN_LOG_HZ) / LOGSTEP, f / F_SP)


def mel_to_hz(m):
    m = np.asarray(m, np.float64)
    return np.where(m >= MIN_LOG_MEL, MIN_LOG_HZ * np.exp(LOGSTEP * (m - MIN_LOG_MEL)), F_SP * m)


def hz_to_mel_htk(f):
    """The textbook formula: m = 2595·log10(1 + f/700)."""
    return 2595 * np.log10(1 + np.asarray(f, np.float64) / 700)


def mel_filters(n_mels=N_MELS):
    """librosa.filters.mel(sr=16000, n_fft=400, n_mels) → W [n_mels, 201], edge points in Hz [n_mels+2]."""
    m_max = hz_to_mel(SAMPLE_RATE / 2)
    pts = mel_to_hz(np.linspace(0, m_max, n_mels + 2))
    f = bin_hz(np.arange(N_FREQ))
    lo, c, hi = pts[:-2, None], pts[1:-1, None], pts[2:, None]
    W = np.maximum(0, np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)))
    W *= 2 / (hi - lo)  # Slaney norm: equal area
    return W.astype(np.float32), pts


def log_mel(y30, n_mels=N_MELS):
    """Every stage of log_mel_spectrogram for a 480000-sample signal."""
    power = stft_power(y30)
    W, pts = mel_filters(n_mels)
    mel = W @ power
    log = np.log10(np.maximum(mel, 1e-10))
    floor = np.maximum(log, log.max() - 8.0)
    scaled = (floor + 4.0) / 4.0
    return dict(power=power, mel=mel, log=log, floor=floor, scaled=scaled, W=W, pts=pts, max=float(log.max()))


_CACHE = {}


def whisper_input():
    """The melody, int16-quantised and padded to 30 s, with all log-mel stages (cached)."""
    if "A" not in _CACHE:
        y = from_int16(to_int16(melody()))
        y30 = pad_or_trim(y)
        _CACHE["A"] = dict(y=y, y30=y30, **log_mel(y30))
    return _CACHE["A"]
