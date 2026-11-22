import numpy as np
import dsp


def test_constants():
    assert (dsp.SAMPLE_RATE, dsp.N_FFT, dsp.HOP_LENGTH, dsp.N_FREQ) == (16000, 400, 160, 201)
    assert (dsp.N_SAMPLES, dsp.N_FRAMES, dsp.N_MELS) == (480000, 3000, 80)


def test_melody_and_int16_roundtrip():
    y = dsp.melody()
    assert y.shape == (64000,) and y.dtype == np.float32
    q = dsp.to_int16(y)
    assert q.dtype == np.int16
    assert np.max(np.abs(dsp.from_int16(q) - y)) <= 0.5 / 32768 + 1e-7


def test_pad_or_trim():
    assert dsp.pad_or_trim(np.ones(10, np.float32)).shape == (480000,)
    assert dsp.pad_or_trim(np.ones(500000, np.float32)).shape == (480000,)


def test_hann_is_periodic():
    w = dsp.hann()
    assert w.shape == (400,) and w[0] == 0 and abs(w[200] - 1) < 1e-6 and w[-1] > 0


def test_stft_frames_and_power():
    y = dsp.pad_or_trim(dsp.from_int16(dsp.to_int16(dsp.melody())))
    X = dsp.stft(y)                       # complex, center=True reflect pad
    assert X.shape == (201, 3001)
    P = dsp.stft_power(y)                 # last frame dropped
    assert P.shape == (201, 3000) and P.min() >= 0


def test_pure_tone_lands_in_its_bin():
    t = np.arange(16000) / 16000
    y = dsp.pad_or_trim(np.sin(2 * np.pi * 1000 * t).astype(np.float32))
    P = dsp.stft_power(y)
    assert int(np.argmax(P[:, 50])) == 25          # 1000 Hz / 40 Hz per bin


def test_slaney_mel_scale():
    assert abs(dsp.hz_to_mel(1000) - 15.0) < 1e-9
    assert abs(dsp.hz_to_mel(500) - 7.5) < 1e-9     # linear below 1 kHz
    for f in [100, 999, 1000, 4000, 8000]:
        assert abs(dsp.mel_to_hz(dsp.hz_to_mel(f)) - f) < 1e-6
    assert abs(dsp.hz_to_mel_htk(700) - 2595 * np.log10(2)) < 1e-9


def test_mel_filters():
    W, pts = dsp.mel_filters()
    assert W.shape == (80, 201) and pts.shape == (82,)
    peaks = W.argmax(axis=1)
    assert np.all(np.diff(peaks) >= 0)
    assert W.min() >= 0
    # Slaney norm: each triangle has area 1 in Hz (approximately, sampled at 40 Hz)
    assert np.all(W.sum(axis=1)[20:] * 40 > 0.8)


def test_log_mel_stages():
    y = dsp.pad_or_trim(dsp.from_int16(dsp.to_int16(dsp.melody())))
    L = dsp.log_mel(y)
    assert L["power"].shape == (201, 3000)
    assert L["mel"].shape == (80, 3000)
    assert L["log"].min() >= -10
    assert abs((L["floor"].max() - L["floor"].min()) - 8) < 1e-5
    assert np.allclose(L["scaled"], (L["floor"] + 4) / 4)
    assert -1.5 < L["scaled"].min() and L["scaled"].max() < 2
