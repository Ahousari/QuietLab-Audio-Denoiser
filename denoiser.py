"""Audio loading, speech-pause suggestions, spectral denoising, and plotting helpers."""
from __future__ import annotations

import io
import warnings
import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter
import soundfile as sf


def load_audio(blob: bytes) -> tuple[np.ndarray, int]:
    """Decode supported audio bytes, returning float32 [samples, channels]."""
    try:
        data, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
        if data.size == 0:
            raise ValueError("The file contains no audio samples.")
    except Exception:
        # librosa delegates compressed formats (notably MP3) to soundfile or
        # audioread/FFmpeg depending on the local installation.
        import librosa
        data, sr = librosa.load(io.BytesIO(blob), sr=None, mono=False)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        else:
            data = data.T
    if not np.isfinite(data).all():
        data = np.nan_to_num(data)
    return data.astype(np.float32, copy=False), int(sr)


def _mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def detect_quiet_regions(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Suggest sufficiently long low-energy intervals using adaptive RMS threshold."""
    x = _mono(audio)
    frame = max(1, int(sr * .025))
    hop = max(1, int(sr * .010))
    if len(x) < frame * 2:
        return []
    n = 1 + (len(x) - frame) // hop
    rms = np.empty(n, dtype=np.float32)
    for i in range(n):
        part = x[i * hop:i * hop + frame]
        rms[i] = np.sqrt(np.mean(part * part) + 1e-12)
    db = 20 * np.log10(rms + 1e-8)
    # Adaptive estimate: choose valleys relative to the recording's level.
    floor = float(np.percentile(db, 20))
    ceiling = float(np.percentile(db, 90))
    threshold = min(floor + 9.0, ceiling - 8.0)
    threshold = max(threshold, -65.0)
    mask = db < threshold
    # Bridge tiny interruptions, then keep intervals at least 300 ms long.
    bridge = max(1, int(.12 * sr / hop))
    for i in range(1, bridge + 1):
        if len(mask) > 2 * i:
            mask[i:-i] |= mask[:-2*i] & mask[2*i:]
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    regions = []
    for a, b in zip(starts, ends):
        start, end = a * hop / sr, min((b * hop + frame) / sr, len(x) / sr)
        if end - start >= .30:
            regions.append((max(0, start), end))
    return regions[:100]


def _stft(x: np.ndarray, sr: int):
    nfft = min(2048, max(512, 2 ** int(np.ceil(np.log2(max(512, sr * .046))))))
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    noverlap = int(nfft * .75)
    f, t, z = signal.stft(x, fs=sr, window="hann", nperseg=nfft,
                          noverlap=noverlap, boundary="zeros", padded=True)
    return f, t, z, nfft, noverlap


def _noise_profile(audio: np.ndarray, sr: int, regions: list[tuple[float, float]]) -> np.ndarray:
    blocks = []
    n = len(audio)
    for a, b in regions:
        i, j = max(0, int(a * sr)), min(n, int(b * sr))
        if j > i:
            blocks.append(audio[i:j])
    if not blocks:
        raise ValueError("Select at least one non-empty noise region.")
    sample = np.concatenate(blocks)
    if len(sample) < int(sr * .12):
        raise ValueError("Noise sample is too short; select at least 0.12 seconds.")
    # Compute per-channel spectrum, then use a robust median over frames.
    channels = sample[:, None] if sample.ndim == 1 else sample
    profiles = []
    for c in range(channels.shape[1]):
        _, _, z, _, _ = _stft(channels[:, c], sr)
        profiles.append(np.median(np.abs(z), axis=1))
    return np.mean(profiles, axis=0)


def denoise_audio(audio: np.ndarray, sr: int, regions: list[tuple[float, float]],
                  strength: float = 4, floor_db: float = -12,
                  smoothing: float = .55) -> np.ndarray:
    """Linked-channel soft spectral reduction with a smoothed Wiener-style mask."""
    source = np.asarray(audio, dtype=np.float32)
    if source.ndim == 1:
        source = source[:, None]
    if source.shape[0] < 1:
        return source.copy()
    profile = _noise_profile(source, sr, regions)
    spectra, meta = [], None
    for c in range(source.shape[1]):
        f, t, z, nfft, noverlap = _stft(source[:, c], sr)
        spectra.append(z)
        meta = (f, t, nfft, noverlap)
    # Same channel-linked mask preserves stereo image and avoids channel drift.
    mag = np.mean([np.abs(z) for z in spectra], axis=0)
    eps = 1e-9
    # Subtract a fraction of estimated noise power; strength > 1 increases it.
    residual = np.maximum(mag ** 2 - max(0.0, strength) * profile[:, None] ** 2, 0.0)
    raw_gain = np.sqrt(residual / (mag ** 2 + eps))
    floor = 10 ** (floor_db / 20)
    gain = floor + (1.0 - floor) * raw_gain
    # Frequency smoothing reduces musical-noise speckles; time smoothing avoids pumping.
    sigma_f = smoothing * 1.2
    sigma_t = smoothing * 1.8
    if smoothing > 0:
        gain = gaussian_filter(gain, sigma=(sigma_f, sigma_t), mode="nearest")
    gain = np.clip(gain, floor, 1.0)
    out = []
    for z in spectra:
        _, y = signal.istft(z * gain, fs=sr, window="hann", nperseg=meta[2],
                            noverlap=meta[3], input_onesided=True, boundary=True)
        if len(y) < len(source):
            y = np.pad(y, (0, len(source) - len(y)))
        out.append(y[:len(source)])
    result = np.stack(out, axis=1).astype(np.float32)
    # Retain headroom if a codec/container overshoot created a peak above unity.
    peak = float(np.max(np.abs(result))) if result.size else 0
    if peak > 1.0:
        result /= peak
    return result if audio.ndim == 2 else result[:, 0]


def make_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    data = np.asarray(audio, dtype=np.float32)
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def waveform_envelope(x: np.ndarray, sr: int, points: int = 6000):
    step = max(1, int(np.ceil(len(x) / points)))
    chunks = [x[i:i + step] for i in range(0, len(x), step)]
    y = np.array([np.max(np.abs(c)) for c in chunks])
    t = (np.arange(len(y)) * step + step / 2) / sr
    return t, y


def frequency_curves(original: np.ndarray, sr: int, regions, processed=None):
    n = min(8192, max(512, 2 ** int(np.floor(np.log2(max(512, len(original)))))))
    f, p = signal.welch(original, sr, nperseg=min(n, len(original)))
    orig = 10 * np.log10(p + 1e-14)
    noise = None
    if regions:
        try:
            profile = _noise_profile(original[:, None], sr, regions)
            # Profile is amplitude; convert to comparable dB magnitude curve.
            fp = np.linspace(0, sr / 2, len(profile))
            noise = 20 * np.log10(np.maximum(np.interp(f, fp, profile), 1e-8))
        except (ValueError, IndexError):
            pass
    proc = None
    if processed is not None:
        pmono = _mono(processed)
        _, pp = signal.welch(pmono, sr, nperseg=min(n, len(pmono)))
        proc = 10 * np.log10(pp + 1e-14)
    valid = f > 0
    return f[valid], orig[valid], noise[valid] if noise is not None else None, proc[valid] if proc is not None else None
