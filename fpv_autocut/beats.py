"""Minimal beat tracker (numpy only): spectral-flux onsets + constant-tempo grid.

Good enough for the steady-tempo music typical of FPV edits. For music with
tempo changes, cut on beats in Resolve instead (or pass --no-beat-sync).
"""

from __future__ import annotations

import subprocess

import numpy as np

from .config import Music

SR = 22050
HOP = 512
NFFT = 2048
PRIOR_WIDTH_OCT = 0.9   # width of the tempo prior, in octaves


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    pcm = np.frombuffer(subprocess.run(cmd, check=True, capture_output=True).stdout, dtype=np.float32)
    if pcm.size == 0:
        raise ValueError(f"{path}: no audio decoded")
    return pcm


def onset_envelope(y: np.ndarray) -> np.ndarray:
    n_frames = 1 + max(0, (len(y) - NFFT) // HOP)
    window = np.hanning(NFFT).astype(np.float32)
    flux = np.zeros(n_frames)
    prev = None
    chunk = 512  # frames per FFT batch keeps memory ~10 MB for any song length
    for c0 in range(0, n_frames, chunk):
        c1 = min(n_frames, c0 + chunk)
        idx = np.arange(NFFT)[None, :] + HOP * np.arange(c0, c1)[:, None]
        mag = np.log1p(np.abs(np.fft.rfft(y[idx] * window, axis=1)) * 10).astype(np.float32)
        if prev is not None:
            mag_all = np.vstack([prev[None, :], mag])
        else:
            mag_all = np.vstack([mag[:1], mag])
        flux[c0:c1] = np.maximum(0, np.diff(mag_all, axis=0)).sum(axis=1)
        prev = mag[-1]
    flux -= np.convolve(flux, np.ones(16) / 16, mode="same")  # remove slow trend
    flux = np.maximum(flux, 0)
    return flux / (flux.max() + 1e-9)


def estimate_tempo(env: np.ndarray, fps: float, lo_bpm: float = 70, hi_bpm: float = 180,
                   prior_bpm: float = 120) -> float:
    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    lags = np.arange(len(ac))
    bpm = np.where(lags > 0, 60 * fps / np.maximum(lags, 1), 0)
    valid = (bpm >= lo_bpm) & (bpm <= hi_bpm)
    # Prefer tempi near ``prior_bpm`` (log-gaussian prior) to avoid half/double picks.
    prior = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1) / prior_bpm) / PRIOR_WIDTH_OCT) ** 2)
    weighted = np.where(valid, ac * prior, -np.inf)
    lag = int(np.argmax(weighted))
    return 60 * fps / lag


def track_beats(path: str, music: Music | None = None) -> tuple[float, np.ndarray]:
    """Return (bpm, beat times in seconds)."""
    m = music or Music()
    y = load_audio(path)
    env = onset_envelope(y)
    fps = SR / HOP
    coarse = estimate_tempo(env, fps, m.bpm_min, m.bpm_max, m.bpm_prior)
    # The autocorrelation lag is an integer number of frames (~1 % BPM error,
    # i.e. ~0.6 s drift per minute). Refine BPM and phase jointly on the grid.
    x = np.arange(len(env))
    best = (-1.0, coarse, 0.0)
    for bpm in np.arange(coarse * 0.97, coarse * 1.03, 0.05):
        period = 60 * fps / bpm
        for phase in np.linspace(0, period, num=max(8, int(period * 2)), endpoint=False):
            pos = np.arange(phase, len(env) - 1, period)
            val = np.interp(pos, x, env).mean()
            if val > best[0]:
                best = (val, bpm, phase)
    _, bpm, phase = best
    period = 60 * fps / bpm
    beats = np.arange(phase, len(env) - 1, period) / fps
    return float(bpm), beats
