"""Turn raw per-bin signals into a 0..1 highlight score per bin.

Signals are percentile-ranked across *all* clips of the session, so the best
moments of the whole day win, not the best moment of each file.
"""

from __future__ import annotations

import numpy as np

from .config import Config


def _rank(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Percentile rank (0..1) of each value among ``values[mask]``; NaN stays NaN.

    Ties get the mid-rank, so a long run of identical values (a still frame)
    lands in the middle of its block instead of at its top.
    """
    out = np.full(values.shape, np.nan)
    ref = values[mask & np.isfinite(values)]
    if ref.size < 2:
        return out
    ref = np.sort(ref)
    fin = np.isfinite(values)
    lo = np.searchsorted(ref, values[fin], side="left")
    hi = np.searchsorted(ref, values[fin], side="right")
    out[fin] = (lo + hi) / 2 / ref.size
    return out


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or x.size == 0:
        return x
    k = np.ones(n) / n
    pad = n // 2
    xp = np.pad(x, (pad, n - 1 - pad), mode="edge")
    return np.convolve(xp, k, mode="valid")


def score_session(signals: list[dict[str, np.ndarray]], cfg: Config) -> tuple[list[np.ndarray], list[dict]]:
    """Return (scores per clip, per-clip debug info with idle/flip masks)."""
    d, w = cfg.detect, cfg.weights
    lengths = [len(s["motion"]) for s in signals]
    cat = {k: np.concatenate([s[k] for s in signals]) for k in signals[0] if k != "sync"}   # "sync" is clip metadata, not a per-bin signal

    luma = cat["luma"]
    usable = np.isfinite(luma) & (luma > d.dark_luma) & (luma < d.blown_luma)

    motion = cat["motion"]
    motion_rank_all = _rank(motion, usable)
    gyro = cat["gyro_mean"]
    idle = np.zeros_like(usable)
    still = np.nan_to_num(motion, nan=0.0) < d.idle_motion_abs
    low_motion = still | (np.nan_to_num(motion_rank_all, nan=0.0) < d.idle_motion_pct)
    has_gyro = np.isfinite(gyro)
    # On the ground / hovering still: no rotation and little image change.
    idle |= has_gyro & (gyro < d.idle_gyro_dps) & low_motion
    # Without gyro, only a practically frozen image counts as idle.
    idle |= ~has_gyro & (np.nan_to_num(motion, nan=0.0) < d.idle_motion_abs / 2)
    active = usable & ~idle

    parts: dict[str, np.ndarray] = {
        "gyro": _rank(cat["gyro_peak"], active),
        "motion": _rank(cat["motion"], active),
        "audio": _rank(cat["audio_db"], active),
        "sharp": _rank(cat["sharp"], active),
    }
    flip = cat["flip_deg"]
    span = max(d.flip_full_deg - d.flip_min_deg, 1e-6)
    parts["flip"] = np.where(np.isfinite(flip), np.clip((flip - d.flip_min_deg) / span, 0, 1), np.nan)

    weights = {"gyro": w.gyro, "flip": w.flip, "motion": w.motion, "audio": w.audio, "sharp": w.sharp}
    num = np.zeros(luma.shape)
    den = np.zeros(luma.shape)
    for name, arr in parts.items():
        ok = np.isfinite(arr)
        num[ok] += weights[name] * arr[ok]
        den[ok] += weights[name]
    raw = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
    raw[~active] = 0.0

    scores, info = [], []
    start = 0
    for n in lengths:
        sl = slice(start, start + n)
        s = _smooth(raw[sl], d.smooth_bins)
        s[~active[sl]] = 0.0
        scores.append(np.clip(s, 0, 1))
        info.append({
            "idle": idle[sl],
            "unusable": ~usable[sl],
            "flip": np.nan_to_num(flip[sl], nan=0.0) >= (d.flip_min_deg + d.flip_full_deg) / 2,
        })
        start += n
    return scores, info
