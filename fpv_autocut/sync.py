"""Automatic gyro <-> video time alignment (numpy only).

The camera's gyro log and its video do not always start on the same instant.
Both signals describe the same physical motion, so the rotation rate from the
gyro and the frame-to-frame image change from the video rise and fall together.
The lag that maximises their correlation is the offset to apply to the gyro
timeline (``gyro_features(..., offset=...)`` adds it to the gyro timestamps).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gyro import GyroLog


@dataclass
class SyncResult:
    offset: float        # seconds to add to gyro timestamps
    corr: float          # Pearson correlation at the best lag (0..1)
    accepted: bool       # False when the match was too weak to trust


def gyro_rate_series(log: GyroLog, fps: float, lowpass_hz: float = 50.0) -> np.ndarray:
    """|omega| in deg/s averaged into ``1/fps`` windows on the (un-shifted) gyro timeline."""
    if len(log.t) < 2:
        return np.zeros(0)
    rate = log.rate_hz
    win = max(1, int(round(rate / lowpass_hz))) if lowpass_hz > 0 else 1
    w = log.gyro
    if win > 1:
        k = np.ones(win) / win
        w = np.stack([np.convolve(log.gyro[:, i], k, mode="same") for i in range(3)], axis=1)
    mag = np.degrees(np.linalg.norm(w, axis=1))
    n = int(np.floor(log.t[-1] * fps)) + 1
    idx = np.floor(log.t * fps).astype(np.int64)
    sums = np.bincount(idx, weights=mag, minlength=n)[:n]
    counts = np.bincount(idx, minlength=n)[:n]
    out = np.zeros(n)
    has = counts > 0
    out[has] = sums[has] / counts[has]
    if (~has).any() and has.any():          # fill gaps in the log by interpolation
        x = np.arange(n)
        out[~has] = np.interp(x[~has], x[has], out[has])
    return out


def _prep(x: np.ndarray, hp_samples: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    bad = ~np.isfinite(x)
    if bad.all():
        return np.zeros_like(x)
    x = np.where(bad, np.nanmean(np.where(bad, np.nan, x)), x)
    x = np.log1p(np.maximum(x, 0))
    if hp_samples > 1 and len(x) > hp_samples:
        x = x - np.convolve(x, np.ones(hp_samples) / hp_samples, mode="same")
    return x


def estimate_offset(gyro_rate: np.ndarray, motion: np.ndarray, fps: float, *, center: float = 0.0,
                    max_shift_s: float = 5.0, min_corr: float = 0.3, min_overlap_s: float = 10.0,
                    highpass_s: float = 3.0) -> SyncResult:
    """Lag (seconds) at which ``gyro_rate`` best matches ``motion``, both sampled at ``fps``.

    A gyro event at gyro time ``t`` shows up in the video at ``t + offset``. The search
    covers ``center +- max_shift_s``. If the correlation stays below ``min_corr`` the
    result is marked not accepted and ``offset`` falls back to ``center``.
    """
    hp = int(round(highpass_s * fps))
    g, v = _prep(gyro_rate, hp), _prep(motion, hp)
    lo = int(round((center - max_shift_s) * fps))
    hi = int(round((center + max_shift_s) * fps))
    min_n = max(3, int(min_overlap_s * fps))
    lags, corrs = [], []
    for m in range(lo, hi + 1):
        # v[j] is compared with g[j - m]
        j0, j1 = max(0, m), min(len(v), len(g) + m)
        if j1 - j0 < min_n:
            continue
        a, b = v[j0:j1], g[j0 - m:j1 - m]
        sa, sb = a.std(), b.std()
        if sa < 1e-9 or sb < 1e-9:
            continue
        lags.append(m)
        corrs.append(float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb)))
    if not lags:
        return SyncResult(center, 0.0, False)
    c = np.asarray(corrs)
    k = int(np.argmax(c))
    best = float(c[k])
    pos = float(lags[k])
    if 0 < k < len(c) - 1 and lags[k - 1] == lags[k] - 1 and lags[k + 1] == lags[k] + 1:
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        denom = y0 - 2 * y1 + y2
        if denom < -1e-12:
            pos += float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))   # sub-sample peak
    if best < min_corr:
        return SyncResult(center, best, False)
    return SyncResult(pos / fps, best, True)
