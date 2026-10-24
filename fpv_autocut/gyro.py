"""Gyroflow .gcsv gyro logs (RunCam Thumb Pro writes these next to each video).

Format (v1.3): a ``GYROFLOW IMU LOG`` / ``CAMERA IMU LOG`` first line,
``key,value`` metadata lines including ``tscale`` (-> seconds) and ``gscale``
(-> rad/s), then a CSV header starting with ``t,gx,gy,gz`` and the samples.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import BIN

MAGIC = ("GYROFLOW IMU LOG", "CAMERA IMU LOG")


@dataclass
class GyroLog:
    t: np.ndarray          # seconds, starting at 0
    gyro: np.ndarray       # (N, 3) rad/s
    meta: dict = field(default_factory=dict)

    @property
    def rate_hz(self) -> float:
        if len(self.t) < 2:
            return 0.0
        return float((len(self.t) - 1) / max(self.t[-1] - self.t[0], 1e-9))


def read_gcsv(path: str | Path) -> GyroLog:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if not lines or lines[0].strip().upper() not in MAGIC:
        raise ValueError(f"{path}: not a gcsv gyro log (first line: {lines[0][:40] if lines else ''!r})")

    meta: dict[str, str] = {}
    header_idx = None
    for i, line in enumerate(lines[1:], start=1):
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("t,"):
            header_idx = i
            break
        key, _, value = s.partition(",")
        meta[key.strip().lower()] = value.strip()
    if header_idx is None:
        raise ValueError(f"{path}: no 't,gx,gy,gz' column header")

    cols = [c.strip().lower() for c in lines[header_idx].split(",")]
    for needed in ("t", "gx", "gy", "gz"):
        if needed not in cols:
            raise ValueError(f"{path}: missing column '{needed}'")

    body = "\n".join(l for l in lines[header_idx + 1:] if l.strip())
    if not body:
        raise ValueError(f"{path}: no samples")
    data = np.loadtxt(io.StringIO(body), delimiter=",", ndmin=2, usecols=range(len(cols)))

    tscale = float(meta.get("tscale", 0.001))
    gscale = float(meta.get("gscale", 1.0))
    t = data[:, cols.index("t")] * tscale
    gyro = data[:, [cols.index("gx"), cols.index("gy"), cols.index("gz")]] * gscale

    order = np.argsort(t, kind="stable")
    t, gyro = t[order], gyro[order]
    t = t - t[0]
    return GyroLog(t=t, gyro=gyro, meta=meta)


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    kernel = np.ones(n) / n
    if x.ndim == 1:
        return np.convolve(x, kernel, mode="same")
    return np.stack([np.convolve(x[:, i], kernel, mode="same") for i in range(x.shape[1])], axis=1)


def gyro_features(
    log: GyroLog,
    duration: float,
    *,
    offset: float = 0.0,
    lowpass_hz: float = 50.0,
    flip_window_s: float = 1.5,
) -> dict[str, np.ndarray]:
    """Per-bin gyro features aligned to the video timeline.

    Returns arrays of length ``ceil(duration / BIN)``:

    * ``gyro_mean`` - mean |omega| in deg/s
    * ``gyro_peak`` - max |omega| in deg/s (after low-pass)
    * ``flip_deg``  - max over axes of |integrated rotation| within a centred
      ``flip_window_s`` window. Signed integration cancels vibration and
      wobble, so ~360 means a real flip/roll, not shaky footage.
    """
    n_bins = max(1, math.ceil(duration / BIN))
    out = {k: np.full(n_bins, np.nan) for k in ("gyro_mean", "gyro_peak", "flip_deg")}
    if len(log.t) < 2:
        return out

    rate = log.rate_hz
    win = max(1, int(round(rate / lowpass_hz))) if lowpass_hz > 0 else 1
    w = _moving_average(log.gyro, win)                       # rad/s, (N,3)
    mag = np.degrees(np.linalg.norm(w, axis=1))              # deg/s

    t = log.t + offset
    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if len(dt) > 1 else 0.0
    dt = np.clip(dt, 0, 0.1)  # gaps in the log must not create fake rotation

    bins = np.floor(t / BIN).astype(np.int64)
    valid = (bins >= 0) & (bins < n_bins)
    if not valid.any():
        return out
    b = bins[valid]
    counts = np.bincount(b, minlength=n_bins).astype(float)
    has = counts > 0

    mean = np.bincount(b, weights=mag[valid], minlength=n_bins)
    out["gyro_mean"][has] = mean[has] / counts[has]

    peak = np.full(n_bins, -np.inf)
    np.maximum.at(peak, b, mag[valid])
    out["gyro_peak"][has] = peak[has]

    # Signed rotation per axis per bin (degrees), then a centred window sum.
    rot = np.zeros((n_bins, 3))
    for ax in range(3):
        rot[:, ax] = np.bincount(b, weights=np.degrees(w[valid, ax] * dt[valid]), minlength=n_bins)
    k = max(1, int(round(flip_window_s / BIN)))
    kernel = np.ones(k)
    win_rot = np.stack([np.convolve(rot[:, ax], kernel, mode="same") for ax in range(3)], axis=1)
    flip = np.abs(win_rot).max(axis=1)
    out["flip_deg"][has] = flip[has]
    return out
