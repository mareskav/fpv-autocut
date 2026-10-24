"""Visual and audio signals decoded with ffmpeg and measured with numpy.

No OpenCV: ffmpeg decodes a tiny grayscale stream (default 160 px wide,
10 fps) into a pipe, numpy measures it. This keeps the dependency list to
numpy only, which matters on Windows ARM64.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import BIN
from .config import Config
from .gyro import gyro_features, read_gcsv
from .media import Clip

CACHE_VERSION = "1"  # bump when signal computation changes, to invalidate old caches
SIGNAL_KEYS = ("motion", "sharp", "luma", "audio_db", "gyro_mean", "gyro_peak", "flip_deg")


def _bin_mean(values: np.ndarray, times: np.ndarray, n_bins: int) -> np.ndarray:
    out = np.full(n_bins, np.nan)
    if len(values) == 0:
        return out
    b = np.floor(times / BIN).astype(np.int64)
    ok = (b >= 0) & (b < n_bins) & np.isfinite(values)
    if not ok.any():
        return out
    sums = np.bincount(b[ok], weights=values[ok], minlength=n_bins)
    counts = np.bincount(b[ok], minlength=n_bins)
    has = counts > 0
    out[has] = sums[has] / counts[has]
    return out


def visual_signals(clip: Clip, cfg: Config, hwaccel: str | None = None) -> dict[str, np.ndarray]:
    d = cfg.detect
    w = int(d.analysis_width)
    h = max(2, int(round(w * clip.height / clip.width / 2)) * 2)
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += [
        "-i", clip.path, "-an",
        "-vf", f"fps={d.analysis_fps},scale={w}:{h}:flags=area,format=gray",
        "-f", "rawvideo", "-",
    ]
    frame_size = w * h
    motion, sharp, luma = [], [], []
    prev = None
    # stderr goes to a file, not a pipe: a damaged video can print more than a pipe buffer holds,
    # which would block ffmpeg while we are still waiting on stdout.
    with tempfile.TemporaryFile() as errf:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf) as proc:
            assert proc.stdout is not None
            while True:
                buf = proc.stdout.read(frame_size)
                if len(buf) < frame_size:
                    break
                f = np.frombuffer(buf, dtype=np.uint8).reshape(h, w).astype(np.float32)
                luma.append(float(f.mean()))
                lap = 4 * f[1:-1, 1:-1] - f[:-2, 1:-1] - f[2:, 1:-1] - f[1:-1, :-2] - f[1:-1, 2:]
                sharp.append(float(lap.var()))
                motion.append(float(np.abs(f - prev).mean()) if prev is not None else np.nan)
                prev = f
            proc.wait()
        errf.seek(0)
        err = errf.read().decode(errors="replace")
    if proc.returncode != 0 and not luma:
        raise RuntimeError(f"ffmpeg failed on {clip.path}:\n{err.strip()[-2000:]}")

    n_bins = max(1, math.ceil(clip.duration / BIN))
    times = np.arange(len(luma)) / d.analysis_fps
    return {
        "motion": _bin_mean(np.array(motion), times, n_bins),
        "sharp": _bin_mean(np.array(sharp), times, n_bins),
        "luma": _bin_mean(np.array(luma), times, n_bins),
    }


def audio_signal(clip: Clip, cfg: Config) -> np.ndarray:
    """Loudness (dB RMS) per bin. Streamed in chunks, so an hour-long clip needs no hour of PCM in RAM."""
    n_bins = max(1, math.ceil(clip.duration / BIN))
    if not clip.has_audio:
        return np.full(n_bins, np.nan)
    sr = int(cfg.detect.audio_sr)
    hop = int(sr * BIN)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", clip.path, "-vn", "-ac", "1", "-ar", str(sr),
           "-f", "f32le", "-"]
    db: list[float] = []
    with tempfile.TemporaryFile() as errf, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf) as proc:
        assert proc.stdout is not None
        chunk_bytes = hop * 4 * 64   # 64 bins per read
        rest = b""
        while True:
            buf = proc.stdout.read(chunk_bytes)
            if not buf:
                break
            rest += buf
            usable = (len(rest) // (hop * 4)) * hop * 4
            if usable:
                frames = np.frombuffer(rest[:usable], dtype=np.float32).reshape(-1, hop)
                rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
                db.extend(20 * np.log10(rms))
                rest = rest[usable:]
        proc.wait()
    out = np.full(n_bins, np.nan)
    if db:  # a failed or empty decode just leaves NaN: loudness is only a minor score input
        m = min(n_bins, len(db))
        out[:m] = np.asarray(db[:m])
    return out


def _cache_key(clip: Clip, cfg: Config, gyro_offset: float) -> str:
    parts = [clip.path, str(os.path.getsize(clip.path)), str(os.path.getmtime(clip.path))]
    if clip.gyro_path:
        parts += [clip.gyro_path, str(os.path.getmtime(clip.gyro_path))]
    d = cfg.detect
    parts += [CACHE_VERSION]
    parts += [str(x) for x in (d.analysis_fps, d.analysis_width, d.audio_sr, d.gyro_lowpass_hz, d.flip_window_s, gyro_offset, BIN)]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _load_cache(path: Path) -> dict[str, np.ndarray] | None:
    """Cached signals, or None if the file is corrupt or incomplete."""
    try:
        with np.load(path) as z:
            sig = {k: z[k] for k in z.files}
    except Exception:  # truncated zip, bad pickle header, I/O error...
        return None
    if any(k not in sig for k in SIGNAL_KEYS):
        return None
    return sig


def _save_cache(path: Path, sig: dict[str, np.ndarray]) -> None:
    """Atomic write: an interrupted run never leaves a half-written .npz."""
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, **sig)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def analyze_clip(clip: Clip, cfg: Config, cache_dir: Path, *, gyro_offset: float = 0.0,
                 hwaccel: str | None = None, log=print) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(clip, cfg, gyro_offset)
    cache = cache_dir / f"{clip.name}-{key}.npz"
    if cache.exists():
        cached = _load_cache(cache)
        if cached is not None:
            return cached
        log(f"  ! discarding unreadable cache {cache.name}")
        cache.unlink(missing_ok=True)

    log(f"  analysing {Path(clip.path).name} ({clip.duration:.0f} s)")
    sig = visual_signals(clip, cfg, hwaccel=hwaccel)
    sig["audio_db"] = audio_signal(clip, cfg)
    n_bins = len(sig["motion"])
    if clip.gyro_path:
        try:
            log_ = read_gcsv(clip.gyro_path)
            g = gyro_features(log_, clip.duration, offset=gyro_offset,
                              lowpass_hz=cfg.detect.gyro_lowpass_hz, flip_window_s=cfg.detect.flip_window_s)
            sig.update(g)
        except (ValueError, OSError) as exc:
            log(f"  ! gyro log ignored for {clip.name}: {exc}")
    for k in SIGNAL_KEYS:
        if k not in sig:
            sig[k] = np.full(n_bins, np.nan)
        elif len(sig[k]) != n_bins:
            arr = np.full(n_bins, np.nan)
            m = min(n_bins, len(sig[k]))
            arr[:m] = sig[k][:m]
            sig[k] = arr
    _save_cache(cache, sig)
    return sig


def analyze_all(clips: list[Clip], cfg: Config, work: Path, *, gyro_offset: float = 0.0,
                hwaccel: str | None = None, jobs: int = 2, log=print) -> list[dict[str, np.ndarray]]:
    cache_dir = work / "cache"
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futures = [ex.submit(analyze_clip, c, cfg, cache_dir, gyro_offset=gyro_offset, hwaccel=hwaccel, log=log)
                   for c in clips]
        return [f.result() for f in futures]


def write_timeline_csv(path: Path, clips: list[Clip], signals: list[dict], scores: list[np.ndarray]) -> None:
    """One row per bin: handy for a human (or Claude) to inspect the scoring."""
    cols = ["clip", "t"] + list(SIGNAL_KEYS) + ["score"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for clip, sig, sc in zip(clips, signals, scores):
            for i in range(len(sc)):
                row = [clip.name, f"{i * BIN:.1f}"]
                for k in SIGNAL_KEYS:
                    v = sig[k][i]
                    row.append("" if not np.isfinite(v) else f"{v:.2f}")
                row.append(f"{sc[i]:.3f}")
                fh.write(",".join(row) + "\n")


def dump_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
