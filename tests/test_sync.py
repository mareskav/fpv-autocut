"""Automatic gyro <-> video alignment."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from synth import make_gcsv, make_video

from fpv_autocut.config import load_config
from fpv_autocut.gyro import GyroLog, read_gcsv
from fpv_autocut.media import Clip
from fpv_autocut.signals import analyze_clip
from fpv_autocut.sync import estimate_offset, gyro_rate_series

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
FPS = 10.0


def _activity(n=400, seed=1) -> np.ndarray:
    """Bursty non-negative signal, like rotation rate over a flight."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for _ in range(14):
        a = int(rng.integers(0, n - 30))
        x[a:a + int(rng.integers(8, 30))] += rng.uniform(40, 200)
    return x


def _shifted(x: np.ndarray, samples: int) -> np.ndarray:
    """What the video sees if events occur ``samples`` later than in the gyro."""
    out = np.zeros_like(x)
    if samples >= 0:
        out[samples:] = x[:len(x) - samples]
    else:
        out[:samples] = x[-samples:]
    return out


@pytest.mark.parametrize("true_shift", [0.0, 1.3, -2.2, 3.7])
def test_estimate_offset_recovers_known_shift(true_shift):
    g = _activity()
    rng = np.random.default_rng(5)
    v = _shifted(g, int(round(true_shift * FPS))) * 0.05 + rng.normal(0, 0.3, len(g))   # video has its own scale and noise
    res = estimate_offset(g, np.abs(v), FPS)
    assert res.accepted and res.corr > 0.5
    assert abs(res.offset - true_shift) <= 0.11


def test_estimate_offset_centre_shifts_the_search():
    g = _activity()
    v = _shifted(g, 80)           # 8 s: outside the default +-5 s window
    assert not estimate_offset(g, v, FPS).accepted or abs(estimate_offset(g, v, FPS).offset - 8.0) > 1
    res = estimate_offset(g, v, FPS, center=8.0)
    assert res.accepted and abs(res.offset - 8.0) <= 0.11


def test_estimate_offset_rejects_unrelated_signals():
    g = _activity(seed=1)
    v = np.random.default_rng(9).uniform(0, 5, len(g))
    res = estimate_offset(g, v, FPS, center=0.25)
    assert not res.accepted and res.offset == 0.25


def test_estimate_offset_flat_signal_is_not_accepted():
    res = estimate_offset(np.zeros(300), np.ones(300), FPS)
    assert not res.accepted and res.offset == 0.0


def test_estimate_offset_tolerates_nan_in_motion():
    g = _activity()
    v = _shifted(g, 10).astype(float)
    v[0] = np.nan
    v[50:55] = np.nan
    res = estimate_offset(g, v, FPS)
    assert res.accepted and abs(res.offset - 1.0) <= 0.11


def test_gyro_rate_series_is_on_gyro_time(tmp_path: Path):
    t = np.arange(0, 20, 1 / 200)
    gyro = np.zeros((len(t), 3))
    gyro[(t >= 5) & (t < 8), 2] = np.radians(100)
    r = gyro_rate_series(GyroLog(t=t, gyro=gyro), FPS, lowpass_hz=0)
    assert len(r) == 200
    assert r[:49].max() < 1 and r[51:79].min() > 99 and r[85:].max() < 1


@needs_ffmpeg
@pytest.mark.parametrize("shift", [1.4, -0.8])
def test_analyze_clip_syncs_a_late_or_early_gyro_log(tmp_path: Path, shift: float):
    src = make_video(tmp_path / "v.mp4", with_audio=False)
    gcsv = make_gcsv(tmp_path / "v.gcsv", shift_s=shift)
    clip = Clip(id=0, path=str(src), render_path=str(src), gyro_path=str(gcsv), duration=40.0,
                fps="30/1", width=640, height=480, has_audio=False)
    assert len(read_gcsv(gcsv).t) > 1000
    cfg = load_config(None, {})
    logs: list[str] = []
    sig = analyze_clip(clip, cfg, tmp_path / "cache", log=logs.append)
    offset, corr, accepted = sig["sync"]
    assert accepted == 1.0 and abs(offset - shift) <= 0.15, logs
    # with the sync applied the flip (18-21 s) lands in the right bins
    flip_bins = np.flatnonzero(sig["flip_deg"] > 250) * 0.5
    assert flip_bins.min() >= 17.0 and flip_bins.max() <= 22.0


@needs_ffmpeg
def test_gyro_sync_can_be_disabled(tmp_path: Path):
    src = make_video(tmp_path / "v.mp4", with_audio=False)
    gcsv = make_gcsv(tmp_path / "v.gcsv", shift_s=1.4)
    clip = Clip(id=0, path=str(src), render_path=str(src), gyro_path=str(gcsv), duration=40.0,
                fps="30/1", width=640, height=480, has_audio=False)
    cfg = load_config(None, {"detect": {"gyro_sync": False}})
    sig = analyze_clip(clip, cfg, tmp_path / "cache", log=lambda m: None)
    assert "sync" not in sig
