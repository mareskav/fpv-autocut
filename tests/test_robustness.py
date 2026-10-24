"""Corrupt clips and damaged cache files must not take the whole run down."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from fpv_autocut import signals
from fpv_autocut.cli import main
from fpv_autocut.config import load_config
from fpv_autocut.media import Clip, discover, probe
from fpv_autocut.signals import SIGNAL_KEYS, _load_cache, _save_cache, analyze_clip

from synth import make_video

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _clip(path: Path, dur=4.0) -> Clip:
    return Clip(id=0, path=str(path), render_path=str(path), gyro_path=None, duration=dur,
                fps="30/1", width=640, height=480, has_audio=False)


def _signals(n=8) -> dict:
    return {k: np.arange(n, dtype=float) for k in SIGNAL_KEYS}


@needs_ffmpeg
def test_discover_skips_truncated_clip(tmp_path: Path):
    good = make_video(tmp_path / "good.mp4")
    data = good.read_bytes()
    (tmp_path / "bad.mp4").write_bytes(data[: len(data) // 20])  # moov atom lost
    (tmp_path / "junk.mov").write_bytes(b"not a video at all")
    logs: list[str] = []
    clips = discover([str(tmp_path)], log=logs.append)
    assert [c.name for c in clips] == ["good"]
    assert [c.id for c in clips] == [0]
    assert any("bad.mp4" in m for m in logs) and any("junk.mov" in m for m in logs)


@needs_ffmpeg
def test_cli_reports_no_usable_clips(tmp_path: Path):
    (tmp_path / "junk.mp4").write_bytes(b"garbage")
    with pytest.raises(SystemExit) as e:
        main(["plan", str(tmp_path), "-o", str(tmp_path / "out")])
    assert "No usable video" in str(e.value)


def test_cache_roundtrip_and_atomic_write(tmp_path: Path):
    p = tmp_path / "c.npz"
    _save_cache(p, _signals())
    assert list(tmp_path.glob("*.tmp*")) == []
    loaded = _load_cache(p)
    assert loaded is not None and set(loaded) == set(SIGNAL_KEYS)


def test_load_cache_rejects_corrupt_and_incomplete(tmp_path: Path):
    p = tmp_path / "c.npz"
    p.write_bytes(b"PK\x03\x04 truncated")
    assert _load_cache(p) is None
    sig = _signals()
    del sig["flip_deg"]
    np.savez_compressed(p, **sig)
    assert _load_cache(p) is None


@needs_ffmpeg
def test_analyze_clip_recovers_from_bad_cache(tmp_path: Path, monkeypatch):
    clip = _clip(make_video(tmp_path / "v.mp4", with_audio=False))
    cfg = load_config(None, {})
    cache_dir = tmp_path / "cache"
    first = analyze_clip(clip, cfg, cache_dir, log=lambda m: None)
    (cache,) = cache_dir.glob("*.npz")
    cache.write_bytes(cache.read_bytes()[:30])  # simulate interrupted write
    logs: list[str] = []
    second = analyze_clip(clip, cfg, cache_dir, log=logs.append)
    assert any("unreadable cache" in m for m in logs)
    for k in SIGNAL_KEYS:
        np.testing.assert_array_equal(first[k], second[k])
    assert _load_cache(cache) is not None


@needs_ffmpeg
def test_cache_key_includes_version(tmp_path: Path, monkeypatch):
    clip = _clip(make_video(tmp_path / "v.mp4", with_audio=False))
    cfg = load_config(None, {})
    k1 = signals._cache_key(clip, cfg, 0.0)
    monkeypatch.setattr(signals, "CACHE_VERSION", "999")
    assert signals._cache_key(clip, cfg, 0.0) != k1
