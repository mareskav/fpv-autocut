"""Corrupt clips and damaged cache files must not take the whole run down."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from synth import make_video

from fpv_autocut import signals
from fpv_autocut.cli import main
from fpv_autocut.config import load_config
from fpv_autocut.media import Clip, discover
from fpv_autocut.signals import SIGNAL_KEYS, _load_cache, _save_cache, analyze_clip

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
    footage = tmp_path / "footage"
    footage.mkdir()
    (footage / "junk.mp4").write_bytes(b"garbage")
    with pytest.raises(SystemExit) as e:
        main(["plan", str(footage), "-o", str(tmp_path / "out")])
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


# ---- Gemini plumbing (no network) -------------------------------------------------------

def test_gemini_proxy_name_unique_per_path():
    from fpv_autocut.gemini import proxy_name
    a = Clip(0, "/d1/GX010001.mp4", "x", None, 1.0, "30/1", 1, 1, False)
    b = Clip(1, "/d2/GX010001.mp4", "x", None, 1.0, "30/1", 1, 1, False)
    assert proxy_name(a) != proxy_name(b) and proxy_name(a).startswith("GX010001-")


def test_gemini_parse_ratings():
    from fpv_autocut.gemini import parse_ratings
    assert parse_ratings('[{"id": 0, "score": 7}, 3, "x"]') == [{"id": 0, "score": 7}]
    assert parse_ratings('{"id": 1}') == [{"id": 1}]
    assert parse_ratings("not json") is None
    assert parse_ratings(None) is None
    assert parse_ratings("42") is None


def test_gemini_wait_until_active(monkeypatch):
    from types import SimpleNamespace as NS

    from fpv_autocut import gemini
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)

    class Files:
        def __init__(self, states):
            self.states = list(states)

        def get(self, name):
            return NS(name=name, state=NS(name=self.states.pop(0)))

    mk = lambda s: NS(name="f", state=NS(name=s))  # noqa: E731
    assert gemini.wait_until_active(NS(files=Files(["PROCESSING", "ACTIVE"])), mk("PROCESSING"), 60).state.name == "ACTIVE"
    assert gemini.wait_until_active(NS(files=Files([])), mk("FAILED"), 60) is None
    assert gemini.wait_until_active(NS(files=Files(["PROCESSING"] * 5)), mk("PROCESSING"), -1) is None


# ---- paths and the output folder ----------------------------------------------------------

def test_concat_quote_escapes_apostrophe():
    from fpv_autocut.render import _concat_quote
    assert _concat_quote(Path("/tmp/it's here/a.mkv")) == r"/tmp/it'\''s here/a.mkv"


def test_out_dir_inside_footage_is_refused(tmp_path: Path):
    from fpv_autocut.cli import check_out_dir
    footage = tmp_path / "footage"
    footage.mkdir()
    clip = footage / "a.mp4"
    clip.write_bytes(b"")
    for bad in (footage, footage / "out", tmp_path / "footage" / "x" / "y"):
        with pytest.raises(SystemExit):
            check_out_dir(bad, [str(footage)])
        with pytest.raises(SystemExit):
            check_out_dir(bad, [str(clip)])
    check_out_dir(tmp_path / "out", [str(footage)])        # sibling is fine
    check_out_dir(tmp_path / "footage2", [str(footage)])   # similar prefix is fine


def test_music_shorter_than_video_warns(monkeypatch):
    from fpv_autocut import pipeline
    from fpv_autocut.planning import Segment
    monkeypatch.setattr(pipeline, "music_duration", lambda p: 10.0)
    logs: list[str] = []
    pipeline._warn_if_music_short("reel", [Segment(0, 0, 30, 1, 1, 1)], "m.mp3", logs.append)
    assert logs and "silent" in logs[0]
    logs.clear()
    pipeline._warn_if_music_short("reel", [Segment(0, 0, 8, 1, 1, 1)], "m.mp3", logs.append)
    assert not logs


# ---- ffmpeg-backed behaviour -------------------------------------------------------------

def _frames_and_audio(p: Path) -> tuple[int, float]:
    import json
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "json", str(p)],
                         capture_output=True, check=True).stdout
    frames = int(json.loads(out)["streams"][0]["nb_read_frames"])
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-vn", "-ac", "1", "-ar", "8000",
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return frames, len(pcm) / 2 / 8000


@needs_ffmpeg
def test_segment_running_past_source_end_keeps_av_in_sync(tmp_path: Path):
    from fpv_autocut.config import Output
    from fpv_autocut.planning import Segment
    from fpv_autocut.render import render_segment
    src = make_video(tmp_path / "v.mp4")                      # 40 s
    clip = Clip(0, str(src), str(src), None, 40.0, "30/1", 640, 480, True)
    out = Output(width=320, height=240, fps=30.0)
    dst = tmp_path / "seg.mkv"
    render_segment(Segment(0, 39.0, 41.0, 1, 1, 40), clip, True, out, dst, draft=True, n_frames=60)
    frames, audio_s = _frames_and_audio(dst)
    assert frames == 60 and abs(audio_s - 2.0) < 0.05


@needs_ffmpeg
def test_contact_sheet_survives_odd_characters_in_path(tmp_path: Path):
    from fpv_autocut.planning import Segment
    from fpv_autocut.render import contact_sheet
    src = make_video(tmp_path / "v.mp4", with_audio=False)
    clip = Clip(0, str(src), str(src), None, 40.0, "30/1", 640, 480, False)
    dst = tmp_path / "100% it's čeština" / "sheet.jpg"
    segs = [Segment(0, 5, 7, 1, 1, 6), Segment(0, 19, 21, 1, 1, 20)]
    assert contact_sheet(segs, [clip], dst).stat().st_size > 1000


@needs_ffmpeg
def test_audio_signal_is_streamed_and_finite(tmp_path: Path):
    src = make_video(tmp_path / "v.mp4")
    clip = Clip(0, str(src), str(src), None, 40.0, "30/1", 640, 480, True)
    db = signals.audio_signal(clip, load_config())
    assert db.shape == (80,) and np.isfinite(db).sum() >= 78
    loud, quiet = db[38:42].mean(), db[4:10].mean()   # flip at 18-21 s is much louder than idle
    assert loud > quiet + 6
