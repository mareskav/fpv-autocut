"""Gaps found in the audit: FCPXML content, Gemini rating with a fake client, odd paths, beat-aligned cuts, CLI errors."""

import json
import shutil
import sys
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from fpv_autocut.beats import track_beats
from fpv_autocut.cli import main
from fpv_autocut.config import Output
from fpv_autocut.fcpxml import write_fcpxml
from fpv_autocut.media import Clip
from fpv_autocut.planning import Segment
from fpv_autocut.render import frame_counts

from synth import make_session

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _clip(i=0, name="a", fps="60/1", audio=True, dur=60.0) -> Clip:
    return Clip(id=i, path=f"/footage/{name}.mp4", render_path=f"/footage/{name}.mp4", gyro_path=None,
                duration=dur, fps=fps, width=1920, height=1080, has_audio=audio)


# ---- FCPXML -----------------------------------------------------------------------------

def _t(s: str) -> Fraction:
    s = s.rstrip("s")
    return Fraction(s)


def test_fcpxml_structure_and_timing(tmp_path: Path):
    clips = [_clip(0, "R&D \"one\"", fps="60/1"), _clip(1, "b", fps="30000/1001", audio=False)]
    segs = [Segment(0, 1.0, 3.55, 0.8, 0.9, 2, tags=["flip"]), Segment(1, 10.0, 12.2, 0.5, 0.6, 11),
            Segment(0, 20.0, 21.1, 0.7, 0.7, 20.5)]
    out = Output(fps=60.0)
    dst = write_fcpxml(tmp_path / "x" / "reel.fcpxml", "reel", segs, clips, out)
    root = ET.parse(dst).getroot()                           # well-formed despite & and " in the name
    assert root.tag == "fcpxml" and root.get("version") == "1.8"

    assets = {a.get("id"): a for a in root.iter("asset")}
    assert set(assets) == {"a0", "a1"}
    assert assets["a0"].get("hasAudio") == "1" and assets["a1"].get("hasAudio") == "0"
    assert assets["a0"].get("src").startswith("file:///")

    spine = list(root.iter("asset-clip"))
    assert len(spine) == 3 and "flip" in spine[0].get("name")
    # clips are laid end to end and the sequence duration is their sum
    pos = Fraction(0)
    for c in spine:
        assert _t(c.get("offset")) == pos
        pos += _t(c.get("duration"))
    seq = next(root.iter("sequence"))
    assert _t(seq.get("duration")) == pos
    # durations are whole frames of the sequence rate, computed on the cumulative timeline
    seq_fps = Fraction(clips[0].fps)
    assert pos * seq_fps == sum(frame_counts(segs, float(seq_fps)))


def test_fcpxml_empty_plan_is_valid(tmp_path: Path):
    dst = write_fcpxml(tmp_path / "e.fcpxml", "e", [], [_clip()], Output(fps=30.0))
    assert ET.parse(dst).getroot().tag == "fcpxml"


def test_fcpxml_audio_channels_from_clip(tmp_path: Path):
    c = _clip()
    c.audio_channels = 1
    root = ET.parse(write_fcpxml(tmp_path / "c.fcpxml", "c", [Segment(0, 1, 3, .5, .5, 2)], [c], Output())).getroot()
    assert next(root.iter("asset")).get("audioChannels") == "1"


@needs_ffmpeg
def test_fcpxml_music_is_connected_under_first_clip(tmp_path: Path):
    from synth import make_music
    music = make_music(tmp_path / "m.mp3", seconds=3)
    segs = [Segment(0, 5.0, 7.0, .5, .5, 6), Segment(0, 20.0, 22.0, .5, .5, 21)]
    root = ET.parse(write_fcpxml(tmp_path / "m.fcpxml", "m", segs, [_clip()], Output(fps=60.0),
                                 music=str(music))).getroot()
    assert {a.get("id") for a in root.iter("asset")} == {"a0", "m0"}
    first = next(root.iter("spine"))[0]
    mus = first.find("asset-clip")
    assert mus.get("ref") == "m0" and mus.get("lane") == "-1"
    assert _t(mus.get("offset")) == _t(first.get("start"))     # offset is in the parent's source time
    assert _t(mus.get("duration")) <= 3 + Fraction(1, 10)      # never longer than the song
    assert len(list(root.iter("spine"))[0]) == 2               # music is nested, not a spine item


def test_fcpxml_vertical_gets_reframe_transform(tmp_path: Path):
    left = Segment(0, 1, 3, .5, .5, 2, crop_x=0.0)
    right = Segment(0, 5, 7, .5, .5, 6, crop_x=1.0)
    root = ET.parse(write_fcpxml(tmp_path / "v.fcpxml", "v", [left, right], [_clip()],
                                 Output(width=1080, height=1920, fps=60.0))).getroot()
    pos = [float(t.get("position").split()[0]) for t in root.iter("adjust-transform")]
    assert len(pos) == 2 and pos[0] > 0 > pos[1]               # left crop shifts the image right, and vice versa
    assert abs(pos[0] + pos[1]) < 1e-6


def test_auto_crop_x_follows_motion():
    from fpv_autocut.planning import auto_crop_x
    cx = np.array([0.1] * 4 + [0.9] * 4)
    motion = np.ones(8)
    seg = lambda a, b: Segment(0, a * 0.5, b * 0.5, .5, .5, a * 0.5)
    aspect = (1080 / 1920, 1920 / 1080)
    assert auto_crop_x(seg(0, 4), cx, motion, aspect[1], aspect[0]) == 0.0
    assert auto_crop_x(seg(4, 8), cx, motion, aspect[1], aspect[0]) == 1.0
    assert auto_crop_x(seg(0, 8), np.full(8, 0.5), motion, aspect[1], aspect[0]) == pytest.approx(0.5)
    assert auto_crop_x(seg(0, 4), cx, np.zeros(8), aspect[1], aspect[0]) == 0.5   # no motion: centred
    assert auto_crop_x(seg(0, 4), cx, motion, 1.0, 2.0) == 0.5                    # source narrower than output


@needs_ffmpeg
def test_probe_reports_audio_channels(tmp_path: Path):
    from fpv_autocut.media import music_duration, probe
    s = make_session(tmp_path / "s")
    assert probe(s["flight"])["audio_channels"] >= 1
    assert music_duration(str(s["music"])) == pytest.approx(60, abs=1)


# ---- Gemini rating with a fake client ---------------------------------------------------

def _fake_genai(monkeypatch, answers: list[str | None]):
    calls = NS(upload=0, generate=0)

    class Files:
        def upload(self, file):
            calls.upload += 1
            return NS(name="files/1", state=NS(name="ACTIVE"))

        def get(self, name):
            return NS(name=name, state=NS(name="ACTIVE"))

        def delete(self, name):
            pass

    class Models:
        def generate_content(self, model, contents, config):
            calls.generate += 1
            return NS(text=answers.pop(0))

    genai = ModuleType("google.genai")
    genai.Client = lambda: NS(files=Files(), models=Models())
    genai.types = NS(GenerateContentConfig=lambda **kw: kw)
    google = ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    return calls


def test_rate_segments_uses_fake_client_and_caches(tmp_path: Path, monkeypatch):
    from fpv_autocut import gemini
    monkeypatch.setattr(gemini, "_proxy", lambda clip, dst: dst)
    answer = json.dumps([{"id": 0, "score": 9, "crop_x": 0.2, "reason": "tight gap"}, {"id": 5, "score": 1}])
    calls = _fake_genai(monkeypatch, [answer])
    clips = [_clip()]
    segs = [Segment(0, 5, 9, 0.5, 0.6, 7), Segment(0, 30, 33, 0.4, 0.5, 31)]

    rated = gemini.rate_segments(segs, clips, tmp_path, "m", log=lambda m: None)
    assert len(rated) == 1 and rated[0][0] is segs[0] and calls.generate == 1   # id 5 is out of range, ignored

    again = gemini.rate_segments(segs, clips, tmp_path, "m", log=lambda m: None)   # served from the JSON cache
    assert len(again) == 1 and calls.generate == 1 and calls.upload == 1

    gemini.apply_ratings(segs, rated, 0.4)
    assert abs(segs[0].score - (0.6 * 0.5 + 0.4 * 0.9)) < 1e-9
    assert segs[0].crop_x == 0.2 and segs[0].note == "tight gap"
    assert segs[1].score == 0.4                                              # unrated segment untouched


def test_rate_segments_skips_clip_on_unusable_answer(tmp_path: Path, monkeypatch):
    from fpv_autocut import gemini
    monkeypatch.setattr(gemini, "_proxy", lambda clip, dst: dst)
    _fake_genai(monkeypatch, ["I could not watch the video"])
    logs: list[str] = []
    rated = gemini.rate_segments([Segment(0, 5, 9, 0.5, 0.6, 7)], [_clip()], tmp_path, "m", log=logs.append)
    assert rated == [] and any("unusable answer" in m for m in logs)
    assert not list((tmp_path / "gemini").glob("*.json"))                       # a bad answer is not cached


# ---- CLI error paths --------------------------------------------------------------------

def test_cli_missing_input_exits_2(tmp_path: Path, capsys):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    assert main(["plan", str(tmp_path / "nope"), "-o", str(tmp_path / "o")]) == 2
    assert "not found" in capsys.readouterr().err


def test_cli_missing_ffmpeg_exits_2(tmp_path: Path, monkeypatch, capsys):
    from fpv_autocut import media
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    assert main(["plan", str(tmp_path), "-o", str(tmp_path / "o")]) == 2
    assert "ffmpeg" in capsys.readouterr().err


def test_cli_bad_config_exits_2(tmp_path: Path, capsys):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    cfg = tmp_path / "c.toml"
    cfg.write_text("[reel]\ntargett = 5\n")
    footage = tmp_path / "f"
    footage.mkdir()
    assert main(["plan", str(footage), "-o", str(tmp_path / "o"), "--config", str(cfg)]) == 2
    assert "targett" in capsys.readouterr().err


def test_cli_render_missing_plan_exits_2(tmp_path: Path, capsys):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    assert main(["render", str(tmp_path / "plan.json")]) == 2
    assert "not found" in capsys.readouterr().err


# ---- e2e: awkward paths, beat-aligned cuts ----------------------------------------------

@needs_ffmpeg
def test_czech_diacritics_and_spaces_in_paths(tmp_path: Path):
    footage = tmp_path / "Záběry z Šumavy 2026"
    files = make_session(footage)
    out = tmp_path / "výstup č. 1"
    assert main(["run", str(footage), "-o", str(out), "--reel", "10", "--long", "20", "--shorts", "1",
                 "--short-len", "6", "--music", str(files["music"]), "--draft", "--only", "reel"]) == 0
    assert (out / "reel_draft.mp4").stat().st_size > 10_000
    assert (out / "reel.fcpxml").exists() and (out / "sheets" / "reel.jpg").exists()
    root = ET.parse(out / "reel.fcpxml").getroot()
    assert any("%C5%A0umavy" in (a.get("src") or "") or "umavy" in (a.get("src") or "") for a in root.iter("asset"))


@needs_ffmpeg
def test_reel_cuts_land_on_beats(tmp_path: Path):
    footage = tmp_path / "f"
    files = make_session(footage)
    out = tmp_path / "o"
    assert main(["plan", str(footage), "-o", str(out), "--reel", "20", "--music", str(files["music"])]) == 0
    plan = json.loads((out / "plan.json").read_text())
    _, beats = track_beats(str(files["music"]))
    bounds = np.cumsum([s["end"] - s["start"] for s in plan["outputs"]["reel"]["segments"]])
    off = np.array([np.abs(beats - b).min() for b in bounds])
    assert len(bounds) >= 3
    assert (off < 0.02).mean() >= 0.75, off
