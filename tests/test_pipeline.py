"""End-to-end test on synthetic footage (needs ffmpeg; ~30 s)."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from synth import make_music, make_session

from fpv_autocut.beats import track_beats
from fpv_autocut.cli import main

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    folder = tmp_path_factory.mktemp("footage")
    files = make_session(folder)
    # A Gyroflow render next to flight1 must be used for rendering, not analysed.
    shutil.copy(files["flight"], folder / "flight1_stabilized.mp4")
    return folder, files


def _probe_frames(p: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=nb_read_frames,width,height", "-of", "json", str(p)],
                         capture_output=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return int(s["nb_read_frames"]), int(s["width"]), int(s["height"])


def test_beat_tracker_tempo(tmp_path):
    bpm, beats = track_beats(str(make_music(tmp_path / "m.mp3", bpm=128, seconds=30)))
    assert abs(bpm - 128) < 0.5
    assert np.allclose(np.diff(beats), 60 / bpm)


def test_end_to_end(session, tmp_path):
    folder, files = session
    out = tmp_path / "out"
    rc = main(["run", str(folder), "-o", str(out), "--reel", "12", "--long", "30", "--short-len", "8",
               "--music", str(files["music"]), "--draft"])
    assert rc == 0
    plan = json.loads((out / "plan.json").read_text())

    clips = plan["clips"]
    assert [Path(c["path"]).name for c in clips] == ["flight1.mp4", "flight2.mp4"]
    assert clips[0]["render_path"].endswith("flight1_stabilized.mp4")
    assert clips[0]["gyro_path"] and clips[1]["gyro_path"] is None

    stats = {s["clip"]: s for s in plan["stats"]}
    assert 12 <= stats["flight1"]["idle_s"] <= 15  # 14 s of ground time in the synthetic clip

    reel = plan["outputs"]["reel"]["segments"]
    flip = [s for s in reel if s["clip"] == 0 and "flip" in s["tags"]]
    assert flip and flip[0]["start"] <= 18.5 and flip[0]["end"] >= 20.5

    frames, w, h = _probe_frames(out / "reel_draft.mp4")
    assert (w, h) == (960, 540)
    assert abs(frames - round(plan["outputs"]["reel"]["duration"] * 60)) <= 1
    _, w, h = _probe_frames(out / "short_01_draft.mp4")
    assert (w, h) == (540, 960)
    assert (out / "reel.fcpxml").exists() and (out / "sheets" / "reel.jpg").exists()

    # Edit the plan (drop a segment) and re-render only the reel.
    plan["outputs"]["reel"]["segments"] = reel[:1]
    (out / "plan.json").write_text(json.dumps(plan))
    assert main(["render", str(out / "plan.json"), "--only", "reel", "--draft"]) == 0
    frames, _, _ = _probe_frames(out / "reel_draft.mp4")
    assert abs(frames - round((reel[0]["end"] - reel[0]["start"]) * 60)) <= 1
