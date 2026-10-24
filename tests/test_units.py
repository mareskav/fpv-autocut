"""Fast unit tests (no video decoding)."""

from pathlib import Path

import numpy as np
import pytest
from synth import DURATION, make_gcsv

from fpv_autocut import BIN
from fpv_autocut.config import Profile, load_config
from fpv_autocut.gemini import apply_ratings
from fpv_autocut.gyro import gyro_features, read_gcsv
from fpv_autocut.media import Clip
from fpv_autocut.planning import Segment, find_segments, select, snap_to_beats
from fpv_autocut.render import frame_counts
from fpv_autocut.scoring import score_session


def _clip(i=0, dur=DURATION):
    return Clip(id=i, path=f"c{i}.mp4", render_path=f"c{i}.mp4", gyro_path=None, duration=dur,
                fps="60/1", width=1920, height=1440, has_audio=True)


def test_gcsv_parse_and_flip_detection(tmp_path: Path):
    log = read_gcsv(make_gcsv(tmp_path / "f.gcsv"))
    assert abs(log.rate_hz - 500) < 1
    f = gyro_features(log, DURATION)
    t = (np.arange(len(f["flip_deg"])) + 0.5) * BIN
    flip_bins = t[f["flip_deg"] >= 270]
    assert flip_bins.min() >= 17.5 and flip_bins.max() <= 21.5, flip_bins
    # 200 Hz vibration (30 deg/s) must not register while idle or cruising.
    idle = (t < 7.5)
    assert np.nanmax(f["gyro_mean"][idle]) < 5
    cruise = (t > 9) & (t < 17)
    assert np.nanmax(f["flip_deg"][cruise]) < 120


def test_gcsv_rejects_garbage(tmp_path: Path):
    p = tmp_path / "x.gcsv"
    p.write_text("hello\n1,2,3\n")
    with pytest.raises(ValueError):
        read_gcsv(p)


def _signals(n, motion, gyro):
    return {
        "motion": np.asarray(motion, float), "sharp": np.full(n, 100.0), "luma": np.full(n, 120.0),
        "audio_db": np.full(n, np.nan), "gyro_mean": np.asarray(gyro, float),
        "gyro_peak": np.asarray(gyro, float) * 1.2, "flip_deg": np.asarray(gyro, float) * 1.5,
    }


def test_scoring_idle_and_peak():
    n = 80
    motion = np.full(n, 8.0)
    gyro = np.full(n, 60.0)
    motion[:16] = 0.2
    gyro[:16] = 2.0          # idle on the ground
    motion[40:46] = 30.0
    gyro[40:46] = 320.0      # big move
    scores, info = score_session([_signals(n, motion, gyro)], load_config())
    s = scores[0]
    assert info[0]["idle"][:16].all()
    assert (s[:16] == 0).all()
    assert 40 <= int(np.argmax(s)) <= 45


def test_find_and_select_respect_lengths():
    n = int(DURATION / BIN)
    score = np.zeros(n)
    score[36:42] = 0.95   # 18-21 s
    score[62:68] = 0.8    # 31-34 s
    score[20:30] = 0.5
    info = {"flip": np.zeros(n, bool)}
    prof = Profile(target=8, min_len=1.5, max_len=6.0, grow_rel=0.65, floor=0.45)
    segs = find_segments(_clip(), score, info, prof)
    assert segs[0].start <= 18 and segs[0].end >= 21
    chosen = select(segs, prof)
    total = sum(s.duration for s in chosen)
    assert abs(total - 8) < 0.1
    assert all(s.duration >= prof.min_len - 1e-6 for s in chosen)
    # chronological, no overlaps
    for a, b in zip(chosen, chosen[1:]):
        assert (a.clip, a.end) <= (b.clip, b.start + 1e-9)


def test_hook_order_puts_best_first():
    segs = [Segment(0, 1, 3, 0.5, 0.6, 2), Segment(0, 10, 12, 0.9, 0.95, 11), Segment(0, 20, 22, 0.7, 0.7, 21)]
    prof = Profile(target=100, min_len=1, max_len=5, grow_rel=0.5, floor=0.1, order="hook")
    out = select(segs, prof)
    assert out[0].start == 10 and [s.start for s in out[1:]] == [1, 20]


def test_snap_to_beats_lands_cuts_on_beats():
    beats = np.arange(0, 60, 60 / 128)
    segs = [Segment(0, 2.0, 5.3, 0.9, 0.9, 3), Segment(0, 10.0, 13.1, 0.8, 0.8, 11),
            Segment(0, 20.0, 22.2, 0.7, 0.7, 21)]
    out = snap_to_beats(segs, beats, [_clip()], min_len=1.5)
    cum = np.cumsum([s.duration for s in out])
    assert np.max([np.min(np.abs(beats - c)) for c in cum]) < 1e-6


def test_frame_counts_do_not_drift():
    segs = [Segment(0, 0, 1.0 / 3, 1, 1, 0) for _ in range(300)]  # 100 s of 1/3 s segments
    counts = frame_counts(segs, 59.94)
    assert sum(counts) == round(100 * 59.94)


def test_gemini_ratings_blend_and_crop():
    seg = Segment(0, 10, 14, 0.5, 0.6, 12)
    rated = [(Segment(0, 9.5, 14.5, 0.5, 0.6, 12), {"id": 0, "score": 9, "crop_x": 0.3, "reason": "tight gap"})]
    apply_ratings([seg], rated, weight=0.4)
    assert abs(seg.score - (0.6 * 0.5 + 0.4 * 0.9)) < 1e-9
    assert seg.crop_x == 0.3 and seg.note == "tight gap"


def test_config_override_and_unknown_key(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text("[reel]\ntarget = 90\n[weights]\ngyro = 0.6\n")
    cfg = load_config(p)
    assert cfg.reel.target == 90 and cfg.weights.gyro == 0.6
    p.write_text("[reel]\ntargett = 90\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_config_rejects_bad_order(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text('[reel]\norder = "random"\n')
    with pytest.raises(ValueError, match="reel.order"):
        load_config(p)


def test_snap_never_grows_into_neighbour():
    beats = np.arange(0, 60, 60 / 128)
    # Two segments of one clip with a 0.4 s gap: growing the first to a beat must stop at the second.
    a = Segment(0, 10.0, 12.6, 0.9, 0.9, 11)
    b = Segment(0, 13.0, 16.0, 0.8, 0.8, 14)
    out = snap_to_beats([a, b], beats, [_clip()], min_len=1.5)
    assert out[0].end <= out[1].start + 1e-9
    assert out[0].start >= 0 and out[1].end <= DURATION


def test_snap_skips_when_no_beat_fits_free_space():
    beats = np.arange(0, 60, 2.0)           # beats too far apart to fit in the 0.2 s gap
    a = Segment(0, 10.0, 11.9, 0.9, 0.9, 11)
    b = Segment(0, 12.1, 14.0, 0.8, 0.8, 13)
    out = snap_to_beats([a, b], beats, [_clip()], min_len=1.5)
    assert out[0].end <= out[1].start + 1e-9


def test_merge_close_joins_near_segments_only():
    from fpv_autocut.planning import _merge_close
    segs = [Segment(0, 0, 2, 0.5, 0.6, 1), Segment(0, 3, 5, 0.9, 0.95, 4), Segment(0, 20, 22, 0.7, 0.7, 21),
            Segment(1, 0, 2, 0.4, 0.4, 1)]
    out = _merge_close(segs, gap=1.5)
    assert [(s.clip, s.start, s.end) for s in out] == [(0, 0, 5), (0, 20, 22), (1, 0, 2)]
    assert out[0].peak == 0.95 and out[0].peak_t == 4
    assert abs(out[0].score - (0.5 * 2 + 0.9 * 2) / 4) < 1e-9


def test_resolve_overlaps_splits_overlap_and_drops_slivers():
    from fpv_autocut.planning import _resolve_overlaps
    segs = [Segment(0, 0, 5, 1, 1, 2), Segment(0, 4, 8, 1, 1, 6), Segment(0, 7.9, 8.0, 1, 1, 7.95)]
    out = _resolve_overlaps(segs)
    assert [(s.start, s.end) for s in out] == [(0, 4.5), (4.5, 7.95)]   # the 0.05 s sliver is dropped
