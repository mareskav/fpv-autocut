"""All tunable numbers in one place.

Every field can be overridden from a TOML file (``--config my.toml``), e.g.::

    [weights]
    gyro = 0.5

    [reel]
    target = 150
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path


ORDERS = ("chrono", "score", "hook")


@dataclass
class Weights:
    """How much each signal contributes to the per-bin highlight score.

    Weights of signals that are missing for a clip (no .gcsv, no audio) are
    dropped and the rest renormalised.
    """

    gyro: float = 0.40    # rotation rate (deg/s), percentile-ranked
    flip: float = 0.25    # flips / rolls / loops detected from integrated gyro
    motion: float = 0.30  # frame-to-frame pixel change: speed + proximity proxy
    audio: float = 0.05   # loudness: throttle punches


@dataclass
class Detect:
    gyro_lowpass_hz: float = 50.0      # same cut-off Gyroflow users use for Thumb Pro
    flip_window_s: float = 1.5         # window for integrating rotation per axis
    flip_min_deg: float = 180.0        # rotation in window where flip bonus starts
    flip_full_deg: float = 360.0       # rotation in window that gives full flip bonus
    idle_gyro_dps: float = 20.0        # below this AND low motion => on the ground / idle
    idle_motion_pct: float = 0.15      # motion percentile below which a bin can be idle
    idle_motion_abs: float = 1.0       # mean |frame diff| (0-255) below which the image is "still"
    dark_luma: float = 18.0            # mean luma below this is unusable
    blown_luma: float = 240.0          # mean luma above this is unusable
    smooth_bins: int = 3               # moving-average window on the final score
    analysis_fps: float = 10.0         # frames per second decoded for visual signals
    analysis_width: int = 160          # width of the decoded analysis frames
    audio_sr: int = 8000               # sample rate the audio is decoded at for loudness


@dataclass
class Profile:
    """Segment shape and total length for one kind of output."""

    target: float               # seconds of output
    min_len: float              # shortest segment
    max_len: float              # longest segment
    grow_rel: float             # grow a segment while score >= peak * grow_rel
    floor: float                # never use segments whose peak score is below this
    pad: float = 0.25           # extra seconds on both sides
    overlap_tol: float = 1.0    # candidates overlapping an already chosen segment by less than this are kept (trimmed)
    min_keep: float = 0.2       # segments shorter than this after trimming are dropped
    order: str = "chrono"       # one of ORDERS; "hook" = best first, rest chronological


@dataclass
class Output:
    width: int = 1920
    height: int = 1080
    fps: float = 60.0
    vcodec: str = "libx264"     # e.g. h264_nvenc, h264_qsv, h264_amf, h264_mf
    crf: int = 18               # libx264 only
    bitrate: str = "30M"        # non-libx264 encoders
    preset: str = "veryfast"
    music_fade_s: float = 2.0
    source_audio_under_music: float = 0.0  # 0 = music only, 0.15 = quiet motors underneath
    edge_fade_s: float = 0.02      # audio fade in/out on every segment, hides clicks at cuts
    source_margin_s: float = 0.5   # extra source read past a segment, so the encoder never starves
    tail_pad_s: float = 1.0        # freeze-frame padding if the source ends before the segment does


@dataclass
class Music:
    bpm_min: float = 70.0       # tempo search range of the beat tracker
    bpm_max: float = 180.0
    bpm_prior: float = 120.0    # tempi near this are preferred, to avoid half/double-time picks


@dataclass
class Config:
    weights: Weights = field(default_factory=Weights)
    detect: Detect = field(default_factory=Detect)
    reel: Profile = field(
        default_factory=lambda: Profile(target=180, min_len=1.5, max_len=6.0, grow_rel=0.65, floor=0.45)
    )
    long: Profile = field(
        default_factory=lambda: Profile(target=720, min_len=4.0, max_len=30.0, grow_rel=0.40, floor=0.30, pad=0.5)
    )
    short: Profile = field(
        default_factory=lambda: Profile(target=30, min_len=1.5, max_len=5.0, grow_rel=0.70, floor=0.50, order="hook")
    )
    shorts_count: int = 2
    short_min_fill: float = 0.5   # a short is dropped if it fills less than this share of its target
    long_merge_gap: float = 3.0   # long version: merge segments of one clip closer than this (s)
    music: Music = field(default_factory=Music)
    horizontal: Output = field(default_factory=Output)
    vertical: Output = field(default_factory=lambda: Output(width=1080, height=1920))
    prefer_stabilized: bool = True
    gemini_model: str = "gemini-flash-latest"
    gemini_weight: float = 0.4  # share of Gemini's rating in the final segment score
    gemini_max_candidates: int = 60  # cost cap: only the strongest moments are sent

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> "Config":
        for name in ("reel", "long", "short"):
            order = getattr(self, name).order
            if order not in ORDERS:
                raise ValueError(f"Config key '{name}.order' must be one of {', '.join(ORDERS)}; got '{order}'")
        return self


def _apply(obj, data: dict, where: str) -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"Unknown config key '{where}{key}'")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Config key '{where}{key}' must be a table")
            _apply(current, value, f"{where}{key}.")
        else:
            setattr(obj, key, type(current)(value) if current is not None else value)


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    cfg = Config()
    if path:
        with open(path, "rb") as fh:
            _apply(cfg, tomllib.load(fh), "")
    if overrides:
        _apply(cfg, overrides, "")
    return cfg.validate()


def config_from_dict(data: dict) -> Config:
    cfg = Config()
    _apply(cfg, data, "")
    return cfg.validate()
