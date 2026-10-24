"""Finding clips, pairing them with gyro logs / stabilized renders, ffprobe."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv"}
STABILIZED_SUFFIX = "_stabilized"  # Gyroflow's default output suffix


class ToolMissing(RuntimeError):
    pass


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise ToolMissing(
                f"'{tool}' not found on PATH. Install ffmpeg (Windows: `winget install Gyan.FFmpeg`)."
            )


@dataclass
class Clip:
    id: int
    path: str                 # original file, used for analysis
    render_path: str          # file used for rendering (stabilized if available)
    gyro_path: str | None
    duration: float
    fps: str                  # rational, e.g. "60000/1001"
    width: int
    height: int
    has_audio: bool

    @property
    def fps_float(self) -> float:
        return float(Fraction(self.fps))

    @property
    def name(self) -> str:
        return Path(self.path).stem

    def to_dict(self) -> dict:
        return asdict(self)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, **kw)


def probe(path: str | Path) -> dict:
    out = run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)]
    ).stdout
    info = json.loads(out)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"{path}: no video stream")
    audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "30/1"
    if rate in ("0/0", "0/1"):
        rate = video.get("r_frame_rate", "30/1")
    duration = float(video.get("duration") or info.get("format", {}).get("duration") or 0.0)
    width, height = int(video["width"]), int(video["height"])
    # Respect rotation metadata (vertical phone footage, etc.).
    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    if abs(rotation) % 180 == 90:
        width, height = height, width
    return {
        "duration": duration,
        "fps": str(Fraction(rate).limit_denominator(1001)),
        "width": width,
        "height": height,
        "has_audio": audio,
    }


def _is_stabilized(p: Path) -> bool:
    return p.stem.lower().endswith(STABILIZED_SUFFIX)


def _find_sibling(p: Path, stem: str, exts: list[str]) -> Path | None:
    for ext in exts:
        for cand in (p.with_name(stem + ext), p.with_name(stem + ext.upper())):
            if cand.exists():
                return cand
    return None


def discover(inputs: list[str], prefer_stabilized: bool = True) -> list[Clip]:
    """Collect source clips from files/directories.

    * ``foo.mp4`` + ``foo.gcsv`` -> gyro log paired.
    * ``foo_stabilized.mp4`` (Gyroflow output) is used for rendering ``foo.mp4``
      when ``prefer_stabilized`` is on; it is never analysed as its own clip.
    """
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(x for x in p.iterdir() if x.suffix.lower() in VIDEO_EXTS))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(item)

    sources = [f for f in files if not _is_stabilized(f)]
    # If the user passed only stabilized files, analyse them directly.
    if not sources:
        sources = files

    clips: list[Clip] = []
    seen: set[Path] = set()
    for f in sources:
        f = f.resolve()
        if f in seen:
            continue
        seen.add(f)
        meta = probe(f)
        gyro = _find_sibling(f, f.stem, [".gcsv"])
        render = f
        if prefer_stabilized and not _is_stabilized(f):
            stab = _find_sibling(f, f.stem + STABILIZED_SUFFIX, [".mp4", ".mov", ".mkv"])
            if stab is not None:
                render = stab.resolve()
        clips.append(
            Clip(
                id=len(clips),
                path=str(f),
                render_path=str(render),
                gyro_path=str(gyro.resolve()) if gyro else None,
                **meta,
            )
        )
    return clips


def render_meta(clip: Clip) -> dict:
    """Probe the render file (may differ from the analysed file in size/fps)."""
    if clip.render_path == clip.path:
        return {"width": clip.width, "height": clip.height, "has_audio": clip.has_audio,
                "duration": clip.duration, "fps": clip.fps}
    return probe(clip.render_path)
