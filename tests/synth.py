"""Synthetic FPV-like footage with known events, for tests and demos.

Timeline of the generated 40 s clip (640x480 @ 30 fps, 4:3 like Thumb Pro 1440p):

    0-8   idle on the ground (still frame, no rotation)
    8-18  slow cruise (slow pan, gentle yaw)
    18-21 FLIP (image rotating ~300 deg/s, strong roll)   <- best moment
    21-31 slow cruise
    31-34 fast low pass (fast pan, strong yaw)            <- second best
    34-40 idle (landed)

Run directly to write files into a folder:  python tests/synth.py OUTDIR
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np

W, H, FPS = 640, 480, 30
EVENTS = [  # (start, end, kind)
    (0, 8, "idle"), (8, 18, "cruise"), (18, 21, "flip"),
    (21, 31, "cruise"), (31, 34, "fast"), (34, 40, "idle"),
]
DURATION = 40


def _ff(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *args], check=True)


def make_video(dst: Path, with_audio: bool = True) -> Path:
    tmp = dst.parent / f".{dst.stem}_parts"
    tmp.mkdir(parents=True, exist_ok=True)
    pattern = tmp / "pattern.png"
    _ff(["-f", "lavfi", "-i", "testsrc2=s=1600x1200:r=1", "-frames:v", "1", str(pattern)])
    parts = []
    for i, (a, b, kind) in enumerate(EVENTS):
        d = b - a
        if kind == "idle":
            vf = f"crop={W}:{H}:400:300"
        elif kind == "cruise":
            vf = f"crop={W}:{H}:x='200+60*t':y=300"
        elif kind == "fast":
            vf = f"crop={W}:{H}:x='100+250*t':y='300+40*sin(6*t)'"
        else:  # flip
            vf = f"crop=1000:1000:300:100,rotate='2*PI*t/1.2':c=black,crop={W}:{H}"
        part = tmp / f"p{i}.mp4"
        _ff(["-loop", "1", "-framerate", str(FPS), "-t", str(d), "-i", str(pattern),
             "-vf", f"{vf},format=yuv420p", "-r", str(FPS), "-c:v", "libx264", "-preset", "ultrafast",
             "-crf", "23", str(part)])
        parts.append(part)
    listing = tmp / "list.txt"
    listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
    video_only = tmp / "video.mp4"
    _ff(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(video_only)])
    if with_audio:
        # "Motor" noise, much louder during the flip and the fast pass.
        loud = "+".join(f"between(t,{a},{b})*0.9" for a, b, k in EVENTS if k in ("flip", "fast"))
        expr = f"(0.05+{loud})*sin(2*PI*180*t)*(0.7+0.3*sin(2*PI*7*t))"
        _ff(["-i", str(video_only), "-f", "lavfi", "-t", str(DURATION), "-i",
             f"aevalsrc='{expr}':s=48000", "-c:v", "copy", "-c:a", "aac", "-shortest", str(dst)])
    else:
        _ff(["-i", str(video_only), "-c", "copy", str(dst)])
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()
    return dst


def gyro_series(rate_hz: int = 500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(t seconds, gyro deg/s (N,3)) matching EVENTS, with 200 Hz motor vibration."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, DURATION, 1 / rate_hz)
    g = np.zeros((len(t), 3))
    for a, b, kind in EVENTS:
        m = (t >= a) & (t < b)
        if kind == "cruise":
            g[m, 2] = 15 + 10 * np.sin(2 * np.pi * 0.5 * t[m])
            g[m, 0] = 8 * np.sin(2 * np.pi * 1.3 * t[m])
        elif kind == "flip":
            g[m, 1] = 300.0
        elif kind == "fast":
            g[m, 2] = 120.0
            g[m, 0] = 40 * np.sin(2 * np.pi * 2 * t[m])
    flying = ~np.isin(np.floor(t), [*range(0, 8), *range(34, 40)])
    g += rng.normal(0, 1.5, g.shape)
    g[flying] += 30 * np.sin(2 * np.pi * 200 * t[flying])[:, None]  # vibration, cancelled by LPF / integration
    return t, g


def make_gcsv(dst: Path, rate_hz: int = 500) -> Path:
    t, g = gyro_series(rate_hz)
    gscale = 0.001  # raw units -> rad/s
    raw = np.round(np.radians(g) / gscale).astype(int)
    tt = np.round(t * 1000).astype(int)
    lines = ["GYROFLOW IMU LOG", "version,1.3", "id,synthetic_thumb", "orientation,YxZ",
             "vendor,runcam", "tscale,0.001", f"gscale,{gscale}", "ascale,0.00048828125", "t,gx,gy,gz,ax,ay,az"]
    lines += [f"{a},{b},{c},{d},0,-2048,0" for a, (b, c, d) in zip(tt, raw)]
    dst.write_text("\n".join(lines) + "\n")
    return dst


def make_music(dst: Path, bpm: float = 128, seconds: float = 60) -> Path:
    period = 60 / bpm
    expr = f"0.8*sin(2*PI*55*t)*exp(-25*mod(t,{period}))+0.1*sin(2*PI*440*t)"
    _ff(["-f", "lavfi", "-t", str(seconds), "-i", f"aevalsrc='{expr}':s=44100", "-c:a", "libmp3lame",
         "-b:a", "192k", str(dst)])
    return dst


def make_session(folder: Path) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    flight = make_video(folder / "flight1.mp4")
    make_gcsv(folder / "flight1.gcsv")
    nogyro = make_video(folder / "flight2.mp4", with_audio=False)
    music = make_music(folder / "music.mp3")
    return {"flight": flight, "nogyro": nogyro, "music": music}


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
    for k, v in make_session(out).items():
        print(k, v)
