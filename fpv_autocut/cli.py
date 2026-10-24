"""Command line interface.

    fpv-autocut run   FOOTAGE... -o OUT [--music song.mp3] [--gemini]   # analyse + plan + render
    fpv-autocut plan  FOOTAGE... -o OUT [...]                         # analyse + plan only
    fpv-autocut render OUT/plan.json [--only reel,short_01] [--draft]  # render (after editing the plan)
    fpv-autocut sheet OUT/plan.json                                    # contact sheets only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from . import __version__
from .config import load_config
from .media import ToolMissing, discover, require_ffmpeg
from .pipeline import build_plan, make_sheets, render_plan


def _log(msg: str) -> None:
    sys.stdout.write(msg + "\n")  # one write per line: safe from parallel workers
    sys.stdout.flush()


def _common_plan_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("inputs", nargs="+", help="video files or folders (pairs foo.mp4 with foo.gcsv)")
    p.add_argument("-o", "--out", default="autocut_out", help="output/work folder (default: autocut_out)")
    p.add_argument("--config", help="TOML file overriding defaults (see fpv_autocut/config.py)")
    p.add_argument("--music", help="music track for reel and shorts; cuts snap to its beats")
    p.add_argument("--no-beat-sync", action="store_true", help="keep music but do not snap cuts to beats")
    p.add_argument("--reel", type=float, help="reel length in seconds (default 180)")
    p.add_argument("--long", type=float, help="long version length in seconds (default 720)")
    p.add_argument("--shorts", type=int, help="number of vertical shorts (default 2)")
    p.add_argument("--short-len", type=float, help="length of each short in seconds (default 30)")
    p.add_argument("--gemini", action="store_true", help="let Gemini rate candidate moments (needs GEMINI_API_KEY)")
    p.add_argument("--no-stabilized", action="store_true", help="ignore *_stabilized.mp4 Gyroflow renders")
    p.add_argument("--gyro-offset", type=float, default=0.0,
                   help="seconds to shift gyro data vs video (centre of the auto-sync search)")
    p.add_argument("--no-gyro-sync", action="store_true",
                   help="do not auto-align gyro to video; use --gyro-offset as is")
    p.add_argument("--hwaccel", help="ffmpeg -hwaccel for analysis decoding, e.g. auto, d3d11va, cuda")
    p.add_argument("-j", "--jobs", type=int, default=2, help="parallel ffmpeg jobs (default 2)")


def _overrides(a: argparse.Namespace) -> dict:
    o: dict = {}
    if a.reel is not None:
        o["reel"] = {"target": a.reel}
    if a.long is not None:
        o["long"] = {"target": a.long}
    if a.short_len is not None:
        o["short"] = {"target": a.short_len}
    if a.shorts is not None:
        o["shorts_count"] = a.shorts
    if a.no_stabilized:
        o["prefer_stabilized"] = False
    if a.no_gyro_sync:
        o["detect"] = {"gyro_sync": False}
    return o


def check_out_dir(out: Path, inputs: list[str]) -> None:
    """Refuse an output folder that is (or sits inside) the footage folder: footage is read-only."""
    out_r = out.resolve()
    for item in inputs:
        p = Path(item).resolve()
        folder = p if p.is_dir() else p.parent
        if out_r == folder or folder in out_r.parents:
            raise SystemExit(
                f"error: output folder {out} is inside the footage folder {folder}. "
                "Pick another -o folder: the footage folder is never written to."
            )


def _plan(a: argparse.Namespace) -> Path:
    cfg = load_config(a.config, _overrides(a))
    out = Path(a.out)
    check_out_dir(out, a.inputs)
    out.mkdir(parents=True, exist_ok=True)
    clips = discover(a.inputs, prefer_stabilized=cfg.prefer_stabilized, log=_log)
    if not clips:
        raise SystemExit("No usable video files found.")
    for c in clips:
        stab = " (render: stabilized)" if c.render_path != c.path else ""
        gyro = "gyro" if c.gyro_path else "NO gyro"
        print(f"  {Path(c.path).name}: {c.duration:.0f} s, {c.width}x{c.height} @ {c.fps_float:.2f}, {gyro}{stab}")
    build_plan(clips, cfg, out, music=a.music, use_gemini=a.gemini, beat_sync=not a.no_beat_sync,
               gyro_offset=a.gyro_offset, hwaccel=a.hwaccel, jobs=a.jobs, log=_log)
    plan = out / "plan.json"
    print(f"Plan: {plan}\nReport: {out / 'report.md'}")
    return plan


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fpv-autocut", description="Automatic FPV highlight editing.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="analyse, plan and render everything")
    _common_plan_args(p_run)
    p_run.add_argument("--draft", action="store_true", help="fast half-resolution preview render")
    p_run.add_argument("--only", help="comma-separated outputs to render, e.g. reel,short_01")

    p_plan = sub.add_parser("plan", help="analyse and write plan.json without rendering")
    _common_plan_args(p_plan)
    p_plan.add_argument("--sheets", action="store_true", help="also write contact sheets")

    p_render = sub.add_parser("render", help="render outputs from an (edited) plan.json")
    p_render.add_argument("plan")
    p_render.add_argument("--only", help="comma-separated outputs, e.g. reel,short_01")
    p_render.add_argument("--draft", action="store_true", help="fast half-resolution preview render")
    p_render.add_argument("--no-fcpxml", action="store_true")
    p_render.add_argument("-j", "--jobs", type=int, default=2)

    p_sheet = sub.add_parser("sheet", help="write contact sheets (one thumbnail per segment)")
    p_sheet.add_argument("plan")

    a = ap.parse_args(argv)
    t0 = time.time()
    try:
        require_ffmpeg()
        if a.cmd in ("run", "plan"):
            plan = _plan(a)
            if a.cmd == "plan":
                if a.sheets:
                    for p in make_sheets(plan):
                        print(f"  {p}")
            else:
                only = a.only.split(",") if a.only else None
                for p in render_plan(plan, only=only, draft=a.draft, jobs=a.jobs, log=_log):
                    print(f"  {p}")
        elif a.cmd == "render":
            only = a.only.split(",") if a.only else None
            for p in render_plan(Path(a.plan), only=only, draft=a.draft, fcpxml=not a.no_fcpxml, jobs=a.jobs, log=_log):
                print(f"  {p}")
        elif a.cmd == "sheet":
            for p in make_sheets(Path(a.plan)):
                print(f"  {p}")
    except ToolMissing as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: not found: {exc}", file=sys.stderr)
        return 2
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    print(f"Done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
