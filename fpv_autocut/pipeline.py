"""Glue: analysis -> plan.json -> rendered outputs.

``plan.json`` is the hand-off point. It lists every output with its segments
(clip, start, end, score, tags, crop_x). Edit it (by hand or with Claude Code)
and run ``fpv-autocut render plan.json`` to re-render without re-analysing.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path

from . import BIN, __version__
from .beats import track_beats
from .config import Config, config_from_dict
from .fcpxml import write_fcpxml
from .media import Clip, music_duration
from .planning import Segment, auto_crop_x, find_segments, select, snap_to_beats
from .render import contact_sheet, render_plan_output
from .scoring import score_session
from .signals import analyze_all, dump_json, write_timeline_csv



def build_plan(clips: list[Clip], cfg: Config, out_dir: Path, *, music: str | None = None,
               use_gemini: bool = False, beat_sync: bool = True, gyro_offset: float = 0.0,
               hwaccel: str | None = None, jobs: int = 2, log=print) -> dict:
    log(f"Analysing {len(clips)} clip(s)...")
    signals = analyze_all(clips, cfg, out_dir, gyro_offset=gyro_offset, hwaccel=hwaccel, jobs=jobs, log=log)
    scores, info = score_session(signals, cfg)
    write_timeline_csv(out_dir / "timeline.csv", clips, signals, scores)

    def cands(prof):
        out: list[Segment] = []
        for c, sc, inf in zip(clips, scores, info):
            out.extend(find_segments(c, sc, inf, prof))
        return out

    reel_c, long_c = cands(cfg.reel), cands(cfg.long)
    short_c = cands(cfg.short)

    if use_gemini:
        from .gemini import apply_ratings, rate_segments
        top = sorted(reel_c, key=lambda s: -s.score)[:cfg.gemini_max_candidates]
        ratings = rate_segments(top, clips, out_dir, cfg.gemini_model, log=log)
        for group in (reel_c, long_c, short_c):
            apply_ratings(group, ratings, cfg.gemini_weight)

    outputs: dict[str, dict] = {}
    bpm, beats = None, None
    if music and beat_sync:
        bpm, beats = track_beats(music, cfg.music)
        log(f"Music: {bpm:.1f} BPM, {len(beats)} beats")

    reel = select(reel_c, cfg.reel)
    if beats is not None:
        reel = snap_to_beats(reel, beats, clips, cfg.reel.min_len)
    if music:
        _warn_if_music_short("reel", reel, music, log)
    outputs["reel"] = _out_entry(reel, "horizontal", music)

    long = select(long_c, cfg.long, merge_gap=cfg.long_merge_gap)
    outputs["long"] = _out_entry(long, "horizontal", None)

    used: list[Segment] = []
    for k in range(cfg.shorts_count):
        segs = select(short_c, cfg.short, exclude=used)
        if sum(s.duration for s in segs) < cfg.short.target * cfg.short_min_fill:
            break
        if beats is not None:
            segs = snap_to_beats(segs, beats, clips, cfg.short.min_len)
        if cfg.auto_crop:
            for s in segs:
                if not s.note:   # a Gemini rating already chose the crop
                    c = clips[s.clip]
                    s.crop_x = auto_crop_x(s, signals[s.clip]["cx"], signals[s.clip]["motion"],
                                           c.width / c.height, cfg.vertical.width / cfg.vertical.height)
        used += segs
        outputs[f"short_{k + 1:02d}"] = _out_entry(segs, "vertical", music)

    plan = {
        "tool": "fpv-autocut",
        "version": __version__,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "bin_s": BIN,
        "music": {"path": music, "bpm": bpm} if music else None,
        "gemini": use_gemini,
        "config": cfg.to_dict(),
        "clips": [c.to_dict() for c in clips],
        "stats": _stats(clips, scores, info),
        "outputs": outputs,
    }
    dump_json(out_dir / "plan.json", plan)
    (out_dir / "report.md").write_text(report_md(plan), encoding="utf-8")
    return plan


def _warn_if_music_short(name: str, segs: list[Segment], music: str, log) -> None:
    total = sum(s.duration for s in segs)
    mlen = music_duration(music)
    if mlen is not None and mlen < total - 0.05:
        log(f"  ! {name}: music is {mlen:.0f} s but the video is {total:.0f} s; the rest will be silent")


def _out_entry(segs: list[Segment], orientation: str, music: str | None) -> dict:
    return {
        "orientation": orientation,
        "music": music,
        "duration": round(sum(s.duration for s in segs), 2),
        "segments": [s.to_dict() for s in segs],
    }


def _stats(clips, scores, info) -> list[dict]:
    out = []
    for c, sc, inf in zip(clips, scores, info):
        out.append({
            "clip": c.name,
            "duration": round(c.duration, 1),
            "gyro": c.gyro_path is not None,
            "idle_s": round(float(inf["idle"].sum()) * BIN, 1),
            "flip_bins": int(inf["flip"].sum()),
            "mean_score": round(float(sc.mean()), 3),
            "max_score": round(float(sc.max()), 3),
        })
    return out


def report_md(plan: dict) -> str:
    clips = plan["clips"]
    lines = [f"# fpv-autocut plan ({plan['created']})", ""]
    lines += ["## Clips", "", "| clip | length | gyro | idle | flip bins | max score |", "|---|---|---|---|---|---|"]
    for s in plan["stats"]:
        lines.append(f"| {s['clip']} | {s['duration']} s | {'yes' if s['gyro'] else 'NO'} | {s['idle_s']} s | "
                     f"{s['flip_bins']} | {s['max_score']} |")
    for name, o in plan["outputs"].items():
        lines += ["", f"## {name} ({o['orientation']}, {o['duration']} s, {len(o['segments'])} segments)", "",
                  "| # | clip | in | out | len | score | tags | note |", "|---|---|---|---|---|---|---|---|"]
        for i, s in enumerate(o["segments"], 1):
            lines.append(f"| {i} | {Path(clips[s['clip']]['path']).stem} | {s['start']:.2f} | {s['end']:.2f} | "
                         f"{s['end'] - s['start']:.2f} | {s['score']:.2f} | {' '.join(s['tags'])} | {s.get('note', '')} |")
    return "\n".join(lines) + "\n"


def load_plan(path: Path) -> tuple[dict, list[Clip], Config]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    clips = [Clip(**c) for c in plan["clips"]]
    cfg = config_from_dict(plan["config"])
    return plan, clips, cfg


def render_plan(plan_path: Path, *, only: list[str] | None = None, draft: bool = False,
                fcpxml: bool = True, sheets: bool = True, jobs: int = 2, log=print) -> list[Path]:
    plan, clips, cfg = load_plan(plan_path)
    out_dir = Path(plan_path).parent
    written: list[Path] = []
    for name, o in plan["outputs"].items():
        if only and name not in only:
            continue
        segs = [Segment.from_dict(s) for s in o["segments"]]
        if not segs:
            log(f"  {name}: empty, skipped")
            continue
        out = cfg.vertical if o["orientation"] == "vertical" else cfg.horizontal
        suffix = "_draft" if draft else ""
        dst = out_dir / f"{name}{suffix}.mp4"
        written.append(render_plan_output(name, segs, clips, out, dst, music=o.get("music"),
                                          draft=draft, jobs=jobs, log=log))
        if fcpxml:
            written.append(write_fcpxml(out_dir / f"{name}.fcpxml", name, segs, clips, out, music=o.get("music")))
        if sheets:
            written.append(contact_sheet(segs, clips, out_dir / "sheets" / f"{name}.jpg"))
    return written


def make_sheets(plan_path: Path, log=print) -> list[Path]:
    plan, clips, _ = load_plan(plan_path)
    out_dir = Path(plan_path).parent
    paths = []
    for name, o in plan["outputs"].items():
        segs = [Segment.from_dict(s) for s in o["segments"]]
        if segs:
            paths.append(contact_sheet(segs, clips, out_dir / "sheets" / f"{name}.jpg"))
    return paths


__all__ = ["build_plan", "render_plan", "make_sheets", "load_plan", "report_md"]
