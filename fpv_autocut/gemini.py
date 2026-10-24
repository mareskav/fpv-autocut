"""Optional: let Gemini watch candidate moments and rate them.

Signals find *where* something happens (rotation, speed); Gemini judges
whether it *looks* good (proximity to trees/buildings, gaps, scenery, a clean
horizon) and where the visual subject is for 9:16 crops.

Needs ``pip install fpv-autocut[gemini]`` and ``GEMINI_API_KEY`` in the env.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from .media import Clip
from .planning import Segment

PROMPT = """You are a senior FPV drone video editor choosing shots for a highlight reel.
The attached video is a low-resolution proxy of one FPV flight (no audio).
Rate each candidate time range below for how much it belongs in a highlight reel.

Reward: close proximity flying (gaps, trees, buildings), dives, flips/rolls/power loops
that read clearly, fast low passes, reveals, striking scenery or light.
Penalise: blurry or shaky frames, sky-only or ground-only frames, props or frame in view
dominating the picture, crashes that end the shot abruptly, nothing happening.

Candidates (seconds from the start of the video):
{candidates}

Answer with a JSON array, one object per candidate, same ids:
[{{"id": <int>, "score": <1-10>, "crop_x": <0.0-1.0 horizontal position of the main visual interest, 0.5 if centred>, "reason": "<max 12 words>"}}]
"""


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def _proxy(clip: Clip, dst: Path) -> Path:
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", clip.path, "-an",
         "-vf", "scale=-2:360,fps=5", "-c:v", "libx264", "-preset", "veryfast", "-crf", "32", str(dst)],
        check=True, capture_output=True,
    )
    return dst


def rate_segments(segments: list[Segment], clips: list[Clip], work: Path, model: str,
                  log=print) -> list[tuple[Segment, dict]]:
    """Return [(rated segment, {"score": 1-10, "crop_x": ..., "reason": ...}), ...]."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Gemini scoring needs: pip install \"fpv-autocut[gemini]\"") from exc

    client = genai.Client()  # reads GEMINI_API_KEY
    gdir = work / "gemini"
    results: list[tuple[Segment, dict]] = []
    by_clip: dict[int, list[Segment]] = {}
    for s in segments:
        by_clip.setdefault(s.clip, []).append(s)

    for cid, segs in by_clip.items():
        clip = clips[cid]
        segs = sorted(segs, key=lambda s: s.start)
        lines = "\n".join(f"- id {i}: {_mmss(s.start)}-{_mmss(s.end)} ({s.start:.1f}-{s.end:.1f} s)"
                          for i, s in enumerate(segs))
        key = hashlib.sha1(f"{model}|{clip.path}|{lines}".encode()).hexdigest()[:16]
        cache = gdir / f"{clip.name}-{key}.json"
        if cache.exists():
            rated = json.loads(cache.read_text(encoding="utf-8"))
        else:
            log(f"  gemini: rating {len(segs)} moments in {clip.name}")
            proxy = _proxy(clip, gdir / f"{clip.name}-proxy.mp4")
            f = client.files.upload(file=str(proxy))
            while getattr(f.state, "name", str(f.state)) == "PROCESSING":
                time.sleep(2)
                f = client.files.get(name=f.name)
            resp = client.models.generate_content(
                model=model,
                contents=[f, PROMPT.format(candidates=lines)],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
            )
            rated = json.loads(resp.text)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(rated, indent=2, ensure_ascii=False), encoding="utf-8")
            try:
                client.files.delete(name=f.name)
            except Exception:  # noqa: BLE001 - cleanup is best effort
                pass
        for item in rated:
            i = int(item.get("id", -1))
            if 0 <= i < len(segs):
                results.append((segs[i], item))
    return results


def _best_match(s: Segment, ratings: list[tuple[Segment, dict]]) -> dict | None:
    best, best_ov = None, 0.0
    for r_seg, item in ratings:
        if r_seg.clip != s.clip:
            continue
        ov = min(s.end, r_seg.end) - max(s.start, r_seg.start)
        if ov > best_ov:
            best, best_ov = item, ov
    return best if best_ov >= min(0.5 * s.duration, 2.0) else None


def apply_ratings(segments: list[Segment], ratings: list[tuple[Segment, dict]], weight: float) -> list[Segment]:
    """Blend Gemini's rating into every segment that overlaps a rated moment."""
    for s in segments:
        r = _best_match(s, ratings)
        if not r:
            continue
        g = min(max(float(r.get("score", 5)), 1), 10) / 10
        s.score = (1 - weight) * s.score + weight * g
        s.crop_x = min(max(float(r.get("crop_x", 0.5)), 0.0), 1.0)
        s.note = str(r.get("reason", ""))[:120]
    return segments
