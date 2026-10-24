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

PROXY_HEIGHT = 360         # proxy sent to Gemini: small and cheap
PROXY_FPS = 5
UPLOAD_TIMEOUT_S = 300.0   # give up on a file that stays in PROCESSING

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


def proxy_name(clip: Clip) -> str:
    """Proxy file name, unique per source path (two flights may both be called GX010001)."""
    return f"{clip.name}-{hashlib.sha1(clip.path.encode()).hexdigest()[:8]}-proxy.mp4"


def _proxy(clip: Clip, dst: Path) -> Path:
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.mp4")  # never leave a half-written proxy under the final name
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", clip.path, "-an",
         "-vf", f"scale=-2:{PROXY_HEIGHT},fps={PROXY_FPS}", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "32", str(tmp)],
        check=True, capture_output=True,
    )
    tmp.replace(dst)
    return dst


def wait_until_active(client, f, timeout: float, poll: float = 2.0):
    """Poll an uploaded file until Gemini has processed it; None on FAILED or timeout."""
    deadline = time.monotonic() + timeout
    while True:
        state = getattr(f.state, "name", str(f.state))
        if state == "ACTIVE":
            return f
        if state == "FAILED" or time.monotonic() > deadline:
            return None
        time.sleep(poll)
        f = client.files.get(name=f.name)


def parse_ratings(text: str | None) -> list[dict] | None:
    """The model's JSON answer as a list of dicts, or None if it is unusable."""
    try:
        data = json.loads(text or "")
    except (TypeError, ValueError):
        return None
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return None
    return [d for d in data if isinstance(d, dict)]


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
            proxy = _proxy(clip, gdir / proxy_name(clip))
            f = client.files.upload(file=str(proxy))
            f = wait_until_active(client, f, timeout=UPLOAD_TIMEOUT_S)
            if f is None:
                log(f"  ! gemini: upload of {clip.name} did not become ready, skipped")
                continue
            resp = client.models.generate_content(
                model=model,
                contents=[f, PROMPT.format(candidates=lines)],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
            )
            rated = parse_ratings(resp.text)
            if rated is None:
                log(f"  ! gemini: unusable answer for {clip.name}, skipped")
                continue
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
