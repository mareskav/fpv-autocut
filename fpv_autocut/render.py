"""Rendering plans to MP4 with ffmpeg, and contact sheets for quick review."""

from __future__ import annotations

import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import Output
from .media import Clip, render_meta
from .planning import Segment


class RenderError(RuntimeError):
    pass


def _ff(cmd: list[str], cwd: Path | None = None) -> None:
    res = subprocess.run(cmd, capture_output=True, cwd=cwd)
    if res.returncode != 0:
        raise RenderError(
            "ffmpeg failed:\n  " + " ".join(cmd) + "\n" + res.stderr.decode(errors="replace").strip()[-3000:]
        )


def _video_filter(out: Output, crop_x: float, draft: bool) -> str:
    w, h = out.width, out.height
    if draft:
        w, h = (w // 2) // 2 * 2, (h // 2) // 2 * 2
    a = out.width / out.height
    cx = min(max(crop_x, 0.0), 1.0)
    return (
        f"crop=w='min(iw,ih*{a:.6f})':h='min(ih,iw/{a:.6f})':x='(iw-ow)*{cx:.4f}':y='(ih-oh)/2',"
        f"scale={w}:{h}:flags=lanczos,setsar=1,fps={out.fps},format=yuv420p,"
        # If the source ends before the segment does, hold the last frame rather than
        # delivering fewer frames than the (already padded) audio.
        f"tpad=stop_mode=clone:stop_duration={out.tail_pad_s}"
    )


def _encode_args(out: Output, draft: bool) -> list[str]:
    if draft:
        return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28"]
    if out.vcodec == "libx264":
        return ["-c:v", "libx264", "-preset", out.preset, "-crf", str(out.crf)]
    return ["-c:v", out.vcodec, "-b:v", out.bitrate]


def _concat_quote(p: Path) -> str:
    """Escape a path for a single-quoted ffmpeg concat-demuxer entry."""
    return p.as_posix().replace("'", "'\\''")


def frame_counts(segments: list[Segment], fps: float) -> list[int]:
    """Whole frames per segment, rounded on the *cumulative* timeline.

    Rounding each segment separately would let cut points drift (up to half a
    frame per cut, ~0.3 s over a 3-minute reel), breaking beat sync.
    """
    counts, cum = [], 0.0
    for s in segments:
        start_f = round(cum * fps)
        cum += s.duration
        counts.append(max(1, round(cum * fps) - start_f))
    return counts


def render_segment(seg: Segment, clip: Clip, has_audio: bool, out: Output, dst: Path, draft: bool,
                   n_frames: int) -> None:
    dur = n_frames / out.fps
    read = dur + out.source_margin_s
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{seg.start:.3f}", "-t", f"{read:.3f}",
           "-i", clip.render_path]
    if has_audio:
        amap = ["-map", "0:v:0", "-map", "0:a:0"]
    else:
        cmd += ["-f", "lavfi", "-t", f"{read:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        amap = ["-map", "0:v:0", "-map", "1:a:0"]
    fade = min(out.edge_fade_s, dur / 4)
    cmd += amap + [
        "-vf", _video_filter(out, seg.crop_x, draft),
        # apad + atrim: audio is exactly as long as the video, even if the source runs short.
        "-af", f"aresample=48000,aformat=channel_layouts=stereo,apad,atrim=0:{dur:.6f},"
               f"afade=t=in:d={fade:.3f},afade=t=out:st={max(dur - fade, 0):.3f}:d={fade:.3f}",
        *_encode_args(out, draft),
        "-frames:v", str(n_frames),
        # PCM in MKV for the parts: AAC adds ~21 ms of priming per part, which
        # would pile up into audible A/V drift after concatenation.
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    _ff(cmd)


def render_plan_output(name: str, segments: list[Segment], clips: list[Clip], out: Output, dst: Path, *,
                       music: str | None = None, draft: bool = False, jobs: int = 2, log=print) -> Path:
    if not segments:
        raise RenderError(f"{name}: no segments to render")
    dst.parent.mkdir(parents=True, exist_ok=True)
    audio_of = {c.id: render_meta(c)["has_audio"] for c in clips}
    counts = frame_counts(segments, out.fps)
    total = sum(counts) / out.fps
    with tempfile.TemporaryDirectory(prefix=f"fpvac-{name}-", dir=dst.parent) as tmp:
        tmpd = Path(tmp)
        parts = [tmpd / f"seg_{i:04d}.mkv" for i in range(len(segments))]
        log(f"  {name}: rendering {len(segments)} segments ({total:.1f} s)")
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            futs = [ex.submit(render_segment, s, clips[s.clip], audio_of[s.clip], out, p, draft, n)
                    for s, p, n in zip(segments, parts, counts)]
            for f in futs:
                f.result()
        listing = tmpd / "concat.txt"
        listing.write_text("".join(f"file '{_concat_quote(p)}'\n" for p in parts), encoding="utf-8")
        joined = tmpd / "joined.mkv"
        _ff(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c", "copy", str(joined)])
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(joined)]
        if music:
            fd = min(out.music_fade_s, total / 4)
            chain = (f"[1:a]aresample=48000,aformat=channel_layouts=stereo,apad,atrim=0:{total:.6f},"
                     f"asetpts=PTS-STARTPTS,afade=t=out:st={max(total - fd, 0):.3f}:d={fd:.3f}[m]")
            if out.source_audio_under_music > 0:
                chain += (f";[0:a]volume={out.source_audio_under_music}[s];"
                          f"[m][s]amix=inputs=2:duration=first:normalize=0[aout]")
                amap = "[aout]"
            else:
                amap = "[m]"
            cmd += ["-i", music, "-filter_complex", chain, "-map", "0:v", "-map", amap]
        else:
            cmd += ["-map", "0:v", "-map", "0:a"]
        cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(dst)]
        _ff(cmd)
    return dst


def contact_sheet(segments: list[Segment], clips: list[Clip], dst: Path, cols: int = 6,
                  thumb: tuple[int, int] = (320, 180)) -> Path:
    """One thumbnail per segment (at its peak), tiled in plan order."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    thumb_w, thumb_h = thumb
    with tempfile.TemporaryDirectory(prefix="fpvac-sheet-", dir=dst.parent) as tmp:
        tmpd = Path(tmp)
        for i, s in enumerate(segments):
            t = min(max(s.peak_t, s.start), max(s.end - 0.05, s.start))
            _ff(["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{t:.3f}", "-i", clips[s.clip].render_path,
                 "-frames:v", "1",
                 "-vf", f"scale={thumb_w}:{thumb_h}:force_original_aspect_ratio=decrease,"
                        f"pad={thumb_w}:{thumb_h}:(ow-iw)/2:(oh-ih)/2",
                 "-q:v", "3", "-update", "1", str(tmpd / f"t_{i:04d}.jpg")])
        cols = max(1, min(cols, len(segments)))
        rows = (len(segments) + cols - 1) // cols
        # Relative pattern + cwd: a '%' in the temp path would otherwise be read as a format spec.
        _ff(["ffmpeg", "-v", "error", "-nostdin", "-y", "-framerate", "1", "-i", "t_%04d.jpg",
             "-vf", f"tile={cols}x{rows}:padding=4:margin=4", "-frames:v", "1", "-q:v", "3",
             "-update", "1", str(dst.resolve())], cwd=tmpd)
    return dst
