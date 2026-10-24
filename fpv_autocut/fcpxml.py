"""FCPXML 1.8 export so DaVinci Resolve can import the cut as an editable timeline.

Resolve: File > Import > Timeline... and pick the .fcpxml. Media is linked to
the original files (stabilized ones if they were used for rendering).
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from xml.sax.saxutils import quoteattr

from .config import Output
from .media import Clip, music_duration, render_meta
from .planning import Segment
from .render import frame_counts


def _t(seconds: float | Fraction, fd: Fraction) -> str:
    frames = round(Fraction(seconds) / fd)
    val = frames * fd
    if val == 0:
        return "0s"
    if val.denominator == 1:
        return f"{val.numerator}s"
    return f"{val.numerator}/{val.denominator}s"


def _fd(fps: str | float) -> Fraction:
    f = Fraction(fps).limit_denominator(1001) if not isinstance(fps, Fraction) else fps
    return 1 / f


def _reframe(seg: Segment, m: dict, out: Output) -> str:
    """adjust-transform that fills a portrait frame with a landscape clip, centred by ``crop_x``.

    Position is in FCPXML units (percent of frame height). Best effort: check the framing in Resolve.
    """
    src_w, src_h = m["width"], m["height"]
    fit = min(out.width / src_w, out.height / src_h)
    fill = out.height / src_h
    scaled_w = src_w * fill
    if fill <= fit or scaled_w <= out.width:
        return ""
    left = min(max(seg.crop_x, 0.0), 1.0) * (scaled_w - out.width)
    x_px = scaled_w / 2 - left - out.width / 2
    return f'<adjust-transform position="{x_px / out.height * 100:.3f} 0" scale="{fill / fit:.5f} {fill / fit:.5f}"/>'


def write_fcpxml(dst: Path, name: str, segments: list[Segment], clips: list[Clip], out: Output,
                 music: str | None = None) -> Path:
    used = sorted({s.clip for s in segments})
    seq_fps = Fraction(clips[segments[0].clip].fps) if segments else Fraction(int(out.fps))
    seq_fd = 1 / seq_fps

    res, formats = [], {}
    res.append(
        f'<format id="r0" name="FFVideoFormatRateUndefined" frameDuration="{_t(seq_fd, seq_fd)}" '
        f'width="{out.width}" height="{out.height}"/>'
    )
    metas = {cid: render_meta(clips[cid]) for cid in used}
    for cid in used:
        m = metas[cid]
        key = (m["width"], m["height"], m["fps"])
        if key not in formats:
            fid = f"r{len(formats) + 1}"
            formats[key] = fid
            res.append(
                f'<format id="{fid}" frameDuration="{_t(_fd(m["fps"]), _fd(m["fps"]))}" '
                f'width="{m["width"]}" height="{m["height"]}"/>'
            )
    for cid in used:
        c, m = clips[cid], metas[cid]
        fd = _fd(m["fps"])
        uri = Path(c.render_path).resolve().as_uri()
        res.append(
            f'<asset id="a{cid}" name={quoteattr(Path(c.render_path).stem)} start="0s" '
            f'duration="{_t(m["duration"], fd)}" hasVideo="1" '
            + (f'hasAudio="1" audioSources="1" audioChannels="{m.get("audio_channels", 2)}" '
               if m["has_audio"] else 'hasAudio="0" ')
            + f'format="{formats[(m["width"], m["height"], m["fps"])]}" '
            f'src={quoteattr(uri)}/>'
        )

    mus_len = music_duration(music) if music and segments else None
    if mus_len:
        res.append(
            f'<asset id="m0" name={quoteattr(Path(music).stem)} start="0s" duration="{_t(Fraction(mus_len).limit_denominator(1000), seq_fd)}" '
            f'hasAudio="1" audioSources="1" audioChannels="2" src={quoteattr(Path(music).resolve().as_uri())}/>'
        )

    spine, offset = [], Fraction(0)
    counts = frame_counts(segments, float(seq_fps))
    total = sum(counts) * seq_fd
    for i, (s, dur_frames) in enumerate(zip(segments, counts)):
        m = metas[s.clip]
        src_fd = _fd(m["fps"])
        dur = dur_frames * seq_fd
        start = _t(Fraction(s.start).limit_denominator(100000), src_fd)
        label = f"{i + 1:02d} {clips[s.clip].name} score {s.score:.2f}" + (" flip" if "flip" in s.tags else "")
        head = (f'<asset-clip ref="a{s.clip}" name={quoteattr(label)} offset="{_t(offset, seq_fd)}" '
                f'start="{start}" duration="{_t(dur, seq_fd)}" '
                f'format="{formats[(m["width"], m["height"], m["fps"])]}" tcFormat="NDF"')
        children = []
        if out.height > out.width:
            adj = _reframe(s, m, out)
            if adj:
                children.append(adj)
        if i == 0 and mus_len:   # music rides under the first clip, from the start of the timeline
            mdur = min(total, Fraction(mus_len).limit_denominator(1000))
            children.append(f'<asset-clip ref="m0" lane="-1" name="music" offset="{start}" start="0s" '
                            f'duration="{_t(mdur, seq_fd)}"/>')
        if children:
            spine.append(head + ">")
            spine.extend("  " + c for c in children)
            spine.append("</asset-clip>")
        else:
            spine.append(head + "/>")
        offset += dur

    xml = "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!DOCTYPE fcpxml>",
        '<fcpxml version="1.8">',
        "  <resources>",
        *("    " + r for r in res),
        "  </resources>",
        "  <library>",
        '    <event name="fpv-autocut">',
        f"      <project name={quoteattr(name)}>",
        f'        <sequence format="r0" duration="{_t(offset, seq_fd)}" tcStart="0s" tcFormat="NDF">',
        "          <spine>",
        *("            " + c for c in spine),
        "          </spine>",
        "        </sequence>",
        "      </project>",
        "    </event>",
        "  </library>",
        "</fcpxml>",
        "",
    ])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(xml, encoding="utf-8")
    return dst
