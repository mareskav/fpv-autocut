"""From per-bin scores to segments, and from segments to output plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import BIN
from .config import Profile
from .media import Clip


@dataclass
class Segment:
    clip: int
    start: float
    end: float
    score: float          # mean score over the segment (0..1)
    peak: float           # highest bin score
    peak_t: float         # time of the peak within the clip
    tags: list[str] = field(default_factory=list)
    crop_x: float = 0.5   # horizontal centre for 9:16 crops (0 = left edge, 1 = right)
    note: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = round(d["start"], 3)
        d["end"] = round(d["end"], 3)
        d["score"] = round(d["score"], 3)
        d["peak"] = round(d["peak"], 3)
        d["peak_t"] = round(d["peak_t"], 3)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def find_segments(clip: Clip, score: np.ndarray, info: dict, prof: Profile) -> list[Segment]:
    """Greedy peak picking: take the highest unused bin, grow around it, repeat."""
    n = len(score)
    used = np.zeros(n, dtype=bool)
    min_b = max(1, int(round(prof.min_len / BIN)))
    max_b = max(min_b, int(round(prof.max_len / BIN)))
    out: list[Segment] = []
    for peak_i in np.argsort(-score, kind="stable"):
        peak = float(score[peak_i])
        if peak < prof.floor:
            break
        if used[peak_i]:
            continue
        thr = peak * prof.grow_rel
        lo = hi = int(peak_i)
        # Grow towards the higher neighbour first, until threshold or max length.
        while hi - lo + 1 < max_b:
            left_ok = lo - 1 >= 0 and not used[lo - 1] and score[lo - 1] >= thr
            right_ok = hi + 1 < n and not used[hi + 1] and score[hi + 1] >= thr
            if not (left_ok or right_ok):
                break
            if left_ok and (not right_ok or score[lo - 1] >= score[hi + 1]):
                lo -= 1
            else:
                hi += 1
        # Enforce minimum length (may include weaker bins, never used ones).
        while hi - lo + 1 < min_b:
            can_l = lo - 1 >= 0 and not used[lo - 1]
            can_r = hi + 1 < n and not used[hi + 1]
            if not (can_l or can_r):
                break
            if can_r and (not can_l or score[hi + 1] >= score[lo - 1]):
                hi += 1
            else:
                lo -= 1
        if hi - lo + 1 < min_b:
            used[lo:hi + 1] = True
            continue
        used[lo:hi + 1] = True
        start = max(0.0, lo * BIN - prof.pad)
        end = min(clip.duration, (hi + 1) * BIN + prof.pad)
        tags = []
        if info["flip"][lo:hi + 1].any():
            tags.append("flip")
        out.append(Segment(
            clip=clip.id, start=start, end=end,
            score=float(score[lo:hi + 1].mean()), peak=peak, peak_t=(peak_i + 0.5) * BIN, tags=tags,
        ))
    return out


def _merge_close(segs: list[Segment], gap: float) -> list[Segment]:
    """Merge segments of the same clip separated by less than ``gap`` seconds."""
    segs = sorted(segs, key=lambda s: (s.clip, s.start))
    merged: list[Segment] = []
    for s in segs:
        if merged and merged[-1].clip == s.clip and s.start - merged[-1].end <= gap:
            m = merged[-1]
            total = m.duration + s.duration
            m.score = (m.score * m.duration + s.score * s.duration) / max(total, 1e-9)
            if s.peak > m.peak:
                m.peak, m.peak_t = s.peak, s.peak_t
            m.end = max(m.end, s.end)
            m.tags = sorted(set(m.tags) | set(s.tags))
        else:
            merged.append(Segment(**asdict(s)))
    return merged


def _resolve_overlaps(segs: list[Segment], min_keep: float = 0.2) -> list[Segment]:
    """Make same-clip segments disjoint so no footage is shown twice."""
    segs = sorted(segs, key=lambda s: (s.clip, s.start))
    out: list[Segment] = []
    for s in segs:
        if out and out[-1].clip == s.clip and s.start < out[-1].end:
            mid = (out[-1].end + s.start) / 2
            out[-1].end = mid
            s.start = mid
        if s.duration > min_keep:
            out.append(s)
    return out


def order_segments(segs: list[Segment], order: str) -> list[Segment]:
    chrono = sorted(segs, key=lambda s: (s.clip, s.start))
    if order == "score":
        return sorted(segs, key=lambda s: -s.score)
    if order == "hook" and segs:
        best = max(segs, key=lambda s: s.peak)
        return [best] + [s for s in chrono if s is not best]
    return chrono


def select(cands: list[Segment], prof: Profile, *, exclude: list[Segment] | None = None,
           merge_gap: float = 0.0) -> list[Segment]:
    """Pick the best segments until their total length reaches ``prof.target``."""
    # Padding makes neighbouring segments overlap slightly; only real overlaps
    # (more than ``overlap_tol``) block a candidate. Small ones are trimmed below.
    tol = prof.overlap_tol

    def overlaps(a: Segment, b: Segment) -> bool:
        return a.clip == b.clip and a.start < b.end - tol and b.start < a.end - tol

    exclude = exclude or []
    chosen: list[Segment] = []
    total = 0.0
    for s in sorted(cands, key=lambda s: -s.score):
        if total >= prof.target:
            break
        if any(overlaps(s, e) for e in exclude) or any(overlaps(s, c) for c in chosen):
            continue
        chosen.append(Segment(**asdict(s)))
        total += s.duration
    if merge_gap > 0:
        chosen = _merge_close(chosen, merge_gap)
    chosen = _resolve_overlaps(chosen, prof.min_keep)
    # Trim the overshoot from the weakest segment(s) so the total lands near the target.
    over = sum(s.duration for s in chosen) - prof.target
    for s in sorted(chosen, key=lambda s: s.score):
        if over <= 0.05:
            break
        cut = min(over, s.duration - prof.min_len)
        if cut <= 0:
            continue
        # Cut from the side further from the peak.
        if (s.peak_t - s.start) > (s.end - s.peak_t):
            s.start += cut
        else:
            s.end -= cut
        over -= cut
    return order_segments(chosen, prof.order)


def _free_space(s: Segment, others: list[Segment], clip_dur: float) -> tuple[float, float]:
    """Source-time interval around ``s`` not taken by other segments of the same clip."""
    lo, hi = 0.0, clip_dur
    for o in others:
        if o.clip != s.clip:
            continue
        if o.end <= s.start + 1e-9:
            lo = max(lo, o.end)
        elif o.start >= s.end - 1e-9:
            hi = min(hi, o.start)
    return lo, hi


def snap_to_beats(segs: list[Segment], beats: np.ndarray, clips: list[Clip], min_len: float) -> list[Segment]:
    """Adjust segment lengths so every cut lands on a music beat."""
    if beats is None or len(beats) < 2:
        return segs
    out: list[Segment] = []
    cur = 0.0
    for i, s in enumerate(segs):
        s = Segment(**asdict(s))
        target_end = cur + s.duration
        # Growing must never run into another segment of the same clip (the same shot twice),
        # so only beats reachable inside the free space around this segment are candidates.
        lo, hi = _free_space(s, out + segs[i + 1:], clips[s.clip].duration)
        cands = beats[(beats >= cur + min_len * 0.75) & (beats <= cur + (hi - lo))]
        if cands.size == 0:
            out.append(s)
            cur += s.duration
            continue
        nb = float(cands[np.argmin(np.abs(cands - target_end))])
        new_dur = nb - cur
        clip_dur = clips[s.clip].duration
        delta = new_dur - s.duration
        if delta < 0:
            # Shorten from the side further from the peak.
            if (s.peak_t - s.start) > (s.end - s.peak_t):
                s.start -= delta
            else:
                s.end += delta
        else:
            grow_end = min(delta, hi - s.end)
            s.end += grow_end
            s.start = max(lo, s.start - (delta - grow_end))
        cur += s.duration
        out.append(s)
    return out
