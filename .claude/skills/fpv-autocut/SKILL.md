---
name: fpv-autocut
description: Cut FPV drone footage into a highlight reel, a long version and vertical shorts with fpv-autocut. Use when the user asks to edit, cut or make a reel/shorts from their flights or a footage folder.
---

# Editing a flight session with fpv-autocut

You are the director; the `fpv-autocut` CLI does the heavy lifting. Don't
decode video yourself. Use the plan, report, contact sheets and timeline CSV.

## 1. Gather inputs

Ask only for what's missing:
- footage folder (`.mp4` + `.gcsv` from the RunCam Thumb Pro; `*_stabilized.mp4` from Gyroflow if they stabilised)
- music track for the reel/shorts (optional)
- output folder (default: `<footage>\edit`)
- lengths if different from defaults (reel 180 s, long 720 s, 2 shorts × 30 s)

Check `ffmpeg -version` works and `fpv-autocut --version` is installed (`pip install -e .` in the repo otherwise).

## 2. Plan

```bash
fpv-autocut plan "<footage>" -o "<out>" --sheets --music "<song>"
```

Add `--gemini` if `GEMINI_API_KEY` is set and the user wants better shot choice.

## 3. Review

1. Read `<out>/report.md`:
   - a clip with `gyro: NO` means a missing `.gcsv`. Tell the user; that clip is scored on image motion only.
   - idle time much lower than expected (flights include ground time) → raise `detect.idle_motion_abs`.
   - zero flip bins in a freestyle session → try `--gyro-offset` or lower `detect.flip_min_deg`.
2. Look at `<out>/sheets/reel.jpg` and each `short_*.jpg` (thumbnails in plan order, left→right, top→bottom). Flag segments that are clearly bad: sky only, ground only, blurry, crash.
3. For a doubtful segment, read the matching rows of `<out>/timeline.csv` (filter by clip and time) to see why it scored high.

## 4. Adjust

Prefer the smallest fix:
- **Remove or reorder a segment**: edit `outputs.<name>.segments` in `plan.json` directly.
- **Systematic problem** (too many slow cruise shots, missed flips): write a TOML with the changed keys (see `fpv-autocut.example.toml`) and re-run `plan --config`. Analysis is cached, so this takes seconds.
- **9:16 crop off-subject**: set that segment's `crop_x` (0 = left, 0.5 = centre, 1 = right).

Explain each change in one line.

## 5. Render

```bash
fpv-autocut render "<out>/plan.json" --draft          # quick preview, half resolution
fpv-autocut render "<out>/plan.json"                  # final
```

Use `--only reel` or `--only short_01` to re-render one output. Tell the user
where the files are, and that `reel.fcpxml` / `long.fcpxml` import into DaVinci
Resolve via *File → Import → Timeline*.
