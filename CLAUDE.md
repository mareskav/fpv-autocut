# fpv-autocut: notes for Claude Code

Python CLI that cuts FPV drone footage automatically: gyro + image motion → per-0.5 s
score → segments → `plan.json` → ffmpeg renders (16:9 reel, 16:9 long, 9:16 shorts)
+ FCPXML for DaVinci Resolve.

## Commands

```bash
pip install -e ".[dev]"            # once
pytest                             # must pass before committing (~40 s, needs ffmpeg)
fpv-autocut plan FOLDER -o OUT --sheets [--music song.mp3] [--gemini]
fpv-autocut render OUT/plan.json [--only reel,short_01] [--draft]
fpv-autocut run FOLDER -o OUT [...]      # plan + render
python tests/synth.py DIR                # synthetic flight with known events
```

## Layout

- `fpv_autocut/config.py`: **every** tunable number (weights, thresholds, profiles, encoders). Add new knobs here, never as magic numbers elsewhere.
- `media.py`: discovery. `foo.mp4` + `foo.gcsv` (gyro) + `foo_stabilized.mp4` (Gyroflow render, used for rendering only).
- `gyro.py`: Gyroflow `.gcsv` parser and per-bin features (`gyro_mean`, `gyro_peak`, `flip_deg`).
- `signals.py`: ffmpeg → numpy visual (motion, sharpness, luma) and audio signals; cached in `OUT/cache/*.npz`.
- `scoring.py`: percentile ranks across the whole session, idle detection, weighted score.
- `planning.py`: greedy segment finding, selection to a target length, beat snapping.
- `beats.py`: numpy-only beat tracker (constant tempo).
- `render.py`: frame-accurate segment renders (MKV + PCM parts → concat → AAC), contact sheets.
- `fcpxml.py`: FCPXML 1.8 export.
- `gemini.py`: optional Gemini rating of candidate moments (needs `GEMINI_API_KEY`).
- `pipeline.py`: glue. `plan.json` is the contract between planning and rendering.

## Rules

- Core dependency is **numpy only**. The user runs Windows ARM64, where OpenCV/numba wheels are unreliable. Optional extras are fine (`[gemini]`).
- Shell out to `ffmpeg`/`ffprobe` with argument lists, never `shell=True`. Paths can contain spaces and Czech diacritics.
- Keep cuts frame-accurate: segment durations are rounded on the cumulative timeline (`render.frame_counts`). Don't round per segment.
- Never write into the user's footage folder except reading; all output goes to `-o OUT`.
- New behaviour needs a test. Use `tests/synth.py` for anything that touches video.
- Gemini code paths cannot run in CI; keep them behind `--gemini` and cover pure logic with unit tests.

## Tuning on real footage

Read `OUT/report.md` (per-clip idle time, flip count, chosen segments), view
`OUT/sheets/*.jpg` (one thumbnail per segment, in order) and grep
`OUT/timeline.csv` for a time range to see the signals behind a score.
Change thresholds with a TOML passed via `--config`, not by editing defaults,
unless the user asks to change defaults.
