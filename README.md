# fpv-autocut

Automatic highlight selection and editing for FPV drone footage.
**English** · [Česky](README.cs.md)

Point it at a folder of flights. It finds the best moments from **gyro data**
(flips, rolls, dives, fast passes) plus **image motion**, and renders:

| Output | Default | Notes |
|---|---|---|
| `reel.mp4` | 3 min, 16:9 | best ~10–15 % of the session, cuts snapped to the music beat |
| `long.mp4` | 12 min, 16:9 | relaxed selection, longer continuous runs, original audio |
| `short_01.mp4`, `short_02.mp4` | 30 s, 9:16 | Instagram / Facebook Reels; strongest moment first |
| `*.fcpxml` | | one per output (shorts included, with a reframing transform); import into DaVinci Resolve |
| `plan.json`, `report.md`, `sheets/*.jpg` | | what was picked and why; edit and re-render |

Built for the **RunCam Thumb Pro**, which writes a Gyroflow `.gcsv` gyro log
next to every video, but any camera works. Without gyro data it falls back to
image motion and sound.

## Why gyro

Image analysis alone can't tell a power loop from a shaky hover. The gyro can:
`fpv-autocut` integrates rotation per axis over a 1.5 s window. Vibration and
wobble cancel out, while a flip adds up to ~360°. Idle time on the ground
(gyro < 20 °/s and a still image) is dropped automatically.

## Install

Requires Python ≥ 3.11 and ffmpeg on `PATH`.

```powershell
winget install Gyan.FFmpeg          # Windows; macOS: brew install ffmpeg
git clone https://github.com/mareskav/fpv-autocut
cd fpv-autocut
uv sync                             # core: numpy only, works on Windows ARM64
uv sync --extra gemini              # optional: Gemini scoring
uv run fpv-autocut --help           # or activate .venv and call fpv-autocut directly
```

## Use

```powershell
# everything: analyse, plan, render
fpv-autocut run D:\FPV\2026-10-24 -o D:\FPV\edit --music D:\music\track.mp3

# fast half-resolution preview first
fpv-autocut run D:\FPV\2026-10-24 -o D:\FPV\edit --music track.mp3 --draft

# plan only, look at report.md and sheets\*.jpg, edit plan.json, then render
fpv-autocut plan D:\FPV\2026-10-24 -o D:\FPV\edit --sheets
fpv-autocut render D:\FPV\edit\plan.json --only reel,short_01
```

Useful options: `--reel 150`, `--long 600`, `--shorts 3`, `--short-len 20`,
`--gemini`, `--config my.toml`, `--hwaccel auto`, `-j 4`.

| Option | Effect |
|---|---|
| `--no-beat-sync` | use the music but don't snap cuts to its beats |
| `--gyro-offset 0.2` | starting guess for the gyro↔video offset (seconds) |
| `--no-gyro-sync` | trust `--gyro-offset` as is instead of aligning by cross-correlation |
| `--no-fcpxml` | (`render`) skip the `.fcpxml` files |

Contact sheets (one thumbnail per segment, in order) can also be made on their own:

```powershell
fpv-autocut sheet D:\FPV\edit\plan.json
```

Gyro logs are aligned to the video automatically; the result is logged as
`gyro sync NAME: +x s (corr c)`. Weak matches are ignored.

Analysis is cached in `OUT/cache/`, so re-planning with different settings
takes seconds.

### Gyroflow

If `foo_stabilized.mp4` (Gyroflow's default output name) sits next to
`foo.mp4`, it is used for rendering. Analysis still runs on the original
file and its `.gcsv`. Stabilise first, then run `fpv-autocut`.

### DaVinci Resolve

*File → Import → Timeline…* and choose `reel.fcpxml`. Clips link to your
original (or stabilised) files, so you can grade, add transitions and fine-tune
cuts at full quality. The music is imported as a connected audio clip under the
first shot. The shorts' 9:16 reframing is a best-effort transform; check the
framing in Resolve. For 4:3 footage on a 16:9 timeline, set
*Project Settings → Image Scaling → Mismatched resolution* to
*Scale full frame with crop*.

### Gemini (optional)

`--gemini` uploads a 360p, 5 fps, silent proxy of each flight that has strong
candidates and asks Gemini to rate up to 60 moments 1–10. It rewards proximity,
gaps, clean flips and scenery, and penalises sky-only or blurry shots. It also
returns where the subject is, which steers the 9:16 crop. Set `GEMINI_API_KEY`
first. Model: `gemini_model` in the config (default `gemini-flash-latest`).
Ratings are cached in `OUT/gemini/`.

## Recommended RunCam Thumb Pro settings

Per the [Gyroflow docs](https://docs.gyroflow.xyz/app/getting-started/supported-cameras/runcam):

- **1440p @ 60 fps**: native 4:3. A 9:16 crop keeps 42 % of the width, vs 32 % from 16:9.
- EIS **off**, distortion correction **off**.
- Shutter 45–90° (1/480–1/240 s at 60 fps), ISO as low as possible.

## Tuning

Every threshold lives in [`fpv_autocut/config.py`](fpv_autocut/config.py) and
can be overridden with a TOML file. See
[`fpv-autocut.example.toml`](fpv-autocut.example.toml). `timeline.csv` shows
every signal and the final score per 0.5 s, so you can see why something was or
wasn't picked.

## Working with Claude Code

The repo includes [`CLAUDE.md`](CLAUDE.md) and a `/fpv-autocut` skill in
`.claude/skills/`. Open Claude Code in this folder and say e.g.
*"cut today's flights from D:\FPV\2026-10-24 with this song"*. It runs the
plan, looks at the contact sheets, adjusts the plan and renders.

## Development

```bash
uv sync                      # installs the dev group (pytest, ruff) too
uv run pytest              # unit tests + end-to-end test on synthetic footage
uv run ruff check .        # lint
uv run python tests/synth.py demo/   # synthetic flight with known flip / fast pass
```

## License

MIT
