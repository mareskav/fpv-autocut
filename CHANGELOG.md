# Changelog

## Unreleased

### Added
- Gyro log is aligned to the video automatically by cross-correlating gyro rate with image motion
  (`--no-gyro-sync` turns it off, `--gyro-offset` sets the starting point).
- Image sharpness contributes to the score (`weights.sharp`).
- 9:16 shorts crop follows the horizontal centre of image motion (`auto_crop`); Gemini's `crop_x` still wins.
- FCPXML: music track connected under the first clip, `.fcpxml` for shorts with a reframing transform,
  audio channel count taken from ffprobe.
- `sheet` command, `--no-beat-sync`, `--no-fcpxml` documented.
- Czech README (`README.cs.md`), ruff lint, Windows ARM64 CI job.

### Changed
- Magic numbers moved to `config.py`; invalid `order` values are rejected up front.
- The version lives only in `fpv_autocut/__init__.py`.

### Fixed
- A corrupt clip no longer aborts the whole run; the signal cache survives truncated or stale files.
- Clear errors (exit code 2) for missing inputs, ffmpeg or bad config.

## 0.1.0
- First release: gyro-driven FPV highlight editing with reel, long and shorts renders and FCPXML export.
