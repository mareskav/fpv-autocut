# TODO

Open items, most useful first.

1. **Try it on real footage.** Everything so far is verified on synthetic flights only.
   Run `plan --sheets` on a real session and review `report.md`, `sheets/` and the
   `gyro sync NAME: +x s (corr c)` log lines. Tune the `sharp` weight and the motion-centred
   9:16 crop (`auto_crop`) through `--config`, not by editing defaults.
2. **Verify the FCPXML in DaVinci Resolve.** Check that the music imports as a connected clip
   under the first shot and that the shorts' `adjust-transform` frames correctly (position units
   are assumed to be a percent of frame height). If not, fix `_reframe` in `fcpxml.py` and add a
   test based on a real export.
3. **First CI run.** Check on GitHub that `test-windows-arm` (ffmpeg via winget) passes. If not,
   fix the install step or keep the job informational only.
4. **Release 0.2.0.** Bump `__version__` in `fpv_autocut/__init__.py` (the only place), rename
   "Unreleased" in `CHANGELOG.md`, tag.
5. **Smaller items.**
   - `ruff format` is not enforced (19 files would be reformatted); decide whether to adopt it.
   - Gyro sync and the 9:16 crop have no test on real data, only synthetic.
   - `auto_crop` can only be turned off through TOML; consider a `--no-auto-crop` CLI flag.
