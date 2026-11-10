# ghostprint v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Mac/Linux command-line tool that takes one photo, video or audio file, reports every place/time/device leak it can find, writes a cleaned copy that looks like an ordinary file, and re-checks the copy, all offline and without writing anything but the output.

**Architecture:** A small media layer pipes everything through ffmpeg ↔ numpy (never via temp files). One module per leak exposes `detect` (and `fix` where v1 fixes it). A pipeline runs detect → fix → re-detect and hands both sets of findings to a report builder that only claims what was measured. flyover-scrub supplies the audio scrub, the ENF detector, and (new, Task 1) a public ridge detector.

**Tech Stack:** Python ≥3.12, `uv`, numpy, scipy, soundfile, Pillow, pillow-heif, flyover-scrub (git tag), pytest; external `ffmpeg`/`ffprobe` and `exiftool` (read-only).

**Spec:** `docs/superpowers/specs/2026-11-10-ghostprint-design.md`

## Global Constraints

- No network access after install: no telemetry, update checks, model downloads, or ADS-B queries.
- No trace: intermediate media never touches disk (ffmpeg → numpy → ffmpeg through pipes); the only files written are the requested output and, if asked, the report. No logs, caches or history.
- The report never repeats leaked values ("GPS location present", never coordinates).
- Never say "clean" when it isn't: a detector that couldn't run is `NOT CHECKED`, never "not found".
- `REMOVED` only when detected on the input and not on the output; a fix run with nothing detected is `SCRUBBED (precaution)`.
- Blend in: outputs look like ordinary phone/platform files: generic, randomised metadata, common encoder settings, no ghostprint-specific signature.
- Default is fix everything fixable; `--keep KIND` opts out per leak.
- No AI models in v1. macOS and Linux CLI. MIT licence. Public repo `lizTheDeveloper/ghostprint`.
- exiftool is used **read-only** (writing with it creates temp files next to the output).
- flyover-scrub is a library dependency pinned to a tag, not copied.

## Review Focus

1. **Rotated phone video** (iPhone portrait stores rotation as metadata): stripping metadata must not leave the output sideways; rotation is baked into pixels. → Task 3 `test_rotated_video_comes_out_upright`.
2. **iPhone HDR (10-bit HLG) video**: output must be 8-bit SDR H.264 that Chrome plays, not 10-bit. → Task 3 `test_hdr_input_becomes_sdr_8bit`.
3. **Video with no audio track / clip too short for a detector**: audio and time-based leaks report `NOT CHECKED` with a reason, never `NOT FOUND`. → Task 4 `test_short_or_silent_is_not_checked`, Task 10 `test_silent_video_report`.
4. **Output path equals input path, or output exists**: refuse; never overwrite the original. → Task 10 `test_refuses_to_overwrite_input`.
5. **Odd frame dimensions** (e.g. 406×721 after a crop): yuv420p needs even sizes; output must still encode. → Task 3 `test_odd_dimensions_encode`.

---

## File Structure

```
pyproject.toml
README.md                       install + use, written for non-technical users
src/ghostprint/
  __init__.py
  media.py        MediaInfo, probe(), pipe readers/writers (Task 2–3)
  blend.py        BlendProfile, random_profile(), COMMON_* tag lists (Task 3)
  findings.py     Status, Finding, LEAKS (Task 4)
  report.py       build_report(), render_text/json/html() (Task 4)
  leaks/
    __init__.py
    metadata.py   (Task 5)
    audio.py      aircraft, hum, gsm detect + combined audio fix (Task 6)
    flicker.py    video flicker detect/fix, photo flicker warn (Task 7)
    prnu.py       camera fingerprint estimate/match/fix (Task 8)
    scene.py      sun/shadow warning, visible-content not-checked (Task 9)
  pipeline.py     run() (Task 10)
  cli.py          argparse entry point `ghostprint` (Task 10)
tests/
  conftest.py     synthetic fixture generators (shared)
  test_*.py       one per module
```

---

### Task 1: Public ridge detector in flyover-scrub (other repo)

**Files (in `~/src/flyover-scrub`):**
- Modify: `src/flyover_scrub/scrub.py` (expose detection)
- Test: `tests/test_core.py`

**Interfaces:**
- Produces: `flyover_scrub.scrub.detect_ridges(audio: np.ndarray, sr: int, cfg: ScrubConfig | None = None) -> list[RidgeInfo]`; `@dataclass RidgeInfo(t_start: float, t_end: float, f_start: float, f_end: float, strength_db: float)`. Only **gliding** ridges ≥ 20 s long whose frequency changes by ≥ 1 % end to end (steady hum excluded); uses its own threshold, not the scrub's deliberately low `ridge_db=4.0`.
- Tag `v0.2.0` pushed to GitHub.

- [ ] **Step 1: Write failing tests** `test_detect_ridges_finds_flyover_not_noise_or_hum`: synthetic flyover from `synthesize_flyover` (as in `test_attack_prefers_truth_and_scrub_removes_it`) mixed at −10 dB → `len(detect_ridges(x, sr)) >= 1` and the first ridge has `f_start > f_end`; pure noise → `[]`; noise + steady 60/120 Hz hum → `[]`.
- [ ] **Step 2:** `uv run pytest -q -k detect_ridges` → FAIL (not defined).
- [ ] **Step 3:** Implement by reusing `_floor`/`_ridges` and filtering by duration, glide and a calibrated `min_db` (start at 8 dB; tune on the fixtures so all three cases pass).
- [ ] **Step 4:** Full suite passes: `uv run pytest -q`.
- [ ] **Step 5:** Commit, `git tag v0.2.0`, push main and tag.

---

### Task 2: Project scaffold and media probing

**Files:** Create `pyproject.toml`, `src/ghostprint/__init__.py`, `src/ghostprint/media.py`, `tests/conftest.py`, `tests/test_media.py`.

**Interfaces:**
- Produces: `@dataclass MediaInfo(path: Path, kind: Literal["photo","video","audio"], duration_s: float | None, width: int | None, height: int | None, fps: float | None, rotation: int, hdr: bool, audio_sr: int | None, audio_channels: int | None)`; `probe(path: Path) -> MediaInfo` (ffprobe JSON over stdout; photos: Pillow + pillow-heif; `hdr` true when color transfer is `arib-std-b67` or `smpte2084`); `require_tools() -> None` raising `ToolMissing(name, install_hint)` for ffmpeg, ffprobe, exiftool.
- `tests/conftest.py` fixtures (all generated in `tmp_path`, never committed media): `make_video(tmp_path, *, seconds=40, fps=30, size=(320, 240), audio=None | (np.ndarray, sr), rotation=0, hdr=False) -> Path`, `make_photo(tmp_path, img: np.ndarray, *, fmt="jpeg", exif: dict | None) -> Path`, `make_audio(tmp_path, x, sr) -> Path`.

- [ ] **Step 1:** `pyproject.toml`: name `ghostprint`, MIT, script `ghostprint = "ghostprint.cli:main"`, deps numpy, scipy, soundfile, pillow, pillow-heif, `flyover-scrub @ git+https://github.com/lizTheDeveloper/flyover-scrub@v0.2.0`; dev pytest.
- [ ] **Step 2: Failing tests** `test_probe_video_photo_audio` (kind, dims, fps≈30, audio_sr for each fixture type), `test_probe_reads_rotation` (rotation=90 fixture → `info.rotation == 90`), `test_require_tools_names_missing_tool` (monkeypatch `shutil.which` to return None for exiftool → `ToolMissing` with `"exiftool"` and an install hint containing `brew` or `apt`).
- [ ] **Step 3:** `uv run pytest -q tests/test_media.py` → FAIL.
- [ ] **Step 4:** Implement.
- [ ] **Step 5:** Tests pass; commit `feat: scaffold and media probing`.

---

### Task 3: Pipe I/O and blend-in profile

**Files:** Modify `src/ghostprint/media.py`; create `src/ghostprint/blend.py`, `tests/test_io.py`, `tests/test_blend.py`.

**Interfaces:**
- Consumes: `MediaInfo`, `probe` (Task 2).
- Produces (media.py):
  - `read_audio(info: MediaInfo) -> tuple[np.ndarray, int] | None` (float32, shape (n, ch); `None` when no audio track), via `ffmpeg … -f f32le pipe:1`.
  - `iter_frames(info: MediaInfo, chunk: int = 32) -> Iterator[np.ndarray]` yielding uint8 (k, H, W, 3), **upright** (ffmpeg autorotate), **SDR** (zscale tone-map when `info.hdr`: `zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p`), **even** W and H (crop 1 px if odd). Re-iterable by calling again (re-reads the input file; no caching).
  - `read_photo(info: MediaInfo) -> np.ndarray` uint8 (H, W, 3), EXIF orientation applied.
  - `write_video(out: Path, frames: Iterable[np.ndarray], fps: float, audio: tuple[np.ndarray, int] | None, profile: BlendProfile) -> None`: one ffmpeg process; raw frames on stdin, raw f32le audio on fd 3 (`os.pipe`, `pass_fds`, input `pipe:3`), two writer threads to avoid deadlock; `-map_metadata -1 -map_chapters -1` then the profile's tags and codec args.
  - `write_audio(out: Path, audio: np.ndarray, sr: int, profile: BlendProfile) -> None`, `write_photo(out: Path, img: np.ndarray, profile: BlendProfile) -> None` (JPEG for JPEG/HEIC input, PNG for PNG; Pillow, no EXIF/XMP/ICC text chunks).
- Produces (blend.py): `@dataclass BlendProfile(container_tags: dict[str, str], video_args: list[str], audio_args: list[str], jpeg_quality: int, mtime: float)`; `random_profile(kind: str, rng: np.random.Generator) -> BlendProfile`; `apply_mtime(path: Path, profile: BlendProfile) -> None` (`os.utime`). Values: `creation_time` = now − uniform(0, 72 h), random seconds; `jpeg_quality` uniform int 85–95; encoder tag chosen from `COMMON_ENCODER_TAGS` (populate from ffprobe of ordinary phone/platform files you have locally; record where each came from in a comment, no file names of private media); video H.264 High yuv420p, CRF chosen 20–24; audio AAC-LC 44.1 kHz 128–160 kbps.

- [ ] **Step 1: Failing tests (test_io.py):** `test_audio_roundtrip_via_pipes` (write then read ≈ equal, corr > 0.99), `test_rotated_video_comes_out_upright` (240×320 rotation=90 fixture with a red top edge → output width/height swapped, no rotation tag in ffprobe, red still at top), `test_hdr_input_becomes_sdr_8bit` (hdr fixture → output `pix_fmt == "yuv420p"`, `color_transfer` in (`bt709`, unset)), `test_odd_dimensions_encode` (321×241 → output exists, even dims), `test_no_temp_files_during_io` (monkeypatch `tempfile.mkstemp`, `mkdtemp`, `NamedTemporaryFile`, `TemporaryDirectory` to raise; set `TMPDIR` to an empty dir; full read→write round-trip; dir still empty).
- [ ] **Step 2: Failing tests (test_blend.py):** `test_profiles_differ_between_runs` (3 profiles → distinct `creation_time` and `mtime`, all within the last 72 h, none in the future), `test_output_has_no_ghostprint_signature` (written video and photo bytes contain no `b"ghostprint"`, encoder tag ∈ `COMMON_ENCODER_TAGS`, no `com.apple.quicktime.location` / GPS keys in ffprobe or exiftool output).
- [ ] **Step 3:** Run → FAIL. **Step 4:** Implement. **Step 5:** Pass; commit `feat: pipe-only media I/O and blend-in profiles`.

---

### Task 4: Findings and the honest report

**Files:** Create `src/ghostprint/findings.py`, `src/ghostprint/report.py`, `tests/test_report.py`.

**Interfaces:**
- Produces (findings.py): `class Status(Enum): DETECTED, NOT_DETECTED, NOT_CHECKED, WARNING`; `@dataclass Finding(leak: str, status: Status, summary: str, strength: float | None = None, reason: str | None = None)`; `LEAKS = ("metadata", "aircraft", "hum", "gsm", "flicker", "fingerprint", "sun", "visible")` (report order).
- Produces (report.py): `@dataclass ReportLine(leak: str, verdict: str, text: str)`; `@dataclass Report(name: str, kind: str, lines: list[ReportLine], headline: str, output: str | None)`; `build_report(name: str, kind: str, before: list[Finding], after: list[Finding] | None, fixed: set[str], output: str | None) -> Report`; `render_text(r) -> str`, `render_json(r) -> str`, `render_html(r) -> str` (single self-contained file, no external assets).
- Verdict rules (exact words): before DETECTED & after NOT_DETECTED → `REMOVED` (`REDUCED` for `fingerprint`); before NOT_DETECTED & leak in `fixed` → `SCRUBBED (precaution)`; DETECTED & after DETECTED → `STILL PRESENT`; NOT_CHECKED → `NOT CHECKED` + reason; WARNING → `WARNING`; NOT_DETECTED & not fixed → `NOT FOUND`; `--check` (after is None): DETECTED → `FOUND`.
- Headline: `"Nothing found that this tool checks for. Look at the picture before you post."` only when every line is `NOT FOUND`/`REMOVED`/`REDUCED`/`SCRUBBED (precaution)`; otherwise it names the count of `STILL PRESENT`, `WARNING`, `NOT CHECKED` lines first.

- [ ] **Step 1: Failing tests:** `test_verdict_rules` (one case per rule above), `test_headline_never_clean_with_warning_or_not_checked`, `test_short_or_silent_is_not_checked` (a `NOT_CHECKED` hum finding with reason `"clip shorter than 30 s"` renders `NOT CHECKED` and the reason, and the headline is not the clean one), `test_report_never_contains_values` (findings whose summaries are produced by later modules are checked in their own tasks; here: `render_html` has no `<script src`/`<link` to external URLs).
- [ ] **Step 2:** FAIL → **Step 3:** implement → **Step 4:** pass → **Step 5:** commit `feat: findings and honest report`.

---

### Task 5: Metadata

**Files:** Create `src/ghostprint/leaks/metadata.py`, `tests/test_metadata.py`.

**Interfaces:**
- Consumes: `Finding`, `Status`; `write_*` from Task 3 (stripping happens there).
- Produces: `read_tags(path: Path) -> dict[str, object]` (`exiftool -j -a -G1 -n <path>`, stdout only); `categories(tags) -> set[str]` from {`"location"`, `"time"`, `"device"`, `"software"`, `"thumbnail"`}; `detect(path: Path) -> Finding` with summary like `"GPS location, capture time and phone model present in the file"` (category names only, never values).

- [ ] **Step 1: Failing tests:** `test_detects_categories_without_values` (photo with EXIF GPS 45.1,−122.2, DateTimeOriginal, Make/Model/Serial → DETECTED, summary mentions location/time/device, and contains none of `"45.1"`, `"122"`, the serial string), `test_clean_output_has_no_identifying_categories` (pass the photo through `write_photo` with a random profile → `categories(read_tags(out))` has no `location`, `device`, `thumbnail`; `time` allowed only as the profile's randomised value).
- [ ] **Steps 2–5:** FAIL → implement → pass → commit `feat: metadata leak`.

---

### Task 6: Audio leaks (aircraft, grid hum, phone-radio buzz) and the audio fix

**Files:** Create `src/ghostprint/leaks/audio.py`, `tests/test_audio_leaks.py`.

**Interfaces:**
- Consumes: `flyover_scrub.scrub.detect_ridges` (Task 1), `flyover_scrub.enf.detect_enf`, `flyover_scrub.scrub.scrub`, `ScrubConfig`.
- Produces: `detect_aircraft(audio, sr) -> Finding`, `detect_hum(audio, sr) -> Finding`, `detect_gsm(audio, sr) -> Finding`, `fix_audio(audio: np.ndarray, sr: int, keep: set[str]) -> np.ndarray` (one `scrub()` pass with `keep_stationary_tones = "hum" in keep`, skipped only if both `aircraft` and `hum` are kept; then GSM comb notch unless `"gsm" in keep`; channels processed independently).
- Rules: clips < 30 s → hum and gsm `NOT_CHECKED` (`"clip shorter than 30 s"`); < 40 s → aircraft `NOT_CHECKED`; audio `None` → all three `NOT_CHECKED` (`"no audio track"`). GSM detection: TDMA buzz = pulse train at 217 Hz; DETECTED when ≥ 4 of the harmonics 217·k (k = 1…8) stand ≥ 6 dB above their ±5 Hz neighbourhood and the 217 Hz envelope periodicity is present.

- [ ] **Step 1: Failing tests:** `test_aircraft_detected_and_removed` (fixture flyover at −10 dB → DETECTED; after `fix_audio` → NOT_DETECTED **and** flyover-scrub's offline attack, run as in its own test, no longer ranks truth in the top 50), `test_hum_detected_and_removed` (wandering 120+240 Hz hum as in flyover-scrub's ENF test → DETECTED → after fix NOT_DETECTED; with `keep={"hum"}` still DETECTED), `test_gsm_detected_and_removed` (577 µs bursts every 4.615 ms of 900 Hz-ish carrier through a crude speaker nonlinearity, −20 dB under speech-shaped noise → DETECTED → after fix NOT_DETECTED; plain noise → NOT_DETECTED), `test_short_or_silent_audio_not_checked`.
- [ ] **Steps 2–5:** FAIL → implement → pass → commit `feat: audio leaks`.

---

### Task 7: Grid flicker (video fix, photo warning)

**Files:** Create `src/ghostprint/leaks/flicker.py`, `tests/test_flicker.py`; add `make_flicker_frames(...)` to `tests/conftest.py`.

**Interfaces:**
- Consumes: `iter_frames` (Task 3).
- Produces: `@dataclass FlickerEstimate(nominal_hz: int | None, snr_db: float, row_rate_hz: float)`; `measure(frames: Callable[[], Iterator[np.ndarray]], fps: float, height: int) -> FlickerEstimate` (pass 1 over the video); `detect(est: FlickerEstimate, duration_s: float) -> Finding` (DETECTED when `snr_db >= 8` at 100 or 120 Hz; `NOT_CHECKED` under 10 s); `class FlickerFixer(est)` with `apply(chunk: np.ndarray, first_frame: int) -> np.ndarray` (pass 2, streaming); `detect_photo(img: np.ndarray) -> Finding` (WARNING when horizontal banding at a plausible flicker period is found, else NOT_DETECTED).
- Approach (pinned because the signal model matters): build a 1-D signal from per-row mean luminance with each frame's low-spatial-frequency content removed, sampled at `row_rate_hz = fps * height` (ignore vertical blanking; search ±3 % around it and keep the best SNR); look for the flicker line at 2× the grid nominal. The fix divides each row by `1 + estimated flicker` at that row's time. Exact estimator and filter are the builder's call.

- [ ] **Step 1: Failing tests:** `test_flicker_detected_and_removed` (synthetic 30 fps, 240 rows, scene × (1 + 0.03·sin(2π·120·t_row)) → DETECTED; after `FlickerFixer` → NOT_DETECTED; mean absolute pixel change outside the flicker < 2 levels), `test_no_flicker_control`, `test_photo_banding_warns`.
- [ ] **Steps 2–5:** FAIL → implement → pass → commit `feat: grid flicker`.

---

### Task 8: Camera fingerprint (PRNU)

**Files:** Create `src/ghostprint/leaks/prnu.py`, `tests/test_prnu.py`; add `make_sensor_images(...)` to conftest.

**Interfaces:**
- Produces: `estimate(images: list[np.ndarray]) -> np.ndarray` (float32 (H, W) fingerprint from denoising residuals, Lukáš/Fridrich style); `@dataclass PRNUMatch(pce: float, scale: float)`; `match(img: np.ndarray, K: np.ndarray) -> PRNUMatch` (PCE via FFT cross-correlation, which covers crop shifts, maximised over scales 0.90–1.10 step 0.01); `detect(m: PRNUMatch | None) -> Finding` (DETECTED when `pce >= 60`; `None` → NOT_CHECKED with reason `"pass --fingerprint-from with a few other photos from this phone"`); `fix(img: np.ndarray, rng: np.random.Generator) -> np.ndarray` (denoise + randomised resample with a small non-uniform warp; applied to every photo and to every video frame using one warp per video).

- [ ] **Step 1: Failing tests:** `test_same_sensor_matches_other_does_not` (8 reference images from sensor A → estimate; new A image PCE ≥ 60; sensor B image PCE < 60), `test_fix_defeats_match_with_search` (`match(fix(imgA), K).pce < 60` even with the scale search), `test_fix_is_visually_mild` (PSNR of fixed vs original ≥ 32 dB, dims within ±2 %).
- [ ] **Steps 2–5:** FAIL → implement → pass → commit `feat: camera fingerprint`.

---

### Task 9: Scene warnings

**Files:** Create `src/ghostprint/leaks/scene.py`, `tests/test_scene.py`.

**Interfaces:**
- Produces: `detect_sun(img: np.ndarray) -> Finding` (WARNING `"outdoor scene with hard shadows: the sun's position can give time of day and latitude"` or NOT_DETECTED; heuristic: bright low-saturation-to-blue sky region in the upper part plus strong straight-edged dark regions; drop to a generic outdoor warning if unreliable, the builder's call); `visible() -> Finding` (always NOT_CHECKED, reason `"readable signs, plates, landmarks or reflections are not checked in v1: look before you post"`). Video: sample one frame per 5 s.

- [ ] **Step 1: Failing tests:** `test_sky_and_shadow_warns`, `test_indoor_flat_does_not_warn`, `test_visible_is_always_not_checked`.
- [ ] **Steps 2–5:** FAIL → implement → pass → commit `feat: scene warnings`.

---

### Task 10: Pipeline, CLI, end-to-end guarantees, README, publish

**Files:** Create `src/ghostprint/pipeline.py`, `src/ghostprint/cli.py`, `tests/test_e2e.py`, `README.md`.

**Interfaces:**
- Consumes: everything above.
- Produces: `run(inp: Path, out: Path | None, *, keep: set[str] = frozenset(), refs: Path | None = None, seed: int | None = None) -> Report` (probe → detect all → if `out`: fix (audio, frames through `FlickerFixer` then `prnu.fix`, photo) → write with `random_profile` → `apply_mtime` → re-detect on `out` → `build_report`); `main(argv: list[str] | None = None) -> int`. CLI: `ghostprint INPUT [-o OUTPUT] [--check] [--report FILE.html|FILE.txt] [--json] [--keep KIND]... [--fingerprint-from DIR]`; exit 0 on success, 2 on usage error, 3 when `ToolMissing`.

- [ ] **Step 1: Failing tests (test_e2e.py):**
  - `test_full_video_run` (40 s fixture with GPS/rotation metadata, flyover, hum and flicker → report lines `REMOVED` for metadata, aircraft, hum, flicker; `NOT CHECKED` for visible; output plays: ffprobe ok).
  - `test_no_network` (monkeypatch `socket.socket` and `socket.create_connection` to raise; full run succeeds).
  - `test_no_trace` (TMPDIR and HOME set to empty dirs; tempfile functions raise; after the run only the output (and `--report` file) exist anywhere under `tmp_path` besides the input).
  - `test_refuses_to_overwrite_input` (`-o` equal to input, or an existing output → exit 2, input unchanged byte for byte).
  - `test_silent_video_report` (video without audio → aircraft/hum/gsm `NOT CHECKED` "no audio track"; headline not the clean one).
  - `test_outputs_do_not_share_a_signature` (three runs on three inputs → distinct creation times and mtimes, no `ghostprint` bytes).
  - `test_report_never_contains_leaked_values` (input GPS 45.1,−122.2 and serial `SN12345` → neither string in text, JSON or HTML report).
- [ ] **Step 2:** FAIL → **Step 3:** implement `run` and `main` → **Step 4:** full suite passes `uv run pytest -q`.
- [ ] **Step 5: README** for non-technical users: install Homebrew/apt + `uv` + ffmpeg + exiftool step by step; one example per media type; what each report word means; what v1 does not check (visible content, voice, face); "verify on your own: run `--check` on the output". Credit colsto and the flyover-scrub research.
- [ ] **Step 6:** Commit `feat: pipeline, CLI and end-to-end guarantees`; create the public repo `lizTheDeveloper/ghostprint` (`gh repo create lizTheDeveloper/ghostprint --public --source . --push`) and push.
