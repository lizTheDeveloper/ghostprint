# ghostprint: design

Status: draft for approval · 2026-11-10

## Intent

ghostprint lets someone post a photo or video without it revealing **where** or **when** it was made, or
linking it to their other posts. The person using it is anonymous and wants to stay that way: the tool knows
nothing about them, never asks, and never talks to the network.

It grew out of [flyover-scrub](https://github.com/lizTheDeveloper/flyover-scrub). That project showed, on real
phone videos and real ADS-B data, that a plane nobody could hear can locate a recording to ~100 m, that
re-encoding and posting to TikTok don't remove it, and that grid hum (ENF) can supply the recording time the
plane attack needs. Existing privacy tools ([MAT2](https://0xacab.org/jvoisin/mat2), ExifCleaner) strip
metadata only and say so. Nothing handles the leaks in the content itself. ghostprint does, and **measures**
its own fixes, the way flyover-scrub does.

## Who it's for

A ghost: someone about to post, half the time a photo and half the time a video, who wants no
place, time or device trail. They run it on their own computer. They may not be technical, so the report
must be plain language.

## Goals (version 1: where and when)

1. **One command, one file in, one cleaned file out, plus a report.**
   `ghostprint input.mov -o clean.mov` writes the cleaned copy and prints the report. `ghostprint --check input`
   reports only.
2. **For every leak: detect → report → fix → re-check.** The report is generated twice: on the input, and again
   on the output, so it shows what was removed and what is left.
3. **A leak gets a fix in v1 only if we can measure the fix** on real files. Everything else is detect-and-warn,
   with honest wording.
4. **Never say "clean" when it isn't.** If something unfixable is detected or can't be checked, the
   summary says so first.

### Leaks in v1

| Leak | Media | Detect | Fix in v1 | How the fix is verified |
|---|---|---|---|---|
| Container and file metadata: GPS, timestamps, device make, model and serial, lens, software, edit history, thumbnails, maker notes, XMP and IPTC | photo, video, audio | read everything (exiftool) | strip all, rewrite the container | re-read the output, which must have no location, time or device fields. File timestamps on disk are set to a neutral value. |
| Aircraft Doppler lines | video, audio | flyover-scrub's tonal ridge detector | flyover-scrub's scrub | the trace detector finds no long gliding ridge in the output. The ADS-B attack needs the network, so it isn't run (see Constraints). |
| Grid hum (ENF) in audio | video, audio | `flyover_scrub.enf.detect_enf` | flyover-scrub's scrub (steady tones are resynthesised across 20–2000 Hz) | `detect_enf` on the output reports none |
| Grid flicker in video (rolling-shutter banding from mains lighting) | video | new: per-frame row-brightness signal, look for the 100/120 Hz flicker and its wander | new: remove the flicker component from per-row brightness | the same detector reports none on the output |
| Grid flicker in a photo | photo | new: banding at the grid's flicker rate across rows | warn only | n/a |
| Phone-radio buzz (GSM/TDMA "dit-dit" pulses at 217 Hz and harmonics, picked up by speakers and mics near a phone) | video, audio | new: comb at 217 Hz multiples with pulse timing | new: notch the comb | detector reports none on the output |
| Camera sensor fingerprint (PRNU) | photo, video | needs a reference: the user points `--fingerprint-from DIR` at a few other photos from the same phone; the tool estimates the fingerprint locally and reports how strongly the input matches it | light denoise plus a small rescale or crop that breaks pixel alignment | correlation of the output with the estimated fingerprint drops to the chance level measured on other cameras |
| Sun angle and shadows | photo, video | warn only: "outdoor scene with hard shadows; the sun's position can give the time of day and latitude" (simple heuristic, no model) | no | n/a |
| Visible content: signs, plates, landmarks, faces of others, screens, window reflections | photo, video | warn only, generic ("anything readable in the picture is not checked in v1") | no (v2 via ComfyUI) | n/a |

## Non-goals (v1)

- **Who** you are: voice change and face replacement are **version 2** (see below).
- Locating or dating other people's recordings. ghostprint audits only the file it's given and never needs the
  user's true location.
- Live streams, calls, or anything not a file.
- Making the media look or sound the same as the original. It should stay natural, but leak removal wins.

## Anti-goals (things that would make it a failure even if it "works")

- **Any network access after install.** No telemetry, no update checks, no model downloads at run time, no ADS-B
  queries. Tests check that no sockets are opened.
- **Leaving a trace.** No logs, caches, history or temp files left behind. The report goes only where the user
  sends it (stdout by default, `--report FILE` optional). The report never repeats the leaked values
  (it says "GPS location present", not the coordinates), so the report can't become a leak itself.
- **False reassurance.** A green "nothing found" when a detector couldn't run, or when an unfixable warning
  applies.
- **Asking who the user is or where they were.**

## Constraints

- Python, packaged with `uv`, MIT licence, public repo `lizTheDeveloper/ghostprint`. flyover-scrub is a
  dependency (library import, not a copy).
- External tools: `ffmpeg` and `exiftool`, both checked at start with a clear message if missing.
- **No AI models in v1.** Everything is signal processing that runs offline on a laptop CPU.
- Formats: at least iPhone and Android photos (HEIC, JPEG, PNG) and videos (MOV, MP4), plus WAV and M4A audio.
  Output keeps the input's format where possible. HEIC may be written as JPEG if HEIC can't be rewritten cleanly
  (builder's call).
- **Video re-encode.** Removing flicker and the sensor fingerprint means re-encoding frames, so video output is
  re-encoded (H.264 or HEVC, quality close to the input). Audio is re-encoded after scrubbing, as flyover-scrub
  does now.
- **Aircraft are not verified against ADS-B.** That would need the network and the true location, so ghostprint
  scrubs unconditionally and verifies with the local ridge detector. flyover-scrub's audit stays the tool for
  people who want the full attack.

## Report

Plain language, ordered by severity, first on the input and then on the output:

```
ghostprint report: IMG_1234.MOV (video, 1:42)

  REMOVED   GPS location and capture time in the file's metadata
  REMOVED   phone model and serial number in the metadata
  REMOVED   grid hum (60 Hz grid): could date this recording to the second
  REMOVED   long gliding tones (aircraft, traffic or HVAC): could locate it from a passing plane
  REMOVED   light flicker from mains lighting: could date it even with the sound removed
  REDUCED   camera fingerprint: matched your other photos (strong) → now at chance
  WARNING   outdoor scene with hard shadows: the sun's position can give time of day and latitude
  NOT CHECKED  readable signs, plates, landmarks or reflections: look before you post

Output: clean.mov (re-encoded; metadata stripped)
```

`--report out.html` writes the same content as one self-contained HTML file. `--json` gives machine-readable
results for tests.

## How it's built

### Decided (in conversation)

- New repo, separate from flyover-scrub, which it uses as a library.
- Leak stripper (where/when) first. Identity (who) is v2.
- Photos and videos are equally important.
- Open source throughout. ML stages (v2) run through a **configurable, local ComfyUI engine**: each ML stage is a
  ComfyUI workflow file kept in the repo, selected in a config file, and run against the user's own ComfyUI on
  `localhost`. Keeping the repo current means updating workflow files as better open models appear.

### My calls (flag if wrong)

- Name `ghostprint`; MIT; same style as flyover-scrub (numpy/scipy, `uv`, pytest).
- One detector + fixer module per leak, all sharing one interface (`detect(media) -> Finding`,
  `fix(media) -> media`), so the report and the re-check are generic and v2 stages slot in the same way.
- The PRNU check runs only when the user supplies reference photos. Without them the report says "camera
  fingerprint: not checked (pass --fingerprint-from with a few other photos from this phone)".
- Default is **fix everything fixable**. `--keep KIND` opts out per leak.
- **PRNU attacker searches over scale and crop.** A plain rescale can be undone, so the fix is applied by default to
  every photo and video (references or not): denoise, plus a randomised resample with a small non-uniform warp. The
  check, when references are given, searches over scale and crop before reporting "at chance". "Chance" is measured
  locally, by correlating against the same fingerprint misaligned (shifted, flipped) rather than against bundled
  data from other cameras. Video is checked against reference *videos* when given; against photos only with the
  scale and crop search.
- **flyover-scrub gets changed as needed.** It's the same author's repo. v1 adds a public, tuned ridge detector
  to it and pins ghostprint to a tagged release. The acceptance test for the aircraft fix is flyover-scrub's offline
  attack on synthetic flyovers (it must fail on the output), not "no ridge detected".
- **Report wording follows what was measured.** `REMOVED` only when the detector fired on the input and not on the
  output. When a fix ran as a precaution with nothing detected, the line says `SCRUBBED (precaution)`, never
  `REMOVED`.

### Left to the builder

- Exact detection thresholds, calibrated on synthetic fixtures **and** on real files where we have them (the
  flyover-scrub study's clips stay private. Calibrate on them locally and never commit them).
- How flicker removal is implemented (per-row gain correction, temporal filtering or something else), as long as
  the detector passes on the output and the picture doesn't visibly band or pump.
- The PRNU estimator (standard Lukáš/Fridrich-style denoise residual plus normalised cross-correlation is the
  expected starting point) and the exact rescale or crop used to defeat it.
- The sun/shadow heuristic, or dropping it to a generic outdoor warning if it's unreliable.
- HTML report styling.

## Testing and verification

- Synthetic fixtures for every detector: hum with real wander, a flyover, rolling-shutter flicker on a rendered
  scene, GSM buzz, and PRNU from synthetic sensors. Each detector must fire on the fixture and not on a control.
- For each fix: the detector fires on the fixture's input and **not** on the output.
- No network: a test runs the full pipeline with sockets blocked.
- No trace: a test checks that the temp directory and home directory are unchanged after a run, apart from the
  requested output.
- Report honesty: a test where a detector is forced to fail must not yield "nothing found".

## Version 2 (identity), recorded so v1 leaves room for it

- **Voice:** a pitch/formant shift (labelled "stops casual recognition; reversible"), and voice conversion to a
  consistent synthetic voice via a ComfyUI workflow (open model, e.g. an RVC-class converter).
- **Face:** replace the user's face with a generated face that belongs to nobody (StyleGAN-class generator, as
  behind thispersondoesnotexist), mapped frame by frame with a face-swap workflow. Only generated faces, never a
  real person's photo. One reusable synthetic face per persona.
- Measured with open speaker- and face-recognition models: does the output still match the original person?
- Models are downloaded once, with explicit consent, then used offline. The repo lists each model's licence;
  many popular ones (InsightFace inswapper, NVIDIA StyleGAN) are non-commercial, and a permissive option is the
  default where one exists.
- Signs, plates and landmark detection and blurring also move here (needs detection models).
