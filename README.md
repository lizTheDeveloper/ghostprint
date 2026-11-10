# ghostprint

Strip the **place**, **time** and **device** leaks out of a photo, video or audio file before you post it,
and get a plain report of what was found, what was removed, and what it couldn't fix.

- **Runs entirely on your computer.** It never uses the network, has no accounts, and sends nothing anywhere.
- **Leaves no trace.** It never writes temporary copies of your media. The only files it writes are the cleaned
  copy and, if you ask, the report.
- **Never asks who you are or where you were.**
- **Cleaned files look ordinary:** no tell-tale "this was scrubbed" signature linking your posts together.

## What it checks

| Leak | What it gives away | ghostprint |
|---|---|---|
| Metadata: GPS, capture time, phone model and serial, editing software, thumbnails | where, when, which phone | **removes**, then writes ordinary-looking, randomised values |
| Long gliding tones from a passing plane | where (planes can be matched to public flight records) | **removes** |
| Grid hum from mains power | when, to the second, and which power grid | **removes** |
| Phone-radio buzz ("dit-dit-dit" from a phone near a speaker) | that a phone was transmitting nearby | **removes** |
| Flicker from mains lighting in video | when, and which grid, even with the sound removed | **removes** (when the frame rate allows it to be checked, see below) |
| Camera sensor fingerprint | links your photos to each other and to your phone | **weakens** it on every file. It can **check** only if you give it a few other photos from your phone |
| Sun position and hard shadows | time of day, latitude | **warns** only |
| Readable signs, plates, landmarks, reflections | where | **not checked**: look before you post |

## Install

You need a Mac or Linux computer and the Terminal. Each step is one line to copy and paste.

**On a Mac:**
1. Install Homebrew (skip if you have it): paste the line from [brew.sh](https://brew.sh) into Terminal.
2. `brew install ffmpeg exiftool uv`
3. `uv tool install git+https://github.com/lizTheDeveloper/ghostprint`

**On Debian or Ubuntu:**
1. `sudo apt install ffmpeg libimage-exiftool-perl pipx && pipx install uv`
2. `uv tool install git+https://github.com/lizTheDeveloper/ghostprint`

Installing downloads ghostprint once. After that it never touches the network.

## Use

```sh
ghostprint IMG_1234.MOV -o clean.mp4                 # clean a video and print the report
ghostprint IMG_5678.HEIC -o clean.jpg                # clean a photo (saved as JPEG)
ghostprint voice-memo.m4a -o clean.m4a               # clean audio
ghostprint IMG_1234.MOV --check                      # report only; writes nothing
ghostprint IMG_5678.HEIC -o clean.jpg --fingerprint-from ~/Pictures/my-phone   # also check the camera fingerprint
ghostprint IMG_1234.MOV -o clean.mp4 --report report.html                      # save the report too
```

Then **check your own work**: run `ghostprint clean.mp4 --check` on the output.

### Reading the report

| Word | Meaning |
|---|---|
| `REMOVED` | found in your file, and gone from the cleaned copy (re-checked) |
| `REDUCED` | the camera fingerprint matched your other photos, and the cleaned copy no longer matches |
| `SCRUBBED (precaution)` | nothing was found, but the fix ran anyway |
| `NOT FOUND` | looked for and not there |
| `STILL PRESENT` | found, and still there after cleaning. Don't post this |
| `WARNING` | something this tool can't fix, like the sun's position |
| `NOT CHECKED` | couldn't be checked, with the reason. Treat it as unknown, not as safe |

The top line says "Nothing found that this tool checks for" only when every line is fine. Signs, plates and
landmarks are never checked in this version, so look at the picture yourself before you post.

## Limits (read these)

- **It can't see what's in the picture.** Faces, street signs, house numbers, reflections in windows or glasses,
  a distinctive view: all still there.
- **Light flicker can't be checked at most common frame rates.** At 30 fps under 60 Hz lighting (or 25 fps under
  50 Hz) the flicker bands barely move between frames, so they can't be told apart from the scene. The report
  says `NOT CHECKED` with that reason.
- **Faint leaks below what its detectors can see** may still be recoverable by a determined analyst. The fixes
  are applied anyway; the detectors are how it checks them.
- **`--keep hum` alone doesn't guarantee the hum survives**: removing plane tones resynthesises the background,
  and faint hum goes with it.
- **The camera-fingerprint fix assumes a whole-picture search.** It warps the image very slightly (straight edges
  can wobble by a pixel), which defeats matching across the whole frame. An analyst matching small patches
  separately might still link it.
- **Video cleaning is slow** (minutes per minute of HD video), because every frame is re-processed.
- **It only protects the copy you post.** The original on your phone, cloud backups and whatever the app you
  post with records are out of its reach.

## Where this comes from

The plane attack was shown by **colsto** in
[*I discovered a sound that leaks your exact location (and it's everywhere)*](https://www.youtube.com/watch?v=UytYrhYLOKI).
The study that tested it on real phone videos, codecs and platforms, and the scrubber ghostprint uses, are in
[flyover-scrub](https://github.com/lizTheDeveloper/flyover-scrub). Grid hum forensics: Grigoras (2005), Cooper
(2008). Light flicker: Garg et al. (2013), Su et al. (2014). Camera fingerprints: Lukáš, Fridrich & Goljan (2006).

MIT licensed.
