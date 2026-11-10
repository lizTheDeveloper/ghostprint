"""Media probing and pipe-only I/O.

Nothing here writes intermediate media to disk: ffmpeg's input is the user's file,
its output is either a pipe to numpy or the requested output file.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff"}
INSTALL_HINT = {
    "ffmpeg": "macOS: brew install ffmpeg · Debian/Ubuntu: sudo apt install ffmpeg",
    "ffprobe": "comes with ffmpeg (macOS: brew install ffmpeg · Debian/Ubuntu: sudo apt install ffmpeg)",
    "exiftool": "macOS: brew install exiftool · Debian/Ubuntu: sudo apt install libimage-exiftool-perl",
}


class UnsafeInput(ValueError):
    """The file isn't plain media: a playlist or list that could make ffmpeg fetch URLs or read other files."""


# Demuxers we accept. Anything else (hls, dash, concat, tee, image2 sequences, ...) is refused.
ALLOWED_DEMUXERS = {"mov", "mp4", "m4a", "3gp", "3g2", "mj2", "matroska", "webm", "wav", "mp3", "aac", "flac", "ogg", "avi", "mpegts"}
_TEXT_LISTS = (b"#EXTM3U", b"ffconcat", b"<?xml", b"<MPD", b"file ")


def input_args(path: Path) -> list[str]:
    """ffmpeg/ffprobe input options for a user file: local file protocol only, never a URL."""
    return ["-protocol_whitelist", "file", "-i", "file:" + str(Path(path).resolve())]


class ToolMissing(RuntimeError):
    def __init__(self, name: str, install_hint: str):
        super().__init__(f"{name} is not installed. Install it with: {install_hint}")
        self.name, self.install_hint = name, install_hint


def require_tools() -> None:
    for name, hint in INSTALL_HINT.items():
        if shutil.which(name) is None:
            raise ToolMissing(name, hint)


@dataclass
class MediaInfo:
    path: Path
    kind: Literal["photo", "video", "audio"]
    duration_s: float | None
    width: int | None
    height: int | None
    fps: float | None
    rotation: int
    hdr: bool
    audio_sr: int | None
    audio_channels: int | None


def _ffprobe(path: Path) -> dict:
    with open(path, "rb") as f:
        head = f.read(64).lstrip(b"\xef\xbb\xbf \t\r\n")
    if head.startswith(_TEXT_LISTS):
        raise UnsafeInput(f"{Path(path).name} is a playlist or list file, not media")
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
                        *input_args(path)], capture_output=True, text=True)
    if r.returncode != 0:
        raise UnsafeInput(f"{Path(path).name} could not be read as media")
    j = json.loads(r.stdout)
    fmt = set(j.get("format", {}).get("format_name", "").split(","))
    if not fmt & ALLOWED_DEMUXERS:
        raise UnsafeInput(f"{Path(path).name} is not a supported media container ({', '.join(sorted(fmt))})")
    return j


def _rotation(stream: dict) -> int:
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            return int(round(float(sd["rotation"]))) % 360
    tag = (stream.get("tags") or {}).get("rotate")
    return int(tag) % 360 if tag else 0


def probe(path: Path) -> MediaInfo:
    path = Path(path)
    if path.suffix.lower() in PHOTO_EXT:
        from PIL import Image, ImageOps
        import pillow_heif

        pillow_heif.register_heif_opener()
        with Image.open(path) as im:
            w, h = ImageOps.exif_transpose(im).size
        return MediaInfo(path, "photo", None, w, h, None, 0, False, None, None)
    j = _ffprobe(path)
    streams = j.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    dur = float(j.get("format", {}).get("duration") or 0) or None
    if v is None:
        return MediaInfo(path, "audio", dur, None, None, None, 0, False,
                         int(a["sample_rate"]) if a else None, int(a["channels"]) if a else None)
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else None
    return MediaInfo(path, "video", dur, int(v["width"]), int(v["height"]), fps, _rotation(v),
                     v.get("color_transfer") in ("arib-std-b67", "smpte2084"),
                     int(a["sample_rate"]) if a else None, int(a["channels"]) if a else None)


# ----------------------------------------------------------------------------- pipe I/O
import os
import threading
from collections.abc import Iterable, Iterator

import numpy as np

TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
           "zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


def read_audio(info: MediaInfo) -> tuple[np.ndarray, int] | None:
    if not info.audio_sr:
        return None
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *input_args(info.path), "-map", "0:a:0", "-vn",
                          "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"], capture_output=True, check=True).stdout
    ch = info.audio_channels or 1
    a = np.frombuffer(raw, dtype="<f4")
    return a[: len(a) // ch * ch].reshape(-1, ch).copy(), info.audio_sr


def frame_size(info: MediaInfo) -> tuple[int, int]:
    """(width, height) of the frames iter_frames yields: upright, even."""
    w, h = info.width, info.height
    if info.rotation in (90, 270):
        w, h = h, w
    return w - w % 2, h - h % 2


def iter_frames(info: MediaInfo, chunk: int = 32) -> Iterator[np.ndarray]:
    """Upright (autorotated), SDR, even-sized uint8 RGB frames, ``chunk`` at a time. Re-reads the file each call."""
    w, h = frame_size(info)
    vf = ([TONEMAP] if info.hdr else []) + [f"crop={w}:{h}:0:0"]
    p = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", *input_args(info.path), "-map", "0:v:0", "-vf", ",".join(vf),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], stdout=subprocess.PIPE)
    fsz = w * h * 3
    try:
        while True:
            buf = p.stdout.read(fsz * chunk)
            if not buf:
                break
            n = len(buf) // fsz
            if n:
                yield np.frombuffer(buf[: n * fsz], np.uint8).reshape(n, h, w, 3)
            if len(buf) < fsz * chunk:
                break
    finally:
        p.stdout.close()
        p.wait()


def read_photo(info: MediaInfo) -> np.ndarray:
    """Upright RGB pixels, converted to sRGB if the file carries another colour profile."""
    import io

    from PIL import Image, ImageCms, ImageOps
    import pillow_heif

    pillow_heif.register_heif_opener()
    with Image.open(info.path) as im:
        im = ImageOps.exif_transpose(im)
        icc = im.info.get("icc_profile")
        im = im.convert("RGB")
        if icc:
            try:
                src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                im = ImageCms.profileToProfile(im, src, ImageCms.createProfile("sRGB"), outputMode="RGB")
            except (OSError, ImageCms.PyCMSError):
                pass
        return np.asarray(im, dtype=np.uint8).copy()


def _tag_args(tags: dict[str, str]) -> list[str]:
    args = ["-map_metadata", "-1", "-map_chapters", "-1", "-fflags", "+bitexact"]
    for k, v in tags.items():
        args += ["-metadata", f"{k}={v}"]
    return args


def write_video(out: Path, frames: Iterable[np.ndarray], fps: float, audio: tuple[np.ndarray, int] | None,
                profile) -> None:
    it = iter(frames)
    first = next(it)
    h, w = first.shape[1:3]
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", f"{fps:.6f}", "-i", "pipe:0"]
    rfd = wfd = None
    if audio is not None:
        a, sr = audio
        rfd, wfd = os.pipe()
        cmd += ["-f", "f32le", "-ar", str(sr), "-ac", str(a.shape[1]), "-i", f"pipe:{rfd}"]
    cmd += ["-map", "0:v"] + (["-map", "1:a"] if audio is not None else [])
    cmd += profile.video_args + (profile.audio_args if audio is not None else [])
    cmd += _tag_args(profile.container_tags) + ["-movflags", "+faststart", "-shortest" if audio is not None else "-y", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, pass_fds=(rfd,) if rfd is not None else ())
    errors = []

    def feed_audio():
        try:
            with os.fdopen(wfd, "wb") as f:
                f.write(np.ascontiguousarray(audio[0], dtype="<f4").tobytes())
        except BrokenPipeError:
            pass
        except Exception as e:  # pragma: no cover
            errors.append(e)

    t = None
    if audio is not None:
        os.close(rfd)
        t = threading.Thread(target=feed_audio, daemon=True)
        t.start()
    try:
        p.stdin.write(np.ascontiguousarray(first).tobytes())
        for c in it:
            p.stdin.write(np.ascontiguousarray(c, dtype=np.uint8).tobytes())
    except BrokenPipeError:
        pass
    finally:
        p.stdin.close()
    if t:
        t.join()
    if p.wait() != 0 or errors:
        raise RuntimeError(f"ffmpeg failed writing {out.name}")


def write_audio(out: Path, audio: np.ndarray, sr: int, profile) -> None:
    a = np.ascontiguousarray(audio, dtype="<f4")
    if a.ndim == 1:
        a = a[:, None]
    codec = ["-c:a", "pcm_s16le"] if out.suffix.lower() == ".wav" else profile.audio_args
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", str(a.shape[1]), "-i", "pipe:0",
           *codec, *_tag_args(profile.container_tags if out.suffix.lower() != ".wav" else {}), str(out)]
    subprocess.run(cmd, input=a.tobytes(), check=True)


def write_photo(out: Path, img: np.ndarray, profile) -> None:
    from PIL import Image

    im = Image.fromarray(np.ascontiguousarray(img, dtype=np.uint8), "RGB")
    if out.suffix.lower() == ".png":
        im.save(out, format="PNG")
    else:
        im.save(out, format="JPEG", quality=profile.jpeg_quality, optimize=True)
