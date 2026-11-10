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
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


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
