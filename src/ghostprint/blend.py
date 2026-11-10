"""Make cleaned files look like ordinary files, not like "cleaned" files.

Every run draws a fresh random profile, so two outputs never share a fixed timestamp,
quality setting or encoder string that would tie one ghost's posts together.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ffmpeg replaces any encoder tag we pass with its own version (rare, so identifying), so outputs are
# written with -fflags +bitexact and carry no encoder tag, like phone camera originals. If a tag is
# present it must be one of these. Muxer tags written by the ffmpeg versions common Linux distributions ship
# (Ubuntu 20.04: 4.2 → Lavf58.29.100; Ubuntu 22.04: 4.4 → Lavf58.76.100; Debian 12: 5.1 → Lavf59.27.100;
# Ubuntu 24.04: 6.1 → Lavf60.16.100; 7.1 → Lavf61.7.100). Countless ordinary MP4s carry one of these.
COMMON_ENCODER_TAGS = ("Lavf58.29.100", "Lavf58.76.100", "Lavf59.27.100", "Lavf60.16.100", "Lavf61.7.100")


@dataclass
class BlendProfile:
    container_tags: dict[str, str]
    video_args: list[str]
    audio_args: list[str]
    jpeg_quality: int
    mtime: float


def random_profile(kind: str, rng: np.random.Generator) -> BlendProfile:
    now = time.time()
    created = now - rng.uniform(0, 72 * 3600)
    mtime = min(now, created + rng.uniform(0, 2 * 3600))
    ct = datetime.fromtimestamp(created, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    crf = int(rng.integers(20, 25))
    abr = int(rng.choice([128, 128, 160]))
    return BlendProfile(
        container_tags={"creation_time": ct},
        video_args=["-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-crf", str(crf),
                    "-preset", "medium", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"],
        audio_args=["-c:a", "aac", "-b:a", f"{abr}k", "-ar", "44100"],
        jpeg_quality=int(rng.integers(85, 96)),
        mtime=float(mtime),
    )


def apply_mtime(path: Path, profile: BlendProfile) -> None:
    os.utime(path, (profile.mtime, profile.mtime))
