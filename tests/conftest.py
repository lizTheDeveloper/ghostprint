"""Synthetic fixtures. Everything is generated in tmp_path; no real media is ever committed."""
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PIL import Image


def _ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)


def make_audio(tmp_path: Path, x: np.ndarray, sr: int, name: str = "a.wav") -> Path:
    p = tmp_path / name
    sf.write(p, x, sr)
    return p


def make_video(tmp_path: Path, *, seconds: float = 40, fps: int = 30, size=(320, 240), audio=None,
               rotation: int = 0, hdr: bool = False, frames: np.ndarray | None = None, name: str = "v.mov",
               gps: bool = False) -> Path:
    """A test video. ``frames`` (n, H, W, 3) uint8 overrides the default test pattern."""
    w, h = size
    out = tmp_path / name
    src = []
    if frames is not None:
        raw = tmp_path / (name + ".rgb")
        raw.write_bytes(np.ascontiguousarray(frames, dtype=np.uint8).tobytes())
        h, w = frames.shape[1:3]
        src += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", str(raw)]
    else:
        src += ["-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps}:duration={seconds}"]
    if audio is not None:
        a = make_audio(tmp_path, audio[0], audio[1], name + ".wav")
        src += ["-i", str(a)]
    args = [*src, "-map", "0:v"] + (["-map", "1:a", "-c:a", "aac", "-b:a", "192k"] if audio is not None else [])
    if hdr:
        args += ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-color_primaries", "bt2020",
                 "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", "-tag:v", "hvc1"]
    else:
        args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
    if gps:
        args += ["-metadata", "location=+45.1000-122.2000/", "-metadata", "com.apple.quicktime.location.ISO6709=+45.1000-122.2000/",
                 "-metadata", "com.apple.quicktime.model=iPhone 12 SN12345", "-metadata", "creation_time=2023-08-12T18:58:04Z"]
    _ff(*args, "-shortest", "-movflags", "use_metadata_tags", str(out))
    if rotation:
        rot = tmp_path / ("r_" + name)
        _ff("-display_rotation", str(rotation), "-i", str(out), "-c", "copy", str(rot))
        out = rot
    return out


def make_photo(tmp_path: Path, img: np.ndarray, *, fmt: str = "jpeg", exif: dict | None = None, name: str | None = None) -> Path:
    p = tmp_path / (name or f"p.{ 'jpg' if fmt == 'jpeg' else fmt}")
    im = Image.fromarray(np.ascontiguousarray(img, dtype=np.uint8))
    kw = {}
    if exif:
        ex = Image.Exif()
        for k, v in exif.items():
            ex[k] = v
        kw["exif"] = ex.tobytes()
    if fmt == "heic":
        import pillow_heif
        pillow_heif.register_heif_opener()
    im.save(p, format=fmt.upper() if fmt != "jpeg" else "JPEG", **kw)
    return p


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def make_flicker_frames(*, seconds=20, fps=30.0, size=(160, 120), flicker_hz=119.94, depth=0.03,
                        readout=0.7, seed=0, noise=1.0):
    """(flickered, clean) uint8 frames of a static scene under mains lighting, rolling-shutter style:
    row y of frame n is exposed at n/fps + (y/H)·readout/fps."""
    rng = np.random.default_rng(seed)
    w, h = size
    from scipy.ndimage import gaussian_filter
    scene = gaussian_filter(rng.uniform(40, 200, (h, w, 3)), (3, 3, 0))
    n = int(seconds * fps)
    t = np.arange(n)[:, None] / fps + (np.arange(h)[None, :] / h) * readout / fps
    gain = 1 + depth * np.sin(2 * np.pi * flicker_hz * t)  # (n, h)
    clean = np.clip(scene[None] + noise * rng.normal(size=(n, h, w, 3)), 0, 255)
    flick = np.clip(clean * gain[:, :, None, None], 0, 255)
    return flick.astype(np.uint8), clean.astype(np.uint8)


def make_sensor_images(K: np.ndarray, n: int, seed: int = 0, noise: float = 1.5) -> list[np.ndarray]:
    """n photos from a sensor whose fingerprint (photo-response non-uniformity) is K (H, W): I = scene·(1+K) + noise."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    h, w = K.shape
    out = []
    for _ in range(n):
        scene = gaussian_filter(rng.uniform(30, 220, (h, w, 3)), (6, 6, 0))
        img = scene * (1 + K[:, :, None]) + noise * rng.normal(size=(h, w, 3))
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


def sensor(seed: int, size=(256, 256), strength=0.02) -> np.ndarray:
    return np.random.default_rng(1000 + seed).normal(0, strength, size)


def make_textured_images(K: np.ndarray, n: int, seed: int = 0, texture: float = 40.0, noise: float = 1.5) -> list[np.ndarray]:
    """Like make_sensor_images, with patches of fine texture (foliage, fabric) that leak into a naive noise residual."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    h, w = K.shape
    out = []
    for _ in range(n):
        scene = gaussian_filter(rng.uniform(30, 220, (h, w, 3)), (6, 6, 0))
        patches = (gaussian_filter(rng.normal(size=(h, w)), 20) > 0)[:, :, None]
        scene = scene + patches * texture * gaussian_filter(rng.normal(size=(h, w, 3)), (0.7, 0.7, 0))
        out.append(np.clip(scene * (1 + K[:, :, None]) + noise * rng.normal(size=(h, w, 3)), 0, 255).astype(np.uint8))
    return out
