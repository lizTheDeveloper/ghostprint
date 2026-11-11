"""Camera sensor fingerprint (PRNU: photo-response non-uniformity).

Every sensor's pixels differ slightly in sensitivity, and that pattern survives in every photo
and video it makes. With a few photos known to come from one phone, an analyst can estimate the
pattern and test whether an "anonymous" photo came from the same phone (Lukáš, Fridrich &
Goljan 2006). Matching is a peak-to-correlation-energy (PCE) test; PCE ≥ 60 is the usual
decision threshold.

The attacker is assumed to search over scale and crop, so a plain resize isn't a defence: the fix
denoises and applies a small random, non-uniform warp, which misaligns the pattern differently in
different parts of the image.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, zoom

from ..findings import Finding, Status

PCE_MATCH = 60.0
SCALES = np.round(np.arange(0.90, 1.1001, 0.01), 2)


def _gray(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float64)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2] if f.ndim == 3 else f


def _residual(g: np.ndarray) -> np.ndarray:
    """Noise residual: image minus a denoised copy, zero-meaned along rows and columns."""
    w = g - gaussian_filter(g, 1.0)
    w -= w.mean(axis=1, keepdims=True)
    w -= w.mean(axis=0, keepdims=True)
    return w


def estimate(images: list[np.ndarray]) -> np.ndarray:
    """Maximum-likelihood fingerprint K from same-sized images of one sensor."""
    num = den = 0.0
    for img in images:
        g = _gray(img)
        num = num + _residual(g) * g
        den = den + g * g
    return (num / np.maximum(den, 1e-6)).astype(np.float32)


@dataclass
class PRNUMatch:
    pce: float
    scale: float


def _pce(w: np.ndarray, kg: np.ndarray) -> float:
    a = w - w.mean()
    b = kg - kg.mean()
    xc = np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))))
    i = np.unravel_index(np.argmax(np.abs(xc)), xc.shape)
    peak = xc[i]
    mask = np.ones_like(xc, bool)
    r = 5
    ys = (np.arange(i[0] - r, i[0] + r + 1) % xc.shape[0])[:, None]
    xs = (np.arange(i[1] - r, i[1] + r + 1) % xc.shape[1])[None, :]
    mask[ys, xs] = False
    return float(peak * peak / np.mean(xc[mask] ** 2))


def match(img: np.ndarray, K: np.ndarray) -> PRNUMatch:
    """Best PCE of the image against K over scales 0.90–1.10; FFT correlation covers crop offsets."""
    g = _gray(img)
    H, W = K.shape
    best = PRNUMatch(0.0, 1.0)
    for s in SCALES:
        gs = g if s == 1.0 else zoom(g, 1.0 / s, order=1)
        canvas = np.zeros((H, W))
        h, w = min(H, gs.shape[0]), min(W, gs.shape[1])
        canvas[:h, :w] = gs[:h, :w]
        wres = np.zeros((H, W))
        wres[:h, :w] = _residual(canvas[:h, :w])
        p = _pce(wres, K * canvas)
        if p > best.pce:
            best = PRNUMatch(p, float(s))
    return best


def detect(m: PRNUMatch | None) -> Finding:
    if m is None:
        return Finding("fingerprint", Status.NOT_CHECKED, "",
                       reason="pass --fingerprint-from with a few other photos from this phone")
    if m.pce >= PCE_MATCH:
        return Finding("fingerprint", Status.DETECTED,
                       "camera sensor fingerprint matches your other photos: links this to the same phone",
                       strength=m.pce)
    return Finding("fingerprint", Status.NOT_DETECTED, "does not match your other photos' sensor fingerprint",
                   strength=m.pce)


def make_warp(shape: tuple[int, int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A random smooth displacement field (~1.5 px, varying over ~12–30 px) plus a slight random rescale.

    Measured on synthetic sensors: ~0.5 px over 30–60 px left PCE in the thousands (regions near the
    field's zero crossings stay aligned); 1.5 px over 12–30 px brings it under 60 at ~38 dB PSNR.
    Straight edges may wobble very slightly."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    s = rng.uniform(0.985, 1.015)
    dy = np.zeros((h, w), np.float32)
    dx = np.zeros((h, w), np.float32)
    for _ in range(6):
        ky, kx = rng.uniform(2 * np.pi / 30, 2 * np.pi / 12, 2)
        py, px = rng.uniform(0, 2 * np.pi, 2)
        dy += 1.5 * np.sin(kx * xx + px) * np.cos(ky * yy + py)
        dx += 1.5 * np.cos(kx * xx + py) * np.sin(ky * yy + px)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    return ((yy - cy) * s + cy + dy).astype(np.float32), ((xx - cx) * s + cx + dx).astype(np.float32)


def fix(img: np.ndarray, rng: np.random.Generator, warp=None, order: int = 3) -> np.ndarray:
    """Denoise lightly, then warp. Pass the same ``warp`` (from make_warp) for every frame of a video.
    Works one channel at a time in float32 to keep memory near 3x the image size."""
    wy, wx = warp if warp is not None else make_warp(img.shape[:2], rng)
    coords = np.stack([wy, wx]).astype(np.float32)
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        f = img[..., c].astype(np.float32)
        f -= 0.8 * (f - gaussian_filter(f, 0.8))  # strip most of the pixel-level noise residual
        out[..., c] = np.clip(np.rint(map_coordinates(f, coords, order=order, mode="reflect")), 0, 255)
    return out
