"""Camera sensor fingerprint (PRNU: photo-response non-uniformity).

Every sensor's pixels differ slightly in sensitivity, and that pattern survives in every photo
and video it makes. With a few photos known to come from one phone, an analyst can estimate the
pattern and test whether an "anonymous" photo came from the same phone (Lukáš, Fridrich &
Goljan 2006). Matching is a peak-to-correlation-energy (PCE) test; PCE ≥ 60 is the usual
decision threshold.

The attacker is assumed to search over scale, crop and rotation, so a plain resize isn't a defence.

Removal (`remove`) needs the user's own reference photos: it divides the photo by (1 + c·K̂), where
K̂ is the fingerprint estimated from those photos and deconvolved for the high-pass filter the
estimate is made through, and c is fitted to this photo then corrected for the noise in K̂
(split-half reliability). Against an analyst's independent estimate this lands at chance level
in tests, at ~40–50 dB PSNR, with no geometric change. An earlier approach (a random warp) defeated
matching but visibly rippled fine texture on real photos, and was dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from ..findings import Finding, Status

PCE_MATCH = 60.0
SCALES = np.round(np.arange(0.90, 1.1001, 0.01), 2)


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float64)
    g = 0.299 * img[..., 0].astype(np.float64)  # a channel at a time: a 12 MP photo is 290 MB as float64
    g += 0.587 * img[..., 1]
    g += 0.114 * img[..., 2]
    return g


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
    rot: int = 0  # quarter turns (np.rot90) that bring the photo back to the sensor's orientation


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
    """Best PCE of the image against K over the four rotations and scales 0.90–1.10; FFT correlation
    covers crop offsets."""
    best = PRNUMatch(0.0, 1.0)
    g = _gray(img)
    landscape = K.shape[1] >= K.shape[0]
    for rot in range(4):
        r = np.rot90(g, rot)
        if r.shape[0] != r.shape[1] and (r.shape[1] > r.shape[0]) != landscape:
            continue  # a portrait frame can't line up with a landscape sensor without turning it
        m = _match_upright(r, K)
        if m.pce > best.pce:
            best = PRNUMatch(m.pce, m.scale, rot)
    return best


def _match_upright(g: np.ndarray, K: np.ndarray) -> PRNUMatch:
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


def reliability(images: list[np.ndarray]) -> float:
    """How much of the estimate from these images is fingerprint rather than noise (0–1): the
    split-half correlation, stepped up to the full set (Spearman–Brown)."""
    if len(images) < 2:
        return 1.0
    a, b = estimate(images[0::2]), estimate(images[1::2])
    r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    r = max(r, 0.05)
    return 2 * r / (1 + r)


def _deconvolve(K: np.ndarray, floor: float = 0.2) -> np.ndarray:
    """Undo the estimate's high-pass (image minus a sigma-1 Gaussian blur) so the whole pattern is removed."""
    from scipy import fft

    H, W = K.shape
    fy = fft.fftfreq(H).astype(np.float32)[:, None]
    fx = fft.rfftfreq(W).astype(np.float32)[None, :]
    passed = 1 - np.exp(-2 * np.pi ** 2 * (fy ** 2 + fx ** 2))
    F = fft.rfft2(K.astype(np.float32))
    F /= np.maximum(passed, floor)
    del passed
    return fft.irfft2(F, s=(H, W)).astype(np.float32)


def remove(img: np.ndarray, K: np.ndarray, rho: float = 1.0) -> np.ndarray | None:
    """The photo with the fingerprint K divided out, or None if the photo isn't the sensor's full frame
    (cropped or resized), where the pattern can't be lined up pixel for pixel."""
    rot = next((r for r in range(4) if np.rot90(img, r).shape[:2] == K.shape), None)
    if rot is None:
        return None
    if img.shape[0] == img.shape[1]:  # square: the shape can't tell the rotation, the correlation can
        rot = match(img, K).rot
    x = np.rot90(img, rot)
    Kd = _deconvolve(K)
    g = _gray(x).astype(np.float32)  # float32 from here: halves the working set
    w = _residual(g)
    b = _residual(Kd * g)
    c = float(np.vdot(w, b) / max(float(np.vdot(b, b)), 1e-20)) / max(rho, 0.05)
    del g, w, b
    Kd *= c
    Kd += 1  # now the per-pixel gain
    out = np.empty_like(x)
    for ch in range(x.shape[2]):
        out[..., ch] = np.clip(np.rint(x[..., ch] / Kd), 0, 255)
    return np.rot90(out, -rot).copy()
