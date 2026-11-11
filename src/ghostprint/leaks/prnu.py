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
MIN_PSNR = 38.0  # removal that would change the photo more than this (visible grain) is declined
SCALES = np.round(np.arange(0.90, 1.1001, 0.01), 2)


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float64)
    g = 0.299 * img[..., 0].astype(np.float64)  # a channel at a time: a 12 MP photo is 290 MB as float64
    g += 0.587 * img[..., 1]
    g += 0.114 * img[..., 2]
    return g


NOISE_SIGMA = 1.0  # noise level (8-bit) the adaptive denoiser assumes; on real JPEGs 1.0 found most (3.0 in Lukáš et al.)


def _zero_mean(w: np.ndarray) -> np.ndarray:
    w -= w.mean(axis=1, keepdims=True)
    w -= w.mean(axis=0, keepdims=True)
    return w


def _hp_residual(g: np.ndarray) -> np.ndarray:
    """Image minus a sigma-1 blur: a linear high-pass, so a fitted strength through it is unbiased."""
    return _zero_mean(g - gaussian_filter(g, 1.0))


def _residual(g: np.ndarray) -> np.ndarray:
    """Noise residual for matching: the high-pass, shrunk where the picture itself is busy (texture, edges).

    An adaptive Wiener shrink over 3–9 px windows, keeping the smallest estimate (after Mihcak et al.
    1999, the denoiser in Lukáš et al.): fine texture otherwise leaks in and drowns the fingerprint."""
    from scipy.ndimage import uniform_filter

    hp = (g - gaussian_filter(g, 1.0)).astype(np.float32)
    s2 = np.float32(NOISE_SIGMA ** 2)
    sq = hp * hp
    out = None
    for win in (3, 5, 7, 9):
        est = hp * (s2 / (np.maximum(uniform_filter(sq, win) - s2, 0) + s2))
        out = est if out is None else np.where(np.abs(est) < np.abs(out), est, out)
    return _zero_mean(out)


def estimate(images: list[np.ndarray]) -> np.ndarray:
    """Maximum-likelihood fingerprint K from same-sized images of one sensor."""
    num = den = 0.0
    for img in images:
        g = _gray(img)
        num = num + _residual(g) * g
        den = den + g * g
    return _zero_mean_phases((num / np.maximum(den, 1e-6)).astype(np.float32))


def _zero_mean_phases(K: np.ndarray) -> np.ndarray:
    """Remove every row and column mean within each 2x2 colour-filter phase (Fridrich's ZeroMeanTotal):
    those patterns are shared by every phone of a model, not unique to this sensor."""
    for i in (0, 1):
        for j in (0, 1):
            s = K[i::2, j::2]
            s -= s.mean(axis=1, keepdims=True)
            s -= s.mean(axis=0, keepdims=True)
    return K


@dataclass
class PRNUMatch:
    pce: float
    scale: float
    rot: int = 0  # quarter turns (np.rot90) that bring the photo back to the sensor's orientation...
    flip: bool = False  # ...after mirroring it left-right first (selfies, EXIF orientations 2/4/5/7)
    comparable: bool = True  # False: a crop (or odd resize) that can't be lined up with the reference frame
    offset_pce: float = 0.0  # best PCE anywhere: catches a shifted photo, but needs OFFSET_PCE to count


def _to_sensor(a: np.ndarray, rot: int, flip: bool) -> np.ndarray:
    return np.rot90(np.fliplr(a) if flip else a, rot)


def _from_sensor(a: np.ndarray, rot: int, flip: bool) -> np.ndarray:
    a = np.rot90(a, -rot)
    return np.fliplr(a) if flip else a


ALIGN_PX = 2  # how far from exact alignment the peak may sit (resampling, demosaicing shifts)
OFFSET_PCE = 1000.0  # anywhere-in-the-frame peaks reach ~40-320 by chance on real photos


def _pce(w: np.ndarray, kg: np.ndarray) -> tuple[float, float]:
    """Peak-to-correlation energy at the aligned position (±ALIGN_PX). Only the aligned position counts:
    the best of millions of offsets is large by chance (PCE 40–160 between different real phones).
    Returns (aligned PCE, best PCE at any offset)."""
    from scipy import fft

    a = (w - w.mean()).astype(np.float32)
    b = (kg - kg.mean()).astype(np.float32)
    F = fft.rfft2(a)
    F *= np.conj(fft.rfft2(b))
    xc = fft.irfft2(F, s=a.shape)
    del F
    r = np.arange(-ALIGN_PX, ALIGN_PX + 1)
    near0 = xc[np.ix_(r % xc.shape[0], r % xc.shape[1])]
    k = np.unravel_index(np.argmax(near0), near0.shape)
    aligned = (int(r[k[0]]) % xc.shape[0], int(r[k[1]]) % xc.shape[1])
    anywhere = np.unravel_index(int(np.argmax(xc)), xc.shape)
    total = float(np.vdot(xc, xc))

    def at(i) -> float:
        peak = float(xc[i])
        if peak <= 0:  # the fingerprint correlates positively; a negative peak is not a match
            return 0.0
        n = 5
        ys = (np.arange(i[0] - n, i[0] + n + 1) % xc.shape[0])[:, None]
        xs = (np.arange(i[1] - n, i[1] + n + 1) % xc.shape[1])[None, :]
        near = xc[ys, xs].astype(np.float64)
        return peak * peak / ((total - float((near ** 2).sum())) / (xc.size - near.size))

    return at(aligned), at(anywhere)


def _comparable(shape: tuple[int, int], K: np.ndarray) -> bool:
    """Whether a photo of this size could be the reference frame, turned and resized within SCALES."""
    H, W = K.shape
    for h, w in (shape, shape[::-1]):
        fh, fw = h / H, w / W
        if abs(fh - fw) < 0.01 and SCALES[0] - 0.005 <= fh <= SCALES[-1] + 0.005:
            return True
    return False


def match(img: np.ndarray, K: np.ndarray, scales=SCALES) -> PRNUMatch:
    """Best PCE of the image against K over the eight rotations and mirrorings and scales 0.90–1.10;
    FFT correlation covers crop offsets."""
    if not _comparable(img.shape[:2], K):
        return PRNUMatch(0.0, 1.0, comparable=False)
    best = PRNUMatch(0.0, 1.0)
    g = _gray(img).astype(np.float32)
    landscape = K.shape[1] >= K.shape[0]
    for flip in (False, True):
        for rot in range(4):
            r = _to_sensor(g, rot, flip)
            if r.shape[0] != r.shape[1] and (r.shape[1] > r.shape[0]) != landscape:
                continue  # a portrait frame can't line up with a landscape sensor without turning it
            m = _match_upright(r, K, scales)
            off = max(best.offset_pce, m.offset_pce)
            if m.pce > best.pce:
                best = PRNUMatch(m.pce, m.scale, rot, flip)
            best.offset_pce = off
    return best


def _match_upright(g: np.ndarray, K: np.ndarray, scales=SCALES) -> PRNUMatch:
    H, W = K.shape
    best = PRNUMatch(0.0, 1.0)
    for s in scales:
        gs = g if s == 1.0 else zoom(g, 1.0 / s, order=1)
        canvas = np.zeros((H, W), np.float32)
        h, w = min(H, gs.shape[0]), min(W, gs.shape[1])
        canvas[:h, :w] = gs[:h, :w]
        del gs
        wres = np.zeros((H, W), np.float32)
        wres[:h, :w] = _residual(canvas[:h, :w])
        canvas *= K
        p, anywhere = _pce(wres, canvas)
        del canvas, wres
        if p > best.pce:
            best = PRNUMatch(p, float(s), offset_pce=max(best.offset_pce, anywhere))
        else:
            best.offset_pce = max(best.offset_pce, anywhere)
    return best


def detect(m: PRNUMatch | None) -> Finding:
    if m is None:
        return Finding("fingerprint", Status.NOT_CHECKED, "",
                       reason="pass --fingerprint-from with a few other photos from this phone")
    if not m.comparable:
        return Finding("fingerprint", Status.NOT_CHECKED, "",
                       reason="this photo is cropped or sized differently from your reference photos, so it can't be lined up with them")
    if m.pce >= PCE_MATCH:
        return Finding("fingerprint", Status.DETECTED,
                       "camera sensor fingerprint matches your other photos: links this to the same phone",
                       strength=m.pce)
    if m.offset_pce >= OFFSET_PCE:
        return Finding("fingerprint", Status.DETECTED,
                       "camera sensor fingerprint matches your other photos at a shifted position (the photo was "
                       "shifted or cropped): links this to the same phone", strength=m.offset_pce)
    return Finding("fingerprint", Status.NOT_DETECTED,
                   "does not match your reference photos (more reference photos, or a stronger analysis, might still link it)",
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
    """The photo with the fingerprint K divided out, or None if it can't be lined up with K pixel for
    pixel (a cropped or resized photo, or one K doesn't recognise: another phone, or too faint to say
    which way round the photo is), or if removing it would visibly change the photo (below MIN_PSNR)."""
    if img.shape[:2] not in (K.shape, K.shape[::-1]):
        return None
    m = match(img, K, scales=(1.0,))  # removal needs the full frame, so only its alignment matters
    if m.pce < PCE_MATCH:
        return None
    x = _to_sensor(img, m.rot, m.flip)
    if x.shape[:2] != K.shape:
        return None
    Kd = _deconvolve(K)
    g = _gray(x).astype(np.float32)  # float32 from here: halves the working set
    w = _hp_residual(g)
    b = _hp_residual(Kd * g)
    c = float(np.vdot(w, b) / max(float(np.vdot(b, b)), 1e-20)) / max(rho, 0.05)
    del g, w, b
    Kd *= c
    Kd += 1  # now the per-pixel gain
    out = np.empty_like(x)
    sq = 0.0
    for ch in range(x.shape[2]):
        out[..., ch] = np.clip(np.rint(x[..., ch] / Kd), 0, 255)
        d = out[..., ch].astype(np.float32) - x[..., ch]
        sq += float(np.vdot(d, d))
    # A noisy estimate (few or grainy reference photos) has to be subtracted hard, and its noise goes in
    # with it: on real iPhone photos that came to 18 dB. Decline rather than damage the photo.
    mse = sq / out.size
    if mse > 0 and 10 * np.log10(255 ** 2 / mse) < MIN_PSNR:
        return None
    return _from_sensor(out, m.rot, m.flip).copy()
