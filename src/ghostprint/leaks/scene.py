"""What the camera saw. v1 only warns: these can't be fixed without understanding the picture."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..findings import Finding, Status

SUN_TEXT = "outdoor scene with hard shadows: the sun's position can give the time of day and latitude"


def _sky_fraction(img: np.ndarray) -> float:
    top = img[: max(1, img.shape[0] // 3)].astype(np.float64)
    r, g, b = top[..., 0], top[..., 1], top[..., 2]
    sky = (b > 140) & (b > r + 15) & (b >= g)
    bright_overcast = (r > 190) & (g > 190) & (b > 190)
    return float(np.mean(sky | bright_overcast))


def _hard_shadow(img: np.ndarray) -> bool:
    """A large dark region in the lower two thirds with a sharp, long boundary against a bright one."""
    low = img[img.shape[0] // 3:].astype(np.float64)
    L = 0.299 * low[..., 0] + 0.587 * low[..., 1] + 0.114 * low[..., 2]
    p10, p90 = np.percentile(L, [10, 90])
    if p90 < 80 or p90 - p10 < 60 or p10 > 0.6 * p90:
        return False
    dark = ndimage.binary_opening(L < (p10 + p90) / 2, iterations=2)
    lab, n = ndimage.label(dark)
    if n == 0:
        return False
    sizes = ndimage.sum(dark, lab, range(1, n + 1))
    big = lab == (1 + int(np.argmax(sizes)))
    if big.mean() < 0.05:
        return False
    edge = big ^ ndimage.binary_erosion(big)
    grad = np.hypot(ndimage.sobel(L, 0), ndimage.sobel(L, 1))
    # sharp: the boundary carries most of the contrast within a couple of pixels
    return edge.sum() > 0.5 * np.sqrt(big.sum()) and np.median(grad[edge]) > 0.5 * (p90 - p10)


def detect_sun(img: np.ndarray) -> Finding:
    if _sky_fraction(img) >= 0.3 and _hard_shadow(img):
        return Finding("sun", Status.WARNING, SUN_TEXT)
    return Finding("sun", Status.NOT_DETECTED, "no sky with hard shadows")


def visible() -> Finding:
    return Finding("visible", Status.NOT_CHECKED, "readable signs, plates, landmarks or reflections",
                   reason="not checked in v1: look before you post")
