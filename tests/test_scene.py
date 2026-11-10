import numpy as np

from ghostprint.findings import Status
from ghostprint.leaks.scene import detect_sun, visible


def _outdoor():
    h, w = 240, 320
    img = np.zeros((h, w, 3), np.float64)
    t = np.linspace(0, 1, 80)[:, None]
    img[:80] = np.stack([110 + 40 * t, 160 + 30 * t, 235 - 10 * t], axis=-1)  # blue sky
    img[80:] = [170, 160, 140]  # sunlit pavement
    yy, xx = np.mgrid[80:h, 0:w]
    shadow = (xx > 60 + 0.8 * (yy - 80)) & (xx < 200 + 0.8 * (yy - 80))  # a building's hard-edged shadow
    img[80:][shadow] = [60, 58, 62]
    return np.clip(img + np.random.default_rng(0).normal(0, 2, img.shape), 0, 255).astype(np.uint8)


def _indoor():
    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w]
    v = 200 - 40 * (((yy - h / 2) / h) ** 2 + ((xx - w / 2) / w) ** 2)
    img = np.stack([v, v * 0.92, v * 0.8], axis=-1)
    return np.clip(img + np.random.default_rng(1).normal(0, 2, img.shape), 0, 255).astype(np.uint8)


def test_sky_and_shadow_warns():
    f = detect_sun(_outdoor())
    assert f.status is Status.WARNING and "sun" in f.summary


def test_indoor_flat_does_not_warn():
    assert detect_sun(_indoor()).status is Status.NOT_DETECTED


def test_visible_is_always_not_checked():
    f = visible()
    assert f.status is Status.NOT_CHECKED and "look before you post" in f.reason
