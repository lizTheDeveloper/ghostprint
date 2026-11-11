import numpy as np

from conftest import make_sensor_images, sensor
from ghostprint.findings import Status
from ghostprint.leaks.prnu import detect, estimate, fix, match


def test_same_sensor_matches_other_does_not():
    KA, KB = sensor(1), sensor(2)
    K = estimate(make_sensor_images(KA, 8, seed=1))
    a = make_sensor_images(KA, 1, seed=50)[0]
    b = make_sensor_images(KB, 1, seed=51)[0]
    assert match(a, K).pce >= 60
    assert match(b, K).pce < 60
    assert detect(match(a, K)).status is Status.DETECTED
    assert detect(None).status is Status.NOT_CHECKED


def test_match_survives_a_plain_rescale():
    from scipy.ndimage import zoom
    KA = sensor(1)
    K = estimate(make_sensor_images(KA, 8, seed=1))
    a = make_sensor_images(KA, 1, seed=52)[0]
    small = np.clip(zoom(a.astype(float), (0.96, 0.96, 1), order=1), 0, 255).astype(np.uint8)
    assert match(small, K).pce >= 60  # the attacker's scale search undoes a plain resize


def test_fix_defeats_match_with_search():
    KA = sensor(1)
    K = estimate(make_sensor_images(KA, 8, seed=1))
    a = make_sensor_images(KA, 1, seed=53)[0]
    assert match(fix(a, np.random.default_rng(0)), K).pce < 60


def test_fix_is_visually_mild():
    a = make_sensor_images(sensor(1), 1, seed=54)[0]
    f = fix(a, np.random.default_rng(1))
    assert abs(f.shape[0] - a.shape[0]) <= 0.02 * a.shape[0] and abs(f.shape[1] - a.shape[1]) <= 0.02 * a.shape[1]
    h, w = min(f.shape[0], a.shape[0]), min(f.shape[1], a.shape[1])
    mse = np.mean((f[:h, :w].astype(float) - a[:h, :w].astype(float)) ** 2)
    assert 10 * np.log10(255 ** 2 / mse) >= 32


def test_video_warp_with_linear_interpolation_still_defeats_match():
    KA = sensor(1)
    K = estimate(make_sensor_images(KA, 8, seed=1))
    a = make_sensor_images(KA, 1, seed=71)[0]
    assert match(fix(a, np.random.default_rng(1), order=1), K).pce < 60
