import numpy as np

from conftest import make_sensor_images, sensor
from ghostprint.findings import Status
from ghostprint.leaks.prnu import detect, estimate, match, reliability, remove


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


def _psnr(a, b):
    return 10 * np.log10(255 ** 2 / np.mean((a.astype(float) - b.astype(float)) ** 2))


def _setup(strength=0.01, n=8):
    KT = sensor(1, strength=strength)
    refs = make_sensor_images(KT, n, seed=1)
    attacker = estimate(make_sensor_images(KT, 32, seed=2))  # the analyst's own photos of the same phone
    return KT, refs, attacker


def test_remove_defeats_an_independent_estimate():
    KT, refs, attacker = _setup()
    x = make_sensor_images(KT, 1, seed=200)[0]
    assert match(x, attacker).pce >= 60
    y = remove(x, estimate(refs), reliability(refs))
    assert match(y, attacker).pce < 60
    assert _psnr(x, y) >= 40


def test_remove_corrects_for_a_noisy_estimate():
    # a faint fingerprint from few photos: the estimate is mostly noise, which shrinks a naive fit
    KT, refs, attacker = _setup(strength=0.005, n=4)
    x = make_sensor_images(KT, 1, seed=201)[0]
    assert match(remove(x, estimate(refs), reliability(refs)), attacker).pce < 60


def test_match_finds_a_rotated_photo():
    KT, refs, _ = _setup()
    K = estimate(refs)
    x = make_sensor_images(KT, 1, seed=202)[0]
    for k in (1, 2, 3):
        assert match(np.rot90(x, k).copy(), K).pce >= 60


def test_remove_handles_a_rotated_photo():
    KT, refs, attacker = _setup()
    x = np.rot90(make_sensor_images(KT, 1, seed=203)[0]).copy()
    y = remove(x, estimate(refs), reliability(refs))
    assert y.shape == x.shape and match(y, attacker).pce < 60


def test_remove_declines_a_photo_of_another_size():
    KT, refs, _ = _setup()
    x = make_sensor_images(KT, 1, seed=204)[0][:200, :230]
    assert remove(x, estimate(refs), reliability(refs)) is None


def test_match_finds_a_mirrored_photo():
    KT, refs, _ = _setup()
    x = make_sensor_images(KT, 1, seed=205)[0]
    for k in (0, 1):
        assert match(np.rot90(np.fliplr(x), k).copy(), estimate(refs)).pce >= 60


def test_remove_handles_a_mirrored_photo():
    KT, refs, attacker = _setup()
    x = np.rot90(np.fliplr(make_sensor_images(KT, 1, seed=206)[0])).copy()
    y = remove(x, estimate(refs), reliability(refs))
    assert y is not None and y.shape == x.shape and match(y, attacker).pce < 60


def test_remove_declines_a_photo_from_another_phone():
    KT, refs, _ = _setup()
    other = make_sensor_images(sensor(2, strength=0.01), 1, seed=207)[0]
    assert remove(other, estimate(refs), reliability(refs)) is None
