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



def test_estimate_has_no_row_or_column_pattern_in_any_colour_phase():
    # rows/columns shared by every phone of a model (readout, demosaicing) would match other phones:
    # on real iPhone photos another phone scored PCE 300 against the raw estimate, 41 after this
    rng = np.random.default_rng(3)
    imgs = [np.clip(im * (1 + 0.02 * rng.normal(size=(1, 256, 1))), 0, 255).astype(np.uint8)
            for im in make_sensor_images(sensor(1), 6, seed=1)]
    K = estimate(imgs)
    for i in (0, 1):
        for j in (0, 1):
            s = K[i::2, j::2]
            assert np.abs(s.mean(axis=0)).max() < 1e-6 and np.abs(s.mean(axis=1)).max() < 1e-6


def test_remove_declines_rather_than_damage_the_photo():
    # few reference photos of a faint fingerprint: the estimate is mostly noise, and subtracting it at
    # the strength needed would add visible grain (real iPhone photos came out at 18 dB)
    KT = sensor(1, strength=0.02)
    refs = make_sensor_images(KT, 2, seed=1, noise=20.0)  # two grainy photos: a match, but a noisy estimate
    x = make_sensor_images(KT, 1, seed=208)[0]
    K = estimate(refs)
    assert match(x, K).pce >= 60
    y = remove(x, K, reliability(refs))
    assert y is None or _psnr(x, y) >= 38


def test_match_only_counts_the_aligned_position():
    # The best of millions of correlation offsets is large by chance alone: on real iPhone photos it gave
    # PCE 40–160 for photos from other phones, while the aligned position read 0–18.
    KT, refs, _ = _setup()
    x = np.roll(make_sensor_images(KT, 1, seed=209)[0], (97, 41), axis=(0, 1))
    assert match(x, estimate(refs)).pce < 60


def test_a_cropped_photo_is_not_checked():
    KT, refs, _ = _setup()
    m = match(make_sensor_images(KT, 1, seed=210)[0][:150, :150], estimate(refs))
    assert detect(m).status is Status.NOT_CHECKED
