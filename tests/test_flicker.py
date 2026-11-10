import numpy as np

from conftest import make_flicker_frames
from ghostprint.findings import Status
from ghostprint.leaks.flicker import FlickerFixer, detect, detect_photo, measure

FPS = 27.0  # neither 100 nor 120 Hz is a multiple of 27 fps, so both grids are checkable


def _src(frames):
    return lambda: (frames[i:i + 32] for i in range(0, len(frames), 32))


def test_flicker_detected_and_removed():
    fl, clean = make_flicker_frames(fps=FPS)
    est = measure(_src(fl), FPS, fl.shape[1])
    assert detect(est, 20.0).status is Status.DETECTED and est.nominal_hz == 60
    fx = FlickerFixer(est)
    fixed = np.concatenate([fx.apply(fl[i:i + 32], i) for i in range(0, len(fl), 32)])
    assert detect(measure(_src(fixed), FPS, fl.shape[1]), 20.0).status is Status.NOT_DETECTED
    assert np.abs(fixed.astype(float) - clean.astype(float)).mean() < 2.0


def test_50hz_grid_detected():
    fl, _ = make_flicker_frames(flicker_hz=99.95, fps=30.0, seed=2)
    est = measure(_src(fl), 30.0, fl.shape[1])
    assert detect(est, 20.0).status is Status.DETECTED and est.nominal_hz == 50


def test_no_flicker_control():
    _, clean = make_flicker_frames(fps=FPS, seed=1)
    assert detect(measure(_src(clean), FPS, clean.shape[1]), 20.0).status is Status.NOT_DETECTED


def test_short_clip_not_checked():
    fl, _ = make_flicker_frames(seconds=5, fps=FPS)
    f = detect(measure(_src(fl), FPS, fl.shape[1]), 5.0)
    assert f.status is Status.NOT_CHECKED and "shorter" in f.reason


def test_photo_banding_warns():
    fl, clean = make_flicker_frames(seconds=0.2, depth=0.08)
    assert detect_photo(fl[0]).status is Status.WARNING
    assert detect_photo(clean[0]).status is Status.NOT_DETECTED


def test_faint_flicker_still_detected():
    fl, _ = make_flicker_frames(depth=0.005, seed=4, fps=FPS)
    assert detect(measure(_src(fl), FPS, fl.shape[1]), 20.0).status is Status.DETECTED


def test_frame_rate_locked_to_grid_is_not_checked():
    """30 fps under 120 Hz flicker: the bands barely move between frames, so they can't be told apart
    from slow scene changes. Report NOT CHECKED rather than guess either way."""
    fl, _ = make_flicker_frames(fps=30.0, flicker_hz=119.94)
    f = detect(measure(_src(fl), 30.0, fl.shape[1]), 20.0)
    assert f.status is Status.NOT_CHECKED and "frame rate" in f.reason
