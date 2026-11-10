import numpy as np
import pytest

from flyover_scrub.attack import whitened_spectrogram
from flyover_scrub.physics import make_track, synthesize_flyover
from ghostprint.findings import Status
from ghostprint.leaks.audio import detect_aircraft, detect_gsm, detect_hum, fix_audio

SR = 8000


def _flyover(rng, seconds=120, level_db=-10):
    trk = make_track(0.0, 260.0, start_xy=(-9000.0, -400.0), altitude=700.0, speed=70.0, heading_deg=5.0,
                     turn_rate_deg_s=-1.0, turn_start=100.0, turn_duration=25.0)
    n = SR * seconds
    bg = rng.normal(size=n)
    plane = synthesize_flyover(trk, np.zeros(3), 70.0, n, SR, f0=90.0, rng=rng)
    plane *= 10 ** (level_db / 20) * bg.std() / plane[plane != 0].std()
    return trk, bg, plane


def _hum(n, rng):
    t = np.arange(n) / SR
    w = 60 + 0.02 * np.sin(2 * np.pi * t / 37) + 0.01 * np.sin(2 * np.pi * t / 11)
    ph = 2 * np.pi * np.cumsum(w) / SR
    return 0.05 * np.sin(2 * ph) + 0.03 * np.sin(4 * ph)


def _gsm(n):
    """GSM TDMA buzz: a 577 µs burst every 4.615 ms (216.7 Hz frame rate), as a speaker demodulates it."""
    t = np.arange(n) / SR
    return 0.08 * (((t % 4.615e-3) < 0.577e-3).astype(float) - 0.125)


def test_aircraft_detected_and_removed():
    rng = np.random.default_rng(3)
    trk, bg, plane = _flyover(rng)
    mix = (bg + plane)[:, None]
    assert detect_aircraft(mix, SR).status is Status.DETECTED
    assert detect_aircraft(bg[:, None], SR).status is Status.NOT_DETECTED
    y = fix_audio(mix, SR, set())
    assert detect_aircraft(y, SR).status is Status.NOT_DETECTED
    # The attack that worked on real recordings (flyover-scrub's trace-fit): it locates the original,
    # and finds no traceable line to fit after the fix.
    from flyover_scrub.trace import locate_trace, trace_ridge
    spec = lambda z: whitened_spectrogram(z, SR, window_s=2.0)
    r0 = trace_ridge(spec(mix[:, 0]), f_range=(70.0, 115.0))
    assert len(r0) >= 40
    s0 = locate_trace(r0, trk, 70.0, (300.0, -200.0), half_width=2000.0, coarse=250.0, fine=50.0).summary(np.zeros(2))
    assert s0["top_error_m"] < 400
    assert len(trace_ridge(spec(y[:, 0]), f_range=(70.0, 115.0))) < 20


def test_hum_detected_and_removed():
    rng = np.random.default_rng(2)
    n = SR * 90
    x = (_hum(n, rng) + 0.3 * rng.normal(size=n))[:, None]
    assert detect_hum(x, SR).status is Status.DETECTED
    assert detect_hum(fix_audio(x, SR, set()), SR).status is Status.NOT_DETECTED
    # keeping hum *and* aircraft skips the scrub, so the hum survives. (Keeping only hum can't promise it:
    # the aircraft scrub resynthesises the background, and faint hum goes with it.)
    assert detect_hum(fix_audio(x, SR, {"hum", "aircraft"}), SR).status is Status.DETECTED


def test_gsm_detected_and_removed():
    rng = np.random.default_rng(5)
    n = SR * 60
    noise = np.cumsum(rng.normal(size=n))  # speech-ish red noise
    noise = 0.3 * (noise - np.convolve(noise, np.ones(400) / 400, "same")) / 20 + 0.05 * rng.normal(size=n)
    x = (noise + _gsm(n))[:, None]
    assert detect_gsm(x, SR).status is Status.DETECTED
    assert detect_gsm(noise[:, None], SR).status is Status.NOT_DETECTED
    assert detect_gsm(fix_audio(x, SR, set()), SR).status is Status.NOT_DETECTED


@pytest.mark.parametrize("seconds", [10, 25])
def test_short_or_silent_audio_not_checked(seconds):
    x = np.random.default_rng(0).normal(size=(SR * seconds, 1))
    for fn in (detect_aircraft, detect_hum, detect_gsm):
        f = fn(x, SR)
        assert f.status is Status.NOT_CHECKED and "shorter" in f.reason
        g = fn(None, SR)
        assert g.status is Status.NOT_CHECKED and g.reason == "no audio track"
