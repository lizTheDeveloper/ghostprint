"""Grid flicker in video: mains lighting pulses at twice the grid frequency (100 or 120 Hz).

A phone's rolling shutter exposes rows one after another, so the pulse shows up as faint
horizontal bands that drift from frame to frame. Like grid hum, the bands can date the video
and say which grid it was shot on, even with the sound removed (Garg et al. 2013, Su et al. 2014).

Model: row y of frame n is exposed at n/fps + (y/H)·readout/fps, readout being the fraction of the
frame time the sensor takes to read out (searched over 0.4–1.0; phones read out in ~10–30 ms).
A flicker at f gives a row-profile ripple with spatial frequency c = f·readout/(fps·H) cycles per
row, whose phase advances by 2π·f/fps per frame. Detection is a matched filter in both: project
each frame's scene-removed row profile onto exp(-i2πcy), de-rotate by the expected per-frame
phase step, and sum coherently over ~1 s blocks. The two grids (50/60 Hz) are told apart by the
per-frame phase step, which the spatial frequency alone can't do.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..findings import Finding, Status

READOUTS = np.linspace(0.4, 1.0, 61)
MIN_S = 10.0
SNR_DB = 8.0
MIN_DEPTH = 0.001  # 0.1 % of brightness; real mains flicker on phone video is ~1–10 %


def _luma(frames: np.ndarray) -> np.ndarray:
    f = frames.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


@dataclass
class FlickerEstimate:
    nominal_hz: int | None
    snr_db: float
    row_rate_hz: float
    cycles_per_row: float = 0.0
    mean_profile: np.ndarray | None = None
    fps: float = 0.0


def _profiles(frames: Callable[[], Iterator[np.ndarray]]) -> np.ndarray:
    return np.concatenate([_luma(c).mean(axis=2) for c in frames()], axis=0)  # (n, H)


def _resid(P: np.ndarray, mean_profile: np.ndarray) -> np.ndarray:
    d = P / np.maximum(mean_profile, 1e-3)[None, :] - 1.0
    return d - d.mean(axis=1, keepdims=True)


def _coherent_power(d: np.ndarray, c: float, step: float, block: int) -> float:
    y = np.arange(d.shape[1])
    z = d @ np.exp(-2j * np.pi * c * y) / d.shape[1]
    z = z * np.exp(-1j * step * np.arange(len(z)))
    nb = len(z) // block
    if nb == 0:
        return 0.0
    return float(np.mean(np.abs(z[: nb * block].reshape(nb, block).mean(axis=1)) ** 2))


def measure(frames: Callable[[], Iterator[np.ndarray]], fps: float, height: int) -> FlickerEstimate:
    P = _profiles(frames)
    mean_profile = P.mean(axis=0)
    d = _resid(P, mean_profile)
    block = max(4, int(round(fps)))
    best = FlickerEstimate(None, -np.inf, fps * height, 0.0, mean_profile, fps)
    for nominal in (50, 60):
        f = 2.0 * nominal
        step = 2 * np.pi * f / fps
        cs = f * READOUTS / (fps * height)
        sig = np.array([_coherent_power(d, c, step, block) for c in cs])
        # chance level: the same sums with a per-frame phase step no flicker has (shifted by a quarter
        # and half turn), so a real ripple averages out there instead of leaking into the floor
        noise = np.median([_coherent_power(d, c, step + off, block) for c in cs[::6] for off in (np.pi / 2, np.pi)]) + 1e-20
        i = int(np.argmax(sig))
        snr = 10 * np.log10(sig[i] / noise)
        if np.sqrt(sig[i]) < MIN_DEPTH / 2:  # coherent amplitude of a ripple of depth D is ~D/2
            snr = min(snr, SNR_DB - 1e-3)
        if snr > best.snr_db:
            best = FlickerEstimate(nominal, float(snr), fps * height / READOUTS[i], float(cs[i]), mean_profile, fps)
    if best.snr_db < SNR_DB:
        best.nominal_hz = None
    return best


def detect(est: FlickerEstimate, duration_s: float) -> Finding:
    if duration_s < MIN_S:
        return Finding("flicker", Status.NOT_CHECKED, "", reason=f"clip shorter than {MIN_S:.0f} s")
    if est.nominal_hz is None:
        return Finding("flicker", Status.NOT_DETECTED, "no mains-light flicker")
    return Finding("flicker", Status.DETECTED,
                   f"flicker from mains lighting ({est.nominal_hz} Hz grid): could date the video even with the sound removed",
                   strength=est.snr_db)


class FlickerFixer:
    """Divides each row by the flicker gain estimated for it. Streams chunk by chunk."""

    def __init__(self, est: FlickerEstimate):
        self.est = est

    def apply(self, chunk: np.ndarray, first_frame: int) -> np.ndarray:
        e = self.est
        if e.nominal_hz is None:
            return chunk
        h = chunk.shape[1]
        d = _resid(_luma(chunk).mean(axis=2), e.mean_profile[:h])
        y = np.arange(h)
        basis = np.exp(2j * np.pi * e.cycles_per_row * y)
        step = 2 * np.pi * 2 * e.nominal_hz / e.fps
        n = first_frame + np.arange(len(chunk))
        # least-squares fit of a·cos + b·sin per frame (a plain projection leaks when the bands
        # don't fit a whole number of cycles in the frame); ripple = Re((a - ib)·basis)
        X = np.stack([basis.real, basis.imag, np.ones(h)], axis=1)  # constant: per-frame exposure, left alone
        coef, *_ = np.linalg.lstsq(X, d.T, rcond=None)
        z = coef[0] - 1j * coef[1]
        # the ripple's phase rotates by `step` per frame (plus a slow drift); smooth it in the rotating frame
        zr = gaussian_filter1d((z * np.exp(-1j * step * n)).real, 2, mode="nearest") \
            + 1j * gaussian_filter1d((z * np.exp(-1j * step * n)).imag, 2, mode="nearest")
        z = zr * np.exp(1j * step * n)
        gain = 1.0 + (z[:, None] * basis[None, :]).real  # (k, h)
        out = chunk.astype(np.float32) / gain[:, :, None, None]
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def detect_photo(img: np.ndarray) -> Finding:
    """Banding at a plausible flicker period in a single photo. Warn-only: blinds and stripes look the same."""
    p = _luma(img[None])[0].mean(axis=1)
    h = len(p)
    r = p / np.maximum(gaussian_filter1d(p, max(2, h / 20)), 1e-3) - 1
    spec = np.abs(np.fft.rfft(r * np.hanning(h))) ** 2
    freqs = np.fft.rfftfreq(h)  # cycles per row
    lo, hi = 100 * 0.010 / h, 120 * 0.040 / h  # readout 10–40 ms over the frame
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        return Finding("flicker", Status.NOT_DETECTED, "no banding")
    ref = np.median(spec[(freqs > 0) & (freqs <= hi * 3)]) + 1e-20
    if 10 * np.log10(spec[band].max() / ref) >= 12.0:
        return Finding("flicker", Status.WARNING,
                       "horizontal banding that may be flicker from mains lighting: could hint at the power grid")
    return Finding("flicker", Status.NOT_DETECTED, "no banding")
