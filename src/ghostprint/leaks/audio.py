"""Audio leaks: aircraft Doppler lines, grid hum (ENF) and phone-radio (GSM/TDMA) buzz.

Detection and the scrub come from flyover-scrub; the GSM comb is detected and notched here.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import filtfilt, iirnotch, stft

from flyover_scrub.enf import detect_enf
from flyover_scrub.scrub import ScrubConfig, detect_ridges, scrub

from ..findings import Finding, Status

GSM_FRAME_HZ = 1 / 4.615e-3  # 216.68 Hz TDMA frame rate
MIN_S = {"aircraft": 40.0, "hum": 30.0, "gsm": 30.0}


def _mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def _gate(leak: str, audio, sr) -> Finding | None:
    if audio is None:
        return Finding(leak, Status.NOT_CHECKED, "", reason="no audio track")
    if len(audio) / sr < MIN_S[leak]:
        return Finding(leak, Status.NOT_CHECKED, "", reason=f"clip shorter than {MIN_S[leak]:.0f} s")
    return None


def detect_aircraft(audio: np.ndarray | None, sr: int) -> Finding:
    if g := _gate("aircraft", audio, sr):
        return g
    ridges = detect_ridges(_mono(audio), sr)
    if not ridges:
        return Finding("aircraft", Status.NOT_DETECTED, "no long gliding tones")
    return Finding("aircraft", Status.DETECTED,
                   f"{len(ridges)} long gliding tone(s), like a passing aircraft: could locate it from flight records",
                   strength=ridges[0].strength_db)


def detect_hum(audio: np.ndarray | None, sr: int) -> Finding:
    if g := _gate("hum", audio, sr):
        return g
    r = detect_enf(_mono(audio), sr)
    if not r.present:
        return Finding("hum", Status.NOT_DETECTED, "no grid hum")
    return Finding("hum", Status.DETECTED,
                   f"grid hum ({r.nominal_hz} Hz power grid): could date the recording and tell which grid it was made on",
                   strength=max(r.line_db.values()))


def _gsm_line_db(x: np.ndarray, sr: int) -> list[float]:
    n = int(sr * 4)
    f, _, Z = stft(x, sr, nperseg=n, noverlap=n // 2)
    P = np.median(np.abs(Z) ** 2, axis=1)
    out = []
    for k in range(1, 9):
        fc = GSM_FRAME_HZ * k
        if fc > sr / 2 - 10:
            break
        on = np.abs(f - fc) <= 0.5
        side = (np.abs(f - fc) >= 2) & (np.abs(f - fc) <= 5)
        out.append(float(10 * np.log10(P[on].max() / np.median(P[side]))))
    return out


def detect_gsm(audio: np.ndarray | None, sr: int) -> Finding:
    if g := _gate("gsm", audio, sr):
        return g
    db = _gsm_line_db(_mono(audio), sr)
    hits = sum(d >= 6 for d in db)
    if hits < 4:
        return Finding("gsm", Status.NOT_DETECTED, "no phone-radio buzz")
    return Finding("gsm", Status.DETECTED,
                   "phone-radio buzz (a phone transmitting on a 2G-style network nearby)", strength=float(max(db)))


def _notch_gsm(x: np.ndarray, sr: int) -> np.ndarray:
    y = x
    k = 1
    while GSM_FRAME_HZ * k < min(sr / 2 - 50, 8000):
        b, a = iirnotch(GSM_FRAME_HZ * k, Q=GSM_FRAME_HZ * k / 2.0, fs=sr)  # ~2 Hz wide
        y = filtfilt(b, a, y)
        k += 1
    return y


def fix_audio(audio: np.ndarray, sr: int, keep: set[str]) -> np.ndarray:
    """One scrub pass (aircraft and, unless kept, hum), then the GSM comb notch (unless kept). Per channel."""
    x = np.asarray(audio, dtype=np.float64)
    two_d = x.ndim == 2
    if not two_d:
        x = x[:, None]
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        y = x[:, c]
        if not ("aircraft" in keep and "hum" in keep):
            y, _ = scrub(y, sr, ScrubConfig(keep_stationary_tones="hum" in keep))
        if "gsm" not in keep:
            y = _notch_gsm(y, sr)
        out[:, c] = y
    out = out.astype(np.float32)
    return out if two_d else out[:, 0]
