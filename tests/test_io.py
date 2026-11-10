import json
import subprocess
import tempfile

import numpy as np
import pytest

from conftest import make_audio, make_video
from ghostprint.blend import random_profile
from ghostprint.media import iter_frames, probe, read_audio, write_audio, write_video


def _streams(p):
    return json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(p)],
                                     capture_output=True, text=True, check=True).stdout)["streams"]


def test_audio_roundtrip_via_pipes(tmp_path, rng):
    sr = 44100
    t = np.arange(sr * 3) / sr
    tone = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 660 * t) + 0.01 * rng.normal(size=t.size)
    x = np.stack([tone, tone[::-1]], axis=1).astype(np.float32)
    a, s = read_audio(probe(make_audio(tmp_path, x, sr)))
    assert s == sr and a.shape == x.shape and a.dtype == np.float32
    out = tmp_path / "o.m4a"
    write_audio(out, a, s, random_profile("audio", rng))
    b, s2 = read_audio(probe(out))
    n = min(len(a), len(b)) - 4096
    lag = int(np.argmax(np.correlate(b[:20000, 0], a[:20000, 0], "full")) - 19999)
    bb = b[max(lag, 0):][:n, 0]
    aa = a[max(-lag, 0):][:n, 0]
    assert np.corrcoef(aa[: len(bb)], bb[: len(aa)])[0, 1] > 0.99


def _roundtrip(tmp_path, v, rng, audio=None):
    info = probe(v)
    out = tmp_path / "out.mp4"
    write_video(out, (c for c in iter_frames(info)), info.fps, audio, random_profile("video", rng))
    return out


def test_rotated_video_comes_out_upright(tmp_path, rng):
    fr = np.zeros((30, 240, 320, 3), np.uint8)
    fr[:, :20, :, 0] = 255  # red band at the top of the stored frame
    v = make_video(tmp_path, frames=fr, rotation=90)
    info = probe(v)
    first = next(iter_frames(info))[0]
    assert first.shape[:2] == (320, 240)  # displayed portrait
    out = _roundtrip(tmp_path, v, rng)
    s = [x for x in _streams(out) if x["codec_type"] == "video"][0]
    assert (s["width"], s["height"]) == (240, 320)
    assert not any("rotation" in d for d in s.get("side_data_list", []))
    up = next(iter_frames(probe(out)))[0]
    assert np.allclose(up, first, atol=40)  # same upright picture (red band wherever display put it)


def test_hdr_input_becomes_sdr_8bit(tmp_path, rng):
    out = _roundtrip(tmp_path, make_video(tmp_path, seconds=1, hdr=True), rng)
    s = [x for x in _streams(out) if x["codec_type"] == "video"][0]
    assert s["pix_fmt"] == "yuv420p" and s.get("color_transfer", "bt709") == "bt709"


def test_odd_dimensions_encode(tmp_path, rng):
    out = _roundtrip(tmp_path, make_video(tmp_path, seconds=1, size=(321, 241)), rng)
    s = [x for x in _streams(out) if x["codec_type"] == "video"][0]
    assert s["width"] % 2 == 0 and s["height"] % 2 == 0


def test_no_temp_files_during_io(tmp_path, rng, monkeypatch):
    sr = 44100
    v = make_video(tmp_path, seconds=2, audio=(0.1 * rng.normal(size=(sr * 2, 2)), sr))
    td = tmp_path / "tmpdir"
    td.mkdir()
    monkeypatch.setenv("TMPDIR", str(td))
    def boom(*a, **k):
        raise AssertionError("temp file created")
    for name in ("mkstemp", "mkdtemp", "NamedTemporaryFile", "TemporaryDirectory", "TemporaryFile", "SpooledTemporaryFile"):
        monkeypatch.setattr(tempfile, name, boom)
    info = probe(v)
    audio = read_audio(info)
    out = tmp_path / "o.mp4"
    write_video(out, iter_frames(info), info.fps, audio, random_profile("video", rng))
    assert out.exists() and list(td.iterdir()) == []
    assert any(x["codec_type"] == "audio" for x in _streams(out))
