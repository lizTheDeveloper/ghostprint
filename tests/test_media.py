import numpy as np
import pytest

from conftest import make_audio, make_photo, make_video
from ghostprint.media import ToolMissing, probe, require_tools


def test_probe_video_photo_audio(tmp_path):
    sr = 44100
    v = make_video(tmp_path, seconds=3, audio=(np.zeros((sr * 3, 2)), sr))
    info = probe(v)
    assert info.kind == "video" and (info.width, info.height) == (320, 240)
    assert abs(info.fps - 30) < 0.01 and info.audio_sr == sr and info.audio_channels == 2
    p = probe(make_photo(tmp_path, np.zeros((60, 80, 3))))
    assert p.kind == "photo" and (p.width, p.height) == (80, 60)
    a = probe(make_audio(tmp_path, np.zeros(16000), 16000))
    assert a.kind == "audio" and a.audio_sr == 16000 and a.width is None


def test_probe_reads_rotation(tmp_path):
    info = probe(make_video(tmp_path, seconds=2, rotation=90))
    assert info.rotation == 90


def test_require_tools_names_missing_tool(monkeypatch):
    import shutil
    real = shutil.which
    monkeypatch.setattr(shutil, "which", lambda n: None if n == "exiftool" else real(n))
    with pytest.raises(ToolMissing) as e:
        require_tools()
    assert e.value.name == "exiftool" and ("brew" in e.value.install_hint or "apt" in e.value.install_hint)
