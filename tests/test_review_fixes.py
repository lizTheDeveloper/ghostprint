"""Regression tests for the final review's Important findings."""
import json
import subprocess

import numpy as np
import pytest

from conftest import make_photo, make_video
from ghostprint.cli import main


def _ffprobe(p):
    return json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(p)],
                                     capture_output=True, text=True, check=True).stdout)["streams"]


def test_video_output_has_no_encoder_signature(tmp_path):
    v = make_video(tmp_path, seconds=2)
    out = tmp_path / "o.mp4"
    assert main([str(v), "-o", str(out)]) == 0
    b = out.read_bytes()
    for sig in (b"x264", b"Lavc", b"Lavf", b"libx264"):
        assert sig not in b, sig


def test_ntsc_frame_rate_is_kept_exactly(tmp_path):
    src = tmp_path / "ntsc.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30000/1001:duration=2",
                    "-c:v", "libx264", str(src)], check=True)
    out = tmp_path / "o.mp4"
    assert main([str(src), "-o", str(out)]) == 0
    v = [s for s in _ffprobe(out) if s["codec_type"] == "video"][0]
    assert v["r_frame_rate"] == "30000/1001"


def test_photo_fix_memory_is_bounded(tmp_path):
    code = (
        "import numpy as np, resource\n"
        "from ghostprint.leaks.prnu import fix\n"
        "img = np.random.default_rng(0).integers(0, 255, (3000, 4000, 3), dtype=np.uint8)\n"
        "fix(img, np.random.default_rng(1))\n"
        "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)\n"
    )
    out = subprocess.run(["uv", "run", "-q", "python", "-c", code], capture_output=True, text=True, check=True).stdout
    rss = int(out.strip().splitlines()[-1])
    rss_mb = rss / (1024 * 1024) if rss > 10_000_000 else rss / 1024  # macOS bytes, Linux KB
    assert rss_mb < 900, rss_mb  # 12 MP photo; was ~2.3 GB


def test_huge_image_is_refused_cleanly(tmp_path, monkeypatch, capsys):
    from ghostprint import media
    monkeypatch.setattr(media, "MAX_PIXELS", 1000)
    p = make_photo(tmp_path, np.zeros((60, 80, 3)))
    assert main([str(p), "-o", str(tmp_path / "o.jpg")]) == 2
    assert "too large" in capsys.readouterr().err


def test_failure_leaves_no_partial_output(tmp_path, monkeypatch):
    from ghostprint.leaks import prnu
    v = make_video(tmp_path, seconds=2)
    calls = {"n": 0}
    real = prnu.fix
    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] > 45:  # after the first chunk has reached ffmpeg
            raise KeyboardInterrupt
        return real(*a, **k)
    monkeypatch.setattr(prnu, "fix", flaky)
    out = tmp_path / "o.mp4"
    with pytest.raises(KeyboardInterrupt):
        main([str(v), "-o", str(out)])
    assert not out.exists()


def test_report_path_equal_to_output_is_refused(tmp_path):
    v = make_video(tmp_path, seconds=1)
    out = tmp_path / "o.mp4"
    assert main([str(v), "-o", str(out), "--report", str(out)]) == 2
    assert not out.exists()


def test_disguised_postscript_photo_is_refused(tmp_path, capsys):
    p = tmp_path / "evil.jpg"
    p.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\nshowpage\n")
    assert main([str(p), "-o", str(tmp_path / "o.jpg")]) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_output_name_with_colon_is_a_local_file(tmp_path, monkeypatch):
    v = make_video(tmp_path, seconds=1)
    monkeypatch.chdir(tmp_path)
    assert main([str(v), "-o", "udp:clean.mp4"]) == 0
    assert (tmp_path / "udp:clean.mp4").exists()


@pytest.mark.parametrize("name", ["o.webp", "o.mp4", "o"])
def test_unsupported_photo_output_extension_is_refused(tmp_path, name):
    p = make_photo(tmp_path, np.zeros((60, 80, 3)))
    assert main([str(p), "-o", str(tmp_path / name)]) == 2
    assert not (tmp_path / name).exists()
