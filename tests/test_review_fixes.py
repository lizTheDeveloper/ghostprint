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
        "from ghostprint.leaks.prnu import remove\n"
        "K = np.random.default_rng(1).normal(0, 0.01, (3000, 4000)).astype(np.float32)\n"
        "img = np.random.default_rng(0).integers(40, 200, (3000, 4000, 3), dtype=np.uint8)\n"
        "for c in range(3): img[..., c] = np.clip(img[..., c] * (1 + K), 0, 255)\n"
        "assert remove(img, K, 0.9) is not None\n"
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
    from ghostprint import pipeline
    v = make_video(tmp_path, seconds=3)
    real = pipeline.iter_frames
    calls = {"n": 0}

    def flaky(info, *a, **k):
        calls["n"] += 1
        for i, chunk in enumerate(real(info, *a, **k)):
            if calls["n"] > 1 and i >= 1:  # the cleaning pass: fail after the first chunk reached ffmpeg
                raise KeyboardInterrupt
            yield chunk

    monkeypatch.setattr(pipeline, "iter_frames", flaky)
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


def test_default_clean_does_not_visibly_alter_a_textured_photo(tmp_path):
    """On real 24 MP iPhone photos the PRNU warp produced a visible rippled-glass pattern on fine texture
    (PSNR 17-22 dB). The default clean must leave pixels essentially as they were (only JPEG loss)."""
    from scipy.ndimage import gaussian_filter
    from PIL import Image
    rng = np.random.default_rng(9)
    tex = gaussian_filter(rng.uniform(0, 255, (600, 800, 3)), (1.2, 1.2, 0))  # fine texture like gravel or fabric
    tex = np.clip((tex - tex.mean()) * 4 + 128, 0, 255).astype(np.uint8)
    p = make_photo(tmp_path, tex, fmt="png", name="tex.png")
    out = tmp_path / "o.png"
    assert main([str(p), "-o", str(out)]) == 0
    b = np.asarray(Image.open(out)).astype(float)
    mse = np.mean((b - tex.astype(float)) ** 2)
    assert 10 * np.log10(255 ** 2 / max(mse, 1e-9)) >= 40


def _fp_case(tmp_path, capsys, *extra):
    from conftest import make_sensor_images, sensor
    from PIL import Image
    refs = tmp_path / "refs"
    refs.mkdir()
    for i, im in enumerate(make_sensor_images(sensor(1), 6, seed=1)):
        Image.fromarray(im).save(refs / f"r{i}.png")
    p = tmp_path / "x.png"
    Image.fromarray(make_sensor_images(sensor(1), 1, seed=90)[0]).save(p)
    out = tmp_path / "o.png"
    assert main([str(p), "-o", str(out), "--fingerprint-from", str(refs), "--json", *extra]) == 0
    return json.loads(capsys.readouterr().out), out


def test_fingerprint_is_removed_with_reference_photos(tmp_path, capsys):
    from conftest import make_sensor_images, sensor
    from ghostprint.leaks.prnu import estimate, match
    from PIL import Image
    r, out = _fp_case(tmp_path, capsys)
    fp = [l for l in r["lines"] if l["leak"] == "fingerprint"][0]
    assert fp["verdict"] == "REDUCED"
    attacker = estimate(make_sensor_images(sensor(1), 32, seed=2))
    assert match(np.asarray(Image.open(out).convert("RGB")), attacker).pce < 60


def test_keep_fingerprint_leaves_it_and_says_so(tmp_path, capsys):
    r, _ = _fp_case(tmp_path, capsys, "--keep", "fingerprint")
    fp = [l for l in r["lines"] if l["leak"] == "fingerprint"][0]
    assert fp["verdict"] == "STILL PRESENT" and "Not clean" in r["headline"]


def test_fingerprint_removal_unverifiable_is_not_claimed(tmp_path, capsys):
    # the held-back re-check photos come from another phone, so they can't recognise this one even
    # before cleaning: a low score afterwards proves nothing
    from conftest import make_sensor_images, sensor
    from PIL import Image
    refs = tmp_path / "refs"
    refs.mkdir()
    mine, other = make_sensor_images(sensor(1), 3, seed=1), make_sensor_images(sensor(2), 3, seed=3)
    for i, im in enumerate(x for pair in zip(mine, other) for x in pair):
        Image.fromarray(im).save(refs / f"r{i}.png")
    p = tmp_path / "x.png"
    Image.fromarray(make_sensor_images(sensor(1), 1, seed=90)[0]).save(p)
    assert main([str(p), "-o", str(tmp_path / "o.png"), "--fingerprint-from", str(refs), "--json"]) == 0
    r = json.loads(capsys.readouterr().out)
    fp = [l for l in r["lines"] if l["leak"] == "fingerprint"][0]
    assert fp["verdict"] == "NOT CHECKED" and "Nothing found" not in r["headline"]
