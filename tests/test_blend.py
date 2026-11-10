import subprocess
import time

import numpy as np

from conftest import make_photo, make_video
from ghostprint.blend import COMMON_ENCODER_TAGS, apply_mtime, random_profile
from ghostprint.media import iter_frames, probe, read_photo, write_photo, write_video


def test_profiles_differ_between_runs():
    ps = [random_profile("video", np.random.default_rng(i)) for i in range(3)]
    ct = {p.container_tags["creation_time"] for p in ps}
    assert len(ct) == 3 and len({p.mtime for p in ps}) == 3
    now = time.time()
    assert all(now - 72 * 3600 <= p.mtime <= now for p in ps)


def _tags(p):
    return subprocess.run(["exiftool", "-a", "-G1", "-s", str(p)], capture_output=True, text=True).stdout


def test_output_has_no_ghostprint_signature(tmp_path):
    rng = np.random.default_rng(3)
    v = make_video(tmp_path, seconds=1, gps=True)
    info = probe(v)
    prof = random_profile("video", rng)
    out = tmp_path / "o.mp4"
    write_video(out, iter_frames(info), info.fps, None, prof)
    apply_mtime(out, prof)
    ph = make_photo(tmp_path, rng.integers(0, 255, (64, 64, 3)), exif={0x010F: "Apple", 0x0110: "iPhone 12"})
    pout = tmp_path / "o.jpg"
    write_photo(pout, read_photo(probe(ph)), random_profile("photo", rng))
    for f in (out, pout):
        b = f.read_bytes()
        assert b"ghostprint" not in b
        t = _tags(f)
        assert "GPS" not in t and "45.1" not in t and "iPhone" not in t and "SN12345" not in t
        assert not any(l.split(":")[0].strip().endswith(("Make", "Model", "SerialNumber")) for l in t.splitlines())
    enc = [l for l in _tags(out).splitlines() if l.split(":")[0].strip().endswith("Encoder")]
    assert all(any(e in l for e in COMMON_ENCODER_TAGS) for l in enc)  # absent, or an ordinary one
