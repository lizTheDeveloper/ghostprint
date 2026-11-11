import json
import socket
import subprocess
import tempfile

import numpy as np
import pytest

from conftest import make_flicker_frames, make_photo, make_video
from flyover_scrub.physics import make_track, synthesize_flyover
from ghostprint.cli import main

SR = 16000


def _soundtrack(seconds, rng):
    trk = make_track(0.0, 260.0, start_xy=(-9000.0, -400.0), altitude=700.0, speed=70.0, heading_deg=5.0)
    n = SR * seconds
    bg = 0.05 * rng.normal(size=n)
    plane = synthesize_flyover(trk, np.zeros(3), 100.0, n, SR, f0=90.0, rng=rng)
    plane *= 10 ** (-10 / 20) * bg.std() / plane[plane != 0].std()
    t = np.arange(n) / SR
    w = 60 + 0.02 * np.sin(2 * np.pi * t / 37) + 0.01 * np.sin(2 * np.pi * t / 11)
    ph = 2 * np.pi * np.cumsum(w) / SR
    hum = 0.01 * np.sin(2 * ph) + 0.006 * np.sin(4 * ph)
    x = bg + plane + hum
    return np.stack([x, x], axis=1)


@pytest.fixture(scope="module")
def leaky_video(tmp_path_factory):
    d = tmp_path_factory.mktemp("in")
    rng = np.random.default_rng(7)
    fl, _ = make_flicker_frames(seconds=60, size=(160, 120), seed=3, fps=27.0)
    return make_video(d, frames=fl, fps=27, audio=(_soundtrack(60, rng), SR), gps=True, name="leaky.mov")


def _run(args, capsys):
    code = main([str(a) for a in args])
    return code, capsys.readouterr().out


def test_full_video_run(leaky_video, tmp_path, capsys):
    out = tmp_path / "clean.mp4"
    code, txt = _run([leaky_video, "-o", out, "--json"], capsys)
    assert code == 0
    r = json.loads(txt)
    v = {l["leak"]: l["verdict"] for l in r["lines"]}
    assert v["metadata"] == "REMOVED" and v["aircraft"] == "REMOVED" and v["hum"] == "REMOVED" and v["flicker"] == "REMOVED"
    assert v["visible"] == "NOT CHECKED"
    assert subprocess.run(["ffprobe", "-v", "error", str(out)]).returncode == 0


def test_check_mode_writes_nothing(leaky_video, tmp_path, capsys):
    before = set(tmp_path.rglob("*"))
    code, txt = _run([leaky_video, "--check"], capsys)
    assert code == 0 and "FOUND" in txt and set(tmp_path.rglob("*")) == before


def test_no_network(leaky_video, tmp_path, capsys, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    code, _ = _run([leaky_video, "-o", tmp_path / "o.mp4"], capsys)
    assert code == 0


def test_no_trace(leaky_video, tmp_path, capsys, monkeypatch):
    work = tmp_path / "work"
    (work / "tmp").mkdir(parents=True)
    (work / "home").mkdir()
    monkeypatch.setenv("TMPDIR", str(work / "tmp"))
    monkeypatch.setenv("HOME", str(work / "home"))
    def boom(*a, **k):
        raise AssertionError("temp file created")
    for name in ("mkstemp", "mkdtemp", "NamedTemporaryFile", "TemporaryDirectory", "TemporaryFile", "SpooledTemporaryFile"):
        monkeypatch.setattr(tempfile, name, boom)
    out, rep = tmp_path / "o.mp4", tmp_path / "r.html"
    before = set(tmp_path.rglob("*"))
    code, _ = _run([leaky_video, "-o", out, "--report", rep], capsys)
    assert code == 0
    assert set(tmp_path.rglob("*")) - before == {out, rep}


def test_refuses_to_overwrite_input(leaky_video, tmp_path, capsys):
    data = leaky_video.read_bytes()
    code, _ = _run([leaky_video, "-o", leaky_video], capsys)
    assert code == 2 and leaky_video.read_bytes() == data
    existing = tmp_path / "x.mp4"
    existing.write_bytes(b"keep")
    code, _ = _run([leaky_video, "-o", existing], capsys)
    assert code == 2 and existing.read_bytes() == b"keep"


def test_silent_video_report(tmp_path, capsys):
    v = make_video(tmp_path, seconds=12)
    code, txt = _run([v, "-o", tmp_path / "o.mp4", "--json"], capsys)
    r = json.loads(txt)
    lines = {l["leak"]: l for l in r["lines"]}
    for k in ("aircraft", "hum", "gsm"):
        assert lines[k]["verdict"] == "NOT CHECKED" and "no audio track" in lines[k]["text"]
    assert "Not clean" in r["headline"]


def test_outputs_do_not_share_a_signature(tmp_path, capsys):
    rng = np.random.default_rng(0)
    outs = []
    for i in range(6):
        p = make_photo(tmp_path, rng.integers(0, 255, (64, 80, 3)), name=f"in{i}.jpg",
                       exif={0x010F: "Apple", 0x0110: "iPhone 12"})
        o = tmp_path / f"out{i}.jpg"
        assert main([str(p), "-o", str(o)]) == 0
        outs.append(o)
    capsys.readouterr()
    assert len({o.stat().st_mtime for o in outs}) == 6
    q = [subprocess.run(["exiftool", "-s3", "-JFIF:all", str(o)], capture_output=True, text=True).stdout for o in outs]
    assert all(b"ghostprint" not in o.read_bytes() for o in outs)
    from PIL import Image
    tables = {repr(Image.open(o).quantization) for o in outs}
    assert len(tables) >= 2  # random quality (11 settings), not one fixed one; all six equal by chance: 1 in 160,000


def test_report_never_contains_leaked_values(leaky_video, tmp_path, capsys):
    rep = tmp_path / "r.html"
    code, txt = _run([leaky_video, "-o", tmp_path / "o.mp4", "--report", rep], capsys)
    _, js = _run([leaky_video, "--check", "--json"], capsys)
    for blob in (txt, js, rep.read_text()):
        assert "45.1" not in blob and "122.2" not in blob and "SN12345" not in blob and "iPhone" not in blob
