import numpy as np

from conftest import make_photo, make_video
from ghostprint.blend import random_profile
from ghostprint.findings import Status
from ghostprint.leaks.metadata import categories, detect, read_tags
from ghostprint.media import iter_frames, probe, read_photo, write_photo, write_video

GPS_EXIF = {0x010F: "Apple", 0x0110: "iPhone 12", 0xA431: "SN12345", 0x0132: "2023:08:12 11:58:04"}


def _photo_with_gps(tmp_path):
    from PIL import Image
    img = Image.fromarray(np.full((40, 60, 3), 128, np.uint8))
    ex = Image.Exif()
    for k, v in GPS_EXIF.items():
        ex[k] = v
    gps = ex.get_ifd(0x8825)
    gps.update({1: "N", 2: (45.0, 6.0, 0.0), 3: "W", 4: (122.0, 12.0, 0.0)})
    p = tmp_path / "gps.jpg"
    img.save(p, exif=ex.tobytes())
    return p


def test_detects_categories_without_values(tmp_path):
    f = detect(_photo_with_gps(tmp_path))
    assert f.status is Status.DETECTED
    s = f.summary.lower()
    assert "location" in s and "time" in s and "phone model" in s
    for leaked in ("45", "122", "SN12345", "iPhone", "2023"):
        assert leaked not in f.summary


def test_video_metadata_detected(tmp_path):
    f = detect(make_video(tmp_path, seconds=1, gps=True))
    assert f.status is Status.DETECTED and "location" in f.summary.lower()


def test_clean_output_has_no_identifying_categories(tmp_path):
    rng = np.random.default_rng(1)
    out = tmp_path / "o.jpg"
    write_photo(out, read_photo(probe(_photo_with_gps(tmp_path))), random_profile("photo", rng))
    assert categories(read_tags(out)) == set()
    v = make_video(tmp_path, seconds=1, gps=True)
    vo = tmp_path / "o.mp4"
    info = probe(v)
    write_video(vo, iter_frames(info), info.fps, None, random_profile("video", rng))
    assert categories(read_tags(vo), allow_container_dates=True) == set()
    assert detect(vo, allow_container_dates=True).status is Status.NOT_DETECTED  # dates there are our random ones
