"""A file that only pretends to be media (a playlist or concat list) must not make ffmpeg
fetch URLs or read other files: that would break the no-network, no-trace promises."""
import pytest

from ghostprint.media import UnsafeInput, probe


@pytest.mark.parametrize("name,body", [
    ("evil.mp4", "#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10,\nhttp://127.0.0.1:9/seg.ts\n#EXT-X-ENDLIST\n"),
    ("evil.mov", "ffconcat version 1.0\nfile '/etc/hosts'\n"),
    ("evil.m3u8", "#EXTM3U\n#EXTINF:10,\nhttp://127.0.0.1:9/a.ts\n"),
])
def test_playlists_and_concat_lists_are_refused(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    with pytest.raises(UnsafeInput):
        probe(p)


def test_every_ffmpeg_call_on_user_input_whitelists_demuxers(tmp_path, monkeypatch):
    import subprocess

    import numpy as np

    from conftest import make_video
    from ghostprint import media

    sr = 44100
    v = make_video(tmp_path, seconds=1, audio=(np.zeros((sr, 2)), sr))
    calls = []
    real_run, real_popen = subprocess.run, subprocess.Popen

    def rec(fn):
        def inner(cmd, *a, **k):
            if cmd and cmd[0] in ("ffmpeg", "ffprobe") and any(str(x).startswith("file:") for x in cmd):
                calls.append(list(cmd))
            return fn(cmd, *a, **k)
        return inner

    monkeypatch.setattr(subprocess, "run", rec(real_run))
    monkeypatch.setattr(subprocess, "Popen", rec(real_popen))
    info = media.probe(v)
    media.read_audio(info)
    list(media.iter_frames(info))
    assert {c[0] for c in calls} == {"ffmpeg", "ffprobe"}  # run() calls Popen internally, so calls repeat
    for c in calls:
        i = c.index("-format_whitelist")
        assert set(c[i + 1].split(",")) <= media.ALLOWED_DEMUXERS and c.index("-i") > i


def test_hls_is_refused_even_without_the_header_check(tmp_path, monkeypatch):
    from ghostprint import media

    monkeypatch.setattr(media, "_TEXT_LISTS", ())
    p = tmp_path / "x.mp4"
    p.write_text("#EXTM3U\n#EXTINF:1,\nother.ts\n")
    with pytest.raises(UnsafeInput):
        probe(p)
