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
