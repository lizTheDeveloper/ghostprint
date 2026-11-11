import json
import re

from ghostprint.findings import LEAKS, Finding, Status
from ghostprint.report import CLEAN_HEADLINE, build_report, render_html, render_json, render_text

D, N, NC, W = Status.DETECTED, Status.NOT_DETECTED, Status.NOT_CHECKED, Status.WARNING


def F(leak, st, summary="s", reason=None):
    return Finding(leak, st, summary, reason=reason)


def verdict(before, after, fixed=frozenset(), leak="hum"):
    r = build_report("x", "video", [F(leak, before)], None if after is None else [F(leak, after)], set(fixed), None)
    return r.lines[0].verdict


def test_verdict_rules():
    assert verdict(D, N, {"hum"}) == "REMOVED"
    assert verdict(D, N, {"fingerprint"}, leak="fingerprint") == "REDUCED"
    assert verdict(N, N, {"hum"}) == "SCRUBBED (precaution)"
    assert verdict(D, D, {"hum"}) == "STILL PRESENT"
    assert verdict(NC, NC) == "NOT CHECKED"
    assert verdict(W, W) == "WARNING"
    assert verdict(N, N) == "NOT FOUND"
    assert verdict(D, None) == "FOUND"


def test_lines_follow_leak_order():
    r = build_report("x", "video", [F("visible", NC), F("metadata", D)], [F("visible", NC), F("metadata", N)], {"metadata"}, "o.mp4")
    assert [l.leak for l in r.lines] == ["metadata", "visible"]
    assert LEAKS.index("metadata") < LEAKS.index("visible")


def test_headline_never_clean_with_warning_or_not_checked():
    ok = build_report("x", "video", [F("hum", D)], [F("hum", N)], {"hum"}, "o")
    assert ok.headline == CLEAN_HEADLINE
    for st in (W, NC):
        r = build_report("x", "video", [F("hum", D), F("sun", st)], [F("hum", N), F("sun", st)], {"hum"}, "o")
        assert r.headline != CLEAN_HEADLINE
    r = build_report("x", "video", [F("hum", D)], [F("hum", D)], {"hum"}, "o")
    assert r.headline != CLEAN_HEADLINE and "still" in r.headline.lower()


def test_short_or_silent_is_not_checked():
    r = build_report("x", "video", [F("hum", NC, reason="clip shorter than 30 s")], [F("hum", NC, reason="clip shorter than 30 s")], {"hum"}, "o")
    txt = render_text(r)
    assert "NOT CHECKED" in txt and "clip shorter than 30 s" in txt and r.headline != CLEAN_HEADLINE


def test_renderers():
    r = build_report("clip.mov", "video", [F("metadata", D, "GPS location present")], [F("metadata", N)], {"metadata"}, "o.mp4")
    assert "REMOVED" in render_text(r) and "GPS location present" in render_text(r)
    j = json.loads(render_json(r))
    assert j["lines"][0]["verdict"] == "REMOVED" and j["headline"] == r.headline
    h = render_html(r)
    assert h.lstrip().lower().startswith("<!doctype html>") and "REMOVED" in h
    assert not re.search(r"<(script|link|img)[^>]+(src|href)=", h)  # self-contained, nothing fetched


def test_html_not_checked_is_not_styled_as_ok():
    r = build_report("x", "video", [F("visible", NC, reason="r"), F("hum", N)], [F("visible", NC, reason="r"), F("hum", N)], set(), "o")
    h = render_html(r)
    assert 'class="v-not-checked"' in h and 'class="v-not-found"' in h
    assert ".v-not-checked td:first-child{color:var(--warn)}" in h


def test_detected_then_gone_is_not_claimed_unless_fixed():
    # e.g. JPEG re-encoding nudges a camera-fingerprint match under threshold: ghostprint didn't treat it
    r = build_report("x", "photo", [F("fingerprint", D)], [F("fingerprint", N)], set(), "o")
    assert r.lines[0].verdict == "NOT REMOVED" and r.headline != CLEAN_HEADLINE
    assert verdict(D, N, set()) == "NOT REMOVED"
