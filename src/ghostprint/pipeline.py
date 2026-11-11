"""detect → fix → write → re-detect, for one file."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .blend import apply_mtime, random_profile
from .findings import Finding, Status
from .leaks import audio as A
from .leaks import flicker as FL
from .leaks import metadata as M
from .leaks import prnu as P
from .leaks import scene as S
from .media import MediaInfo, frame_size, iter_frames, probe, read_audio, read_photo, require_tools, write_audio, write_photo, write_video
from .report import Report, build_report

KEEPABLE = ("aircraft", "hum", "gsm", "flicker", "fingerprint")
OUTPUT_EXT = {"photo": (".jpg", ".jpeg", ".png"), "video": (".mp4", ".mov", ".m4v"), "audio": (".m4a", ".wav")}
SUN_EVERY_S = 5.0


class UsageError(ValueError):
    pass


def _load_refs(refs: Path) -> list[np.ndarray]:
    imgs = []
    for p in sorted(Path(refs).iterdir()):
        try:
            info = probe(p)
        except Exception:
            continue
        if info.kind == "photo":
            imgs.append(read_photo(info))
        elif info.kind == "video":
            imgs.append(next(iter_frames(info, chunk=1))[0])
    if not imgs:
        raise UsageError(f"no photos or videos found in {refs}")
    shape = max({im.shape for im in imgs}, key=[im.shape for im in imgs].count)
    return [im for im in imgs if im.shape == shape]


class _Analysis:
    """One pass over a video: row profiles for flicker and a frame every few seconds for the sun check."""

    def __init__(self, info: MediaInfo):
        self.info = info
        self.samples: list[np.ndarray] = []

    def frames(self):
        every = max(1, int(round((self.info.fps or 30) * SUN_EVERY_S)))
        n = 0
        self.samples = []
        for chunk in iter_frames(self.info):
            for i in range(len(chunk)):
                if (n + i) % every == 0:
                    self.samples.append(chunk[i].copy())
            n += len(chunk)
            yield chunk


def _detect(info: MediaInfo, K: np.ndarray | None, *, is_output: bool) -> tuple[list[Finding], dict]:
    f = [M.detect(info.path, allow_container_dates=is_output)]
    ctx: dict = {}
    if info.kind in ("video", "audio"):
        a = read_audio(info)
        ctx["audio"] = a
        x, sr = (a if a is not None else (None, info.audio_sr or 1))
        f += [A.detect_aircraft(x, sr), A.detect_hum(x, sr), A.detect_gsm(x, sr)]
    if info.kind == "video":
        an = _Analysis(info)
        est = FL.measure(an.frames, info.fps or 30.0, frame_size(info)[1])
        ctx["flicker"] = est
        f.append(FL.detect(est, info.duration_s or 0.0))
        probe_img = an.samples[0] if an.samples else None
        sun = [S.detect_sun(s) for s in an.samples]
        f.append(next((x for x in sun if x.status is Status.WARNING), Finding("sun", Status.NOT_DETECTED, "no sky with hard shadows")))
    elif info.kind == "photo":
        img = read_photo(info)
        ctx["img"] = img
        probe_img = img
        f += [FL.detect_photo(img), S.detect_sun(img)]
    else:
        probe_img = None
    if info.kind != "audio":
        f.append(P.detect(P.match(probe_img, K) if (K is not None and probe_img is not None) else None))
        f.append(S.visible())
    return f, ctx


def run(inp: Path, out: Path | None, *, keep: set[str] = frozenset(), refs: Path | None = None,
        seed: int | None = None) -> Report:
    require_tools()
    inp = Path(inp)
    if not inp.is_file():
        raise UsageError(f"no such file: {inp}")
    bad = set(keep) - set(KEEPABLE)
    if bad:
        raise UsageError(f"--keep takes {', '.join(KEEPABLE)} (not {', '.join(sorted(bad))})")
    if out is not None:
        out = Path(out)
        if out.resolve() == inp.resolve():
            raise UsageError("the output would overwrite the original; choose another name")
        if out.exists():
            raise UsageError(f"{out} already exists; choose another name")
    info = probe(inp)
    if out is not None and out.suffix.lower() not in OUTPUT_EXT[info.kind]:
        raise UsageError(f"write a {info.kind} as {' or '.join(OUTPUT_EXT[info.kind])}")
    K = P.estimate(_load_refs(refs)) if refs else None
    before, ctx = _detect(info, K, is_output=False)
    if out is None:
        return build_report(inp.name, info.kind, before, None, set(), None)

    try:
        return _clean(inp, out, info, ctx, before, K, keep, seed)
    except BaseException:
        # never leave a half-written file at the user's chosen name: it would be easy to post by mistake
        if out.exists():
            out.unlink()
        raise


def progress(msg: str) -> None:
    import sys

    if sys.stderr.isatty():
        sys.stderr.write("\r" + msg.ljust(40) + ("\n" if not msg else ""))
        sys.stderr.flush()


def _clean(inp: Path, out: Path, info: MediaInfo, ctx: dict, before: list[Finding], K, keep, seed) -> Report:
    rng = np.random.default_rng(seed)
    profile = random_profile(info.kind, rng)
    fixed = {"metadata"}
    audio = ctx.get("audio")
    if audio is not None and not {"aircraft", "hum", "gsm"} <= set(keep):
        audio = (A.fix_audio(audio[0], audio[1], set(keep)), audio[1])
        fixed |= {"aircraft", "hum", "gsm"} - set(keep)
    if info.kind == "video":
        est = ctx["flicker"]
        fixer = FL.FlickerFixer(est) if ("flicker" not in keep and est.nominal_hz is not None) else None
        warp = None
        if "fingerprint" not in keep:
            fw, fh = frame_size(info)
            warp = P.make_warp((fh, fw), rng)
            fixed.add("fingerprint")
        if fixer:
            fixed.add("flicker")

        total = max(1, int(round((info.duration_s or 0) * (info.fps or 30))))

        def frames():
            n = 0
            for chunk in iter_frames(info):
                c = fixer.apply(chunk, n) if fixer else chunk
                if warp is not None:
                    c = np.stack([P.fix(fr, rng, warp, order=1) for fr in c])
                n += len(chunk)
                progress(f"cleaning video: {min(100, 100 * n // total)}%")
                yield c

        write_video(out, frames(), info.fps_frac or info.fps or 30.0, audio, profile)
        progress("")
    elif info.kind == "photo":
        img = ctx["img"]
        if "fingerprint" not in keep:
            img = P.fix(img, rng)
            fixed.add("fingerprint")
        write_photo(out, img, profile)
    else:
        write_audio(out, audio[0], audio[1], profile)
    apply_mtime(out, profile)
    after, _ = _detect(probe(out), K, is_output=True)
    return build_report(inp.name, info.kind, before, after, fixed, out.name)
