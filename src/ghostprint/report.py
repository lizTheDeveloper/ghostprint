"""The report: only claims what was measured, and never says "clean" when it isn't."""
from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass

from .findings import LABELS, LEAKS, Finding, Status

CLEAN_HEADLINE = "Nothing found that this tool checks for. Look at the picture before you post."
OK_VERDICTS = {"NOT FOUND", "REMOVED", "REDUCED", "SCRUBBED (precaution)"}


@dataclass
class ReportLine:
    leak: str
    verdict: str
    text: str


@dataclass
class Report:
    name: str
    kind: str
    lines: list[ReportLine]
    headline: str
    output: str | None


def _verdict(b: Finding, a: Finding | None, fixed: set[str]) -> str:
    if b.status is Status.NOT_CHECKED:
        return "NOT CHECKED"
    if b.status is Status.WARNING:
        return "WARNING"
    if a is None:
        return "FOUND" if b.status is Status.DETECTED else "NOT FOUND"
    if a.status is Status.NOT_CHECKED:
        return "NOT CHECKED"
    if b.status is Status.DETECTED:
        if a.status is Status.DETECTED:
            return "STILL PRESENT"
        return "REDUCED" if b.leak == "fingerprint" else "REMOVED"
    if a.status is Status.DETECTED:
        return "STILL PRESENT"
    return "SCRUBBED (precaution)" if b.leak in fixed else "NOT FOUND"


def _text(b: Finding, a: Finding | None, verdict: str) -> str:
    base = b.summary or LABELS.get(b.leak, b.leak)
    if verdict == "SCRUBBED (precaution)":
        base = LABELS.get(b.leak, b.leak) + " (none detected; removed anyway)"
    reason = (a.reason if a is not None and a.reason else None) or b.reason
    return f"{base} ({reason})" if verdict == "NOT CHECKED" and reason else base


def build_report(name: str, kind: str, before: list[Finding], after: list[Finding] | None, fixed: set[str],
                 output: str | None) -> Report:
    aft = {f.leak: f for f in after or []}
    order = {k: i for i, k in enumerate(LEAKS)}
    lines = []
    for b in sorted(before, key=lambda f: order.get(f.leak, len(LEAKS))):
        a = aft.get(b.leak) if after is not None else None
        v = _verdict(b, a, fixed)
        lines.append(ReportLine(b.leak, v, _text(b, a, v)))
    bad = [l for l in lines if l.verdict not in OK_VERDICTS]
    if not bad:
        headline = CLEAN_HEADLINE
    else:
        n = {v: sum(l.verdict == v for l in bad) for v in ("STILL PRESENT", "FOUND", "WARNING", "NOT CHECKED")}
        parts = []
        if n["STILL PRESENT"]:
            parts.append(f"{n['STILL PRESENT']} leak(s) still present after cleaning")
        if n["FOUND"]:
            parts.append(f"{n['FOUND']} leak(s) found")
        if n["WARNING"]:
            parts.append(f"{n['WARNING']} warning(s) this tool can't fix")
        if n["NOT CHECKED"]:
            parts.append(f"{n['NOT CHECKED']} thing(s) not checked")
        headline = "Not clean: " + ", ".join(parts) + ". Read the lines below before you post."
    return Report(name, kind, lines, headline, output)


def render_text(r: Report) -> str:
    w = max([len(l.verdict) for l in r.lines] + [8])
    out = [f"ghostprint report: {r.name} ({r.kind})", "", f"  {r.headline}", ""]
    out += [f"  {l.verdict:<{w}}  {l.text}" for l in r.lines]
    if r.output:
        out += ["", f"Output: {r.output}"]
    return "\n".join(out) + "\n"


def render_json(r: Report) -> str:
    return json.dumps(asdict(r), indent=2)


def render_html(r: Report) -> str:
    e = html.escape
    rows = "\n".join(f'<tr class="v-{e(l.verdict.split(" (")[0].lower().replace(" ", "-"))}"><td>{e(l.verdict)}</td><td>{e(l.text)}</td></tr>'
                     for l in r.lines)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ghostprint report</title>
<style>
:root{{--bg:#fff;--fg:#1a1a1a;--dim:#666;--ok:#1b7f3b;--bad:#b3261e;--warn:#9a6700}}
@media (prefers-color-scheme:dark){{:root{{--bg:#121212;--fg:#eee;--dim:#aaa;--ok:#6fd38a;--bad:#ff8a80;--warn:#f2c14e}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;max-width:46rem;margin:2rem auto;padding:0 16px}}
h1{{font-size:1.2rem}} p.h{{font-weight:600}} table{{border-collapse:collapse;width:100%}}
td{{padding:.4rem .5rem;border-bottom:1px solid #8884;vertical-align:top}} td:first-child{{white-space:nowrap;font-weight:600}}
.v-removed td:first-child,.v-reduced td:first-child,.v-scrubbed td:first-child,.v-not-found td:first-child{{color:var(--ok)}}
.v-still-present td:first-child,.v-found td:first-child{{color:var(--bad)}}
.v-warning td:first-child{{color:var(--warn)}}
.v-not-checked td:first-child{{color:var(--warn)}}
.dim{{color:var(--dim)}}
</style></head><body>
<h1>ghostprint report: {e(r.name)} <span class="dim">({e(r.kind)})</span></h1>
<p class="h">{e(r.headline)}</p>
<table>{rows}</table>
{f'<p class="dim">Output: {e(r.output)}</p>' if r.output else ''}
</body></html>
"""
