"""ghostprint INPUT [-o OUTPUT] [--check] [--report FILE] [--json] [--keep KIND]... [--fingerprint-from DIR]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .media import ToolMissing, UnsafeInput
from .pipeline import KEEPABLE, UsageError, run
from .report import render_html, render_json, render_text

HELP = """Strip the place, time and device leaks from a photo, video or audio file, and report what was found.

Runs entirely on this computer and never uses the network. Writes only the output file
(and the report, if you ask for one).

  ghostprint IMG_1234.MOV -o clean.mp4        clean a copy and print the report
  ghostprint IMG_1234.MOV --check             report only, write nothing
  ghostprint photo.heic -o clean.jpg --fingerprint-from ~/my-phone-photos
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ghostprint", description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="where to write the cleaned copy")
    ap.add_argument("--check", action="store_true", help="report only; write nothing")
    ap.add_argument("--report", type=Path, help="also save the report (.html, or .txt)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--keep", action="append", default=[], choices=KEEPABLE, metavar="KIND",
                    help=f"don't fix this leak ({', '.join(KEEPABLE)}); repeatable")
    ap.add_argument("--fingerprint-from", type=Path, metavar="DIR",
                    help="a folder of other photos from the same phone, to check the camera fingerprint")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    if not a.check and a.output is None:
        print("ghostprint: say where to write the cleaned copy with -o, or use --check", file=sys.stderr)
        return 2
    if a.report is not None and a.report.exists():
        print(f"ghostprint: {a.report} already exists; choose another name", file=sys.stderr)
        return 2
    try:
        r = run(a.input, None if a.check else a.output, keep=set(a.keep), refs=a.fingerprint_from)
    except ToolMissing as e:
        print(f"ghostprint: {e}", file=sys.stderr)
        return 3
    except (UsageError, UnsafeInput) as e:
        print(f"ghostprint: {e}", file=sys.stderr)
        return 2
    print(render_json(r) if a.json else render_text(r), end="")
    if a.report is not None:
        a.report.write_text(render_html(r) if a.report.suffix.lower() in (".html", ".htm") else render_text(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
