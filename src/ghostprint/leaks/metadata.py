"""Metadata leaks: what the file says about where, when and on what it was made.

exiftool is used read-only. The finding names categories, never values.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..findings import Finding, Status

WORDS = {"location": "GPS location", "time": "capture time", "device": "phone model or serial number",
         "software": "editing software or edit history", "thumbnail": "an embedded thumbnail"}
_SKIP_GROUPS = ("ExifTool", "System", "File", "Composite")
# Dates every MP4/MOV container carries. On an output they are the randomised values ghostprint wrote.
CONTAINER_DATES = ("CreateDate", "ModifyDate", "TrackCreateDate", "TrackModifyDate", "MediaCreateDate", "MediaModifyDate")


def read_tags(path: Path) -> dict[str, object]:
    out = subprocess.run(["exiftool", "-j", "-a", "-G1", "-n", "--", str(Path(path).resolve())],
                         capture_output=True, text=True).stdout
    try:
        return json.loads(out)[0]
    except (ValueError, IndexError):
        return {}


def _category(group: str, name: str) -> str | None:
    n = name.lower()
    if "gps" in n or "location" in n or "iso6709" in n or group.startswith("GPS"):
        return "location"
    if "thumbnail" in n or "previewimage" in n or "jpgfromraw" in n:
        return "thumbnail"
    if n in ("make", "model", "lensmake", "lensmodel", "lensinfo", "serialnumber", "bodyserialnumber",
             "internalserialnumber", "camerauniqueid", "uniquecameramodel", "ownername", "cameraownername") \
            or "serial" in n:
        return "device"
    if n in ("software", "creatortool", "historysoftwareagent", "historyaction", "historywhen", "processingsoftware"):
        return "software"
    if ("date" in n or "time" in n) and n not in ("duration", "timescale", "mediatimescale", "previewtime",
                                                    "posterTime".lower(), "selectiontime", "currenttime", "previewduration",
                                                    "selectionduration", "trackduration", "mediaduration"):
        return "time"
    return None


def categories(tags: dict[str, object], *, allow_container_dates: bool = False) -> set[str]:
    cats = set()
    for key, val in tags.items():
        if ":" not in key or val in (None, "", 0, "0000:00:00 00:00:00"):
            continue
        group, name = key.split(":", 1)
        if group.startswith(_SKIP_GROUPS):
            continue
        if allow_container_dates and name in CONTAINER_DATES:
            continue
        c = _category(group, name)
        if c:
            cats.add(c)
    return cats


def detect(path: Path, *, allow_container_dates: bool = False) -> Finding:
    cats = categories(read_tags(path), allow_container_dates=allow_container_dates)
    if not cats:
        return Finding("metadata", Status.NOT_DETECTED, "no identifying metadata")
    words = [WORDS[c] for c in ("location", "time", "device", "software", "thumbnail") if c in cats]
    return Finding("metadata", Status.DETECTED, "in the file's metadata: " + ", ".join(words), strength=float(len(cats)))
