"""What a detector found. Summaries name the kind of leak, never its value."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

LEAKS = ("metadata", "aircraft", "hum", "gsm", "flicker", "fingerprint", "sun", "visible")

LABELS = {
    "metadata": "identifying metadata in the file",
    "aircraft": "long gliding tones (aircraft, traffic or HVAC): could locate it from a passing plane",
    "hum": "grid hum: could date the recording and tell which power grid it was made on",
    "gsm": "phone-radio buzz: shows a phone was transmitting nearby",
    "flicker": "light flicker from mains lighting: could date it even with the sound removed",
    "fingerprint": "camera sensor fingerprint: can link this to other photos from the same phone",
    "sun": "sun position and shadows",
    "visible": "readable signs, plates, landmarks or reflections",
}


class Status(Enum):
    DETECTED = "detected"
    NOT_DETECTED = "not_detected"
    NOT_CHECKED = "not_checked"
    WARNING = "warning"


@dataclass
class Finding:
    leak: str
    status: Status
    summary: str
    strength: float | None = None
    reason: str | None = None
