"""
Turning an uploaded file name into something safe to store and show.

WHY
    An upload's file name is attacker-controlled text. It is never used as a path in this
    project - images are decoded in memory and clips go to a temp file with a generated name -
    but it IS written into the violation log, shown on the dashboard and put into audit rows.
    A name like `../../etc/passwd` or one with a newline in it should not survive that journey
    intact, because the next person to add a feature may well use it as a path.

WHAT IT DOES
    Keeps the last path segment only, drops anything that is not a safe character, collapses
    the result and caps its length. An empty result becomes "upload" rather than "".
"""
from __future__ import annotations

import re

SAFE = re.compile(r"[^A-Za-z0-9._-]+")
MAX_LENGTH = 80
FALLBACK = "upload"


def safe_filename(name: str, fallback: str = FALLBACK) -> str:
    """`../../evil name.jpg` -> `evil_name.jpg`. Never returns a path, never returns empty."""
    if not name:
        return fallback
    # last segment only: both separators, because an upload can come from either OS
    tail = str(name).replace("\\", "/").split("/")[-1]
    tail = tail.replace("\x00", "")
    cleaned = SAFE.sub("_", tail).strip("._-")
    cleaned = re.sub(r"_{2,}", "_", cleaned)
    if not cleaned or set(cleaned) <= {"."}:
        return fallback
    return cleaned[:MAX_LENGTH]
