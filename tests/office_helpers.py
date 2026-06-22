"""Handlers for the office worker's process, which must be importable from a new process."""

from __future__ import annotations

import time


def hang_forever(path: str) -> str:
    time.sleep(3600)
    return ""


def fail_on_bad_names(path: str) -> str:
    if "bad" in path:
        raise ValueError("this file can't be converted")
    return "converted"
