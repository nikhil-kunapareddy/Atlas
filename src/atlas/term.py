"""Terminal output helpers.

Colour is disabled when stdout is not a TTY, so piping `atlas search` into a
file or another program yields clean text, and when `NO_COLOR` is set, per the
convention at https://no-color.org.

This lives in its own module so both halves of the CLI can use it without
importing each other.
"""

from __future__ import annotations

import os
import sys

ENABLED = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if ENABLED else text


def dim(text: str) -> str:
    return _c(text, "2")


def bold(text: str) -> str:
    return _c(text, "1")


def cyan(text: str) -> str:
    return _c(text, "36")


def yellow(text: str) -> str:
    return _c(text, "33")


def green(text: str) -> str:
    return _c(text, "32")


def red(text: str) -> str:
    return _c(text, "31")
