"""Identify a file by its bytes rather than its name.

Extensions lie. Real corpora are full of `.xlsx` files that are actually legacy
OLE2 workbooks, `.pdf` files that are HTML error pages, and `.doc` files that
are RTF. Dispatching purely on extension means those surface as bewildering
parser errors — "File is not a zip file" tells a user nothing about what went
wrong or what to do.

This module answers one question: *what does this file actually look like?* It
is used to explain failures, not to drive dispatch. Extension-based dispatch is
right almost always; sniffing earns its place on the rare file where it isn't,
by turning a confusing error into an accurate one.
"""

from __future__ import annotations

from pathlib import Path

# (offset, signature, label). Ordered most-specific first: a JAR is a ZIP, and
# an OOXML document is a ZIP, so the plain ZIP entry has to come last.
SIGNATURES: tuple[tuple[int, bytes, str], ...] = (
    (0, b"%PDF-", "pdf"),
    (0, b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "ole2"),  # legacy .doc/.xls/.ppt
    (0, b"{\\rtf", "rtf"),
    (0, b"\xff\xd8\xff", "jpeg"),
    (0, b"\x89PNG\r\n\x1a\n", "png"),
    (0, b"GIF87a", "gif"),
    (0, b"GIF89a", "gif"),
    (0, b"BM", "bmp"),
    (0, b"II*\x00", "tiff"),
    (0, b"MM\x00*", "tiff"),
    (0, b"ID3", "mp3"),
    (0, b"\xff\xfb", "mp3"),
    (0, b"OggS", "ogg"),
    (0, b"fLaC", "flac"),
    (0, b"\x1a\x45\xdf\xa3", "matroska"),
    (4, b"ftyp", "mp4"),
    (0, b"\x1f\x8b", "gzip"),
    (0, b"BZh", "bzip2"),
    (0, b"7z\xbc\xaf\x27\x1c", "7z"),
    (0, b"Rar!", "rar"),
    (0, b"\x7fELF", "elf"),
    (0, b"SQLite format 3\x00", "sqlite"),
    (0, b"PK\x03\x04", "zip"),  # last: ooxml, odf and jars all start this way
)

# What each sniffed format implies about the extensions it should carry.
EXPECTED_EXTENSIONS: dict[str, frozenset[str]] = {
    "pdf": frozenset({".pdf"}),
    "ole2": frozenset({".doc", ".xls", ".ppt", ".msg", ".db"}),
    "zip": frozenset({".zip", ".docx", ".xlsx", ".pptx", ".docm", ".xlsm", ".pptm",
                      ".odt", ".ods", ".odp", ".jar", ".epub", ".apk"}),
    "rtf": frozenset({".rtf", ".doc"}),
    "jpeg": frozenset({".jpg", ".jpeg", ".jpe"}),
    "png": frozenset({".png"}),
    "gif": frozenset({".gif"}),
    "tiff": frozenset({".tif", ".tiff"}),
    "bmp": frozenset({".bmp"}),
    "mp3": frozenset({".mp3"}),
    "ogg": frozenset({".ogg", ".oga", ".ogv", ".opus"}),
    "flac": frozenset({".flac"}),
    "matroska": frozenset({".mkv", ".webm"}),
    "mp4": frozenset({".mp4", ".m4a", ".m4v", ".mov", ".3gp"}),
}

# Human-readable notes for the formats Atlas cannot read, so a warning can say
# what to do rather than only what failed.
UNSUPPORTED: dict[str, str] = {
    "ole2": "a legacy OLE2 Office file (.doc/.xls/.ppt); Atlas reads the modern "
            "OOXML formats only — convert it with LibreOffice to index it",
    "rtf": "an RTF document; Atlas does not parse RTF",
    "7z": "a 7-Zip archive",
    "rar": "a RAR archive",
    "elf": "a compiled binary",
}


def sniff(path: Path, head: bytes | None = None) -> str | None:
    """The format `path` actually is, by magic bytes, or None if unrecognised."""
    if head is None:
        try:
            with open(path, "rb") as handle:
                head = handle.read(32)
        except OSError:
            return None
    for offset, signature, label in SIGNATURES:
        if head[offset : offset + len(signature)] == signature:
            return label
    return None


def mislabelled(path: Path, head: bytes | None = None) -> str | None:
    """Describe the mismatch when a file's bytes contradict its extension.

    Returns None when the extension is consistent, unrecognised, or when there
    is nothing useful to say — a false alarm here is worse than silence, since
    it would attach a scary warning to a perfectly good file.
    """
    actual = sniff(path, head)
    if actual is None:
        return None
    expected = EXPECTED_EXTENSIONS.get(actual)
    suffix = path.suffix.lower()
    if expected is None or not suffix or suffix in expected:
        return None
    return f"content is {actual}, not {suffix}"


def explain(path: Path) -> str | None:
    """A one-line explanation for why a file may have failed to parse."""
    actual = sniff(path)
    if actual is None:
        return None
    note = UNSUPPORTED.get(actual)
    mismatch = mislabelled(path)
    if note and mismatch:
        return f"{mismatch} — {note}"
    if note:
        return f"this is {note}"
    return mismatch
