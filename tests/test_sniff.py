"""Content sniffing for mislabelled files.

Every case here comes from a real corpus: the Govdocs1 archives contain legacy
OLE2 workbooks named `.xlsx`, HTML error pages named `.pdf`, and RTF named
`.doc`. The value of sniffing is not dispatch — extension dispatch is right
almost always — it is turning "File is not a zip file" into a sentence that says
what the file is and what to do about it.
"""

from __future__ import annotations

import pytest

from atlas.sniff import explain, mislabelled, sniff

OLE2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 24
ZIP = b"PK\x03\x04" + b"\x00" * 28
PDF = b"%PDF-1.4\n" + b"\x00" * 24
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 28
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
MP4 = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 16
RTF = b"{\\rtf1\\ansi" + b"\x00" * 20


def write(tmp_path, name: str, blob: bytes):
    path = tmp_path / name
    path.write_bytes(blob)
    return path


@pytest.mark.parametrize(
    "blob,expected",
    [
        (OLE2, "ole2"),
        (ZIP, "zip"),
        (PDF, "pdf"),
        (JPEG, "jpeg"),
        (PNG, "png"),
        (MP4, "mp4"),
        (RTF, "rtf"),
        (b"just some prose, no signature at all", None),
    ],
)
def test_sniff_identifies_by_magic_bytes(tmp_path, blob, expected):
    assert sniff(write(tmp_path, "f.bin", blob)) == expected


def test_specific_signatures_win_over_the_generic_zip(tmp_path):
    """A .docx and a .jar are both ZIPs, so `zip` must be the last signature
    checked — otherwise it would shadow anything layered on top of it."""
    from atlas.sniff import SIGNATURES

    labels = [label for _, _, label in SIGNATURES]
    assert labels[-1] == "zip"


# -- the case that motivated this -----------------------------------------


def test_legacy_workbook_named_xlsx_is_diagnosed(tmp_path):
    path = write(tmp_path, "500968.xlsx", OLE2)
    assert mislabelled(path) == "content is ole2, not .xlsx"

    message = explain(path)
    assert "ole2" in message
    assert "OOXML" in message, "must say what Atlas can read"
    assert "LibreOffice" in message, "must say what to do about it"


def test_correctly_named_files_produce_no_warning(tmp_path):
    """A false alarm is worse than silence — it maligns a good file."""
    for name, blob in (
        ("a.xlsx", ZIP), ("a.docx", ZIP), ("a.pptx", ZIP), ("a.odt", ZIP),
        ("a.pdf", PDF), ("a.jpg", JPEG), ("a.jpeg", JPEG), ("a.png", PNG),
        ("a.mp4", MP4), ("a.m4a", MP4), ("a.doc", OLE2), ("a.xls", OLE2),
        ("a.rtf", RTF),
    ):
        assert mislabelled(write(tmp_path, name, blob)) is None, name


def test_unknown_content_and_extensionless_files_are_left_alone(tmp_path):
    assert mislabelled(write(tmp_path, "notes.md", b"# hello\n")) is None
    assert mislabelled(write(tmp_path, "Makefile", OLE2)) is None, (
        "no extension means nothing to contradict"
    )


def test_unreadable_file_does_not_raise(tmp_path):
    assert sniff(tmp_path / "does-not-exist") is None
    assert mislabelled(tmp_path / "does-not-exist") is None


def test_explain_covers_formats_atlas_cannot_read(tmp_path):
    assert "legacy" in explain(write(tmp_path, "a.doc", OLE2))
    assert "RTF" in explain(write(tmp_path, "a.rtf", RTF))
    assert explain(write(tmp_path, "a.pdf", PDF)) is None, "nothing to say about a good pdf"
