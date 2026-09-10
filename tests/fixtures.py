"""Builders for real files to run extractors against.

Tests use genuine PDFs, workbooks, images, and WAV files rather than mocks,
because every bug these extractors are likely to have lives in the parsing —
mocking the parser would test nothing. Everything here is written with the
standard library or a dependency Atlas already declares, so the test suite adds
no dependency of its own.
"""

from __future__ import annotations

import struct
import wave
from collections.abc import Sequence
from pathlib import Path

# -- PDF -------------------------------------------------------------------


def make_pdf(path: Path, pages: Sequence[str], title: str | None = None) -> Path:
    """Write a valid multi-page PDF with selectable text on each page.

    Hand-assembled rather than generated with reportlab so the suite stays
    dependency-free. Object numbering: 1 catalog, 2 page tree, 3 font, then one
    page object and one content stream per page.
    """
    objects: dict[int, bytes] = {}
    font_id = 3
    first_page_id = 4
    page_ids = [first_page_id + 2 * i for i in range(len(pages))]
    content_ids = [pid + 1 for pid in page_ids]

    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = b" ".join(b"%d 0 R" % pid for pid in page_ids)
    objects[2] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(pages))
    objects[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for page_id, content_id, text in zip(page_ids, content_ids, pages, strict=True):
        objects[page_id] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
            % (font_id, content_id)
        )
        objects[content_id] = _content_stream(text)

    info_id = max(objects) + 1
    if title:
        objects[info_id] = b"<< /Title (%s) >>" % _escape(title)

    return _assemble(path, objects, info_id if title else None)


def _content_stream(text: str) -> bytes:
    """One text-showing stream, one line of the input per line on the page."""
    lines = [line for line in text.splitlines() if line.strip()] or [" "]
    body = [b"BT", b"/F1 12 Tf", b"14 TL", b"72 720 Td"]
    for line in lines:
        body.append(b"(%s) Tj T*" % _escape(line))
    body.append(b"ET")
    stream = b"\n".join(body)
    return b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)


def _escape(text: str) -> bytes:
    out = text.encode("latin-1", "replace")
    for char, replacement in ((b"\\", b"\\\\"), (b"(", b"\\("), (b")", b"\\)")):
        out = out.replace(char, replacement)
    return out


def _assemble(path: Path, objects: dict[int, bytes], info_id: int | None) -> Path:
    chunks = [b"%PDF-1.4\n"]
    offsets: dict[int, int] = {}
    position = len(chunks[0])

    for number in sorted(objects):
        offsets[number] = position
        blob = b"%d 0 obj\n%s\nendobj\n" % (number, objects[number])
        chunks.append(blob)
        position += len(blob)

    highest = max(objects)
    xref_at = position
    xref = [b"xref\n", b"0 %d\n" % (highest + 1), b"0000000000 65535 f \n"]
    for number in range(1, highest + 1):
        xref.append(b"%010d 00000 n \n" % offsets.get(number, 0))
    chunks.extend(xref)

    trailer = b"trailer\n<< /Size %d /Root 1 0 R" % (highest + 1)
    if info_id:
        trailer += b" /Info %d 0 R" % info_id
    trailer += b" >>\nstartxref\n%d\n%%%%EOF\n" % xref_at
    chunks.append(trailer)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(chunks))
    return path


# -- spreadsheets ----------------------------------------------------------


def make_xlsx(path: Path, sheets: dict[str, Sequence[Sequence[object]]]) -> Path:
    """Write a workbook with one sheet per entry in `sheets`."""
    import openpyxl

    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    for name, rows in sheets.items():
        worksheet = workbook.create_sheet(title=name[:31])
        for row in rows:
            worksheet.append(list(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(str(path))
    return path


def make_csv(path: Path, rows: Sequence[Sequence[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(",".join(str(c) for c in row) for row in rows) + "\n")
    return path


# -- images ----------------------------------------------------------------


def make_image(path: Path, size: tuple[int, int] = (320, 240), color: str = "navy") -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(str(path))
    return path


# -- audio -----------------------------------------------------------------


def make_wav(path: Path, seconds: float = 1.0, rate: int = 8000) -> Path:
    """A short silent WAV — enough for ffprobe to report real duration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack(f"<{frames}h", *([0] * frames)))
    return path


# -- a whole folder --------------------------------------------------------


def make_corpus(root: Path) -> Path:
    """A small multimodal folder used by the pipeline tests.

    Deliberately seeded with entities that recur across modalities — the invoice
    id, the email address, and the amount each appear in more than one file — so
    tests can assert that the graph actually joins them.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(exist_ok=True)
    (root / "media").mkdir(exist_ok=True)

    (root / "README.md").write_text(
        "# Northwind Q3\n\n"
        "Owner ana@acme.example. Budget $42,500.00, due 2024-09-30.\n"
        "Tracking INV-2024-0912. See [the brief](docs/brief.md).\n"
    )
    (root / "docs" / "brief.md").write_text(
        "# Brief\n\n"
        "INV-2024-0912 covers the migration. Contact ana@acme.example.\n"
        "Signed off 2024-09-30 for $42,500.00.\n"
    )
    make_pdf(
        root / "docs" / "report.pdf",
        [
            "Quarterly Report\nPrepared for Acme Corp\nInvoice INV-2024-0912",
            "Findings\nTotal spend was $42,500.00 in the period.\nContact ana@acme.example",
        ],
        title="Quarterly Report",
    )
    make_csv(
        root / "docs" / "customers.csv",
        [
            ("customer_id", "name", "email", "plan", "mrr", "signed_on"),
            (1, "Acme Corp", "ops@acme.example", "enterprise", "4200.00", "2024-01-15"),
            (2, "Globex", "hi@globex.example", "pro", "890.00", "2024-03-02"),
        ],
    )
    make_image(root / "media" / "chart.png")
    make_wav(root / "media" / "standup.wav")
    return root
