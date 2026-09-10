"""Extractors, run against real files rather than mocks.

Every bug these are likely to have lives in the parsing, so the fixtures build
genuine PDFs, workbooks, images and WAVs. The shared contract each extractor
must honour — produce chunks, anchor them, and never raise on a bad file — is
asserted here so a new modality can be checked against the same bar.
"""

from __future__ import annotations

import pytest

import fixtures
from atlas.extract import ExtractOptions, for_path, modality_for
from atlas.extract.base import ExtractContext
from atlas.graph import Modality, NodeKind


def run(path, **option_overrides):
    """Run whichever extractor claims `path` and return its Extraction."""
    extractor = for_path(path)
    assert extractor is not None, f"nothing claims {path.name}"
    ctx = ExtractContext(
        path=path,
        rel_path=path.name,
        doc_uid=f"doc:{path.name}",
        digest="deadbeef",
        size=path.stat().st_size,
        options=ExtractOptions(**option_overrides),
        # No caching in tests: each call should exercise the real extractor.
        cached=lambda producer, version, compute: compute(),
    )
    return extractor.extract(ctx)


def kinds(result):
    return {n.kind for n in result.nodes}


def chunks(result):
    return [n for n in result.nodes if n.kind == NodeKind.CHUNK]


# -- dispatch --------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a.pdf", Modality.PDF),
        ("a.xlsx", Modality.SHEET),
        ("a.csv", Modality.SHEET),
        ("a.png", Modality.IMAGE),
        ("a.mp3", Modality.AUDIO),
        ("a.mp4", Modality.VIDEO),
        ("a.pptx", Modality.SLIDES),
        ("a.md", Modality.TEXT),
        ("Dockerfile", Modality.TEXT),
    ],
)
def test_registry_routes_by_extension(tmp_path, name, expected):
    assert modality_for(tmp_path / name) is expected


def test_unknown_binary_types_are_unclaimed(tmp_path):
    assert for_path(tmp_path / "archive.zip") is None


# -- pdf -------------------------------------------------------------------


def test_pdf_produces_pages_chunks_and_metadata(tmp_path):
    path = fixtures.make_pdf(
        tmp_path / "r.pdf",
        ["Page one mentions INV-2024-0912.", "Page two totals $42,500.00."],
        title="Quarterly Report",
    )
    result = run(path)

    assert result.modality is Modality.PDF
    assert result.title == "Quarterly Report"
    assert result.props["pages"] == 2
    assert kinds(result) == {NodeKind.PAGE, NodeKind.CHUNK}

    pages = sorted(
        (n for n in result.nodes if n.kind == NodeKind.PAGE), key=lambda n: n.props["page"]
    )
    assert [n.props["page"] for n in pages] == [1, 2]
    assert "p.2" in pages[1].name, "page number must be visible in the citable name"
    # Every chunk carries the page it came from, so an answer can cite it.
    assert all("page" in c.props for c in chunks(result))
    assert "$42,500.00" in result.text


def test_pdf_pages_are_linked_in_reading_order(tmp_path):
    path = fixtures.make_pdf(tmp_path / "r.pdf", ["one", "two", "three"])
    result = run(path)
    next_edges = [e for e in result.edges if e.kind == "next"]
    assert len(next_edges) == 2


def test_corrupt_pdf_warns_instead_of_raising(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4\nthis is not a pdf at all")
    result = run(path)
    assert result.nodes == []
    assert result.warnings, "a broken file must explain itself"


def test_pdf_with_no_text_layer_says_so(tmp_path):
    path = fixtures.make_pdf(tmp_path / "scan.pdf", [" "])
    result = run(path)
    assert any("scanned" in w or "no extractable text" in w for w in result.warnings)


# -- spreadsheets ----------------------------------------------------------


def test_xlsx_yields_sheets_columns_and_row_chunks(tmp_path):
    path = fixtures.make_xlsx(
        tmp_path / "book.xlsx",
        {
            "Customers": [
                ("customer_id", "name", "email", "mrr", "signed_on"),
                (1, "Acme", "ops@acme.example", 4200.0, "2024-01-15"),
                (2, "Globex", "hi@globex.example", 890.0, "2024-03-02"),
            ],
            "Notes": [("note",), ("second sheet",)],
        },
    )
    result = run(path)

    assert result.props["sheets"] == 2
    assert kinds(result) == {NodeKind.SHEET, NodeKind.COLUMN, NodeKind.CHUNK}

    columns = {n.name: n for n in result.nodes if n.kind == NodeKind.COLUMN}
    assert set(columns) >= {"customer_id", "name", "email", "mrr", "signed_on"}
    assert columns["mrr"].props["dtype"] == "number"
    assert columns["signed_on"].props["dtype"] == "date"
    assert columns["name"].props["dtype"] == "text"
    # Sample values are what make "which sheet holds customer emails?" work.
    assert "ops@acme.example" in columns["email"].body


def test_csv_header_detection_skips_headerless_data(tmp_path):
    with_header = fixtures.make_csv(tmp_path / "a.csv", [("name", "qty"), ("bolt", 4)])
    assert any(n.kind == NodeKind.COLUMN for n in run(with_header).nodes)

    numeric_first_row = fixtures.make_csv(tmp_path / "b.csv", [(1, 2), (3, 4)])
    assert not any(n.kind == NodeKind.COLUMN for n in run(numeric_first_row).nodes)


def test_column_type_survives_a_stray_value(tmp_path):
    """Real spreadsheets have an 'N/A' in an otherwise numeric column."""
    rows = [("amount",)] + [(str(i),) for i in range(30)] + [("N/A",)]
    path = fixtures.make_csv(tmp_path / "c.csv", rows)
    column = next(n for n in run(path).nodes if n.kind == NodeKind.COLUMN)
    assert column.props["dtype"] == "number"


def test_row_chunks_repeat_the_header(tmp_path):
    """A block of bare values is unreadable in an agent's context."""
    rows = [("name", "qty")] + [(f"item{i}", i) for i in range(50)]
    path = fixtures.make_csv(tmp_path / "d.csv", rows)
    row_chunks = chunks(run(path))
    assert len(row_chunks) >= 2
    assert all(c.body.startswith("name | qty") for c in row_chunks)


# -- images ----------------------------------------------------------------


def test_image_records_dimensions_and_stays_searchable(tmp_path):
    path = fixtures.make_image(tmp_path / "chart-q3.png", size=(640, 480))
    result = run(path)
    assert result.props["width"] == 640
    assert result.props["height"] == 480
    assert result.props["format"] == "PNG"
    # No OCR and no vision model, but the node must still be findable.
    assert "chart" in result.text and "PNG image" in result.text


# -- audio -----------------------------------------------------------------


def test_audio_without_whisper_still_lands_with_metadata(tmp_path):
    path = fixtures.make_wav(tmp_path / "standup.wav", seconds=1.0)
    result = run(path, transcribe=True)
    assert result.modality is Modality.AUDIO
    assert "standup" in result.text
    # ffprobe may or may not be installed; when it is, duration must be right.
    if "duration" in result.props:
        assert 0.5 < result.props["duration"] < 2.0


def test_transcription_can_be_switched_off(tmp_path):
    path = fixtures.make_wav(tmp_path / "a.wav")
    result = run(path, transcribe=False)
    assert not any(n.kind == NodeKind.SEGMENT for n in result.nodes)


# -- text ------------------------------------------------------------------


def test_markdown_title_headings_and_relative_links(tmp_path):
    (tmp_path / "docs").mkdir()
    path = tmp_path / "docs" / "a.md"
    path.write_text("# The Title\n\nBody text here.\n\nSee [spec](../spec.md) and [x](http://e.com).\n")
    extractor = for_path(path)
    ctx = ExtractContext(
        path=path,
        rel_path="docs/a.md",
        doc_uid="doc:docs/a.md",
        digest="d",
        size=path.stat().st_size,
        options=ExtractOptions(),
        cached=lambda p, v, c: c(),
    )
    result = extractor.extract(ctx)

    assert result.title == "The Title"
    references = [e for e in result.edges if e.kind == "references"]
    assert len(references) == 1, "only the relative link becomes an edge"
    assert references[0].dst == "doc:spec.md", "resolved against the file's own folder"


def test_html_tags_are_stripped(tmp_path):
    path = tmp_path / "p.html"
    path.write_text("<html><head><title>Report</title><style>b{}</style></head>"
                    "<body><p>Total was 500</p><script>x=1</script></body></html>")
    result = run(path)
    assert result.title == "Report"
    assert "Total was 500" in result.text
    assert "script" not in result.text and "x=1" not in result.text


# -- library noise ---------------------------------------------------------


def test_parser_chatter_is_silenced_during_extraction(tmp_path, capfd):
    """Real PDFs make pypdf extremely loud.

    On a corpus of government PDFs it emitted thousands of lines about missing
    font tooling — none of it actionable, since Atlas never renders glyphs — and
    it buried the warnings Atlas raises on purpose.
    """
    import logging

    from atlas.extract.base import quiet

    logging.getLogger("pypdf").setLevel(logging.DEBUG)
    with quiet():
        logging.getLogger("pypdf").warning("fontTools is required to fully parse…")
        import warnings as w

        w.warn("Unknown extension is not supported", UserWarning, stacklevel=1)

    captured = capfd.readouterr()
    assert "fontTools" not in captured.err
    assert "Unknown extension" not in captured.err


def test_quiet_restores_logging_levels_afterwards():
    """Silencing must be scoped: a global mute would hide real problems."""
    import logging

    from atlas.extract.base import quiet

    logger = logging.getLogger("pypdf")
    logger.setLevel(logging.DEBUG)
    with quiet():
        assert logger.level == logging.ERROR
    assert logger.level == logging.DEBUG
