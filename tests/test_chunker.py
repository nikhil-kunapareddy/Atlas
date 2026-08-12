from __future__ import annotations

from atlas.chunker import chunk_text, is_probably_binary, should_index, split_identifiers


def test_chunks_carry_line_ranges():
    text = "\n".join(f"line {i}" for i in range(1, 401))
    chunks = chunk_text(text, "big.txt")

    assert len(chunks) > 1
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 400
    for chunk in chunks:
        assert chunk.start_line <= chunk.end_line


def test_line_ranges_point_at_real_content():
    lines = [f"row {i}" for i in range(1, 121)]
    chunks = chunk_text("\n".join(lines), "rows.txt")

    for chunk in chunks:
        first_body_line = chunk.text.splitlines()[0]
        assert lines[chunk.start_line - 1] == first_body_line


def test_markdown_headings_become_chunk_headings():
    text = (
        "# Title\n\n## Setup\n\n"
        + "Install the thing.\n" * 30
        + "\n## Teardown\n\n"
        + "Remove the thing.\n" * 30
    )
    chunks = chunk_text(text, "guide.md")
    headings = {c.heading for c in chunks}
    assert "Setup" in headings
    assert "Teardown" in headings


def test_code_symbols_become_chunk_headings():
    text = "def parse_config(path):\n" + "    pass\n" * 5
    chunks = chunk_text(text, "conf.py")
    assert chunks[0].heading == "def parse_config"


def test_camel_case_is_split_for_search():
    extra = split_identifiers("class TokenValidator: pass")
    assert "token" in extra
    assert "validator" in extra


def test_search_body_includes_path_tokens():
    chunks = chunk_text("nothing relevant here", "src/auth/session.py")
    assert "auth" in chunks[0].search_body
    # The stored text stays clean; only the search body is augmented.
    assert chunks[0].text == "nothing relevant here"


def test_empty_and_whitespace_files_produce_no_chunks():
    assert chunk_text("", "empty.txt") == []
    assert chunk_text("\n\n   \n", "blank.txt") == []


def test_binary_detection(tmp_path):
    assert is_probably_binary(b"\x00\x01\x02")
    assert not is_probably_binary(b"plain text")

    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG")
    assert not should_index(png)

    src = tmp_path / "x.py"
    src.write_text("print(1)")
    assert should_index(src)


def test_oversized_files_are_skipped(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("x" * 1_000_001)
    assert not should_index(big)
