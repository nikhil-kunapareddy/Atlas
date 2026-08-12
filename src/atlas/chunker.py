"""Turn a file into line-anchored chunks.

Chunks stay small enough to paste into an agent's context without crowding it,
and every chunk carries its line range so recall can cite `path:12-48` — a
coding agent can then open exactly that span instead of re-reading the file.
"""

from __future__ import annotations

import re
from pathlib import Path

from .store import ChunkInput

TARGET_CHARS = 1200
OVERLAP_LINES = 2
MAX_FILE_BYTES = 1_000_000

# Extensions we never index. Anything else gets a binary sniff instead of a
# guess, so unusual-but-textual files (.env.example, Dockerfile.prod) still land.
BINARY_EXTENSIONS = frozenset(
    """
    .png .jpg .jpeg .gif .bmp .ico .webp .tif .tiff .svgz .heic .avif
    .mp3 .wav .flac .ogg .m4a .aac .mp4 .mov .avi .mkv .webm .wmv
    .zip .tar .gz .bz2 .xz .7z .rar .jar .war .whl .egg .dmg .iso
    .pdf .doc .docx .xls .xlsx .ppt .pptx .odt .ods
    .so .dylib .dll .exe .bin .o .a .obj .class .pyc .pyo .wasm
    .db .sqlite .sqlite3 .mdb .parquet .avro .pkl .npy .npz .h5 .pt .pth .safetensors
    .ttf .otf .woff .woff2 .eot
    .lock
    """.split()
)

MARKDOWN_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)")

# Deliberately shallow: this is a breadcrumb for the reader, not a parser.
CODE_SYMBOL = re.compile(
    r"""^[ \t]{0,4}(?:export\s+)?(?:default\s+)?(?:public\s+|private\s+|protected\s+)?
        (?:async\s+)?(?:static\s+)?
        (?P<kind>def|class|func|function|fn|type|struct|interface|impl|trait|enum|module)
        \s+(?P<name>[A-Za-z_$][\w$]*)""",
    re.VERBOSE,
)

CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def is_probably_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def should_index(path: Path) -> bool:
    """Cheap pre-read filter. Content checks happen in `read_text`."""
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return False
    # Dotfiles are usually config worth indexing; these few are pure noise.
    if path.name in {".DS_Store", ".gitattributes", ".gitmodules"}:
        return False
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return False
    except OSError:
        return False
    return True


def read_text(path: Path) -> str | None:
    """Return decoded text, or None if the file is binary or unreadable."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if is_probably_binary(data):
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("latin-1")
        except UnicodeDecodeError:
            return None


def split_identifiers(text: str) -> list[str]:
    """Extra tokens so `user authentication` can match `UserAuthenticator`.

    unicode61 already splits snake_case on underscores; camelCase needs help.
    """
    extra: set[str] = set()
    for ident in IDENTIFIER.findall(text):
        parts = [p for p in CAMEL_BOUNDARY.split(ident) if len(p) > 2]
        if len(parts) > 1:
            extra.update(p.lower() for p in parts)
    return sorted(extra)


def _search_body(text: str, heading: str | None, rel_path: str) -> str:
    """What actually goes into FTS: the text plus retrieval hints.

    The path is included so `atlas recall "auth config"` can hit files named for
    the concept even when the body never spells it out.
    """
    parts = [text, rel_path.replace("/", " ").replace("_", " ").replace("-", " ")]
    if heading:
        parts.append(heading)
    extra = split_identifiers(text)
    if extra:
        parts.append(" ".join(extra))
    return "\n".join(parts)


def _heading_for(line: str, is_markdown: bool, current: str | None) -> str | None:
    if is_markdown:
        match = MARKDOWN_HEADING.match(line)
        if match:
            return match.group(2).strip()
        return current
    match = CODE_SYMBOL.match(line)
    if match:
        return f"{match.group('kind')} {match.group('name')}"
    return current


def chunk_text(text: str, rel_path: str) -> list[ChunkInput]:
    """Split `text` into overlapping, line-anchored chunks."""
    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        return []

    is_markdown = rel_path.lower().endswith((".md", ".mdx", ".markdown", ".rst", ".txt"))
    chunks: list[ChunkInput] = []

    buffer: list[str] = []
    buffer_chars = 0
    start_line = 1
    heading: str | None = None
    chunk_heading: str | None = None

    def flush(end_line: int) -> None:
        nonlocal buffer, buffer_chars, chunk_heading
        body = "\n".join(buffer).strip()
        if body:
            chunks.append(
                ChunkInput(
                    ord=len(chunks),
                    start_line=start_line,
                    end_line=end_line,
                    heading=chunk_heading,
                    text=body,
                    search_body=_search_body(body, chunk_heading, rel_path),
                )
            )
        buffer = []
        buffer_chars = 0
        chunk_heading = None

    # Below this, a chunk is still just the tail of the previous section and
    # can be relabelled; above it, a heading change is a real boundary.
    boundary_floor = TARGET_CHARS // 3

    for lineno, line in enumerate(lines, start=1):
        previous_heading = heading
        heading = _heading_for(line, is_markdown, heading)

        if heading != previous_heading and buffer:
            if buffer_chars >= boundary_floor:
                # A new section or top-level symbol is a natural boundary.
                # Break here rather than straddling two topics in one chunk.
                flush(lineno - 1)
                start_line = lineno
            else:
                # Whatever we've buffered is preamble; this chunk is really
                # about the section that just started.
                chunk_heading = heading

        if not buffer:
            start_line = lineno
            chunk_heading = heading

        buffer.append(line)
        buffer_chars += len(line) + 1

        if buffer_chars >= TARGET_CHARS:
            flush(lineno)
            overlap = lines[max(0, lineno - OVERLAP_LINES) : lineno]
            if overlap and lineno < len(lines):
                buffer = list(overlap)
                buffer_chars = sum(len(x) + 1 for x in overlap)
                start_line = lineno - len(overlap) + 1
                chunk_heading = heading

    if buffer:
        flush(len(lines))

    # Re-number after the fact: markdown boundary flushes can drop empty chunks.
    return [
        ChunkInput(
            ord=i,
            start_line=c.start_line,
            end_line=c.end_line,
            heading=c.heading,
            text=c.text,
            search_body=c.search_body,
        )
        for i, c in enumerate(chunks)
    ]
