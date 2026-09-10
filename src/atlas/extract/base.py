"""The extractor contract.

An extractor turns one file into a piece of graph: a description of the
document itself, plus whatever structure lives inside it — pages, sheets,
columns, chunks, time segments. It does not touch the database, does not walk
the filesystem, and does not decide what the document *means*. That separation
is what makes the modalities comparable: a PDF extractor and an audio extractor
produce the same shape of output, so the pipeline treats them identically and
new modalities are additive.

Two conventions every extractor follows:

**Fail soft, never silently.** A missing optional dependency, a corrupt file, or
an encrypted PDF must not abort a build over a thousand files. Extractors raise
`ExtractionError` (or return warnings) and the pipeline records the document as
a metadata-only node. A folder always produces a graph; the graph just says
less about the files it could not read.

**Everything expensive is content-addressed.** Transcription and OCR are keyed
by file digest through `ExtractContext.cached`, so re-running a build over an
unchanged folder re-reads nothing and re-bills nothing.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..graph import Edge, EdgeKind, Modality, Node, NodeKind, child_uid


class ExtractionError(Exception):
    """The file could not be read as its apparent type."""


class MissingDependency(ExtractionError):
    """An optional extra is not installed.

    Carries the install hint so the CLI can tell the user exactly what to run
    rather than surfacing a bare ImportError.
    """

    def __init__(self, module: str, extra: str):
        self.module = module
        self.extra = extra
        super().__init__(
            f"{module} is not installed — run: pip install 'atlas-context[{extra}]'"
        )


def require(module: str, extra: str) -> Any:
    """Import an optional dependency, or explain how to get it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised via extractors
        raise MissingDependency(module, extra) from exc


def have(module: str) -> bool:
    """Whether an optional dependency is importable, for capability reporting."""
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


# Parsing libraries are chatty about files they can still read. On a real
# corpus pypdf alone emits thousands of lines about missing font tooling, which
# buries the warnings Atlas raises deliberately and makes the CLI unusable. The
# information is not actionable — Atlas does not render glyphs — so it is
# silenced for the duration of an extraction, not globally.
NOISY_LOGGERS = ("pypdf", "PIL", "openpyxl", "pdfminer", "fontTools")


@contextlib.contextmanager
def quiet():
    """Silence third-party parser chatter while an extractor runs."""
    levels = {}
    for name in NOISY_LOGGERS:
        logger = logging.getLogger(name)
        levels[name] = logger.level
        logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)


@dataclass
class ExtractContext:
    """Everything an extractor is given about the file it is reading."""

    path: Path
    rel_path: str
    doc_uid: str
    digest: str
    size: int
    options: ExtractOptions
    # Content-addressed cache: cached(producer, version, compute) runs `compute`
    # only if this exact file has not been through that producer before.
    cached: Callable[[str, str, Callable[[], Any]], Any]

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()


@dataclass
class ExtractOptions:
    """Knobs the user set on the command line.

    Defaults are chosen so `atlas <folder>` works offline, for free, on a laptop.
    """

    transcribe: bool = True
    whisper_model: str = "base"
    ocr: bool = False
    keyframes: int = 4
    max_file_bytes: int = 200 * 1024 * 1024
    max_text_bytes: int = 5 * 1024 * 1024
    max_sheet_rows: int = 50_000
    language: str | None = None
    # Where extracted artefacts (video keyframes) are written. Set by the
    # pipeline to `<store>/media`; None disables anything that writes files.
    media_dir: Path | None = None


@dataclass
class Extraction:
    """What an extractor produces for one file.

    `props` lands on the document node; `nodes` and `edges` describe everything
    inside it. Extractors never create the document node themselves — the
    pipeline does, so that a failed extraction still yields one.
    """

    modality: Modality
    title: str = ""
    props: dict[str, Any] = field(default_factory=dict)
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    text: str = ""
    warnings: list[str] = field(default_factory=list)

    def add(self, node: Node, *, contained_by: str | None = None) -> None:
        self.nodes.append(node)
        if contained_by:
            self.edges.append(Edge(contained_by, node.uid, EdgeKind.CONTAINS))

    def warn(self, message: str) -> None:
        self.warnings.append(message)


class Extractor(ABC):
    """Base class for every modality.

    `name` and `version` identify the producer in the derived-work cache.
    Bumping `version` invalidates only that extractor's cached output, which is
    the intended way to ship an extraction improvement without forcing users to
    rebuild graphs from scratch.
    """

    name: str = "extractor"
    version: str = "1"
    modality: Modality = Modality.UNKNOWN
    extensions: frozenset[str] = frozenset()

    def matches(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    @abstractmethod
    def extract(self, ctx: ExtractContext) -> Extraction:
        """Read `ctx.path` and describe it as graph."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name} v{self.version}>"


# -- shared helpers --------------------------------------------------------


def text_chunks(
    parent_uid: str,
    text: str,
    rel_path: str,
    extra_props: dict[str, Any] | None = None,
    start_ord: int = 0,
) -> tuple[list[Node], list[Edge], int]:
    """Split `text` into CHUNK nodes hanging off `parent_uid`.

    Every modality funnels through here once it has produced text, so a chunk
    from a PDF page, a Word document, and a video transcript are the same kind
    of thing and rank against each other in search. Returns the nodes, the
    CONTAINS/NEXT edges, and the next free ordinal so a caller looping over
    pages can keep chunk numbering unique across the whole document.
    """
    from ..chunker import chunk_text  # local import: avoids a cycle at module load

    nodes: list[Node] = []
    edges: list[Edge] = []
    previous_uid: str | None = None
    ordinal = start_ord

    for chunk in chunk_text(text, rel_path):
        uid = child_uid(parent_uid, NodeKind.CHUNK, ordinal)
        # `path` is set unconditionally: every chunk must be able to say which
        # file it came from, or a search result cannot be cited. Callers may
        # override it via extra_props, but they cannot omit it.
        props: dict[str, Any] = {
            "path": rel_path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "ord": ordinal,
        }
        if chunk.heading:
            props["heading"] = chunk.heading
        if extra_props:
            props.update(extra_props)

        name = _chunk_name(rel_path, chunk.heading, props)
        nodes.append(Node(uid=uid, kind=NodeKind.CHUNK, name=name, body=chunk.text, props=props))
        edges.append(Edge(parent_uid, uid, EdgeKind.CONTAINS))
        if previous_uid:
            edges.append(Edge(previous_uid, uid, EdgeKind.NEXT))
        previous_uid = uid
        ordinal += 1

    return nodes, edges, ordinal


def _chunk_name(rel_path: str, heading: str | None, props: dict[str, Any]) -> str:
    """A short human label — this is what shows up in search results."""
    base = Path(rel_path).name
    if "page" in props:
        base = f"{base} p.{props['page']}"
    elif "sheet" in props:
        base = f"{base} [{props['sheet']}]"
    elif "start_time" in props:
        base = f"{base} {_timestamp(props['start_time'])}"
    else:
        base = f"{base}:{props['start_line']}-{props['end_line']}"
    return f"{base} — {heading}" if heading else base


def _timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def clean(text: str, limit: int | None = None) -> str:
    """Normalise extracted text: strip control characters, collapse blank runs.

    PDF and OCR output is full of form feeds, soft hyphens, and NULs that break
    both FTS tokenisation and terminal output.
    """
    if not text:
        return ""
    out = text.replace("\x00", "").replace("\xad", "").replace("\f", "\n")
    lines = [line.rstrip() for line in out.splitlines()]
    result: list[str] = []
    blanks = 0
    for line in lines:
        if line:
            blanks = 0
            result.append(line)
        else:
            blanks += 1
            if blanks <= 1:
                result.append("")
    joined = "\n".join(result).strip()
    return joined[:limit] if limit else joined
