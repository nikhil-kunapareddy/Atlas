"""The build pipeline: a folder on disk becomes a graph.

This is the whole of `atlas <folder>`. It walks the tree, hands each file to the
extractor that understands it, and writes the result into the graph, mirroring
the directory hierarchy as FOLDER nodes so the shape of the folder survives into
the shape of the graph.

Three properties are worth stating up front, because the rest of the file exists
to preserve them:

**A build always produces a graph.** No single file can fail the run. A corrupt
PDF, an encrypted spreadsheet, a video with no ffmpeg — each becomes a document
node carrying a warning, and the walk continues. Partial knowledge beats a
traceback.

**A build is idempotent and incremental.** Node identity comes from path, so
re-running upserts in place. Files unchanged since last time are recognised by
`(mtime, size)` and never opened; files that were touched but not modified are
caught by digest. Running `atlas <folder>` twice in a row is close to free.

**Deleting a file deletes what it contributed, and nothing else.** Structural
nodes are owned by their file and cascade away with it; shared entities survive
and are swept only once nothing mentions them.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import entities as entity_scan
from .extract import ExtractContext, Extraction, ExtractOptions, for_path
from .extract.base import MissingDependency, quiet
from .graph import (
    Edge,
    EdgeKind,
    GraphStore,
    Modality,
    Node,
    NodeKind,
    Provenance,
    document_uid,
    entity_uid,
    folder_uid,
)
from .indexer import DEFAULT_IGNORE_DIRS
from .sniff import explain, mislabelled, sniff
from .store import Store

ProgressFn = Callable[[str], None]

DIGEST_BLOCK = 1024 * 1024

# Files that are never worth a node.
IGNORED_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini", ".localized"})


@dataclass
class BuildStats:
    """What a build did, in enough detail to explain itself afterwards."""

    root: str = ""
    files_seen: int = 0
    files_indexed: int = 0
    files_unchanged: int = 0
    files_skipped: int = 0
    files_removed: int = 0
    nodes_written: int = 0
    edges_written: int = 0
    entities_linked: int = 0
    unreadable: Counter = field(default_factory=Counter)
    edges_dropped: int = 0
    orphans_collected: int = 0
    elapsed: float = 0.0
    by_modality: Counter = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = [
            f"{self.files_indexed} files",
            f"{self.nodes_written} nodes",
            f"{self.edges_written} edges",
        ]
        if self.files_unchanged:
            bits.append(f"{self.files_unchanged} unchanged")
        if self.files_removed:
            bits.append(f"{self.files_removed} removed")
        if self.files_skipped:
            bits.append(f"{self.files_skipped} skipped")
        if self.errors:
            bits.append(f"{len(self.errors)} errors")
        return f"{', '.join(bits)} in {self.elapsed:.1f}s"


def build(
    root: Path,
    store: Store,
    graph: GraphStore,
    options: ExtractOptions | None = None,
    progress: ProgressFn | None = None,
    prune: bool = True,
) -> BuildStats:
    """Convert `root` into graph, incrementally."""
    started = time.time()
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        raise NotADirectoryError(f"{root} is a file — pass the folder that contains it")

    options = options or ExtractOptions()
    stats = BuildStats(root=str(root))
    source_id = store.add_source(root)
    known = store.files_for_source(source_id)
    seen: set[str] = set()

    # The folder itself, so every document has an ancestor chain to walk up.
    root_uid = folder_uid(".")
    graph.upsert_node(
        Node(
            uid=root_uid,
            kind=NodeKind.FOLDER,
            name=root.name or str(root),
            body=f"root folder {root.name}",
            props={"path": ".", "absolute": str(root), "root": True},
        )
    )
    folders_done: set[str] = {root_uid}

    for path in discover(root, options):
        stats.files_seen += 1
        key = str(path)
        seen.add(key)

        try:
            stat = path.stat()
        except OSError as exc:
            stats.errors.append(f"{path.name}: {exc}")
            continue

        existing = known.get(key)
        if existing and existing.mtime == stat.st_mtime and existing.size == stat.st_size:
            stats.files_unchanged += 1
            continue

        digest = _digest(path)
        if digest is None:
            stats.errors.append(f"{path.name}: unreadable")
            continue

        if existing and existing.digest == digest:
            # Touched but not modified — refresh stat data so the cheap check
            # catches it next time, and leave the graph alone.
            store.touch_file(existing.id, stat.st_mtime, stat.st_size)
            stats.files_unchanged += 1
            continue

        rel = _relative(path, root)
        file_id = store.upsert_file(source_id, key, stat.st_mtime, stat.st_size, digest)
        # Everything this file previously contributed goes before it is re-read,
        # so a shrinking document cannot leave stale pages behind.
        graph.clear_file(file_id)

        _ensure_folders(graph, rel, folders_done)
        _index_file(graph, path, rel, digest, stat.st_size, file_id, options, stats)

        stats.files_indexed += 1
        if progress and stats.files_indexed % 10 == 0:
            progress(f"  {stats.files_indexed} files → {graph.stats.nodes_written} nodes")

    if prune:
        for path_str, row in known.items():
            if path_str not in seen:
                graph.clear_file(row.id)
                store.delete_file(row.id)
                stats.files_removed += 1

    stats.edges_dropped = graph.resolve_pending()
    stats.orphans_collected = graph.gc_orphans() + graph.prune_empty_folders()
    store.mark_source_indexed(source_id)
    graph.commit()

    stats.nodes_written = graph.stats.nodes_written
    stats.edges_written = graph.stats.edges_written
    stats.elapsed = time.time() - started
    return stats


# -- per-file work ---------------------------------------------------------


def _index_file(
    graph: GraphStore,
    path: Path,
    rel: str,
    digest: str,
    size: int,
    file_id: int,
    options: ExtractOptions,
    stats: BuildStats,
) -> None:
    """Turn one file into nodes and edges."""
    doc_uid = document_uid(rel)
    extractor = for_path(path)
    modality = extractor.modality if extractor else Modality.UNKNOWN

    extraction: Extraction | None = None
    if extractor is not None:
        ctx = ExtractContext(
            path=path,
            rel_path=rel,
            doc_uid=doc_uid,
            digest=digest,
            size=size,
            options=options,
            cached=_cache_for(graph, digest),
        )
        try:
            with quiet():
                extraction = extractor.extract(ctx)
        except MissingDependency as exc:
            stats.warnings.append(f"{rel}: {exc}")
        except Exception as exc:
            # An extractor bug or an exotic malformed file must not end the run.
            stats.errors.append(f"{rel}: {type(exc).__name__}: {exc}")

    if extraction is not None:
        modality = extraction.modality

    # Extensions lie, and real corpora are full of files whose bytes disagree
    # with their name. Sniffing only when something went wrong keeps the cost
    # off the happy path while turning "File is not a zip file" into an
    # explanation the reader can act on.
    detected = sniff(path)
    mismatch = mislabelled(path)
    failed = extraction is None or not extraction.nodes

    if extractor is None:
        # A file no extractor claims still becomes a node — the folder listing
        # should be complete — but silence here is the wrong answer. On a real
        # archive a sixth of the files can be legacy Office formats, and a user
        # who is not told will assume they were indexed.
        reason = explain(path) or f"no extractor handles {path.suffix or 'this file type'}"
        stats.unreadable[path.suffix.lower() or "(none)"] += 1
        stats.warnings.append(f"{rel}: {reason}")
        unreadable_note = reason
    else:
        unreadable_note = None
    if mismatch and (failed or extraction is None or extraction.warnings):
        note = explain(path) or mismatch
        if extraction is not None:
            extraction.warnings = [f"{w} ({note})" for w in extraction.warnings] or [note]
        else:
            stats.warnings.append(f"{rel}: {note}")

    if extraction is not None:
        for warning in extraction.warnings:
            stats.warnings.append(f"{rel}: {warning}")

    stats.by_modality[str(modality)] += 1

    props: dict[str, Any] = {
        "path": rel,
        "name": path.name,
        "extension": path.suffix.lower(),
        "modality": str(modality),
        "size": size,
        "digest": digest,
    }
    if detected:
        props["detected_format"] = detected
    if unreadable_note:
        props["unreadable"] = unreadable_note
    if mismatch:
        # Recorded on the node, not just logged, so a query can find every file
        # in the folder whose name misrepresents it.
        props["mislabelled"] = mismatch
    if extraction:
        props.update(extraction.props)
    if extraction and extraction.warnings:
        props["warnings"] = extraction.warnings

    title = (extraction.title if extraction and extraction.title else path.stem) or path.name
    # The document body is a short synopsis rather than the full text: the text
    # already lives in the chunks below it, and duplicating it here would double
    # the index and let one long file dominate every search.
    body = _document_body(title, rel, props, extraction)

    graph.upsert_node(
        Node(uid=doc_uid, kind=NodeKind.DOCUMENT, name=title, body=body, props=props),
        owner_file_id=file_id,
    )
    graph.add_edge(
        Edge(folder_uid(str(Path(rel).parent)), doc_uid, EdgeKind.CONTAINS), owner_file_id=file_id
    )

    if extraction is None:
        return

    for node in extraction.nodes:
        graph.upsert_node(node, owner_file_id=file_id)
    for edge in extraction.edges:
        graph.add_edge(edge, owner_file_id=file_id)

    stats.entities_linked += _link_entities(graph, extraction, doc_uid, file_id)


def _link_entities(
    graph: GraphStore, extraction: Extraction, doc_uid: str, file_id: int
) -> int:
    """Attach literal entities to the chunks that mention them.

    Mentions hang off chunks, not documents, so an answer can point at the exact
    span — and so an entity's neighbourhood is a set of passages rather than a
    set of whole files.
    """
    linked = 0
    for node in extraction.nodes:
        if node.kind != NodeKind.CHUNK or not node.body:
            continue
        for mention in entity_scan.extract(node.body):
            uid = entity_uid(mention.type, mention.value)
            graph.upsert_node(
                Node(
                    uid=uid,
                    kind=NodeKind.ENTITY,
                    name=mention.display,
                    body=f"{mention.type}: {mention.display}",
                    props={"entity_type": str(mention.type), "value": mention.value},
                )
                # No owner: entities are shared across every file that names them.
            )
            graph.add_edge(
                Edge(node.uid, uid, EdgeKind.MENTIONS, provenance=Provenance.EXTRACTED),
                owner_file_id=file_id,
            )
            linked += 1
    return linked


def _document_body(
    title: str, rel: str, props: dict[str, Any], extraction: Extraction | None
) -> str:
    """A searchable synopsis of a document."""
    parts = [title, rel.replace("/", " ").replace("_", " ").replace("-", " ")]
    for key in ("author", "subject", "keywords", "artist", "album", "camera_model"):
        if props.get(key):
            parts.append(str(props[key]))
    if extraction and extraction.text:
        parts.append(extraction.text[:1000])
    return "\n".join(p for p in parts if p)


def _cache_for(graph: GraphStore, digest: str) -> Callable[[str, str, Callable[[], Any]], Any]:
    """Bind the derived-work cache to one file's digest."""

    def cached(producer: str, version: str, compute: Callable[[], Any]) -> Any:
        hit = graph.cache_get(digest, producer, version)
        if hit is not None:
            return hit
        value = compute()
        if value is not None:
            graph.cache_put(digest, producer, version, value)
        return value

    return cached


def _ensure_folders(graph: GraphStore, rel: str, done: set[str]) -> None:
    """Create the FOLDER chain above a file, once per folder per build."""
    parent = Path(rel).parent
    parts = [p for p in parent.parts if p not in (".", "")]
    trail = ""
    previous = folder_uid(".")
    for part in parts:
        trail = f"{trail}/{part}" if trail else part
        uid = folder_uid(trail)
        if uid not in done:
            graph.upsert_node(
                Node(
                    uid=uid,
                    kind=NodeKind.FOLDER,
                    name=part,
                    body=trail.replace("/", " ").replace("_", " ").replace("-", " "),
                    props={"path": trail},
                )
            )
            graph.add_edge(Edge(previous, uid, EdgeKind.CONTAINS))
            done.add(uid)
        previous = uid


# -- walking ---------------------------------------------------------------


def discover(root: Path, options: ExtractOptions) -> Iterator[Path]:
    """Yield every candidate file under `root`.

    Unlike the text-only indexer this does *not* filter by extension — binary
    files are the point here. Filtering happens later, when no extractor claims
    a path.
    """
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            d for d in dirnames if d not in DEFAULT_IGNORE_DIRS and not d.startswith(".")
        )
        for name in sorted(filenames):
            if name in IGNORED_NAMES or name.startswith("._"):
                continue
            path = Path(dirpath) / name
            if path.is_symlink() or not path.is_file():
                continue
            try:
                if path.stat().st_size > options.max_file_bytes:
                    continue
            except OSError:
                continue
            yield path


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.name


def _digest(path: Path) -> str | None:
    """SHA-256 of the file, read in blocks so a 4GB video does not need 4GB."""
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(DIGEST_BLOCK):
                hasher.update(block)
    except OSError:
        return None
    return hasher.hexdigest()
