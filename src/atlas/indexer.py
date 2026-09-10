"""Walk folders and keep the chunk index in sync with what's on disk.

Reindexing is incremental: unchanged files are recognised by (mtime, size) and
skipped without a read, changed-but-identical files (touched, reformatted back)
are caught by digest, and files that vanished are dropped along with their
chunks. Running `atlas index` twice in a row should be nearly free.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import chunk_text, read_text, should_index
from .store import Store

DEFAULT_IGNORE_DIRS = frozenset(
    [".git", ".hg", ".svn", ".atlas", "node_modules", "bower_components", "vendor", ".venv", "venv", "env", ".env", "virtualenv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox", "dist", "build", "out", "target", ".next", ".nuxt", ".parcel-cache", ".turbo", ".svelte-kit", ".idea", ".vscode", ".gradle", ".terraform", "coverage", "htmlcov", ".cache"]
)

ProgressFn = Callable[[str], None]


@dataclass
class IndexStats:
    files_seen: int = 0
    files_indexed: int = 0
    files_unchanged: int = 0
    files_skipped: int = 0
    files_removed: int = 0
    chunks_written: int = 0
    elapsed: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = [
            f"{self.files_indexed} indexed",
            f"{self.files_unchanged} unchanged",
            f"{self.chunks_written} chunks",
        ]
        if self.files_removed:
            bits.append(f"{self.files_removed} removed")
        if self.files_skipped:
            bits.append(f"{self.files_skipped} skipped")
        if self.errors:
            bits.append(f"{len(self.errors)} errors")
        return f"{', '.join(bits)} in {self.elapsed:.1f}s"


def _git_tracked_files(root: Path) -> list[Path] | None:
    """Ask git for the file list so .gitignore is respected for free."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=str(root),
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    names = result.stdout.decode("utf-8", "replace").split("\0")
    return [root / n for n in names if n]


def _walk_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in DEFAULT_IGNORE_DIRS and not d.startswith(".atlas")
        ]
        for name in filenames:
            yield Path(dirpath) / name


def discover_files(root: Path) -> Iterator[Path]:
    """Yield candidate files under `root`, respecting .gitignore when possible."""
    root = root.resolve()
    if root.is_file():
        yield root
        return

    tracked = _git_tracked_files(root)
    candidates: Iterator[Path]
    if tracked is not None:
        candidates = iter(tracked)
    else:
        candidates = _walk_files(root)

    for path in candidates:
        parts = set(path.parts)
        if parts & DEFAULT_IGNORE_DIRS:
            continue
        if not path.is_file() or path.is_symlink():
            continue
        yield path


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def index_path(
    store: Store,
    target: Path,
    progress: ProgressFn | None = None,
    prune: bool = True,
) -> IndexStats:
    """Index (or reindex) `target` into `store`."""
    started = time.time()
    target = Path(target).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(target)

    stats = IndexStats()
    source_id = store.add_source(target)
    known = store.files_for_source(source_id)
    seen_paths: set[str] = set()

    for path in discover_files(target):
        stats.files_seen += 1
        key = str(path)
        seen_paths.add(key)

        try:
            stat = path.stat()
        except OSError as exc:
            stats.errors.append(f"{path}: {exc}")
            continue

        existing = known.get(key)
        if existing and existing.mtime == stat.st_mtime and existing.size == stat.st_size:
            stats.files_unchanged += 1
            continue

        if not should_index(path):
            stats.files_skipped += 1
            # A file that used to be indexable and no longer is (grew past the
            # size cap, say) must not leave stale chunks behind.
            if existing:
                store.delete_file(existing.id)
                stats.files_removed += 1
            continue

        digest = _digest(path)
        if digest is None:
            stats.errors.append(f"{path}: unreadable")
            continue

        if existing and existing.digest == digest:
            store.touch_file(existing.id, stat.st_mtime, stat.st_size)
            stats.files_unchanged += 1
            continue

        text = read_text(path)
        if text is None:
            stats.files_skipped += 1
            if existing:
                store.delete_file(existing.id)
                stats.files_removed += 1
            continue

        try:
            rel = str(path.relative_to(target)) if target.is_dir() else path.name
        except ValueError:
            rel = path.name

        chunks = chunk_text(text, rel)
        file_id = store.upsert_file(source_id, key, stat.st_mtime, stat.st_size, digest)
        store.delete_file_chunks(file_id)
        stats.chunks_written += store.add_chunks(file_id, chunks)
        stats.files_indexed += 1

        if progress and stats.files_indexed % 25 == 0:
            progress(f"  {stats.files_indexed} files indexed…")

    if prune:
        for path_str, row in known.items():
            if path_str not in seen_paths:
                store.delete_file(row.id)
                stats.files_removed += 1

    store.mark_source_indexed(source_id)
    store.conn.commit()
    stats.elapsed = time.time() - started
    return stats


def reindex_all(store: Store, progress: ProgressFn | None = None) -> IndexStats:
    """Refresh every source already registered in this store."""
    total = IndexStats()
    started = time.time()
    for row in store.sources():
        path = Path(row["path"])
        if not path.exists():
            store.remove_source(path)
            continue
        if progress:
            progress(f"indexing {path}")
        stats = index_path(store, path, progress=progress)
        total.files_seen += stats.files_seen
        total.files_indexed += stats.files_indexed
        total.files_unchanged += stats.files_unchanged
        total.files_skipped += stats.files_skipped
        total.files_removed += stats.files_removed
        total.chunks_written += stats.chunks_written
        total.errors.extend(stats.errors)
    total.elapsed = time.time() - started
    return total
