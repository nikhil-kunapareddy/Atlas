from __future__ import annotations

import os
import subprocess

from atlas.indexer import discover_files, index_path


def test_index_populates_files_and_chunks(project_store, repo):
    stats = index_path(project_store, repo)

    assert stats.files_indexed >= 4
    counts = project_store.counts()
    assert counts["files"] >= 4
    assert counts["chunks"] > 0
    assert counts["sources"] == 1


def test_binary_files_are_not_indexed(project_store, repo):
    index_path(project_store, repo)
    paths = [r["path"] for r in project_store.conn.execute("SELECT path FROM files")]
    assert not any(p.endswith(".png") for p in paths)


def test_reindex_is_incremental(project_store, repo):
    first = index_path(project_store, repo)
    assert first.files_indexed > 0

    second = index_path(project_store, repo)
    assert second.files_indexed == 0
    assert second.files_unchanged == first.files_indexed
    assert second.chunks_written == 0


def test_modified_file_is_reindexed(project_store, repo):
    index_path(project_store, repo)

    target = repo / "src" / "billing.py"
    target.write_text("def compute_invoice_total(items):\n    return Decimal(0)\n")
    os.utime(target, (target.stat().st_atime, target.stat().st_mtime + 10))

    stats = index_path(project_store, repo)
    assert stats.files_indexed == 1

    stored = project_store.conn.execute(
        "SELECT c.text FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.path LIKE ?",
        ("%billing.py",),
    ).fetchall()
    assert len(stored) == 1
    assert "Decimal(0)" in stored[0]["text"]
    assert "item.price" not in stored[0]["text"]


def test_touched_but_identical_file_is_not_rechunked(project_store, repo):
    index_path(project_store, repo)
    target = repo / "src" / "auth.py"
    os.utime(target, (target.stat().st_atime, target.stat().st_mtime + 100))

    stats = index_path(project_store, repo)
    # Digest matches, so it counts as unchanged and writes no chunks.
    assert stats.files_indexed == 0
    assert stats.chunks_written == 0


def test_deleted_files_are_pruned(project_store, repo):
    index_path(project_store, repo)
    (repo / "src" / "billing.py").unlink()

    stats = index_path(project_store, repo)
    assert stats.files_removed == 1
    paths = [r["path"] for r in project_store.conn.execute("SELECT path FROM files")]
    assert not any(p.endswith("billing.py") for p in paths)


def test_deleted_file_chunks_leave_no_orphans(project_store, repo):
    index_path(project_store, repo)
    (repo / "src" / "billing.py").unlink()
    index_path(project_store, repo)

    orphans = project_store.conn.execute(
        "SELECT count(*) FROM chunks WHERE file_id NOT IN (SELECT id FROM files)"
    ).fetchone()[0]
    assert orphans == 0

    # The FTS side must be cleaned too, or deleted content stays searchable.
    fts_rows = project_store.conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
    chunk_rows = project_store.counts()["chunks"]
    assert fts_rows == chunk_rows


def test_removing_a_source_clears_its_files(project_store, repo):
    index_path(project_store, repo)
    assert project_store.remove_source(repo) is True
    counts = project_store.counts()
    assert counts["files"] == 0
    assert counts["chunks"] == 0


def test_default_ignore_dirs_are_skipped(project_store, tmp_path):
    root = tmp_path / "proj"
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / "node_modules" / "pkg" / "index.js").write_text("module.exports = 1")
    (root / "app.js").write_text("console.log('hi')")

    index_path(project_store, root)
    paths = [r["path"] for r in project_store.conn.execute("SELECT path FROM files")]
    assert any(p.endswith("app.js") for p in paths)
    assert not any("node_modules" in p for p in paths)


def test_gitignore_is_respected(tmp_path, project_store):
    root = tmp_path / "gitrepo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".gitignore").write_text("secret.txt\n")
    (root / "secret.txt").write_text("do not index me")
    (root / "public.txt").write_text("index me please")

    found = {p.name for p in discover_files(root)}
    assert "public.txt" in found
    assert "secret.txt" not in found
