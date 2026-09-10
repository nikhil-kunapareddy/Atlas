"""SQLite storage: indexed file chunks plus agent-written facts.

Both scopes use the same schema, so one `Store` class serves either. The FTS5
tables are plain (not external-content) and kept in sync by hand: every write
path here owns both sides, and manual sync avoids depending on
`contentless_delete`, which needs SQLite 3.43+.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import (
    GLOBAL_SCOPE,
    PROJECT_SCOPE,
    find_project_root,
    global_db_path,
    project_db_path,
)

SCHEMA_VERSION = 1

# Porter stemming earns its keep on prose ("indexing" -> "index"); unicode61
# splits identifiers on underscores and punctuation, which is what we want for
# code. camelCase is handled by the indexer, which appends split forms.
TOKENIZER = "porter unicode61 remove_diacritics 2"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id              INTEGER PRIMARY KEY,
    path            TEXT NOT NULL UNIQUE,
    added_at        REAL NOT NULL,
    last_indexed_at REAL
);

CREATE TABLE IF NOT EXISTS files (
    id         INTEGER PRIMARY KEY,
    source_id  INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    path       TEXT NOT NULL UNIQUE,
    mtime      REAL NOT NULL,
    size       INTEGER NOT NULL,
    digest     TEXT NOT NULL,
    indexed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS files_by_source ON files(source_id);

CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    ord        INTEGER NOT NULL,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    heading    TEXT,
    text       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_file ON chunks(file_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(body, tokenize="{TOKENIZER}");

CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY,
    text       TEXT NOT NULL,
    tags       TEXT NOT NULL DEFAULT '',
    origin     TEXT NOT NULL DEFAULT 'agent',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    hits       INTEGER NOT NULL DEFAULT 0,
    last_hit_at REAL,
    pinned     INTEGER NOT NULL DEFAULT 0
);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(body, tokenize="{TOKENIZER}");

-- Populated only when the optional embedding extra is installed.
CREATE TABLE IF NOT EXISTS vectors (
    kind    TEXT NOT NULL,
    item_id INTEGER NOT NULL,
    model   TEXT NOT NULL,
    dim     INTEGER NOT NULL,
    vec     BLOB NOT NULL,
    PRIMARY KEY (kind, item_id)
);
"""


@dataclass(frozen=True)
class FileRow:
    id: int
    path: str
    mtime: float
    size: int
    digest: str


@dataclass(frozen=True)
class Fact:
    id: int
    text: str
    tags: str
    origin: str
    created_at: float
    updated_at: float
    hits: int
    pinned: bool
    scope: str = ""

    @property
    def tag_list(self) -> list[str]:
        return [t for t in self.tags.split(",") if t]


class Store:
    """One SQLite database, either the project store or the global store."""

    def __init__(self, path: Path, scope: str):
        self.path = Path(path)
        self.scope = scope
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self._migrate()

    # -- lifecycle ---------------------------------------------------------

    def _migrate(self) -> None:
        self.conn.executescript(SCHEMA)
        current = self.get_meta("schema_version")
        if current is None:
            self.set_meta("schema_version", str(SCHEMA_VERSION))
        elif int(current) > SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.path} was written by a newer Atlas "
                f"(schema v{current} > v{SCHEMA_VERSION}). Upgrade Atlas."
            )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- meta --------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # -- sources -----------------------------------------------------------

    def add_source(self, path: Path) -> int:
        resolved = str(Path(path).resolve())
        self.conn.execute(
            "INSERT INTO sources(path, added_at) VALUES(?, ?) "
            "ON CONFLICT(path) DO NOTHING",
            (resolved, time.time()),
        )
        row = self.conn.execute("SELECT id FROM sources WHERE path = ?", (resolved,)).fetchone()
        return int(row["id"])

    def mark_source_indexed(self, source_id: int) -> None:
        self.conn.execute(
            "UPDATE sources SET last_indexed_at = ? WHERE id = ?", (time.time(), source_id)
        )

    def sources(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM sources ORDER BY path"))

    def remove_source(self, path: Path) -> bool:
        """Drop a source and everything indexed under it."""
        resolved = str(Path(path).resolve())
        row = self.conn.execute("SELECT id FROM sources WHERE path = ?", (resolved,)).fetchone()
        if row is None:
            return False
        source_id = int(row["id"])
        file_ids = [
            int(r["id"])
            for r in self.conn.execute("SELECT id FROM files WHERE source_id = ?", (source_id,))
        ]
        for file_id in file_ids:
            self.delete_file_chunks(file_id)
        self.conn.execute("DELETE FROM files WHERE source_id = ?", (source_id,))
        self.conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        self.conn.commit()
        return True

    # -- files and chunks --------------------------------------------------

    def files_for_source(self, source_id: int) -> dict[str, FileRow]:
        rows = self.conn.execute(
            "SELECT id, path, mtime, size, digest FROM files WHERE source_id = ?",
            (source_id,),
        )
        return {
            r["path"]: FileRow(r["id"], r["path"], r["mtime"], r["size"], r["digest"])
            for r in rows
        }

    def upsert_file(
        self, source_id: int, path: str, mtime: float, size: int, digest: str
    ) -> int:
        self.conn.execute(
            """
            INSERT INTO files(source_id, path, mtime, size, digest, indexed_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                source_id = excluded.source_id,
                mtime     = excluded.mtime,
                size      = excluded.size,
                digest    = excluded.digest,
                indexed_at = excluded.indexed_at
            """,
            (source_id, path, mtime, size, digest, time.time()),
        )
        row = self.conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
        return int(row["id"])

    def touch_file(self, file_id: int, mtime: float, size: int) -> None:
        """Content is unchanged; just refresh stat data so we skip it next time."""
        self.conn.execute(
            "UPDATE files SET mtime = ?, size = ?, indexed_at = ? WHERE id = ?",
            (mtime, size, time.time(), file_id),
        )

    def delete_file_chunks(self, file_id: int) -> None:
        ids = [
            int(r["id"])
            for r in self.conn.execute("SELECT id FROM chunks WHERE file_id = ?", (file_id,))
        ]
        if not ids:
            return
        self.conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(i,) for i in ids])
        self.conn.executemany(
            "DELETE FROM vectors WHERE kind = 'chunk' AND item_id = ?", [(i,) for i in ids]
        )
        self.conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

    def delete_file(self, file_id: int) -> None:
        self.delete_file_chunks(file_id)
        self.conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    def add_chunks(self, file_id: int, chunks: Sequence[ChunkInput]) -> int:
        """Insert chunks and their FTS rows. Returns the number written."""
        for chunk in chunks:
            cur = self.conn.execute(
                "INSERT INTO chunks(file_id, ord, start_line, end_line, heading, text) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (file_id, chunk.ord, chunk.start_line, chunk.end_line, chunk.heading, chunk.text),
            )
            self.conn.execute(
                "INSERT INTO chunks_fts(rowid, body) VALUES(?, ?)",
                (cur.lastrowid, chunk.search_body),
            )
        return len(chunks)

    # -- facts -------------------------------------------------------------

    def add_fact(self, text: str, tags: Iterable[str] = (), origin: str = "agent",
                 pinned: bool = False) -> int:
        now = time.time()
        tag_str = ",".join(sorted({t.strip() for t in tags if t.strip()}))
        cur = self.conn.execute(
            "INSERT INTO facts(text, tags, origin, created_at, updated_at, pinned) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (text, tag_str, origin, now, now, int(pinned)),
        )
        fact_id = int(cur.lastrowid)
        self.conn.execute(
            "INSERT INTO facts_fts(rowid, body) VALUES(?, ?)", (fact_id, _fact_body(text, tag_str))
        )
        self.conn.commit()
        return fact_id

    def update_fact(self, fact_id: int, text: str, tags: Iterable[str] | None = None) -> None:
        row = self.conn.execute("SELECT tags FROM facts WHERE id = ?", (fact_id,)).fetchone()
        if row is None:
            raise KeyError(fact_id)
        tag_str = (
            ",".join(sorted({t.strip() for t in tags if t.strip()}))
            if tags is not None
            else row["tags"]
        )
        self.conn.execute(
            "UPDATE facts SET text = ?, tags = ?, updated_at = ? WHERE id = ?",
            (text, tag_str, time.time(), fact_id),
        )
        self.conn.execute("DELETE FROM facts_fts WHERE rowid = ?", (fact_id,))
        self.conn.execute(
            "INSERT INTO facts_fts(rowid, body) VALUES(?, ?)", (fact_id, _fact_body(text, tag_str))
        )
        self.conn.execute("DELETE FROM vectors WHERE kind = 'fact' AND item_id = ?", (fact_id,))
        self.conn.commit()

    def delete_fact(self, fact_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        if cur.rowcount == 0:
            return False
        self.conn.execute("DELETE FROM facts_fts WHERE rowid = ?", (fact_id,))
        self.conn.execute("DELETE FROM vectors WHERE kind = 'fact' AND item_id = ?", (fact_id,))
        self.conn.commit()
        return True

    def get_fact(self, fact_id: int) -> Fact | None:
        row = self.conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return _row_to_fact(row, self.scope) if row else None

    def all_facts(self, tag: str | None = None) -> list[Fact]:
        rows = self.conn.execute(
            "SELECT * FROM facts ORDER BY pinned DESC, updated_at DESC"
        ).fetchall()
        facts = [_row_to_fact(r, self.scope) for r in rows]
        if tag:
            facts = [f for f in facts if tag in f.tag_list]
        return facts

    def record_fact_hits(self, fact_ids: Iterable[int]) -> None:
        now = time.time()
        self.conn.executemany(
            "UPDATE facts SET hits = hits + 1, last_hit_at = ? WHERE id = ?",
            [(now, i) for i in fact_ids],
        )
        self.conn.commit()

    # -- stats -------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        def one(sql: str) -> int:
            return int(self.conn.execute(sql).fetchone()[0])

        return {
            "sources": one("SELECT count(*) FROM sources"),
            "files": one("SELECT count(*) FROM files"),
            "chunks": one("SELECT count(*) FROM chunks"),
            "facts": one("SELECT count(*) FROM facts"),
            "vectors": one("SELECT count(*) FROM vectors"),
        }


@dataclass(frozen=True)
class ChunkInput:
    ord: int
    start_line: int
    end_line: int
    heading: str | None
    text: str
    search_body: str


def _fact_body(text: str, tags: str) -> str:
    return f"{text}\n{tags.replace(',', ' ')}" if tags else text


def _row_to_fact(row: sqlite3.Row, scope: str) -> Fact:
    return Fact(
        id=int(row["id"]),
        text=row["text"],
        tags=row["tags"],
        origin=row["origin"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        hits=int(row["hits"]),
        pinned=bool(row["pinned"]),
        scope=scope,
    )


def open_store(scope: str, root: Path | None = None, create: bool = True) -> Store | None:
    """Open one scope's store, or None if the project scope has no project."""
    if scope == GLOBAL_SCOPE:
        return Store(global_db_path(), GLOBAL_SCOPE)
    if root is None:
        root = find_project_root()
    if root is None:
        return None
    path = project_db_path(root)
    if not create and not path.exists():
        return None
    return Store(path, PROJECT_SCOPE)


def open_stores(root: Path | None = None, create: bool = False) -> Iterator[Store]:
    """Yield project store (when there is one) then global, in rank order."""
    project = open_store(PROJECT_SCOPE, root, create=create)
    if project is not None:
        yield project
    yield Store(global_db_path(), GLOBAL_SCOPE)
