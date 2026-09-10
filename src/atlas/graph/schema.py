"""SQL for the graph tables.

The graph lives in the same SQLite file as the rest of Atlas, on top of the
`sources` and `files` tables that already track what has been indexed and
whether it changed. Reusing them is what makes incremental rebuilds work: a
file's mtime/size/digest is already the trigger, so the graph only has to
answer "what did this file put in the graph, and how do I take it back out?"

That question is answered by `owner_file_id`, and the rule it encodes is the
one non-obvious thing in this file:

    Structural nodes are OWNED by the file they came from. Entity nodes are
    NOT owned by anything, because they are shared.

A page belongs to exactly one PDF; delete the PDF and the page should go. But
the entity `Acme Corp` may be mentioned by two hundred files, and deleting one
of them must not delete the entity. So `MENTIONS` edges are owned by the file
that produced them, entity nodes are not owned at all, and an entity that ends
up with no edges is swept later by `gc_orphans()`. Getting this backwards
produces a graph that silently loses shared nodes on reindex.
"""

from __future__ import annotations

# Matches the tokenizer used by the lexical chunk index so a query behaves the
# same whichever table answers it.
TOKENIZER = "porter unicode61 remove_diacritics 2"

GRAPH_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS nodes (
    id            INTEGER PRIMARY KEY,
    uid           TEXT NOT NULL UNIQUE,
    kind          TEXT NOT NULL,
    name          TEXT NOT NULL,
    body          TEXT NOT NULL DEFAULT '',
    props         TEXT NOT NULL DEFAULT '{{}}',
    -- NULL for shared nodes (entities, topics). Set for anything that lives
    -- inside exactly one file and must die with it.
    owner_file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS nodes_by_kind  ON nodes(kind);
CREATE INDEX IF NOT EXISTS nodes_by_owner ON nodes(owner_file_id);

-- Plain (not external-content) FTS, kept in sync by hand in GraphStore. Same
-- reasoning as the chunk index: contentless_delete needs SQLite 3.43+, and
-- Atlas supports older runtimes.
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts
    USING fts5(body, tokenize="{TOKENIZER}");

CREATE TABLE IF NOT EXISTS edges (
    id            INTEGER PRIMARY KEY,
    src_id        INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    dst_id        INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,
    weight        REAL NOT NULL DEFAULT 1.0,
    provenance    TEXT NOT NULL DEFAULT 'extracted',
    props         TEXT NOT NULL DEFAULT '{{}}',
    owner_file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    created_at    REAL NOT NULL
);

-- One edge per (src, dst, kind): re-running a build re-asserts rather than
-- duplicating. Two files claiming the same relationship collapse to one edge,
-- which is correct — the relationship exists once.
CREATE UNIQUE INDEX IF NOT EXISTS edges_unique ON edges(src_id, dst_id, kind);
CREATE INDEX IF NOT EXISTS edges_out   ON edges(src_id, kind);
CREATE INDEX IF NOT EXISTS edges_in    ON edges(dst_id, kind);
CREATE INDEX IF NOT EXISTS edges_owner ON edges(owner_file_id);
-- `counts()` groups by kind on every call (the agent's overview tool uses
-- it); without this it degenerates to a full scan of the edge table.
-- Measured on a 100MB / 420k-edge corpus: 81ms -> 18ms, costing 7.5MB of
-- index. A deliberate trade, and the cheapest one to reverse if the disk
-- matters more than the latency.
CREATE INDEX IF NOT EXISTS edges_by_kind ON edges(kind);

-- Content-addressed cache for anything expensive to produce: Whisper
-- transcripts, OCR text, LLM enrichment. Keyed by the digest of the input plus
-- the name and version of the producer, so re-indexing an unchanged file never
-- re-runs a model, and changing a producer's version invalidates just its own
-- entries.
CREATE TABLE IF NOT EXISTS derived_cache (
    digest     TEXT NOT NULL,
    producer   TEXT NOT NULL,
    version    TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (digest, producer, version)
);
"""
