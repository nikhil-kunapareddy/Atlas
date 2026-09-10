"""Reading and writing the property graph.

`GraphStore` wraps a SQLite connection and offers two surfaces:

* a **write** surface used by the build pipeline — `upsert_node`, `add_edge`,
  `clear_file`, `gc_orphans`;
* a **read** surface used by the CLI, the MCP server, and the agent —
  `node`, `neighbors`, `search`, `shortest_path`, `subgraph`, `hubs`.

Writes are idempotent by `uid`. Running a build twice over an unchanged folder
produces the same graph, not two copies of it, and that property is what lets
`atlas <folder>` be safe to re-run at any time.

Edges are addressed by `uid`, not row id, so an extractor may emit an edge to a
node that a later extractor will create — a document referencing a sibling it
has not reached yet. Unresolved edges are parked and retried in
`resolve_pending()`, which the pipeline calls once every file has been read.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .model import (
    Edge,
    EdgeKind,
    Node,
    NodeKind,
)
from .schema import GRAPH_SCHEMA

# One query per this many frontier nodes, keeping the bound parameter count
# well under SQLite's limit.
FRONTIER_BATCH = 400
# Give up rather than traverse an entire dense graph looking for a path.
MAX_VISITED = 50_000


@dataclass
class _Side:
    """One half of a bidirectional search: what it has seen, and where next."""

    root: int
    frontier: list[int] = field(default_factory=list)
    parent: dict[int, int | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frontier = [self.root]
        self.parent = {self.root: None}


@dataclass(frozen=True)
class NodeRow:
    """A node as read back out of the database."""

    id: int
    uid: str
    kind: str
    name: str
    body: str
    props: dict[str, Any]
    owner_file_id: int | None = None

    @property
    def path(self) -> str | None:
        """The source file this node came from, when it has one."""
        return self.props.get("path")

    def preview(self, limit: int = 200) -> str:
        text = " ".join(self.body.split())
        return text[: limit - 1] + "…" if len(text) > limit else text


@dataclass(frozen=True)
class EdgeRow:
    id: int
    src_uid: str
    dst_uid: str
    kind: str
    weight: float
    provenance: str
    props: dict[str, Any]


@dataclass(frozen=True)
class Neighbor:
    """A node reached from another, plus the edge that got you there."""

    node: NodeRow
    edge_kind: str
    direction: str  # "out" or "in"
    weight: float
    provenance: str
    label: str | None = None


@dataclass
class WriteStats:
    nodes_written: int = 0
    nodes_updated: int = 0
    edges_written: int = 0
    edges_dropped: int = 0

    def merge(self, other: WriteStats) -> None:
        self.nodes_written += other.nodes_written
        self.nodes_updated += other.nodes_updated
        self.edges_written += other.edges_written
        self.edges_dropped += other.edges_dropped


class GraphStore:
    """The graph half of an Atlas database."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(GRAPH_SCHEMA)
        self.conn.commit()
        self.stats = WriteStats()
        # uid -> row id. Saves a SELECT per edge endpoint during a build; the
        # graph for one folder comfortably fits, and it is dropped with the
        # store.
        self._uid_cache: dict[str, int] = {}
        self._pending_edges: list[tuple[Edge, int | None]] = []

    # -- writing -----------------------------------------------------------

    def upsert_node(self, node: Node, owner_file_id: int | None = None) -> int:
        """Insert `node`, or merge it into the existing node with that uid.

        Merging matters for entities: the tenth file to mention `Acme Corp`
        should enrich that node, not clobber what the first nine recorded. So
        props are merged rather than replaced, and the longest body wins — for
        an entity the body is a description, and a longer one is strictly more
        useful than a shorter one.
        """
        now = time.time()
        props_json = json.dumps(node.props, sort_keys=True, default=str)
        existing = self.conn.execute(
            "SELECT id, body, props FROM nodes WHERE uid = ?", (node.uid,)
        ).fetchone()

        if existing is None:
            cur = self.conn.execute(
                "INSERT INTO nodes(uid, kind, name, body, props, owner_file_id,"
                " created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    node.uid,
                    str(node.kind),
                    node.name,
                    node.body,
                    props_json,
                    owner_file_id,
                    now,
                    now,
                ),
            )
            node_id = int(cur.lastrowid or 0)
            self.conn.execute(
                "INSERT INTO nodes_fts(rowid, body) VALUES(?, ?)",
                (node_id, _search_body(node)),
            )
            self.stats.nodes_written += 1
        else:
            node_id = int(existing["id"])
            merged_props = {**json.loads(existing["props"] or "{}"), **node.props}
            body = node.body if len(node.body) >= len(existing["body"] or "") else existing["body"]
            self.conn.execute(
                "UPDATE nodes SET kind = ?, name = ?, body = ?, props = ?,"
                " owner_file_id = COALESCE(?, owner_file_id), updated_at = ?"
                " WHERE id = ?",
                (
                    str(node.kind),
                    node.name,
                    body,
                    json.dumps(merged_props, sort_keys=True, default=str),
                    owner_file_id,
                    now,
                    node_id,
                ),
            )
            self.conn.execute("DELETE FROM nodes_fts WHERE rowid = ?", (node_id,))
            self.conn.execute(
                "INSERT INTO nodes_fts(rowid, body) VALUES(?, ?)",
                (node_id, _search_body(Node(node.uid, NodeKind(node.kind), node.name, body, merged_props))),
            )
            self.stats.nodes_updated += 1

        self._uid_cache[node.uid] = node_id
        return node_id

    def add_edge(self, edge: Edge, owner_file_id: int | None = None) -> bool:
        """Write an edge, or park it if either endpoint does not exist yet.

        Returns True when the edge was written now. A False return is not a
        failure — `resolve_pending()` will try again once the build has seen
        every file.
        """
        src_id = self._lookup(edge.src)
        dst_id = self._lookup(edge.dst)
        if src_id is None or dst_id is None:
            self._pending_edges.append((edge, owner_file_id))
            return False
        self._write_edge(src_id, dst_id, edge, owner_file_id)
        return True

    def _write_edge(
        self, src_id: int, dst_id: int, edge: Edge, owner_file_id: int | None
    ) -> None:
        # A re-asserted edge keeps the strongest claim about itself: the higher
        # weight, and the more trustworthy provenance. Otherwise a later LLM
        # guess could quietly downgrade a fact read straight off the bytes.
        self.conn.execute(
            """
            INSERT INTO edges(src_id, dst_id, kind, weight, provenance, props,
                              owner_file_id, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET
                weight     = MAX(edges.weight, excluded.weight),
                provenance = CASE
                    WHEN edges.provenance = 'extracted' THEN edges.provenance
                    WHEN excluded.provenance = 'extracted' THEN excluded.provenance
                    ELSE edges.provenance
                END,
                props = excluded.props
            """,
            (
                src_id,
                dst_id,
                str(edge.kind),
                edge.weight,
                str(edge.provenance),
                json.dumps(edge.props, sort_keys=True, default=str),
                owner_file_id,
                time.time(),
            ),
        )
        self.stats.edges_written += 1

    def resolve_pending(self) -> int:
        """Retry parked edges. Returns how many were dropped as unresolvable."""
        dropped = 0
        for edge, owner in self._pending_edges:
            src_id = self._lookup(edge.src)
            dst_id = self._lookup(edge.dst)
            if src_id is None or dst_id is None:
                dropped += 1
                continue
            self._write_edge(src_id, dst_id, edge, owner)
        self._pending_edges.clear()
        self.stats.edges_dropped += dropped
        return dropped

    def _lookup(self, uid: str) -> int | None:
        cached = self._uid_cache.get(uid)
        if cached is not None:
            return cached
        row = self.conn.execute("SELECT id FROM nodes WHERE uid = ?", (uid,)).fetchone()
        if row is None:
            return None
        self._uid_cache[uid] = int(row["id"])
        return int(row["id"])

    def clear_file(self, file_id: int) -> None:
        """Remove everything a file contributed, before re-reading it.

        Only owned rows go: shared entity nodes survive, and lose just the
        edges this file asserted. `gc_orphans()` collects any that are left
        with nothing pointing at them.
        """
        rows = self.conn.execute(
            "SELECT id FROM nodes WHERE owner_file_id = ?", (file_id,)
        ).fetchall()
        ids = [int(r["id"]) for r in rows]
        if ids:
            self.conn.executemany("DELETE FROM nodes_fts WHERE rowid = ?", [(i,) for i in ids])
            for uid_row in self.conn.execute(
                "SELECT uid FROM nodes WHERE owner_file_id = ?", (file_id,)
            ):
                self._uid_cache.pop(uid_row["uid"], None)
        self.conn.execute("DELETE FROM edges WHERE owner_file_id = ?", (file_id,))
        self.conn.execute("DELETE FROM nodes WHERE owner_file_id = ?", (file_id,))

    def gc_orphans(self) -> int:
        """Delete shared nodes that nothing references any more."""
        rows = self.conn.execute(
            """
            SELECT n.id FROM nodes n
            WHERE n.owner_file_id IS NULL
              AND n.kind IN (?, ?)
              AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.src_id = n.id OR e.dst_id = n.id)
            """,
            (str(NodeKind.ENTITY), str(NodeKind.TOPIC)),
        ).fetchall()
        ids = [int(r["id"]) for r in rows]
        if not ids:
            return 0
        self.conn.executemany("DELETE FROM nodes_fts WHERE rowid = ?", [(i,) for i in ids])
        self.conn.executemany("DELETE FROM nodes WHERE id = ?", [(i,) for i in ids])
        self._uid_cache.clear()
        return len(ids)

    def prune_empty_folders(self) -> int:
        """Drop FOLDER nodes that no longer contain anything.

        Runs to a fixed point: emptying `a/b/c` can empty `a/b`, which can empty
        `a`. The root folder is kept regardless, so an emptied source still has
        a node explaining that it was indexed and found nothing.
        """
        removed = 0
        while True:
            rows = self.conn.execute(
                """
                SELECT n.id FROM nodes n
                WHERE n.kind = ?
                  AND json_extract(n.props, '$.root') IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM edges e WHERE e.src_id = n.id AND e.kind = ?
                  )
                """,
                (str(NodeKind.FOLDER), str(EdgeKind.CONTAINS)),
            ).fetchall()
            ids = [int(r["id"]) for r in rows]
            if not ids:
                return removed
            self.conn.executemany("DELETE FROM nodes_fts WHERE rowid = ?", [(i,) for i in ids])
            self.conn.executemany("DELETE FROM nodes WHERE id = ?", [(i,) for i in ids])
            removed += len(ids)
            self._uid_cache.clear()

    def commit(self) -> None:
        self.conn.commit()

    # -- derived-work cache ------------------------------------------------

    def cache_get(self, digest: str, producer: str, version: str) -> Any | None:
        row = self.conn.execute(
            "SELECT payload FROM derived_cache WHERE digest = ? AND producer = ? AND version = ?",
            (digest, producer, version),
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def cache_put(self, digest: str, producer: str, version: str, payload: Any) -> None:
        self.conn.execute(
            "INSERT INTO derived_cache(digest, producer, version, payload, created_at)"
            " VALUES(?, ?, ?, ?, ?)"
            " ON CONFLICT(digest, producer, version) DO UPDATE SET payload = excluded.payload",
            (digest, producer, version, json.dumps(payload, default=str), time.time()),
        )

    # -- reading -----------------------------------------------------------

    def node(self, uid: str) -> NodeRow | None:
        row = self.conn.execute("SELECT * FROM nodes WHERE uid = ?", (uid,)).fetchone()
        return _to_node(row) if row else None

    def nodes_of_kind(self, kind: NodeKind | str, limit: int = 100) -> list[NodeRow]:
        rows = self.conn.execute(
            "SELECT * FROM nodes WHERE kind = ? ORDER BY name LIMIT ?", (str(kind), limit)
        )
        return [_to_node(r) for r in rows]

    def neighbors(
        self,
        uid: str,
        kinds: Sequence[EdgeKind | str] | None = None,
        direction: str = "both",
        node_kinds: Sequence[NodeKind | str] | None = None,
        limit: int = 50,
    ) -> list[Neighbor]:
        """One hop out from `uid`.

        `direction` is "out", "in", or "both". Direction is preserved in the
        result because it changes the meaning: a chunk *mentions* an entity,
        and an entity is *mentioned by* a chunk.
        """
        node_id = self._lookup(uid)
        if node_id is None:
            return []

        clauses: list[str] = []
        params: list[str] = []
        if kinds:
            clauses.append(f"e.kind IN ({','.join('?' * len(kinds))})")
            params.extend(str(k) for k in kinds)
        edge_filter = (" AND " + " AND ".join(clauses)) if clauses else ""

        out: list[Neighbor] = []
        if direction in ("out", "both"):
            rows = self.conn.execute(
                f"""SELECT n.*, e.kind AS ekind, e.weight, e.provenance, e.props AS eprops
                    FROM edges e JOIN nodes n ON n.id = e.dst_id
                    WHERE e.src_id = ?{edge_filter}""",
                (node_id, *params),
            ).fetchall()
            out.extend(_to_neighbor(r, "out") for r in rows)
        if direction in ("in", "both"):
            rows = self.conn.execute(
                f"""SELECT n.*, e.kind AS ekind, e.weight, e.provenance, e.props AS eprops
                    FROM edges e JOIN nodes n ON n.id = e.src_id
                    WHERE e.dst_id = ?{edge_filter}""",
                (node_id, *params),
            ).fetchall()
            out.extend(_to_neighbor(r, "in") for r in rows)

        if node_kinds:
            wanted = {str(k) for k in node_kinds}
            out = [n for n in out if n.node.kind in wanted]
        out.sort(key=lambda n: -n.weight)
        return out[:limit]

    def search(
        self,
        query: str,
        kinds: Sequence[NodeKind | str] | None = None,
        limit: int = 20,
    ) -> list[tuple[NodeRow, float]]:
        """Full-text search over node bodies, ranked by BM25 (lower is better,
        so it is negated into a score where higher is better)."""
        match = _fts_query(query)
        if not match:
            return []
        sql = [
            "SELECT n.*, bm25(nodes_fts) AS rank FROM nodes_fts",
            "JOIN nodes n ON n.id = nodes_fts.rowid",
            "WHERE nodes_fts MATCH ?",
        ]
        params: list[Any] = [match]
        if kinds:
            sql.append(f"AND n.kind IN ({','.join('?' * len(kinds))})")
            params.extend(str(k) for k in kinds)
        sql.append("ORDER BY rank LIMIT ?")
        params.append(limit)
        try:
            rows = self.conn.execute(" ".join(sql), params).fetchall()
        except sqlite3.OperationalError:
            # A malformed FTS expression should return nothing, not explode in
            # the middle of an agent's tool call.
            return []
        return [(_to_node(r), -float(r["rank"])) for r in rows]

    def shortest_path(self, src_uid: str, dst_uid: str, max_depth: int = 6) -> list[NodeRow]:
        """Shortest path between two nodes, treating edges as undirected.

        Undirected on purpose: "how is this invoice connected to that email
        thread?" is a question about connectivity, and insisting on edge
        direction would answer "not at all" for most genuinely related pairs.

        The search runs from **both ends at once**, always expanding whichever
        frontier is smaller. That matters more here than in a typical graph: a
        popular entity is mentioned by thousands of chunks, so a single-ended
        search fans out over most of the graph the moment it touches one.
        Levels are expanded with one query per batch of frontier nodes rather
        than one query per node.

        Returning as soon as the two sides touch is safe, which is not
        obvious — the usual worry is that the first meeting found is a hop
        longer than one found later in the same level. It cannot be here.
        Say the near side reaches node `m`, and the far side had already reached
        `m` at *less* than its current level. The far side would then have
        expanded `m` on a later level, so every neighbour of `m` is known to it
        — including the near-side node that just reached `m`. That node would
        have been a meeting point one level earlier, and the search would have
        already stopped. So every node the far side offers as a meeting point
        sits at exactly its current level, all meetings in a level are
        equidistant, and the first one found is as short as any other.

        This was checked empirically as well as argued: first-meeting and
        pick-the-best-in-the-level agree on every reachable pair across tens of
        thousands of randomised graphs.
        """
        src_id, dst_id = self._lookup(src_uid), self._lookup(dst_uid)
        if src_id is None or dst_id is None:
            return []
        if src_id == dst_id:
            node = self.node(src_uid)
            return [node] if node else []

        forward = _Side(src_id)
        backward = _Side(dst_id)

        for _ in range(max_depth):
            if not forward.frontier or not backward.frontier:
                break
            near, far = (
                (forward, backward)
                if len(forward.frontier) <= len(backward.frontier)
                else (backward, forward)
            )
            meeting = self._expand(near, far)
            if meeting is not None:
                return self._stitch(meeting, forward, backward)
            if len(forward.parent) + len(backward.parent) > MAX_VISITED:
                # A pathological pair in a dense graph. Reporting "no path
                # within this bound" beats spending a minute proving one exists.
                break
        return []

    def _expand(self, near: _Side, far: _Side) -> int | None:
        """Advance `near` by one level, returning a meeting point if one appears."""
        discovered: list[int] = []
        found: int | None = None

        for start in range(0, len(near.frontier), FRONTIER_BATCH):
            batch = near.frontier[start : start + FRONTIER_BATCH]
            placeholders = ",".join("?" * len(batch))
            rows = self.conn.execute(
                f"SELECT src_id AS came_from, dst_id AS reached FROM edges"
                f" WHERE src_id IN ({placeholders})"
                f" UNION ALL "
                f"SELECT dst_id AS came_from, src_id AS reached FROM edges"
                f" WHERE dst_id IN ({placeholders})",
                (*batch, *batch),
            )
            for row in rows:
                reached = int(row["reached"])
                if reached in near.parent:
                    continue
                near.parent[reached] = int(row["came_from"])
                discovered.append(reached)
                if found is None and reached in far.parent:
                    found = reached
            if found is not None:
                break

        near.frontier = discovered
        return found

    def _stitch(self, meeting: int, forward: _Side, backward: _Side) -> list[NodeRow]:
        """Walk the two parent chains outward from where they met."""
        chain: list[int] = []
        cursor: int | None = meeting
        while cursor is not None:
            chain.append(cursor)
            cursor = forward.parent[cursor]
        chain.reverse()
        cursor = backward.parent[meeting]
        while cursor is not None:
            chain.append(cursor)
            cursor = backward.parent[cursor]
        return [n for n in (self._node_by_id(i) for i in chain) if n]

    def subgraph(self, uid: str, depth: int = 1, limit: int = 200) -> tuple[list[NodeRow], list[EdgeRow]]:
        """Everything within `depth` hops of `uid`, as nodes plus the edges among them."""
        start = self._lookup(uid)
        if start is None:
            return [], []
        seen = {start}
        frontier = [start]
        for _ in range(depth):
            if not frontier or len(seen) >= limit:
                break
            placeholders = ",".join("?" * len(frontier))
            rows = self.conn.execute(
                f"SELECT dst_id AS other FROM edges WHERE src_id IN ({placeholders})"
                f" UNION SELECT src_id AS other FROM edges WHERE dst_id IN ({placeholders})",
                (*frontier, *frontier),
            ).fetchall()
            frontier = []
            for row in rows:
                other = int(row["other"])
                if other not in seen and len(seen) < limit:
                    seen.add(other)
                    frontier.append(other)

        ids = list(seen)
        placeholders = ",".join("?" * len(ids))
        nodes = [
            _to_node(r)
            for r in self.conn.execute(f"SELECT * FROM nodes WHERE id IN ({placeholders})", ids)
        ]
        edges = [
            EdgeRow(
                id=int(r["id"]),
                src_uid=r["src_uid"],
                dst_uid=r["dst_uid"],
                kind=r["kind"],
                weight=float(r["weight"]),
                provenance=r["provenance"],
                props=json.loads(r["props"] or "{}"),
            )
            for r in self.conn.execute(
                f"""SELECT e.*, s.uid AS src_uid, d.uid AS dst_uid FROM edges e
                    JOIN nodes s ON s.id = e.src_id
                    JOIN nodes d ON d.id = e.dst_id
                    WHERE e.src_id IN ({placeholders}) AND e.dst_id IN ({placeholders})""",
                (*ids, *ids),
            )
        ]
        return nodes, edges

    def hubs(self, kind: NodeKind | str | None = None, limit: int = 20) -> list[tuple[NodeRow, int]]:
        """The most-connected nodes — graphify's "god nodes".

        Useful as an opening move for an agent with no idea what is in a folder:
        the highest-degree entities are usually what the folder is *about*.

        Degree is computed in one grouped pass over the edge indexes rather than
        with a correlated subquery per node, which on a real corpus is the
        difference between milliseconds and a scan of every node in the graph.
        Nodes with no edges are not hubs and do not appear.
        """
        sql = """
            WITH degree AS (
                SELECT node_id, sum(hits) AS degree FROM (
                    SELECT src_id AS node_id, count(*) AS hits FROM edges GROUP BY src_id
                    UNION ALL
                    SELECT dst_id AS node_id, count(*) AS hits FROM edges GROUP BY dst_id
                ) GROUP BY node_id
            )
            SELECT n.*, d.degree AS degree
            FROM degree d JOIN nodes n ON n.id = d.node_id
        """
        params: list[Any] = []
        if kind:
            sql += " WHERE n.kind = ?"
            params.append(str(kind))
        sql += " ORDER BY d.degree DESC, n.name LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [(_to_node(r), int(r["degree"])) for r in rows]

    def all_nodes(self, limit: int = 100_000) -> list[NodeRow]:
        """Every node, for export. Ordered by kind so output is stable."""
        rows = self.conn.execute(
            "SELECT * FROM nodes ORDER BY kind, uid LIMIT ?", (limit,)
        )
        return [_to_node(r) for r in rows]

    def all_edges(self, limit: int = 500_000) -> list[EdgeRow]:
        rows = self.conn.execute(
            """SELECT e.*, s.uid AS src_uid, d.uid AS dst_uid FROM edges e
               JOIN nodes s ON s.id = e.src_id
               JOIN nodes d ON d.id = e.dst_id
               ORDER BY e.id LIMIT ?""",
            (limit,),
        )
        return [
            EdgeRow(
                id=int(r["id"]),
                src_uid=r["src_uid"],
                dst_uid=r["dst_uid"],
                kind=r["kind"],
                weight=float(r["weight"]),
                provenance=r["provenance"],
                props=json.loads(r["props"] or "{}"),
            )
            for r in rows
        ]

    def counts(self) -> dict[str, int]:
        nodes_by_kind = {
            r["kind"]: int(r["n"])
            for r in self.conn.execute("SELECT kind, count(*) AS n FROM nodes GROUP BY kind")
        }
        edges_by_kind = {
            r["kind"]: int(r["n"])
            for r in self.conn.execute("SELECT kind, count(*) AS n FROM edges GROUP BY kind")
        }
        return {
            "nodes": sum(nodes_by_kind.values()),
            "edges": sum(edges_by_kind.values()),
            **{f"node:{k}": v for k, v in sorted(nodes_by_kind.items())},
            **{f"edge:{k}": v for k, v in sorted(edges_by_kind.items())},
        }

    def _node_by_id(self, node_id: int) -> NodeRow | None:
        row = self.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return _to_node(row) if row else None


# -- helpers ---------------------------------------------------------------


def _to_node(row: sqlite3.Row) -> NodeRow:
    return NodeRow(
        id=int(row["id"]),
        uid=row["uid"],
        kind=row["kind"],
        name=row["name"],
        body=row["body"] or "",
        props=json.loads(row["props"] or "{}"),
        owner_file_id=row["owner_file_id"] if "owner_file_id" in set(row.keys()) else None,
    )


def _to_neighbor(row: sqlite3.Row, direction: str) -> Neighbor:
    props = json.loads(row["eprops"] or "{}")
    return Neighbor(
        node=_to_node(row),
        edge_kind=row["ekind"],
        direction=direction,
        weight=float(row["weight"]),
        provenance=row["provenance"],
        label=props.get("label"),
    )


def _search_body(node: Node) -> str:
    """What goes into FTS: the body, plus the name and any searchable props.

    The name is folded in so a query can find a document by its filename even
    when the text never says it — the same trick the chunk index uses for paths.
    """
    parts = [node.body, node.name.replace("/", " ").replace("_", " ").replace("-", " ")]
    for key in ("path", "heading", "title", "sheet", "speaker", "alias"):
        value = node.props.get(key)
        if isinstance(value, str) and value:
            parts.append(value.replace("/", " ").replace("_", " "))
    return "\n".join(p for p in parts if p)


_FTS_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 expression.

    Users and agents type quotes, hyphens, and colons, all of which are FTS5
    operators. Rather than teach callers to escape, extract bare tokens and
    quote each one — a slightly blunter query that never raises.
    """
    tokens = _FTS_TOKEN.findall(query)
    return " OR ".join(f'"{t}"' for t in tokens if len(t) > 1)
