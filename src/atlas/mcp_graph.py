"""Graph tools for the MCP server.

Registered onto the same server as the fact tools so an agent sees one Atlas.
Kept in its own module because the two halves answer different questions and
their tool descriptions are long — descriptions are load-bearing here, since an
agent chooses a tool from nothing but the text.

Every tool returns human-readable text with `uid`s embedded rather than JSON.
The uid is the handle for the next call, so it has to be visible; the surrounding
prose is what lets the model tell a page from a spreadsheet column without
parsing anything.
"""

from __future__ import annotations

from typing import Any

from .config import PROJECT_SCOPE, find_project_root, project_db_path
from .graph import EdgeKind, GraphStore, NodeKind
from .store import Store

GRAPH_INSTRUCTIONS = """\
Atlas also holds a graph of a folder's contents — PDFs by page, spreadsheets by \
sheet and column, recordings by timestamp, images by metadata — joined by the \
entities they mention.

Use `atlas_graph_search` to find content, then `atlas_graph_neighbors` to \
traverse. The key move: to find everything about an invoice number, an email \
address, or a date, look up its entity node and get its incoming neighbours — \
that returns every passage in every file that mentions it, across formats.\
"""

MAX_TEXT = 4000


def _open() -> tuple[Store, GraphStore] | None:
    root = find_project_root()
    if root is None:
        return None
    path = project_db_path(root)
    if not path.exists():
        return None
    store = Store(path, PROJECT_SCOPE)
    return store, GraphStore(store.conn)


def _need_graph() -> str:
    return (
        "No graph here. Build one first by running `atlas <folder>` in a "
        "terminal, which converts that folder's documents, spreadsheets, "
        "images and recordings into a queryable graph."
    )


def _line(node: Any, extra: str = "") -> str:
    label = f"[{node.kind}] {node.name}"
    path = node.props.get("path")
    where = f"  ({path})" if path and path not in node.name else ""
    return f"{label}{where}{extra}\n    uid: {node.uid}"


def register(server: Any) -> None:
    """Add the graph tools to `server`."""

    @server.tool(
        name="atlas_graph_overview",
        description=(
            "What the indexed folder contains: how many nodes and edges, every "
            "document with its type (pdf, spreadsheet, image, audio, video, "
            "text), and the entities mentioned most often across files. Call "
            "this first when you do not yet know what is in the folder."
        ),
    )
    def atlas_graph_overview() -> str:
        opened = _open()
        if opened is None:
            return _need_graph()
        store, graph = opened
        try:
            counts = graph.counts()
            lines = [
                f"{counts.get('nodes', 0)} nodes, {counts.get('edges', 0)} edges",
                "",
                "Node kinds: "
                + ", ".join(f"{k[5:]}={v}" for k, v in counts.items() if k.startswith("node:")),
                "",
                "Documents:",
            ]
            for node in graph.nodes_of_kind(NodeKind.DOCUMENT, limit=60):
                lines.append(f"  [{node.props.get('modality', '?')}] {node.props.get('path')}")
                lines.append(f"      uid: {node.uid}")

            hubs = graph.hubs(NodeKind.ENTITY, limit=15)
            if hubs:
                lines += ["", "Most-mentioned entities (these join files together):"]
                for node, degree in hubs:
                    kind = node.props.get("entity_type", "entity")
                    lines.append(f"  {node.name}  [{kind}, {degree} mentions]")
                    lines.append(f"      uid: {node.uid}")
            return "\n".join(lines)
        finally:
            store.close()

    @server.tool(
        name="atlas_graph_search",
        description=(
            "Full-text search across everything extracted from the folder: PDF "
            "pages, spreadsheet columns and rows, transcripts, document text. "
            "Search for words that would appear INSIDE the documents, not words "
            "describing them. Returns previews with uids; call "
            "atlas_graph_node for a passage's full text. Optionally restrict to "
            "one node kind, e.g. 'column' to find spreadsheet columns or "
            "'segment' to find moments in a recording."
        ),
    )
    def atlas_graph_search(query: str, kind: str | None = None, limit: int = 10) -> str:
        opened = _open()
        if opened is None:
            return _need_graph()
        store, graph = opened
        try:
            kinds = [NodeKind(kind)] if kind else None
        except ValueError:
            store.close()
            return f"unknown kind {kind!r}; valid kinds: {', '.join(k.value for k in NodeKind)}"
        try:
            hits = graph.search(query, kinds=kinds, limit=max(1, min(limit, 25)))
            if not hits:
                return f"No matches for {query!r}. Try different words, or atlas_graph_overview."
            return "\n\n".join(
                f"{i}. {_line(node)}\n    {node.preview(300)}"
                for i, (node, _) in enumerate(hits, start=1)
            )
        finally:
            store.close()

    @server.tool(
        name="atlas_graph_node",
        description=(
            "The full text and all properties of one node, by uid. Use after a "
            "search to read a passage in full — search results are truncated. "
            "Properties carry the citation details: page number, timestamp, "
            "sheet name, column type, source file path."
        ),
    )
    def atlas_graph_node(uid: str) -> str:
        opened = _open()
        if opened is None:
            return _need_graph()
        store, graph = opened
        try:
            node = graph.node(uid)
            if node is None:
                return f"No node with uid {uid!r}. Use atlas_graph_search to find one."
            lines = [f"{node.name}", f"kind: {node.kind}", f"uid: {node.uid}"]
            for key, value in sorted(node.props.items()):
                lines.append(f"{key}: {value}")
            if node.body:
                lines += ["", node.body[:MAX_TEXT]]
                if len(node.body) > MAX_TEXT:
                    lines.append(f"… ({len(node.body) - MAX_TEXT} more characters)")
            return "\n".join(lines)
        finally:
            store.close()

    @server.tool(
        name="atlas_graph_neighbors",
        description=(
            "Everything one hop from a node — the main way to traverse. On an "
            "ENTITY use direction='in' to get every passage in every file that "
            "mentions it, which is how you connect a spreadsheet row to a PDF "
            "to a recording. On a DOCUMENT use direction='out' to list its "
            "pages, sheets or time segments. Each result notes how the edge was "
            "established: 'extracted' was read off the bytes, 'llm' was inferred "
            "by a model and may be wrong."
        ),
    )
    def atlas_graph_neighbors(
        uid: str, direction: str = "both", edge_kind: str | None = None, limit: int = 20
    ) -> str:
        opened = _open()
        if opened is None:
            return _need_graph()
        store, graph = opened
        try:
            if graph.node(uid) is None:
                return f"No node with uid {uid!r}."
            try:
                kinds = [EdgeKind(edge_kind)] if edge_kind else None
            except ValueError:
                return (
                    f"unknown edge kind {edge_kind!r}; valid: "
                    f"{', '.join(k.value for k in EdgeKind)}"
                )
            found = graph.neighbors(
                uid, kinds=kinds, direction=direction, limit=max(1, min(limit, 50))
            )
            if not found:
                return f"{uid} has no neighbours in that direction."
            return "\n".join(
                _line(
                    n.node,
                    f"  — {'→' if n.direction == 'out' else '←'} {n.edge_kind}"
                    + ("" if n.provenance == "extracted" else f" [{n.provenance}]"),
                )
                for n in found
            )
        finally:
            store.close()

    @server.tool(
        name="atlas_graph_path",
        description=(
            "The shortest chain of relationships connecting two nodes. Use to "
            "explain how two documents, or a document and an entity, relate to "
            "each other when it is not obvious."
        ),
    )
    def atlas_graph_path(source_uid: str, target_uid: str, max_depth: int = 6) -> str:
        opened = _open()
        if opened is None:
            return _need_graph()
        store, graph = opened
        try:
            chain = graph.shortest_path(source_uid, target_uid, max_depth=max_depth)
            if not chain:
                return f"No connection within {max_depth} hops."
            return f"{len(chain) - 1} hops:\n" + "\n".join(
                f"  {'  ' * i}{'└─ ' if i else ''}[{n.kind}] {n.name}" for i, n in enumerate(chain)
            )
        finally:
            store.close()
