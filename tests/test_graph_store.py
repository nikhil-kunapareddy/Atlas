"""Storage behaviour: identity, incremental rewrites, and traversal.

These are the properties the rest of the system assumes. If upserts stop being
idempotent, every rebuild silently doubles the graph; if `clear_file` takes
shared entities with it, reindexing one file quietly deletes knowledge that came
from others. Both failures are invisible without a test.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from atlas.graph import (
    Edge,
    EdgeKind,
    Node,
    NodeKind,
    Provenance,
    child_uid,
    document_uid,
    entity_uid,
    folder_uid,
    normalize_entity,
)
from atlas.graph.model import EntityType


def _doc(graph, uid="doc:a.md", name="A", body="alpha beta", file_id=None):
    return graph.upsert_node(
        Node(uid=uid, kind=NodeKind.DOCUMENT, name=name, body=body), owner_file_id=file_id
    )


# -- identity --------------------------------------------------------------


def test_uids_are_deterministic_and_posix():
    assert document_uid("docs\\a.pdf") == "doc:docs/a.pdf"
    assert folder_uid("") == "dir:."
    assert child_uid("doc:a.pdf", NodeKind.PAGE, 3) == "doc:a.pdf#page:3"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Acme Corp.", "acme corp"),
        ("  ACME   CORP  ", "acme corp"),
        ('"Acme Corp"', "acme corp"),
        ("Café", "cafe"),
    ],
)
def test_entity_normalisation_merges_surface_forms(raw, expected):
    assert normalize_entity(raw) == expected
    assert entity_uid(EntityType.ORG, raw) == f"ent:org:{expected}"


def test_normalisation_does_not_over_merge():
    """Distinct entities must stay distinct — a wrong merge is unrecoverable."""
    assert normalize_entity("Acme Corp") != normalize_entity("Acme Corps")
    assert normalize_entity("acme") != normalize_entity("acme inc")


def test_self_loops_are_rejected():
    with pytest.raises(ValueError):
        Edge("a", "a", EdgeKind.CONTAINS)


# -- upsert ----------------------------------------------------------------


def test_upsert_is_idempotent(graph):
    first = _doc(graph)
    second = _doc(graph)
    assert first == second
    assert graph.counts()["nodes"] == 1


def test_upsert_merges_props_and_keeps_longest_body(graph):
    graph.upsert_node(Node("ent:x", NodeKind.ENTITY, "X", "short", {"a": 1}))
    graph.upsert_node(Node("ent:x", NodeKind.ENTITY, "X", "a much longer body", {"b": 2}))
    node = graph.node("ent:x")
    assert node.props == {"a": 1, "b": 2}
    assert node.body == "a much longer body"

    # A shorter body must not clobber the longer one already recorded.
    graph.upsert_node(Node("ent:x", NodeKind.ENTITY, "X", "tiny", {}))
    assert graph.node("ent:x").body == "a much longer body"


def test_repeated_edges_collapse_to_one(graph):
    _doc(graph, "doc:a")
    _doc(graph, "doc:b")
    for _ in range(3):
        graph.add_edge(Edge("doc:a", "doc:b", EdgeKind.REFERENCES))
    assert graph.counts()["edges"] == 1


def test_extracted_provenance_is_not_downgraded_by_a_later_guess(graph):
    _doc(graph, "doc:a")
    _doc(graph, "doc:b")
    graph.add_edge(Edge("doc:a", "doc:b", EdgeKind.REFERENCES, provenance=Provenance.EXTRACTED))
    graph.add_edge(Edge("doc:a", "doc:b", EdgeKind.REFERENCES, provenance=Provenance.LLM))
    assert graph.all_edges()[0].provenance == "extracted"


# -- forward references ----------------------------------------------------


def test_edges_to_not_yet_created_nodes_are_parked_then_resolved(graph):
    _doc(graph, "doc:a")
    assert graph.add_edge(Edge("doc:a", "doc:later", EdgeKind.REFERENCES)) is False
    assert graph.counts()["edges"] == 0

    _doc(graph, "doc:later")
    assert graph.resolve_pending() == 0
    assert graph.counts()["edges"] == 1


def test_permanently_unresolvable_edges_are_dropped_and_counted(graph):
    _doc(graph, "doc:a")
    graph.add_edge(Edge("doc:a", "doc:never", EdgeKind.REFERENCES))
    assert graph.resolve_pending() == 1
    assert graph.counts()["edges"] == 0


# -- incremental rewrite ---------------------------------------------------


def test_clear_file_removes_owned_nodes_but_spares_shared_entities(graph, project_store):
    source_id = project_store.add_source(".")
    file_id = project_store.upsert_file(source_id, "/a.md", 1.0, 10, "d1")
    other_id = project_store.upsert_file(source_id, "/b.md", 1.0, 10, "d2")

    graph.upsert_node(Node("doc:a.md", NodeKind.DOCUMENT, "A"), owner_file_id=file_id)
    graph.upsert_node(Node("doc:b.md", NodeKind.DOCUMENT, "B"), owner_file_id=other_id)
    graph.upsert_node(Node("ent:shared", NodeKind.ENTITY, "shared"))
    graph.add_edge(Edge("doc:a.md", "ent:shared", EdgeKind.MENTIONS), owner_file_id=file_id)
    graph.add_edge(Edge("doc:b.md", "ent:shared", EdgeKind.MENTIONS), owner_file_id=other_id)

    graph.clear_file(file_id)

    assert graph.node("doc:a.md") is None, "the file's own node must go"
    assert graph.node("doc:b.md") is not None, "another file's node must survive"
    assert graph.node("ent:shared") is not None, "a shared entity must survive"
    assert graph.gc_orphans() == 0, "still referenced by b.md, so not an orphan"


def test_gc_collects_entities_once_nothing_mentions_them(graph, project_store):
    source_id = project_store.add_source(".")
    file_id = project_store.upsert_file(source_id, "/a.md", 1.0, 10, "d1")
    graph.upsert_node(Node("doc:a.md", NodeKind.DOCUMENT, "A"), owner_file_id=file_id)
    graph.upsert_node(Node("ent:lonely", NodeKind.ENTITY, "lonely"))
    graph.add_edge(Edge("doc:a.md", "ent:lonely", EdgeKind.MENTIONS), owner_file_id=file_id)

    graph.clear_file(file_id)
    assert graph.gc_orphans() == 1
    assert graph.node("ent:lonely") is None


def test_empty_folders_are_pruned_but_the_root_survives(graph):
    graph.upsert_node(Node("dir:.", NodeKind.FOLDER, "root", props={"root": True}))
    graph.upsert_node(Node("dir:empty", NodeKind.FOLDER, "empty", props={"path": "empty"}))
    graph.add_edge(Edge("dir:.", "dir:empty", EdgeKind.CONTAINS))

    assert graph.prune_empty_folders() == 1
    assert graph.node("dir:empty") is None
    assert graph.node("dir:.") is not None


# -- reading ---------------------------------------------------------------


def test_search_ranks_and_filters_by_kind(graph):
    _doc(graph, "doc:a", "A", "the quarterly revenue report")
    graph.upsert_node(Node("doc:a#chunk:0", NodeKind.CHUNK, "chunk", "quarterly revenue detail"))

    assert [n.uid for n, _ in graph.search("quarterly revenue")] != []
    only_chunks = graph.search("quarterly", kinds=[NodeKind.CHUNK])
    assert {n.kind for n, _ in only_chunks} == {"chunk"}


def test_search_survives_fts_operator_characters(graph):
    """Agents type quotes and colons; a bad query must return [], not raise."""
    _doc(graph, "doc:a", "A", "alpha beta")
    for hostile in ['"unclosed', "a AND OR b", "path:*", "NEAR(", "-- ;"]:
        assert isinstance(graph.search(hostile), list)


def test_neighbors_report_direction(graph):
    _doc(graph, "doc:a")
    graph.upsert_node(Node("ent:x", NodeKind.ENTITY, "X"))
    graph.add_edge(Edge("doc:a", "ent:x", EdgeKind.MENTIONS))

    out = graph.neighbors("doc:a", direction="out")
    assert [(n.node.uid, n.direction) for n in out] == [("ent:x", "out")]
    incoming = graph.neighbors("ent:x", direction="in")
    assert [(n.node.uid, n.direction) for n in incoming] == [("doc:a", "in")]
    assert graph.neighbors("doc:missing") == []


def test_shortest_path_is_undirected_and_bounded(graph):
    for uid in ("doc:a", "doc:b", "doc:c"):
        _doc(graph, uid)
    graph.add_edge(Edge("doc:a", "doc:b", EdgeKind.REFERENCES))
    graph.add_edge(Edge("doc:c", "doc:b", EdgeKind.REFERENCES))  # points the other way

    chain = graph.shortest_path("doc:a", "doc:c")
    assert [n.uid for n in chain] == ["doc:a", "doc:b", "doc:c"]
    assert graph.shortest_path("doc:a", "doc:c", max_depth=1) == []
    assert graph.shortest_path("doc:a", "doc:nowhere") == []


def test_hubs_orders_by_degree(graph):
    _doc(graph, "doc:hub")
    for i in range(3):
        _doc(graph, f"doc:leaf{i}")
        graph.add_edge(Edge("doc:hub", f"doc:leaf{i}", EdgeKind.REFERENCES))
    top, degree = graph.hubs(limit=1)[0]
    assert top.uid == "doc:hub"
    assert degree == 3


def test_derived_cache_round_trips_and_is_versioned(graph):
    graph.cache_put("digest1", "whisper", "1", {"segments": [1, 2]})
    assert graph.cache_get("digest1", "whisper", "1") == {"segments": [1, 2]}
    assert graph.cache_get("digest1", "whisper", "2") is None, "a version bump must miss"
    assert graph.cache_get("digest2", "whisper", "1") is None


# -- path finding ----------------------------------------------------------
#
# The search is bidirectional, which is fast but easy to get subtly wrong. These
# pin the properties that a first-meeting-wins implementation would violate.


def _chain(graph, *uids):
    for uid in uids:
        _doc(graph, uid)
    for before, after in pairwise(uids):
        graph.add_edge(Edge(before, after, EdgeKind.REFERENCES))


def test_path_is_the_shortest_when_two_routes_exist(graph):
    """A short route and a long route between the same pair; the short one wins."""
    _chain(graph, "doc:a", "doc:b", "doc:z")                      # 2 hops
    _chain(graph, "doc:a", "doc:p", "doc:q", "doc:r", "doc:z")    # 4 hops

    chain = [n.uid for n in graph.shortest_path("doc:a", "doc:z")]
    assert chain == ["doc:a", "doc:b", "doc:z"]


def test_path_is_shortest_when_routes_differ_only_on_the_far_side(graph):
    """The case that breaks accepting the first meeting point found.

    Both `mid1` and `mid2` sit one hop from the source, so they land in the same
    forward level — but `mid1` is two hops from the target and `mid2` is one.
    An implementation that returns whichever it happens to see first is
    order-dependent and wrong half the time.
    """
    for uid in ("doc:src", "doc:mid1", "doc:mid2", "doc:far", "doc:dst"):
        _doc(graph, uid)
    graph.add_edge(Edge("doc:src", "doc:mid1", EdgeKind.REFERENCES))
    graph.add_edge(Edge("doc:mid1", "doc:far", EdgeKind.REFERENCES))
    graph.add_edge(Edge("doc:far", "doc:dst", EdgeKind.REFERENCES))
    graph.add_edge(Edge("doc:src", "doc:mid2", EdgeKind.REFERENCES))
    graph.add_edge(Edge("doc:mid2", "doc:dst", EdgeKind.REFERENCES))

    chain = [n.uid for n in graph.shortest_path("doc:src", "doc:dst")]
    assert chain == ["doc:src", "doc:mid2", "doc:dst"]


def test_path_endpoints_and_adjacency_are_intact(graph):
    """Every consecutive pair in the returned chain must really be joined."""
    _chain(graph, "doc:a", "doc:b", "doc:c", "doc:d", "doc:e")
    chain = [n.uid for n in graph.shortest_path("doc:a", "doc:e")]

    assert chain[0] == "doc:a" and chain[-1] == "doc:e"
    assert len(chain) == len(set(chain)), "a path must not revisit a node"
    for before, after in pairwise(chain):
        both = {n.node.uid for n in graph.neighbors(before, direction="both", limit=50)}
        assert after in both, f"{before} and {after} are not actually adjacent"


def test_path_through_a_high_degree_hub_stays_fast_and_correct(graph):
    """A hub entity mentioned by many chunks is the shape that made this slow."""
    graph.upsert_node(Node("ent:hub", NodeKind.ENTITY, "hub"))
    for i in range(600):
        uid = f"doc:c{i}"
        _doc(graph, uid)
        graph.add_edge(Edge(uid, "ent:hub", EdgeKind.MENTIONS))

    chain = [n.uid for n in graph.shortest_path("doc:c0", "doc:c599")]
    assert chain == ["doc:c0", "ent:hub", "doc:c599"]


def test_path_respects_max_depth_in_both_directions(graph):
    _chain(graph, "doc:a", "doc:b", "doc:c", "doc:d")
    assert graph.shortest_path("doc:a", "doc:d", max_depth=2) == []
    assert len(graph.shortest_path("doc:a", "doc:d", max_depth=3)) == 4


def test_hubs_excludes_unconnected_nodes(graph):
    """Degree is computed from edges, so a node with none is not a hub."""
    _doc(graph, "doc:lonely")
    assert graph.hubs() == []


def test_path_length_matches_a_reference_bfs_on_random_graphs(graph):
    """Property test against a plain single-source BFS.

    Hand-built examples are weak evidence for a bidirectional search: whether a
    given case discriminates depends on which side expands first, which depends
    on frontier sizes. Randomised graphs checked against an obviously-correct
    reference catch what a fixed example may miss by luck.
    """
    import random
    from collections import deque

    rng = random.Random(1234)
    nodes = [f"doc:n{i}" for i in range(40)]
    for uid in nodes:
        _doc(graph, uid)

    adjacency: dict[str, set[str]] = {uid: set() for uid in nodes}
    for _ in range(90):
        a, b = rng.sample(nodes, 2)
        if b in adjacency[a]:
            continue
        graph.add_edge(Edge(a, b, EdgeKind.REFERENCES))
        adjacency[a].add(b)
        adjacency[b].add(a)

    def reference(start: str, goal: str) -> int | None:
        seen = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current == goal:
                return seen[current]
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen[neighbour] = seen[current] + 1
                    queue.append(neighbour)
        return None

    checked = 0
    for _ in range(60):
        a, b = rng.sample(nodes, 2)
        expected = reference(a, b)
        chain = [n.uid for n in graph.shortest_path(a, b, max_depth=12)]
        if expected is None or expected > 12:
            assert chain == [], f"{a}->{b} is unreachable but a path was returned"
            continue
        assert chain, f"{a}->{b} is reachable in {expected} hops but nothing was returned"
        assert len(chain) - 1 == expected, (
            f"{a}->{b}: returned {len(chain) - 1} hops, shortest is {expected}"
        )
        assert chain[0] == a and chain[-1] == b
        for before, after in pairwise(chain):
            assert after in adjacency[before]
        checked += 1
    assert checked > 20, "the random graph was too sparse to be a real test"
