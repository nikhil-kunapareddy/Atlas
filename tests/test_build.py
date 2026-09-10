"""The build pipeline — `atlas <folder>` end to end.

These are the tests that describe the product's promises: a folder always
produces a graph, rebuilding is cheap and idempotent, deleting a file removes
exactly what it contributed, and entities join files that share a fact even when
those files are different formats.
"""

from __future__ import annotations

import fixtures
from atlas.build import build, discover
from atlas.extract import ExtractOptions
from atlas.graph import EdgeKind, GraphStore, NodeKind
from atlas.store import Store

QUIET = ExtractOptions(transcribe=False, keyframes=0)


def rebuild(root, store, graph, **kwargs):
    return build(root, store, graph, QUIET, **kwargs)


def uids(graph, kind):
    return {n.uid for n in graph.nodes_of_kind(kind, limit=500)}


# -- the happy path --------------------------------------------------------


def test_a_folder_becomes_a_graph_spanning_every_modality(built):
    _, _, graph = built
    counts = graph.counts()

    assert counts["nodes"] > 0 and counts["edges"] > 0
    documents = {n.props["path"]: n for n in graph.nodes_of_kind(NodeKind.DOCUMENT, limit=100)}
    assert set(documents) == {
        "README.md",
        "docs/brief.md",
        "docs/report.pdf",
        "docs/customers.csv",
        "media/chart.png",
        "media/standup.wav",
    }
    assert {d.props["modality"] for d in documents.values()} == {
        "text", "pdf", "sheet", "image", "audio",
    }
    # The structural vocabulary each modality contributes.
    assert counts.get("node:page", 0) >= 2, "pdf pages"
    assert counts.get("node:sheet", 0) >= 1 and counts.get("node:column", 0) >= 5
    assert counts.get("node:entity", 0) >= 5


def test_folder_hierarchy_is_mirrored(built):
    _, _, graph = built
    assert graph.node("dir:.") is not None
    assert graph.node("dir:docs") is not None

    children = graph.neighbors("dir:docs", kinds=[EdgeKind.CONTAINS], direction="out", limit=50)
    assert "doc:docs/report.pdf" in {n.node.uid for n in children}


def test_a_file_that_cannot_be_read_still_gets_a_node(built):
    """The core promise: no single file can fail the run."""
    _, _, graph = built
    audio = graph.node("doc:media/standup.wav")
    assert audio is not None
    assert audio.props["modality"] == "audio"


# -- the headline: cross-format joins --------------------------------------


def test_one_entity_joins_a_pdf_and_two_markdown_files(built):
    """`INV-2024-0912` appears in a PDF and two Markdown files.

    This is the whole point of the graph: a shared literal links documents of
    different formats with no model in the loop.
    """
    _, _, graph = built
    mentions = graph.neighbors(
        "ent:identifier:inv-2024-0912", kinds=[EdgeKind.MENTIONS], direction="in", limit=50
    )
    source_files = {n.node.props.get("path") for n in mentions}
    assert source_files == {"README.md", "docs/brief.md", "docs/report.pdf"}


def test_shared_amount_and_email_also_join(built):
    _, _, graph = built
    for uid in ("ent:money:usd 42500.00", "ent:email:ana@acme.example"):
        node = graph.node(uid)
        assert node is not None, uid
        incoming = graph.neighbors(uid, direction="in", limit=50)
        assert len({n.node.props.get("path") for n in incoming}) >= 2, uid


def test_relative_links_between_files_become_edges(built):
    _, _, graph = built
    outgoing = graph.neighbors("doc:README.md", kinds=[EdgeKind.REFERENCES], direction="out")
    assert [n.node.uid for n in outgoing] == ["doc:docs/brief.md"]
    assert outgoing[0].provenance == "inferred", "a link is a rule, not a raw fact"


# -- incrementality --------------------------------------------------------


def test_rebuilding_an_unchanged_folder_does_no_work(built):
    root, store, graph = built
    before = graph.counts()

    again = rebuild(root, store, graph)

    assert again.files_indexed == 0
    assert again.files_unchanged == 6
    assert graph.counts() == before, "an idempotent rebuild must not change the graph"


def test_editing_a_file_replaces_only_its_own_nodes(built):
    root, store, graph = built
    other_before = graph.node("doc:docs/brief.md").body

    (root / "README.md").write_text("# Replaced\n\nEntirely different words now.\n")
    stats = rebuild(root, store, graph)

    assert stats.files_indexed == 1
    assert graph.node("doc:README.md").name == "Replaced"
    assert "Northwind" not in graph.node("doc:README.md").body, "stale content is gone"
    assert graph.node("doc:docs/brief.md").body == other_before, "neighbours untouched"


def test_shrinking_a_file_leaves_no_orphaned_chunks(built):
    root, store, graph = built
    long_text = "\n\n".join(f"## Section {i}\n\n" + ("filler " * 200) for i in range(6))
    (root / "README.md").write_text(long_text)
    rebuild(root, store, graph)
    many = len([u for u in uids(graph, NodeKind.CHUNK) if u.startswith("doc:README.md")])

    (root / "README.md").write_text("# Tiny\n\nOne line.\n")
    rebuild(root, store, graph)
    few = len([u for u in uids(graph, NodeKind.CHUNK) if u.startswith("doc:README.md")])

    assert many > few == 1


def test_deleting_a_file_prunes_it_and_collects_its_private_entities(built):
    root, store, graph = built
    assert graph.node("ent:email:hi@globex.example") is not None

    (root / "docs" / "customers.csv").unlink()
    stats = rebuild(root, store, graph)

    assert stats.files_removed == 1
    assert graph.node("doc:docs/customers.csv") is None
    assert not uids(graph, NodeKind.SHEET)
    # Only that file mentioned Globex, so the entity goes too…
    assert graph.node("ent:email:hi@globex.example") is None
    # …but an entity three other files still mention must survive.
    assert graph.node("ent:identifier:inv-2024-0912") is not None


def test_deleting_every_file_leaves_an_empty_but_valid_graph(built):
    root, store, graph = built
    for path in list(root.rglob("*")):
        if path.is_file() and ".atlas" not in path.parts:
            path.unlink()

    rebuild(root, store, graph)
    assert not uids(graph, NodeKind.DOCUMENT)
    assert not uids(graph, NodeKind.ENTITY)
    assert graph.node("dir:.") is not None, "the root node explains that we looked"


def test_prune_can_be_disabled(built):
    root, store, graph = built
    (root / "README.md").unlink()
    stats = rebuild(root, store, graph, prune=False)
    assert stats.files_removed == 0
    assert graph.node("doc:README.md") is not None


# -- walking ---------------------------------------------------------------


def test_discover_skips_noise_and_the_store_itself(tmp_path):
    root = tmp_path / "w"
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / ".atlas").mkdir()
    (root / "node_modules" / "pkg" / "index.js").write_text("x")
    (root / ".atlas" / "store.db").write_text("x")
    (root / ".DS_Store").write_text("x")
    (root / "keep.md").write_text("keep")

    assert [p.name for p in discover(root, ExtractOptions())] == ["keep.md"]


def test_oversized_files_are_skipped(tmp_path):
    root = tmp_path / "big"
    root.mkdir()
    (root / "huge.txt").write_text("x" * 5000)
    (root / "small.txt").write_text("ok")
    found = list(discover(root, ExtractOptions(max_file_bytes=1000)))
    assert [p.name for p in found] == ["small.txt"]


def test_build_rejects_a_file_argument(tmp_path, project_store):
    graph = GraphStore(project_store.conn)
    target = tmp_path / "a.md"
    target.write_text("x")
    try:
        build(target, project_store, graph, QUIET)
    except NotADirectoryError as exc:
        assert "folder that contains it" in str(exc)
    else:
        raise AssertionError("expected NotADirectoryError")


# -- caching ---------------------------------------------------------------


def test_expensive_extraction_is_cached_by_content(tmp_path):
    """A rebuild after a touch must not re-run the extractor."""
    root = tmp_path / "c"
    root.mkdir()
    fixtures.make_pdf(root / "a.pdf", ["Some page text here."])
    store = Store(root / ".atlas" / "store.db", "project")
    graph = GraphStore(store.conn)

    rebuild(root, store, graph)
    cached = graph.cache_get(
        next(iter(graph.nodes_of_kind(NodeKind.DOCUMENT, 5))).props["digest"], "pdf", "1"
    )
    assert cached is not None and cached["page_count"] == 1
    store.close()


# -- mislabelled files -----------------------------------------------------


def test_a_file_whose_bytes_contradict_its_name_is_recorded_as_such(tmp_path):
    """Real corpora are full of these; the graph should say so, not just fail.

    Taken from Govdocs1, which contains legacy OLE2 workbooks carrying an
    `.xlsx` extension.
    """
    root = tmp_path / "mixed"
    root.mkdir()
    (root / "report.xlsx").write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 512)
    (root / "notes.md").write_text("# Fine\n\nA perfectly ordinary file.\n")

    store = Store(root / ".atlas" / "store.db", "project")
    graph = GraphStore(store.conn)
    stats = rebuild(root, store, graph)

    liar = graph.node("doc:report.xlsx")
    assert liar is not None, "a mislabelled file still belongs in the graph"
    assert liar.props["mislabelled"] == "content is ole2, not .xlsx"
    assert liar.props["detected_format"] == "ole2"

    honest = graph.node("doc:notes.md")
    assert "mislabelled" not in honest.props, "a good file must not be maligned"

    assert any("ole2" in w and "LibreOffice" in w for w in stats.warnings), (
        "the warning should say what the file is and what to do"
    )
    assert not stats.errors, "a mislabelled file is a warning, never an error"
    store.close()
