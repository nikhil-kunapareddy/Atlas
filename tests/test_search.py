from __future__ import annotations

from atlas.indexer import index_path
from atlas.memory import remember
from atlas.search import build_match_query, recall


def _recall(stores, query, **kwargs):
    kwargs.setdefault("track", False)
    return recall(query, stores=stores, **kwargs)


def test_match_query_survives_hostile_input():
    # Any of these are FTS5 syntax errors if passed through raw.
    for hostile in ['unbalanced "quote', "AND OR NOT", "*", "a(b)c", "x AND"]:
        match = build_match_query(hostile)
        assert match is None or '"' in match


def test_match_query_is_none_for_empty_input():
    assert build_match_query("") is None
    assert build_match_query("   !!  ") is None


def test_hostile_queries_do_not_raise(project_store, repo):
    index_path(project_store, repo)
    for hostile in ['unbalanced "quote', "AND OR NOT", "*", "", "((("]:
        _recall([project_store], hostile)  # must not raise


def test_recall_finds_indexed_content(project_store, repo):
    index_path(project_store, repo)
    results = _recall([project_store], "redis before pytest")

    assert results
    assert any("Redis" in r.text for r in results)


def test_chunk_results_carry_openable_refs(project_store, repo):
    index_path(project_store, repo)
    results = _recall([project_store], "gateway gRPC")

    chunks = [r for r in results if r.kind == "chunk"]
    assert chunks
    top = chunks[0]
    assert top.path.endswith("architecture.md")
    assert top.start_line >= 1 and top.end_line >= top.start_line
    assert ":" in top.ref


def test_camel_case_identifiers_are_findable_by_word(project_store, repo):
    index_path(project_store, repo)
    results = _recall([project_store], "token validator")
    assert any(r.path and r.path.endswith("auth.py") for r in results if r.kind == "chunk")


def test_facts_outrank_file_chunks(project_store, repo):
    index_path(project_store, repo)
    remember("Redis must be running before pytest", store=project_store)

    results = _recall([project_store], "redis pytest")
    assert results[0].kind == "fact"


def test_pinned_facts_surface_for_unrelated_queries(project_store, repo):
    index_path(project_store, repo)
    remember("Never edit generated/ by hand", pinned=True, store=project_store)

    results = _recall([project_store], "invoice totals")
    assert any(r.pinned and "generated" in r.text for r in results)


def test_pinned_facts_can_be_suppressed(project_store):
    remember("Never edit generated/ by hand", pinned=True, store=project_store)
    results = _recall([project_store], "invoice", include_pinned=False)
    assert not any(r.pinned for r in results)


def test_pinned_facts_are_not_duplicated(project_store):
    remember("Always run make fmt before committing", pinned=True, store=project_store)
    results = _recall([project_store], "make fmt before committing")

    matching = [r for r in results if r.kind == "fact" and "make fmt" in r.text]
    assert len(matching) == 1


def test_project_scope_outranks_global(project_store, global_store):
    remember("Deploy with the standard pipeline", store=global_store)
    remember("Deploy with the standard pipeline", store=project_store)

    results = _recall([project_store, global_store], "deploy pipeline")
    assert results[0].scope == "project"


def test_scope_filter_excludes_other_scope(project_store, global_store):
    remember("A global habit worth keeping", store=global_store)
    results = _recall([project_store, global_store], "global habit", scopes=("project",))
    assert not any(r.scope == "global" for r in results)


def test_facts_only_omits_chunks(project_store, repo):
    index_path(project_store, repo)
    remember("Redis is needed for tests", store=project_store)

    results = _recall([project_store], "redis", kinds=("fact",))
    assert results
    assert all(r.kind == "fact" for r in results)


def test_k_limits_results(project_store, repo):
    index_path(project_store, repo)
    assert len(_recall([project_store], "the", k=2)) <= 2


def test_recall_tracks_fact_usage(project_store):
    result = remember("Tracked fact about caching", store=project_store)
    recall("caching", stores=[project_store], track=True)

    assert project_store.get_fact(result.fact_id).hits == 1


def test_deleted_content_stops_being_searchable(project_store, repo):
    index_path(project_store, repo)
    assert _recall([project_store], "compute invoice total")

    (repo / "src" / "billing.py").unlink()
    index_path(project_store, repo)

    results = _recall([project_store], "compute invoice total")
    assert not any(r.path and r.path.endswith("billing.py") for r in results if r.kind == "chunk")
