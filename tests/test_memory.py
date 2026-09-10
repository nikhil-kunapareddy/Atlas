from __future__ import annotations

import pytest

from atlas.memory import export_markdown, remember, similarity


def test_remember_creates_a_fact(project_store):
    result = remember("Tests need Redis running", store=project_store)
    assert result.action == "created"
    assert project_store.counts()["facts"] == 1


def test_near_identical_facts_are_merged(project_store):
    remember("Tests are run with pytest, not unittest", store=project_store)
    second = remember("tests are run with pytest and not unittest", store=project_store)

    assert second.action == "updated"
    assert project_store.counts()["facts"] == 1


def test_distinct_facts_are_kept_separate(project_store):
    remember("Tests need Redis running", store=project_store)
    remember("Deploys go through the release pipeline", store=project_store)
    assert project_store.counts()["facts"] == 2


def test_merge_preserves_tags_from_both_writes(project_store):
    remember("Run tests with make test", tags=["testing"], store=project_store)
    result = remember("run the tests with make test", tags=["ci"], store=project_store)

    fact = project_store.get_fact(result.fact_id)
    assert set(fact.tag_list) == {"testing", "ci"}


def test_related_facts_are_reported(project_store):
    remember("Deploys go through the release pipeline", store=project_store)
    result = remember("Deploys through the release pipeline require approval", store=project_store)

    assert result.action == "created"
    assert result.related, "a partially overlapping fact should be flagged as related"


def test_updating_a_fact_refreshes_the_search_index(project_store):
    result = remember("The cache uses memcached", store=project_store)
    project_store.update_fact(result.fact_id, "The cache uses redis")

    rows = project_store.conn.execute(
        "SELECT body FROM facts_fts WHERE rowid = ?", (result.fact_id,)
    ).fetchall()
    assert len(rows) == 1, "stale FTS rows would make old text searchable forever"
    assert "redis" in rows[0]["body"]
    assert "memcached" not in rows[0]["body"]


def test_empty_text_is_rejected(project_store):
    with pytest.raises(ValueError):
        remember("   ", store=project_store)


def test_deleting_a_fact_clears_its_index_row(project_store):
    result = remember("Something forgettable", store=project_store)
    assert project_store.delete_fact(result.fact_id) is True

    rows = project_store.conn.execute("SELECT count(*) FROM facts_fts").fetchone()[0]
    assert rows == 0
    assert project_store.delete_fact(result.fact_id) is False


def test_similarity_bounds():
    assert similarity("identical text here", "identical text here") == 1.0
    assert similarity("completely different", "wholly unrelated words") == 0.0


def test_export_markdown_groups_pinned_and_tagged(project_store):
    remember("Always use uv", pinned=True, store=project_store)
    remember("Migrations live in db/migrations", tags=["layout"], store=project_store)

    md = export_markdown(project_store.all_facts())
    assert "## Always" in md
    assert "Always use uv" in md
    assert "## layout" in md


def test_export_markdown_handles_empty():
    assert "No facts recorded yet" in export_markdown([])
