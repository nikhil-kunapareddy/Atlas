"""LLM enrichment, driven by a fake client.

The properties worth protecting are about trust and money: guessed knowledge
must be labelled as guessed, an API failure must leave the deterministic graph
intact, and the same text must never be paid for twice.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atlas import enrich as enrich_mod
from atlas.graph import EdgeKind, Node, NodeKind


def tool_use(passages):
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="tool_use", name="record_knowledge", input={"passages": passages})
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class BoomClient:
    def __init__(self):
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        raise RuntimeError("api exploded")


@pytest.fixture
def chunked(graph, project_store):
    source_id = project_store.add_source(".")
    file_id = project_store.upsert_file(source_id, "/a.md", 1.0, 10, "d")
    body = (
        "Ana Rodriguez leads the migration for Acme Robotics, which acquired "
        "Globex last spring. The work is coordinated out of the Lisbon office."
    )
    graph.upsert_node(
        Node("doc:a.md#chunk:0", NodeKind.CHUNK, "a.md:1-3", body, {"path": "a.md"}),
        owner_file_id=file_id,
    )
    return graph, body


# -- writing ---------------------------------------------------------------


def test_entities_and_relations_are_written(chunked):
    graph, _ = chunked
    client = FakeClient(
        [
            tool_use(
                [
                    {
                        "id": "0",
                        "entities": [
                            {"name": "Ana Rodriguez", "type": "person"},
                            {"name": "Acme Robotics", "type": "org"},
                            {"name": "Lisbon", "type": "place"},
                        ],
                        "relations": [
                            {
                                "source": "Ana Rodriguez",
                                "target": "Acme Robotics",
                                "label": "leads migration for",
                            }
                        ],
                    }
                ]
            )
        ]
    )
    stats = enrich_mod.enrich(graph, client=client)

    assert stats.entities == 3 and stats.relations == 1
    assert graph.node("ent:person:ana rodriguez") is not None
    assert graph.node("ent:place:lisbon") is not None

    related = graph.neighbors(
        "ent:person:ana rodriguez", kinds=[EdgeKind.RELATED_TO], direction="out"
    )
    assert related[0].label == "leads migration for"


def test_guessed_knowledge_is_labelled_as_guessed(chunked):
    """An agent must be able to tell an inference from a fact."""
    graph, _ = chunked
    client = FakeClient(
        [tool_use([{"id": "0", "entities": [{"name": "Acme", "type": "org"}], "relations": []}])]
    )
    enrich_mod.enrich(graph, client=client)

    mention = graph.neighbors("ent:org:acme", direction="in")[0]
    assert mention.provenance == "llm"
    assert graph.node("ent:org:acme").props["source"] == "llm"


def test_relations_between_unextracted_entities_are_dropped(chunked):
    """A relation must be traceable to the passage that supports it."""
    graph, _ = chunked
    client = FakeClient(
        [
            tool_use(
                [
                    {
                        "id": "0",
                        "entities": [{"name": "Acme", "type": "org"}],
                        "relations": [
                            {"source": "Acme", "target": "Someone Unmentioned", "label": "knows"},
                            {"source": "Acme", "target": "Acme", "label": "is"},
                        ],
                    }
                ]
            )
        ]
    )
    stats = enrich_mod.enrich(graph, client=client)
    assert stats.relations == 0


def test_unknown_entity_types_fall_back_to_concept(chunked):
    graph, _ = chunked
    client = FakeClient(
        [tool_use([{"id": "0", "entities": [{"name": "X", "type": "nonsense"}], "relations": []}])]
    )
    enrich_mod.enrich(graph, client=client)
    assert graph.node("ent:concept:x") is not None


# -- money -----------------------------------------------------------------


def test_identical_text_is_never_paid_for_twice(chunked):
    graph, _ = chunked
    payload = [{"id": "0", "entities": [{"name": "Acme", "type": "org"}], "relations": []}]
    client = FakeClient([tool_use(payload)])

    first = enrich_mod.enrich(graph, client=client)
    second = enrich_mod.enrich(graph, client=client)

    assert client.calls == 1, "the second run must be served entirely from cache"
    assert first.chunks_sent == 1
    assert second.chunks_cached == 1 and second.chunks_sent == 0
    assert second.input_tokens == 0


def test_cost_is_computed_from_real_usage(chunked):
    graph, _ = chunked
    client = FakeClient([tool_use([{"id": "0", "entities": [], "relations": []}])])
    stats = enrich_mod.enrich(graph, model="claude-haiku-4-5", client=client)
    # 1000 in @ $1/Mtok + 200 out @ $5/Mtok
    assert stats.cost("claude-haiku-4-5") == pytest.approx(0.002)


def test_estimate_runs_without_spending_anything(chunked):
    graph, _ = chunked
    forecast = enrich_mod.estimate(graph, "claude-opus-5")
    assert forecast["chunks"] == 1
    assert forecast["cost"] > 0
    assert enrich_mod.estimate(graph, "claude-haiku-4-5")["cost"] < forecast["cost"]


# -- failure -------------------------------------------------------------


def test_an_api_failure_leaves_the_extracted_graph_untouched(chunked):
    graph, _ = chunked
    before = graph.counts()

    stats = enrich_mod.enrich(graph, client=BoomClient())

    assert stats.errors and "api exploded" in stats.errors[0]
    assert graph.counts() == before
    assert stats.entities == 0


def test_prose_instead_of_a_tool_call_is_recorded_as_an_error(chunked):
    graph, _ = chunked
    client = FakeClient(
        [
            SimpleNamespace(
                content=[SimpleNamespace(type="text", text="I could not do that")],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            )
        ]
    )
    stats = enrich_mod.enrich(graph, client=client)
    assert stats.entities == 0
    assert any("structured" in e for e in stats.errors)


def test_short_chunks_are_not_sent(graph, project_store):
    source_id = project_store.add_source(".")
    file_id = project_store.upsert_file(source_id, "/a.md", 1.0, 10, "d")
    graph.upsert_node(Node("doc:a.md#chunk:0", NodeKind.CHUNK, "tiny", "too short"), file_id)
    client = FakeClient([])
    stats = enrich_mod.enrich(graph, client=client)
    assert client.calls == 0 and stats.chunks_seen == 0
