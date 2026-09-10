"""The agent's tool-use loop, driven by a scripted fake client.

No network and no API key: the loop, the tool dispatch, and the message
protocol are what these tests are about, and a real model would make them slow
and non-deterministic without testing any more of our code.

The protocol details asserted here are the ones that fail silently in
production — replaying assistant content verbatim (thinking blocks must survive)
and returning every tool_result in a single user message.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atlas.agent import GraphAgent
from atlas.graph import Edge, EdgeKind, Node, NodeKind


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def thinking_block():
    return SimpleNamespace(type="thinking", thinking="…")


def tool_block(name, args, block_id="tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=args, id=block_id)


def reply(content, stop_reason):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


class FakeClient:
    """Replays a scripted list of responses and records what it was sent."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.script:
            raise AssertionError("the agent made more calls than the script allows")
        return self.script.pop(0)


@pytest.fixture
def populated(graph):
    graph.upsert_node(Node("doc:a.pdf", NodeKind.DOCUMENT, "Report", "quarterly report"))
    graph.upsert_node(
        Node("doc:a.pdf#page:1", NodeKind.PAGE, "a.pdf p.1", "x" * 5000, {"page": 1})
    )
    graph.upsert_node(Node("ent:identifier:inv-1", NodeKind.ENTITY, "INV-1"))
    graph.add_edge(Edge("doc:a.pdf", "doc:a.pdf#page:1", EdgeKind.CONTAINS))
    graph.add_edge(Edge("doc:a.pdf#page:1", "ent:identifier:inv-1", EdgeKind.MENTIONS))
    return graph


def agent_with(graph, script, **kwargs):
    return GraphAgent(graph, client=FakeClient(script), **kwargs)


# -- the loop --------------------------------------------------------------


def test_answers_directly_when_no_tool_is_needed(populated):
    agent = agent_with(populated, [reply([text_block("42")], "end_turn")])
    assert agent.ask("how many?") == "42"
    assert agent.trace.calls == []


def test_runs_a_tool_then_answers(populated):
    agent = agent_with(
        populated,
        [
            reply([tool_block("search_graph", {"query": "quarterly"})], "tool_use"),
            reply([text_block("Found it in a.pdf.")], "end_turn"),
        ],
    )
    assert agent.ask("what is here?") == "Found it in a.pdf."
    assert agent.trace.calls == [("search_graph", {"query": "quarterly"})]
    assert agent.trace.input_tokens == 20


def test_assistant_content_is_replayed_verbatim(populated):
    """Thinking blocks must survive the round trip or the next call 400s."""
    assistant = [thinking_block(), tool_block("overview", {})]
    client = FakeClient([reply(assistant, "tool_use"), reply([text_block("done")], "end_turn")])
    GraphAgent(populated, client=client).ask("?")

    replayed = client.requests[1]["messages"][1]
    assert replayed["role"] == "assistant"
    assert replayed["content"] is assistant, "content must be passed through unchanged"


def test_parallel_tool_results_go_back_in_one_message(populated):
    """Splitting them teaches the model to stop calling tools in parallel."""
    calls = [
        tool_block("overview", {}, "tu_a"),
        tool_block("get_node", {"uid": "doc:a.pdf"}, "tu_b"),
    ]
    client = FakeClient([reply(calls, "tool_use"), reply([text_block("ok")], "end_turn")])
    GraphAgent(populated, client=client).ask("?")

    user_turn = client.requests[1]["messages"][2]
    assert user_turn["role"] == "user"
    assert [b["tool_use_id"] for b in user_turn["content"]] == ["tu_a", "tu_b"]


def test_step_limit_stops_an_agent_that_never_concludes(populated):
    script = [reply([tool_block("overview", {})], "tool_use") for _ in range(5)]
    agent = agent_with(populated, script, max_steps=3)
    answer = agent.ask("loop forever")
    assert "Stopped after 3 steps" in answer
    assert len(agent.trace.calls) == 3


def test_refusal_is_reported_not_crashed(populated):
    agent = agent_with(populated, [reply([], "refusal")])
    assert "declined" in agent.ask("something disallowed")


# -- tool behaviour --------------------------------------------------------


def test_tool_errors_come_back_as_tool_results_not_exceptions(populated):
    client = FakeClient(
        [
            reply([tool_block("get_node", {"uid": "doc:missing"})], "tool_use"),
            reply([text_block("no such node")], "end_turn"),
        ]
    )
    GraphAgent(populated, client=client).ask("?")
    result = client.requests[1]["messages"][2]["content"][0]
    assert "no node with uid" in result["content"]


def test_unknown_tool_and_bad_arguments_are_flagged(populated):
    for block, expected in (
        (tool_block("nonexistent", {}), "no such tool"),
        (tool_block("get_node", {"wrong": 1}), "bad arguments"),
    ):
        client = FakeClient([reply([block], "tool_use"), reply([text_block("ok")], "end_turn")])
        GraphAgent(populated, client=client).ask("?")
        result = client.requests[1]["messages"][2]["content"][0]
        assert result["is_error"] is True
        assert expected in result["content"]


def test_long_node_text_is_truncated_with_a_flag(populated):
    agent = agent_with(populated, [])
    payload = agent._get_node("doc:a.pdf#page:1")
    assert payload["truncated"] is True
    assert len(payload["text"]) <= 1200


def test_overview_summarises_without_dumping_the_graph(populated):
    agent = agent_with(populated, [])
    payload = agent._overview()
    assert payload["totals"]["nodes"] == 3
    assert {d["uid"] for d in payload["documents"]} == {"doc:a.pdf"}
    assert payload["top_entities"][0]["name"] == "INV-1"


def test_entity_traversal_returns_the_mentioning_chunks(populated):
    agent = agent_with(populated, [])
    payload = agent._neighbors("ent:identifier:inv-1", direction="in")
    assert [n["uid"] for n in payload["neighbors"]] == ["doc:a.pdf#page:1"]
    assert payload["neighbors"][0]["edge"] == "mentions"


def test_tools_advertise_only_real_kinds(populated):
    """The enums in the schemas must track the graph vocabulary."""
    agent = agent_with(populated, [])
    schema = next(t for t in agent._tools if t["name"] == "get_neighbors")
    assert set(schema["input_schema"]["properties"]["edge_kind"]["enum"]) == {
        k.value for k in EdgeKind
    }


# -- failure messages ------------------------------------------------------


class AuthlessClient:
    """Credentials resolve lazily, so auth failures surface on the first call."""

    def __init__(self, exc):
        self._exc = exc
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        raise self._exc


def test_missing_credentials_explain_themselves(populated):
    from atlas.agent import AgentUnavailable

    client = AuthlessClient(TypeError("Could not resolve authentication method."))
    with pytest.raises(AgentUnavailable) as caught:
        GraphAgent(populated, client=client).ask("?")

    message = str(caught.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "no credentials" in message or "needs no credentials" in message


@pytest.mark.parametrize(
    "exc,expected",
    [
        (RuntimeError("rate limit exceeded"), "rate limited"),
        (OSError("connection refused"), "could not reach the API"),
        (ValueError("something else entirely"), "ValueError"),
    ],
)
def test_other_api_failures_are_summarised(populated, exc, expected):
    from atlas.agent import AgentUnavailable

    with pytest.raises(AgentUnavailable) as caught:
        GraphAgent(populated, client=AuthlessClient(exc)).ask("?")
    assert expected in str(caught.value)
