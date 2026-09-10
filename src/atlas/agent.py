"""The agent that answers questions about a graph.

`atlas ask "..."` runs a tool-use loop: Claude gets a handful of graph
primitives and decides how to combine them. The graph is never dumped into the
prompt — it is traversed, one call at a time, which is what lets a folder far
larger than any context window still be answerable.

The loop is written out by hand rather than delegated to the SDK's tool runner.
Two reasons: every tool result passes through `_format`, which is where the
token budget is actually enforced, and `--trace` needs to show the user each
call as it happens. Both want a loop we own.

Tool design follows one rule: **return citable identifiers, never prose.**
Every result carries the `uid` it came from, so the model can walk from a search
hit to its neighbours to its source document, and the final answer can name
`report.pdf p.2` instead of asserting something unattributable.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .graph import EdgeKind, GraphStore, NodeKind

DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 16_000

# Ceilings on what one tool call may return. A folder can contain a 40MB
# transcript; without these an agent can blow its own context in one call.
MAX_RESULTS = 25
MAX_BODY_CHARS = 1_200
MAX_TOOL_CHARS = 12_000


class AgentUnavailable(RuntimeError):
    """The agent cannot run — no SDK, or no credentials."""


SYSTEM = """\
You answer questions about a folder that has been converted into a property \
graph. You cannot see the folder. You can only see what the tools return, so \
investigate before answering.

THE GRAPH

Node kinds:
  folder    a directory
  document  one file of any type: pdf, spreadsheet, image, audio, video, text
  page      one page of a PDF, or one slide of a deck
  sheet     one worksheet of a workbook
  column    one column of a sheet, with inferred type and sample values
  chunk     a span of text. THE TEXT LIVES HERE. Pages, sheets and segments
            carry only a short preview; their chunks carry the full content.
  segment   a time range of audio or video, with start_time and end_time
  frame     a keyframe sampled from a video
  entity    something mentioned across files: an email, url, date, amount,
            phone number, or identifier such as an invoice or ticket number

Edge kinds:
  contains      folder->file, file->page, sheet->column, page->chunk
  next          ordering between consecutive pages, chunks or segments
  mentions      chunk->entity
  references    document->document, from a link between files
  derived_from  a chunk and what it was derived from
  related_to    entity->entity, labelled in the edge's props
  has_frame     segment->frame

Every edge carries a provenance: `extracted` was read directly off the bytes and
is exact; `inferred` was resolved by a rule; `llm` was proposed by a model and
may be wrong. Weight `related_to` and `llm` edges accordingly, and say so when
an answer leans on one.

HOW TO WORK

1. Call `overview` first when you do not already know what the folder holds.
2. `search_graph` to find a starting point. Search matches text, so search for
   words that would appear IN the documents, not for words describing them.
3. Entities are the join. To find everything about an invoice number, a person's
   email, or a date, find the entity node and call `get_neighbors` with
   direction "in" — that returns every chunk in every file that mentions it,
   which is how you connect a spreadsheet row to a PDF to a recording.
4. `get_node` on a chunk gives you its full text. Search previews are truncated;
   read the chunk before quoting it.
5. `find_path` explains how two things are connected when it is not obvious.

ANSWERING

Cite where each claim came from, using the human-readable name — `report.pdf
p.2`, `customers.csv [Sheet1]`, `standup.m4a @ 14:32`. If the graph does not
support an answer, say what you looked for and what was missing. Do not
speculate about file contents you have not read. If a document node carries a
`warnings` prop, that file was only partly readable — mention it if it matters \
to the question."""


@dataclass
class Trace:
    """A record of what the agent did, for `--trace` and for tests."""

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    def note(self, name: str, args: dict[str, Any]) -> None:
        self.calls.append((name, args))


class GraphAgent:
    """A conversation with one graph."""

    def __init__(
        self,
        graph: GraphStore,
        model: str = DEFAULT_MODEL,
        max_steps: int = 12,
        client: Any | None = None,
    ):
        self.graph = graph
        self.model = model
        self.max_steps = max_steps
        self.trace = Trace()
        self._client = client or _make_client()
        self._tools = _tool_schemas()
        self._handlers: dict[str, Callable[..., Any]] = {
            "overview": self._overview,
            "search_graph": self._search,
            "get_node": self._get_node,
            "get_neighbors": self._neighbors,
            "find_path": self._path,
        }

    # -- the loop ----------------------------------------------------------

    def ask(self, question: str, trace: bool = False) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

        for _ in range(self.max_steps):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM,
                    tools=self._tools,
                    thinking={"type": "adaptive"},
                    messages=messages,
                )
            except Exception as exc:
                # Credentials resolve lazily on the first request, so an auth
                # problem surfaces here rather than at construction. Translate
                # it: the raw SDK message does not say what to do about it.
                raise AgentUnavailable(_explain(exc)) from exc
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.trace.input_tokens += getattr(usage, "input_tokens", 0) or 0
                self.trace.output_tokens += getattr(usage, "output_tokens", 0) or 0

            if response.stop_reason == "refusal":
                return "The model declined to answer this question."

            # Append the whole content list, not just the text: thinking blocks
            # must be replayed unchanged on the next turn.
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                return _text_of(response) or "(no answer)"

            results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                if trace:
                    print(f"  · {block.name}({_brief(block.input)})")
                results.append(self._run(block))
            # Every tool_result for one assistant turn goes back in ONE user
            # message; splitting them teaches the model to stop calling tools
            # in parallel.
            messages.append({"role": "user", "content": results})

        return (
            f"Stopped after {self.max_steps} steps without reaching an answer. "
            "Re-run with --max-steps to allow more, or ask something narrower."
        )

    def _run(self, block: Any) -> dict[str, Any]:
        handler = self._handlers.get(block.name)
        self.trace.note(block.name, dict(block.input))
        if handler is None:
            return _result(block.id, f"no such tool: {block.name}", error=True)
        try:
            payload = handler(**block.input)
        except TypeError as exc:
            return _result(block.id, f"bad arguments: {exc}", error=True)
        except Exception as exc:
            return _result(block.id, f"{type(exc).__name__}: {exc}", error=True)
        return _result(block.id, _format(payload))

    # -- tools -------------------------------------------------------------

    def _overview(self) -> dict[str, Any]:
        counts = self.graph.counts()
        return {
            "totals": {"nodes": counts.get("nodes", 0), "edges": counts.get("edges", 0)},
            "node_kinds": {k[5:]: v for k, v in counts.items() if k.startswith("node:")},
            "edge_kinds": {k[5:]: v for k, v in counts.items() if k.startswith("edge:")},
            "documents": [
                {"uid": n.uid, "name": n.name, "modality": n.props.get("modality")}
                for n in self.graph.nodes_of_kind(NodeKind.DOCUMENT, limit=60)
            ],
            "top_entities": [
                {"uid": n.uid, "name": n.name, "type": n.props.get("entity_type"), "mentions": d}
                for n, d in self.graph.hubs(NodeKind.ENTITY, limit=20)
            ],
        }

    def _search(self, query: str, kind: str | None = None, limit: int = 10) -> Any:
        kinds = [NodeKind(kind)] if kind else None
        hits = self.graph.search(query, kinds=kinds, limit=min(limit, MAX_RESULTS))
        if not hits:
            return {"query": query, "results": [], "hint": "no matches; try other words"}
        return {
            "query": query,
            "results": [
                {
                    "uid": n.uid,
                    "kind": n.kind,
                    "name": n.name,
                    "score": round(s, 2),
                    "preview": n.preview(300),
                }
                for n, s in hits
            ],
        }

    def _get_node(self, uid: str) -> Any:
        node = self.graph.node(uid)
        if node is None:
            return {"error": f"no node with uid {uid!r}"}
        return {
            "uid": node.uid,
            "kind": node.kind,
            "name": node.name,
            "props": node.props,
            "text": node.body[:MAX_BODY_CHARS],
            "truncated": len(node.body) > MAX_BODY_CHARS,
        }

    def _neighbors(
        self,
        uid: str,
        edge_kind: str | None = None,
        direction: str = "both",
        limit: int = 20,
    ) -> Any:
        if self.graph.node(uid) is None:
            return {"error": f"no node with uid {uid!r}"}
        kinds = [EdgeKind(edge_kind)] if edge_kind else None
        found = self.graph.neighbors(
            uid, kinds=kinds, direction=direction, limit=min(limit, MAX_RESULTS)
        )
        return {
            "uid": uid,
            "neighbors": [
                {
                    "uid": n.node.uid,
                    "kind": n.node.kind,
                    "name": n.node.name,
                    "edge": n.edge_kind,
                    "direction": n.direction,
                    "provenance": n.provenance,
                    **({"label": n.label} if n.label else {}),
                    "preview": n.node.preview(200),
                }
                for n in found
            ],
        }

    def _path(self, source_uid: str, target_uid: str, max_depth: int = 6) -> Any:
        chain = self.graph.shortest_path(source_uid, target_uid, max_depth=max_depth)
        if not chain:
            return {"connected": False, "hops": None}
        return {
            "connected": True,
            "hops": len(chain) - 1,
            "path": [{"uid": n.uid, "kind": n.kind, "name": n.name} for n in chain],
        }


# -- schemas ---------------------------------------------------------------


def _tool_schemas() -> list[dict[str, Any]]:
    node_kinds = [k.value for k in NodeKind]
    edge_kinds = [k.value for k in EdgeKind]
    return [
        {
            "name": "overview",
            "description": (
                "What is in this graph: node and edge counts, every document with its "
                "modality, and the most-mentioned entities. Call this first when you do "
                "not yet know what the folder contains."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "search_graph",
            "description": (
                "Full-text search over node text, ranked by relevance. Matches the "
                "CONTENT of documents, so search for words that would appear inside "
                "them. Returns previews only — call get_node for a full chunk."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "words to look for"},
                    "kind": {
                        "type": "string",
                        "enum": node_kinds,
                        "description": "restrict to one node kind, e.g. 'chunk' or 'column'",
                    },
                    "limit": {"type": "integer", "description": "max results (default 10)"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_node",
            "description": (
                "Everything about one node: its full text, and all of its properties "
                "(page number, timestamp, column type, file path, warnings)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"uid": {"type": "string"}},
                "required": ["uid"],
            },
        },
        {
            "name": "get_neighbors",
            "description": (
                "One hop out from a node. This is the main way to traverse. On an "
                "entity use direction 'in' to get every chunk that mentions it across "
                "every file; on a document use direction 'out' to list its pages, "
                "sheets or segments."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "uid": {"type": "string"},
                    "edge_kind": {"type": "string", "enum": edge_kinds},
                    "direction": {"type": "string", "enum": ["out", "in", "both"]},
                    "limit": {"type": "integer"},
                },
                "required": ["uid"],
            },
        },
        {
            "name": "find_path",
            "description": (
                "The shortest chain of relationships connecting two nodes, for "
                "explaining how two files or entities relate."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "source_uid": {"type": "string"},
                    "target_uid": {"type": "string"},
                    "max_depth": {"type": "integer"},
                },
                "required": ["source_uid", "target_uid"],
            },
        },
    ]


# -- helpers ---------------------------------------------------------------


def _make_client() -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise AgentUnavailable(
            "the agent needs the llm extra — pip install 'atlas-context[llm]'"
        ) from exc
    try:
        # A bare constructor also picks up an `ant auth login` profile, so an
        # unset ANTHROPIC_API_KEY is not on its own an error.
        return anthropic.Anthropic()
    except Exception as exc:
        raise AgentUnavailable(f"could not create an Anthropic client: {exc}") from exc


def _explain(exc: Exception) -> str:
    """Turn an SDK exception into something a user can act on."""
    text = str(exc)
    lowered = f"{type(exc).__name__} {text}".lower()

    if any(word in lowered for word in ("authentication", "api_key", "credentials", "401")):
        return (
            "no Anthropic credentials found. Set one of:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "  ant auth login\n"
            "Building and querying a graph needs no credentials — only `atlas ask` "
            "and `atlas enrich` do."
        )
    if "rate" in lowered and "limit" in lowered:
        return f"rate limited by the API — wait and retry. ({text})"
    if any(word in lowered for word in ("connection", "timeout", "network")):
        return f"could not reach the API — check your network. ({text})"
    return f"{type(exc).__name__}: {text}"


def _result(tool_use_id: str, content: str, error: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if error:
        block["is_error"] = True
    return block


def _format(payload: Any) -> str:
    text = json.dumps(payload, indent=1, default=str)
    if len(text) <= MAX_TOOL_CHARS:
        return text
    return text[:MAX_TOOL_CHARS] + "\n… truncated; narrow the query or raise the limit"


def _text_of(response: Any) -> str:
    return "\n".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()


def _brief(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={json.dumps(v, default=str)[:48]}" for k, v in args.items())
