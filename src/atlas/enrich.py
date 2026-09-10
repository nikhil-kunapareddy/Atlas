"""Optional semantic enrichment.

`entities.py` finds what a regex can prove: emails, dates, amounts, invoice
numbers. This module finds what only a reader can: the people, organisations,
places and concepts a document is *about*, and how they relate.

It is off by default and costs money, so three things are true of every design
choice here:

**Nothing it writes is confused with fact.** Entities and relations from this
pass carry `Provenance.LLM`, and the agent's system prompt tells it to hedge on
those. A deterministic edge and a guessed one are never indistinguishable.

**It never pays twice for the same content.** Results are cached against the
SHA-256 of the chunk text, not the file, so moving a paragraph between documents
is free, and re-running enrichment after editing one file re-bills only that
file's chunks.

**It degrades to a no-op.** No API key, no `anthropic` package, or a mid-run
API error leaves the deterministic graph exactly as it was.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from .graph import (
    Edge,
    EdgeKind,
    EntityType,
    GraphStore,
    Node,
    NodeKind,
    Provenance,
    entity_uid,
)

VERSION = "1"
DEFAULT_MODEL = "claude-opus-5"

# How much chunk text goes into one request. Large enough that the instructions
# are amortised, small enough that one failure loses little work.
BATCH_CHARS = 12_000
MAX_CHUNKS_PER_BATCH = 12
MIN_CHUNK_CHARS = 120

# Per million tokens, for the estimate printed before a run. Input/output.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

SEMANTIC_TYPES = ("person", "org", "place", "product", "event", "concept")

SYSTEM = """\
You extract structured knowledge from passages of documents so they can be \
linked in a graph.

Return, for each passage you are given:
  entities  — the people, organisations, places, products, events and concepts \
the passage is genuinely ABOUT. Use the name as written. Do not extract emails, \
URLs, dates, amounts or ticket numbers: those are already handled exactly, and \
duplicating them adds noise.
  relations — connections stated or clearly implied BETWEEN two extracted \
entities, with a short lowercase verb phrase as the label ("works at", \
"acquired", "reports to", "supersedes").

Rules that matter more than coverage:
- Extract only what the passage supports. Do not use outside knowledge, and do \
not infer a relation because it is plausible in general.
- Skip document furniture: headers, footers, "Table of Contents", "Best \
regards", page numbers, boilerplate legal text.
- Prefer the most specific correct name ("Acme Robotics" over "the company"), \
and use the same spelling every time an entity recurs so it resolves to one node.
- A passage with nothing worth extracting should return empty lists. That is a \
normal and useful answer.\
"""

TOOL = {
    "name": "record_knowledge",
    "description": "Record the entities and relations found in each passage.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "passages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string", "description": "the passage id given to you"},
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": "string", "enum": list(SEMANTIC_TYPES)},
                                },
                                "required": ["name", "type"],
                            },
                        },
                        "relations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "source": {"type": "string"},
                                    "target": {"type": "string"},
                                    "label": {"type": "string"},
                                },
                                "required": ["source", "target", "label"],
                            },
                        },
                    },
                    "required": ["id", "entities", "relations"],
                },
            }
        },
        "required": ["passages"],
    },
}


@dataclass
class EnrichStats:
    chunks_seen: int = 0
    chunks_sent: int = 0
    chunks_cached: int = 0
    entities: int = 0
    relations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: list[str] = field(default_factory=list)

    def cost(self, model: str) -> float:
        rate_in, rate_out = PRICES.get(model, PRICES[DEFAULT_MODEL])
        return (self.input_tokens * rate_in + self.output_tokens * rate_out) / 1e6

    def summary(self, model: str) -> str:
        bits = [
            f"{self.chunks_sent} passages read",
            f"{self.entities} entities",
            f"{self.relations} relations",
        ]
        if self.chunks_cached:
            bits.append(f"{self.chunks_cached} cached")
        if self.errors:
            bits.append(f"{len(self.errors)} errors")
        return f"{', '.join(bits)} — ${self.cost(model):.4f}"


def estimate(graph: GraphStore, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """What a full enrichment pass would cost, before spending anything."""
    rows = graph.conn.execute(
        "SELECT count(*) AS n, coalesce(sum(length(body)), 0) AS chars"
        " FROM nodes WHERE kind = ? AND length(body) >= ?",
        (str(NodeKind.CHUNK), MIN_CHUNK_CHARS),
    ).fetchone()
    chunks, chars = int(rows["n"]), int(rows["chars"])
    # ~4 characters per token, plus the system prompt once per batch, and an
    # output roughly a fifth the size of the input.
    batches = max(1, chunks // MAX_CHUNKS_PER_BATCH)
    input_tokens = chars / 4 + batches * len(SYSTEM) / 4
    output_tokens = input_tokens * 0.2
    rate_in, rate_out = PRICES.get(model, PRICES[DEFAULT_MODEL])
    return {
        "chunks": chunks,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "cost": (input_tokens * rate_in + output_tokens * rate_out) / 1e6,
        "model": model,
    }


def enrich(
    graph: GraphStore,
    model: str = DEFAULT_MODEL,
    limit: int | None = None,
    progress: Callable[[str], None] | None = None,
    client: Any | None = None,
) -> EnrichStats:
    """Add semantic entities and relations to every text-bearing chunk."""
    stats = EnrichStats()
    api = client or _client()

    pending: list[tuple[str, str]] = []
    for uid, body in _chunks(graph, limit):
        stats.chunks_seen += 1
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        cached = graph.cache_get(digest, f"enrich-{model}", VERSION)
        if cached is not None:
            stats.chunks_cached += 1
            _write(graph, uid, cached, stats)
            continue
        pending.append((uid, body))

        if _batch_full(pending):
            _run_batch(graph, api, model, pending, stats, progress)
            pending = []

    if pending:
        _run_batch(graph, api, model, pending, stats, progress)

    graph.resolve_pending()
    graph.commit()
    return stats


# -- batching --------------------------------------------------------------


def _chunks(graph: GraphStore, limit: int | None) -> Iterator[tuple[str, str]]:
    sql = (
        "SELECT uid, body FROM nodes WHERE kind = ? AND length(body) >= ? ORDER BY uid"
    )
    params: list[Any] = [str(NodeKind.CHUNK), MIN_CHUNK_CHARS]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    for row in graph.conn.execute(sql, params).fetchall():
        yield row["uid"], row["body"]


def _batch_full(pending: Sequence[tuple[str, str]]) -> bool:
    if len(pending) >= MAX_CHUNKS_PER_BATCH:
        return True
    return sum(len(body) for _, body in pending) >= BATCH_CHARS


def _run_batch(
    graph: GraphStore,
    client: Any,
    model: str,
    batch: list[tuple[str, str]],
    stats: EnrichStats,
    progress: Callable[[str], None] | None,
) -> None:
    """One API call for a group of passages."""
    numbered = {str(i): (uid, body) for i, (uid, body) in enumerate(batch)}
    prompt = "\n\n".join(
        f"<passage id=\"{i}\">\n{body[:4000]}\n</passage>" for i, (_, body) in numbered.items()
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=8000,
            system=SYSTEM,
            tools=[TOOL],
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        stats.errors.append(f"{type(exc).__name__}: {exc}")
        return

    usage = getattr(response, "usage", None)
    if usage is not None:
        stats.input_tokens += getattr(usage, "input_tokens", 0) or 0
        stats.output_tokens += getattr(usage, "output_tokens", 0) or 0
    stats.chunks_sent += len(batch)

    payload = _tool_payload(response)
    if payload is None:
        # The model answered in prose instead of calling the tool. Nothing to
        # write, but the deterministic graph is untouched — that is the point.
        stats.errors.append("model returned no structured result for a batch")
        return

    for passage in payload.get("passages", []):
        entry = numbered.get(str(passage.get("id")))
        if entry is None:
            continue
        uid, body = entry
        record = {
            "entities": passage.get("entities", []),
            "relations": passage.get("relations", []),
        }
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        graph.cache_put(digest, f"enrich-{model}", VERSION, record)
        _write(graph, uid, record, stats)

    if progress:
        progress(f"  {stats.chunks_sent} passages · ${stats.cost(model):.4f}")


def _tool_payload(response: Any) -> dict[str, Any] | None:
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "tool_use" and block.name == TOOL["name"]:
            return dict(block.input)
    return None


# -- writing ---------------------------------------------------------------


def _write(graph: GraphStore, chunk_uid: str, record: dict[str, Any], stats: EnrichStats) -> None:
    """Turn one passage's result into nodes and edges.

    The owning file is read off the chunk so MENTIONS edges are deleted with it
    on reindex, exactly like the deterministic ones.
    """
    row = graph.conn.execute(
        "SELECT owner_file_id FROM nodes WHERE uid = ?", (chunk_uid,)
    ).fetchone()
    owner = row["owner_file_id"] if row else None

    resolved: dict[str, str] = {}
    for item in record.get("entities", []):
        name = str(item.get("name", "")).strip()
        raw_type = str(item.get("type", "concept")).strip().lower()
        if not name or len(name) > 120:
            continue
        try:
            etype = EntityType(raw_type)
        except ValueError:
            etype = EntityType.CONCEPT
        uid = entity_uid(etype, name)
        graph.upsert_node(
            Node(
                uid=uid,
                kind=NodeKind.ENTITY,
                name=name,
                body=f"{etype}: {name}",
                props={"entity_type": str(etype), "source": "llm"},
            )
        )
        graph.add_edge(
            Edge(chunk_uid, uid, EdgeKind.MENTIONS, provenance=Provenance.LLM),
            owner_file_id=owner,
        )
        resolved[name.casefold()] = uid
        stats.entities += 1

    for item in record.get("relations", []):
        source = resolved.get(str(item.get("source", "")).strip().casefold())
        target = resolved.get(str(item.get("target", "")).strip().casefold())
        label = str(item.get("label", "")).strip()[:60]
        # Only relate entities this same passage actually named, so a relation
        # can always be traced back to the text that supports it.
        if not source or not target or source == target or not label:
            continue
        graph.add_edge(
            Edge(
                source,
                target,
                EdgeKind.RELATED_TO,
                provenance=Provenance.LLM,
                props={"label": label, "evidence": chunk_uid},
            ),
            owner_file_id=owner,
        )
        stats.relations += 1


def _client() -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "enrichment needs the llm extra — pip install 'atlas-context[llm]'"
        ) from exc
    return anthropic.Anthropic()
