"""Recall: rank facts and file chunks across both scopes.

Scoring is deliberately simple and explainable. Within each result list bm25 is
min-max normalised to 0..1, then multiplied by weights that encode three
judgements:

  * a hand-written fact beats a file chunk, because someone chose to record it
  * project knowledge beats global knowledge
  * pinned facts are the always-on layer, the part that replaces CLAUDE.md

Pinned facts are returned whether or not they match the query, since the point
of pinning is "the agent should know this regardless of what it asked".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .config import GLOBAL_SCOPE, PROJECT_SCOPE
from .store import Store, open_stores

FACT_WEIGHT = 1.30
CHUNK_WEIGHT = 1.0
SCOPE_WEIGHT = {PROJECT_SCOPE: 1.0, GLOBAL_SCOPE: 0.82}
PINNED_SCORE = 10.0
# How much a perfect cosine match can add on top of the lexical score. Kept
# below 1.0 so semantics reorder the lexical ranking rather than replace it.
SEMANTIC_WEIGHT = 0.75

QUOTED = re.compile(r'"([^"]+)"')
WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*")


@dataclass
class Result:
    kind: str  # "fact" | "chunk"
    scope: str
    score: float
    text: str
    ref: str
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    heading: str | None = None
    fact_id: int | None = None
    chunk_id: int | None = None
    tags: list[str] = field(default_factory=list)
    pinned: bool = False

    def snippet(self, width: int = 240) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= width else flat[: width - 1].rstrip() + "…"

    def to_dict(self) -> dict:
        data = {
            "kind": self.kind,
            "scope": self.scope,
            "score": round(self.score, 4),
            "ref": self.ref,
            "text": self.text,
        }
        if self.kind == "chunk":
            data.update(
                path=self.path,
                start_line=self.start_line,
                end_line=self.end_line,
                heading=self.heading,
            )
        else:
            data.update(id=self.fact_id, tags=self.tags, pinned=self.pinned)
        return data


def build_match_query(query: str) -> str | None:
    """Turn free text into a valid FTS5 MATCH expression.

    Raw user input is never safe to hand to MATCH — a stray quote or a bare
    `AND` is a syntax error — so tokens are extracted and re-quoted. Quoted
    spans in the input survive as phrases.
    """
    terms: list[str] = []
    remainder = query

    for phrase in QUOTED.findall(query):
        cleaned = phrase.replace('"', "").strip()
        if cleaned:
            terms.append(f'"{cleaned}"')
    remainder = QUOTED.sub(" ", remainder)

    for word in WORD.findall(remainder):
        token = word.strip("./-")
        if len(token) < 2:
            continue
        terms.append(f'"{token}"')

    if not terms:
        return None
    return " OR ".join(terms)


def query_terms(query: str) -> list[str]:
    """Lowercased bare terms, for highlighting and snippet centring."""
    terms = [p.lower() for p in QUOTED.findall(query)]
    terms += [w.lower().strip("./-") for w in WORD.findall(QUOTED.sub(" ", query))]
    return [t for t in terms if len(t) >= 2]


def _normalise(rows: Sequence[tuple[float, object]]) -> list[tuple[float, object]]:
    """Map raw bm25 (lower is better, negative) onto 0..1 (higher is better)."""
    if not rows:
        return []
    scores = [-r[0] for r in rows]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    out = []
    for score, payload in zip(scores, [r[1] for r in rows]):
        norm = 1.0 if span <= 0 else (score - lo) / span
        # Keep a floor so a sole result doesn't collapse to zero.
        out.append((0.35 + 0.65 * norm, payload))
    return out


def _search_facts(store: Store, match: str, limit: int) -> list[Result]:
    rows = store.conn.execute(
        """
        SELECT f.id, f.text, f.tags, f.pinned, bm25(facts_fts) AS score
        FROM facts_fts
        JOIN facts f ON f.id = facts_fts.rowid
        WHERE facts_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (match, limit),
    ).fetchall()

    weight = FACT_WEIGHT * SCOPE_WEIGHT.get(store.scope, 1.0)
    results = []
    for norm, row in _normalise([(r["score"], r) for r in rows]):
        if row["pinned"]:
            continue  # already surfaced by the pinned layer
        results.append(
            Result(
                kind="fact",
                scope=store.scope,
                score=norm * weight,
                text=row["text"],
                ref=f"fact#{row['id']}",
                fact_id=int(row["id"]),
                tags=[t for t in row["tags"].split(",") if t],
                pinned=False,
            )
        )
    return results


def _search_chunks(store: Store, match: str, limit: int) -> list[Result]:
    rows = store.conn.execute(
        """
        SELECT c.id, c.text, c.heading, c.start_line, c.end_line,
               fl.path AS path, bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        JOIN files  fl ON fl.id = c.file_id
        WHERE chunks_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (match, limit),
    ).fetchall()

    weight = CHUNK_WEIGHT * SCOPE_WEIGHT.get(store.scope, 1.0)
    results = []
    for norm, row in _normalise([(r["score"], r) for r in rows]):
        path = row["path"]
        results.append(
            Result(
                kind="chunk",
                scope=store.scope,
                score=norm * weight,
                text=row["text"],
                ref=f"{_display_path(path)}:{row['start_line']}-{row['end_line']}",
                path=path,
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                heading=row["heading"],
                chunk_id=int(row["id"]),
            )
        )
    return results


def _result_key(r: Result) -> tuple | None:
    if r.kind == "fact" and r.fact_id is not None:
        return ("fact", r.fact_id)
    if r.kind == "chunk" and r.chunk_id is not None:
        return ("chunk", r.chunk_id)
    return None


def _hydrate(store: Store, kind: str, item_id: int, score: float) -> Result | None:
    """Build a Result for a vector-only hit that lexical search missed."""
    if kind == "fact":
        row = store.conn.execute(
            "SELECT id, text, tags, pinned FROM facts WHERE id = ?", (item_id,)
        ).fetchone()
        if row is None or row["pinned"]:
            return None
        return Result(
            kind="fact",
            scope=store.scope,
            score=score,
            text=row["text"],
            ref=f"fact#{row['id']}",
            fact_id=int(row["id"]),
            tags=[t for t in row["tags"].split(",") if t],
        )

    row = store.conn.execute(
        """
        SELECT c.id, c.text, c.heading, c.start_line, c.end_line, fl.path AS path
        FROM chunks c JOIN files fl ON fl.id = c.file_id
        WHERE c.id = ?
        """,
        (item_id,),
    ).fetchone()
    if row is None:
        return None
    return Result(
        kind="chunk",
        scope=store.scope,
        score=score,
        text=row["text"],
        ref=f"{_display_path(row['path'])}:{row['start_line']}-{row['end_line']}",
        path=row["path"],
        start_line=int(row["start_line"]),
        end_line=int(row["end_line"]),
        heading=row["heading"],
        chunk_id=int(row["id"]),
    )


def _apply_semantic(
    store: Store, results: list[Result], embedder, query: str, wanted_kinds: set[str]
) -> list[Result]:
    """Add a cosine bonus to lexical hits and admit strong vector-only hits."""
    from .embeddings import vector_search

    hits = vector_search(store, embedder, query)
    if not hits:
        return results

    scope_weight = SCOPE_WEIGHT.get(store.scope, 1.0)
    in_scope = [r for r in results if r.scope == store.scope]
    by_key = {k: r for r in in_scope if (k := _result_key(r)) is not None}

    for (kind, item_id), cosine in hits.items():
        if kind not in wanted_kinds or cosine <= 0:
            continue
        bonus = SEMANTIC_WEIGHT * cosine * scope_weight
        existing = by_key.get((kind, item_id))
        if existing is not None:
            existing.score += bonus
        else:
            hydrated = _hydrate(store, kind, item_id, bonus)
            if hydrated is not None:
                results.append(hydrated)
    return results


def _pinned_facts(store: Store) -> list[Result]:
    rows = store.conn.execute(
        "SELECT id, text, tags FROM facts WHERE pinned = 1 ORDER BY updated_at DESC"
    ).fetchall()
    weight = SCOPE_WEIGHT.get(store.scope, 1.0)
    return [
        Result(
            kind="fact",
            scope=store.scope,
            score=PINNED_SCORE * weight,
            text=row["text"],
            ref=f"fact#{row['id']}",
            fact_id=int(row["id"]),
            tags=[t for t in row["tags"].split(",") if t],
            pinned=True,
        )
        for row in rows
    ]


def _display_path(path: str) -> str:
    """Shorten absolute paths against cwd and $HOME for readable refs."""
    p = Path(path)
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        pass
    try:
        return "~/" + str(p.relative_to(Path.home()))
    except ValueError:
        return str(p)


def recall(
    query: str,
    k: int = 8,
    root: Path | None = None,
    scopes: Iterable[str] = (PROJECT_SCOPE, GLOBAL_SCOPE),
    kinds: Iterable[str] = ("fact", "chunk"),
    include_pinned: bool = True,
    track: bool = True,
    semantic: bool = False,
    stores: Sequence[Store] | None = None,
) -> list[Result]:
    """Rank facts and chunks for `query` across the requested scopes."""
    match = build_match_query(query)
    wanted_scopes = set(scopes)
    wanted_kinds = set(kinds)

    owned = stores is None
    store_list = list(stores) if stores is not None else list(open_stores(root))

    embedder = None
    if semantic:
        from .embeddings import load_embedder

        embedder = load_embedder()

    results: list[Result] = []
    try:
        for store in store_list:
            if store.scope not in wanted_scopes:
                continue
            if include_pinned and "fact" in wanted_kinds:
                results.extend(_pinned_facts(store))
            if match is not None:
                # Over-fetch so cross-scope fusion has something to choose from.
                pool = max(k * 3, 20)
                if "fact" in wanted_kinds:
                    results.extend(_search_facts(store, match, pool))
                if "chunk" in wanted_kinds:
                    results.extend(_search_chunks(store, match, pool))
            if embedder is not None:
                results = _apply_semantic(store, results, embedder, query, wanted_kinds)

        results.sort(key=lambda r: r.score, reverse=True)
        top = _dedupe(results)[:k]

        if track:
            for store in store_list:
                ids = [
                    r.fact_id
                    for r in top
                    if r.kind == "fact" and r.scope == store.scope and r.fact_id is not None
                ]
                if ids:
                    store.record_fact_hits(ids)
        return top
    finally:
        if owned:
            for store in store_list:
                store.close()


def _dedupe(results: list[Result]) -> list[Result]:
    """Collapse overlapping chunks from the same file; keep the best-scoring one."""
    seen: set[tuple] = set()
    out: list[Result] = []
    for r in results:
        if r.kind == "fact":
            key = ("fact", r.scope, r.fact_id)
        else:
            key = ("chunk", r.path, r.start_line)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out
