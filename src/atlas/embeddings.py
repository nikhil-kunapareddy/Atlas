"""Optional semantic layer.

Core Atlas is lexical and dependency-free. This module is only live when the
`embeddings` extra is installed; every entry point degrades to None rather than
raising, so `import atlas` never depends on torch being present.

Vectors are stored as float32 blobs, L2-normalised at write time, so similarity
is a plain dot product.
"""

from __future__ import annotations

import array
import math
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from .store import Store

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass
class SentenceTransformerEmbedder:
    name: str
    dim: int
    _model: object

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(  # type: ignore[attr-defined]
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, v)) for v in vectors]


def load_embedder(model: str | None = None) -> Embedder | None:
    """Return an embedder, or None when the extra isn't installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    name = model or DEFAULT_MODEL
    st = SentenceTransformer(name)
    return SentenceTransformerEmbedder(
        name=name, dim=int(st.get_sentence_embedding_dimension()), _model=st
    )


def pack(vec: Sequence[float]) -> bytes:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return array.array("f", [x / norm for x in vec]).tobytes()


def unpack(blob: bytes) -> array.array:
    arr = array.array("f")
    arr.frombytes(blob)
    return arr


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def build_embeddings(
    store: Store,
    embedder: Embedder,
    progress: Callable[[str], None] | None = None,
) -> int:
    """Embed every chunk and fact that doesn't already have a current vector."""
    written = 0

    for kind, sql in (
        (
            "chunk",
            """
            SELECT c.id AS id, c.text AS text FROM chunks c
            LEFT JOIN vectors v ON v.kind = 'chunk' AND v.item_id = c.id AND v.model = ?
            WHERE v.item_id IS NULL
            """,
        ),
        (
            "fact",
            """
            SELECT f.id AS id, f.text AS text FROM facts f
            LEFT JOIN vectors v ON v.kind = 'fact' AND v.item_id = f.id AND v.model = ?
            WHERE v.item_id IS NULL
            """,
        ),
    ):
        rows = store.conn.execute(sql, (embedder.name,)).fetchall()
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start : start + BATCH_SIZE]
            vectors = embedder.encode([r["text"] for r in batch])
            store.conn.executemany(
                "INSERT INTO vectors(kind, item_id, model, dim, vec) VALUES(?, ?, ?, ?, ?) "
                "ON CONFLICT(kind, item_id) DO UPDATE SET "
                "model = excluded.model, dim = excluded.dim, vec = excluded.vec",
                [
                    (kind, int(r["id"]), embedder.name, embedder.dim, pack(v))
                    for r, v in zip(batch, vectors)
                ],
            )
            written += len(batch)
            if progress and written % (BATCH_SIZE * 4) == 0:
                progress(f"{written} embedded…")
        store.conn.commit()

    return written


def vector_search(
    store: Store, embedder: Embedder, query: str, limit: int = 40
) -> dict[tuple[str, int], float]:
    """Brute-force cosine search. Fine to tens of thousands of chunks.

    Beyond that this wants a real ANN index; the interface here is the seam
    where one would slot in.
    """
    rows = store.conn.execute(
        "SELECT kind, item_id, vec FROM vectors WHERE model = ?", (embedder.name,)
    ).fetchall()
    if not rows:
        return {}

    query_vec = embedder.encode([query])[0]
    norm = math.sqrt(sum(x * x for x in query_vec)) or 1.0
    query_vec = [x / norm for x in query_vec]

    scored = [
        ((r["kind"], int(r["item_id"])), dot(query_vec, unpack(r["vec"]))) for r in rows
    ]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return dict(scored[:limit])
