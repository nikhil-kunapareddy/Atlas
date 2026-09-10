#!/usr/bin/env python3
"""Measure how Atlas performs on a real folder.

    python scripts/make_corpus.py --out ./benchmark-corpus --size-mb 100
    python scripts/benchmark.py --corpus ./benchmark-corpus

Reports four things, because they fail independently:

* **cold build** — throughput in MB/s and files/s from an empty database,
  broken down by modality so a slow extractor is visible rather than averaged
  away;
* **warm rebuild** — the no-op case. This should be dominated by hashing, and
  it is the number that decides whether re-running Atlas is cheap enough to put
  in a hook;
* **incremental** — one file edited. Should cost roughly one file, not one
  folder;
* **query latency** — p50/p95 over the read paths an agent actually calls, since
  a graph that is slow to query is not useful however fast it built.

Peak RSS is tracked throughout: streaming extraction is a design claim, and this
is what would catch it regressing into loading whole files into memory.
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import shutil
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from atlas.build import build  # noqa: E402
from atlas.extract import ExtractOptions, missing_extras  # noqa: E402
from atlas.graph import GraphStore, NodeKind  # noqa: E402
from atlas.store import Store  # noqa: E402

MB = 1024 * 1024


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return usage / MB if sys.platform == "darwin" else usage / 1024


def folder_bytes(root: Path) -> tuple[int, int]:
    total = count = 0
    for path in root.rglob("*"):
        if path.is_file() and ".atlas" not in path.parts:
            total += path.stat().st_size
            count += 1
    return total, count


def open_graph(root: Path) -> tuple[Store, GraphStore]:
    store = Store(root / ".atlas" / "store.db", "project")
    return store, GraphStore(store.conn)


def timed(fn):
    started = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - started


def run_build(root: Path, options: ExtractOptions, label: str) -> dict:
    store, graph = open_graph(root)
    try:
        stats, elapsed = timed(lambda: build(root, store, graph, options))
        counts = graph.counts()
    finally:
        store.close()
    return {
        "label": label,
        "seconds": elapsed,
        "files_indexed": stats.files_indexed,
        "files_unchanged": stats.files_unchanged,
        "files_removed": stats.files_removed,
        "nodes": counts.get("nodes", 0),
        "edges": counts.get("edges", 0),
        "by_modality": dict(stats.by_modality),
        "warnings": len(stats.warnings),
        "errors": stats.errors[:10],
        "node_kinds": {k[5:]: v for k, v in counts.items() if k.startswith("node:")},
        "edge_kinds": {k[5:]: v for k, v in counts.items() if k.startswith("edge:")},
    }


def sample_terms(graph, sample: int = 400) -> tuple[str, str]:
    """Pick a common and a rare term from the corpus itself.

    Hardcoding query words measures how often they miss, not how fast search is.
    The first version of this benchmark asked a corpus of government documents
    about "invoice shipment" — vocabulary from the synthetic generator — and
    reported 0.04ms because almost nothing matched. Terms have to come from the
    text actually indexed.
    """
    from collections import Counter

    words: Counter[str] = Counter()
    rows = graph.conn.execute(
        "SELECT body FROM nodes WHERE kind = 'chunk' AND length(body) > 200"
        " ORDER BY id LIMIT ?",
        (sample,),
    )
    for row in rows:
        for token in re.findall(r"[A-Za-z]{4,}", row["body"][:2000]):
            words[token.lower()] += 1

    if not words:
        return "the", "zzz"
    ranked = words.most_common()
    common = " ".join(w for w, _ in ranked[:2])
    # Something seen a handful of times: rare enough to be selective, present
    # enough that the query is not a guaranteed miss.
    tail = [w for w, n in ranked if 2 <= n <= 5]
    rare = tail[len(tail) // 2] if tail else ranked[-1][0]
    return common, rare


def measure_queries(root: Path, repeats: int = 20) -> dict:
    """Latency of the read paths the agent and MCP tools use."""
    store, graph = open_graph(root)
    try:
        common, rare = sample_terms(graph)
        hub_entities = graph.hubs(NodeKind.ENTITY, limit=5)
        busiest = hub_entities[0][0].uid if hub_entities else None
        documents = graph.nodes_of_kind(NodeKind.DOCUMENT, limit=2)
        pair = [d.uid for d in documents][:2]

        cases: dict[str, callable] = {
            f"search common ({common})": lambda: graph.search(common, limit=20),
            f"search rare ({rare})": lambda: graph.search(rare, limit=20),
            "search kind-filtered": lambda: graph.search(
                common, kinds=[NodeKind.CHUNK], limit=20
            ),
            "hubs": lambda: graph.hubs(limit=20),
            "counts": graph.counts,
        }
        if busiest:
            cases["neighbors (busiest entity)"] = lambda: graph.neighbors(
                busiest, direction="in", limit=50
            )
        if len(pair) == 2:
            cases["shortest_path (2 docs)"] = lambda: graph.shortest_path(pair[0], pair[1])

        results = {}
        for name, call in cases.items():
            hits = len(call() or [])
            samples = []
            for _ in range(repeats):
                _, seconds = timed(call)
                samples.append(seconds * 1000)
            samples.sort()
            results[name] = {
                "p50_ms": round(statistics.median(samples), 2),
                "p95_ms": round(samples[int(len(samples) * 0.95) - 1], 2),
                "results": hits,
            }
        return results
    finally:
        store.close()


def graph_health(root: Path) -> dict:
    """Sanity signals: is the graph actually connected and joined?"""
    store, graph = open_graph(root)
    try:
        counts = graph.counts()
        entities = counts.get("node:entity", 0)
        mentions = counts.get("edge:mentions", 0)
        multi_file = graph.conn.execute(
            """
            SELECT count(*) AS n FROM (
                SELECT e.dst_id
                FROM edges e
                JOIN nodes d ON d.id = e.dst_id
                JOIN nodes s ON s.id = e.src_id
                WHERE e.kind = 'mentions' AND d.kind = 'entity'
                GROUP BY e.dst_id
                HAVING count(DISTINCT json_extract(s.props, '$.path')) > 1
            )
            """
        ).fetchone()["n"]
        orphan_docs = graph.conn.execute(
            """SELECT count(*) AS n FROM nodes n WHERE n.kind = 'document'
               AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.src_id = n.id)"""
        ).fetchone()["n"]
        return {
            "entities": entities,
            "mentions": mentions,
            "entities_spanning_multiple_files": multi_file,
            "documents_with_no_children": orphan_docs,
            "mentions_per_entity": round(mentions / entities, 1) if entities else 0,
        }
    finally:
        store.close()


def fmt_build(row: dict, size_mb: float, files: int) -> str:
    """One build result. Throughput is only shown for the cold run.

    A warm or single-file run divided by corpus size produces an impressive and
    entirely meaningless MB/s, so those rows report wall time instead.
    """
    head = f"  {row['label']:<22} {row['seconds']:8.2f}s"
    if row.get("show_throughput") and row["seconds"] > 0.05:
        head += f"   {size_mb / row['seconds']:6.1f} MB/s   {files / row['seconds']:6.0f} files/s"
    elif row["files_indexed"]:
        head += f"   ({row['files_indexed']} file(s) re-read)"
    else:
        head += "   (nothing re-read)"
    return (
        head
        + f"\n  {'':<22} {row['files_indexed']} indexed, {row['files_unchanged']} unchanged, "
        f"{row['nodes']:,} nodes, {row['edges']:,} edges"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", default="./benchmark-corpus")
    parser.add_argument("--json", metavar="FILE", help="also write results as JSON")
    parser.add_argument("--transcribe", action="store_true",
                        help="run speech transcription (slow; needs the media extra)")
    parser.add_argument("--keep", action="store_true", help="keep the graph afterwards")
    args = parser.parse_args()

    root = Path(args.corpus).expanduser().resolve()
    if not root.is_dir():
        print(f"no corpus at {root} — run scripts/make_corpus.py first", file=sys.stderr)
        return 1

    total_bytes, files = folder_bytes(root)
    size_mb = total_bytes / MB
    options = ExtractOptions(transcribe=args.transcribe, keyframes=0,
                             media_dir=root / ".atlas" / "media")

    print(f"Corpus  {size_mb:.1f} MB · {files} files · {root}")
    missing = missing_extras()
    if missing:
        print(f"Missing extras: {', '.join(missing)} — those modalities degrade to metadata")
    if not args.transcribe:
        print("Transcription off (pass --transcribe to include it)")
    print()

    # Always start cold, or the first number measures nothing.
    shutil.rmtree(root / ".atlas", ignore_errors=True)

    results = {"corpus_mb": round(size_mb, 1), "files": files, "builds": []}

    print("BUILD")
    cold = run_build(root, options, "cold (empty db)")
    cold["show_throughput"] = True
    print(fmt_build(cold, size_mb, cold["files_indexed"]))
    results["builds"].append(cold)

    warm = run_build(root, options, "warm (no changes)")
    print(fmt_build(warm, size_mb, files))
    results["builds"].append(warm)

    victim = next(root.rglob("engineering/docs/note-*.md"), None)
    if victim:
        victim.write_text(victim.read_text() + "\n\nAppended for the incremental test.\n")
        incremental = run_build(root, options, "incremental (1 edit)")
        print(fmt_build(incremental, size_mb, 1))
        results["builds"].append(incremental)

    print()
    print("GRAPH")
    print("  nodes  " + "  ".join(f"{k}={v:,}" for k, v in sorted(cold["node_kinds"].items())))
    print("  edges  " + "  ".join(f"{k}={v:,}" for k, v in sorted(cold["edge_kinds"].items())))
    print("  files  " + "  ".join(f"{k}={v}" for k, v in sorted(cold["by_modality"].items())))
    db = (root / ".atlas" / "store.db").stat().st_size / MB
    print(f"  store  {db:.1f} MB on disk ({db / size_mb * 100:.0f}% of the corpus)")
    results["store_mb"] = round(db, 1)

    health = graph_health(root)
    results["health"] = health
    print()
    print("JOINS")
    print(f"  {health['entities']:,} entities · {health['mentions']:,} mentions "
          f"({health['mentions_per_entity']} per entity)")
    print(f"  {health['entities_spanning_multiple_files']:,} entities appear in >1 file "
          "— these are the cross-document links")
    if health["documents_with_no_children"]:
        print(f"  {health['documents_with_no_children']} documents yielded no structure")

    print()
    print("QUERIES")
    queries = measure_queries(root)
    results["queries"] = queries
    for name, row in queries.items():
        print(f"  {name:<32} p50 {row['p50_ms']:7.2f} ms   p95 {row['p95_ms']:7.2f} ms"
              f"   ({row['results']} hits)")

    results["peak_rss_mb"] = round(peak_rss_mb(), 1)
    print()
    print(f"Peak RSS {results['peak_rss_mb']:.0f} MB")
    if cold["errors"]:
        print("\nErrors:")
        for error in cold["errors"]:
            print(f"  {error}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.json}")
    if not args.keep:
        shutil.rmtree(root / ".atlas", ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
