"""The graph half of the command line: `atlas <folder>` and everything you can
ask of the result.

Kept separate from `cli.py` so each file stays readable, and registered into the
same parser so the user sees one tool.

Where the graph lives: building a folder creates `<folder>/.atlas/store.db`,
making that folder its own Atlas root. Running any query command from inside it
therefore finds the right graph with no arguments, the same way git finds its
repo — and `--store` overrides when you want to query from elsewhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .build import build
from .config import PROJECT_SCOPE, STORE_DIRNAME, find_project_root, project_db_path
from .extract import ExtractOptions, capabilities, missing_extras
from .graph import EdgeKind, GraphStore, NodeKind
from .store import Store
from .term import bold, cyan, dim, green, yellow


def _fail(message: str) -> int:
    print(f"atlas: {message}", file=sys.stderr)
    return 1


# -- store resolution ------------------------------------------------------


def _resolve_root(explicit: str | None) -> Path | None:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        return root if root.is_dir() else None
    return find_project_root()


def _open(args: argparse.Namespace, create: bool = False) -> tuple[Store, GraphStore] | None:
    """Open the graph the user means, or explain why we can't find one."""
    root = _resolve_root(getattr(args, "store", None))
    if root is None:
        print(
            "atlas: no graph here. Build one first:\n  atlas <folder>",
            file=sys.stderr,
        )
        return None
    path = project_db_path(root)
    if not create and not path.exists():
        print(f"atlas: no graph at {path}. Build one:\n  atlas {root}", file=sys.stderr)
        return None
    store = Store(path, PROJECT_SCOPE)
    return store, GraphStore(store.conn)


def _options(args: argparse.Namespace, root: Path) -> ExtractOptions:
    return ExtractOptions(
        transcribe=not getattr(args, "no_transcribe", False),
        whisper_model=getattr(args, "whisper", "base"),
        ocr=getattr(args, "ocr", False),
        keyframes=0 if getattr(args, "no_keyframes", False) else getattr(args, "keyframes", 4),
        language=getattr(args, "language", None),
        media_dir=root / STORE_DIRNAME / "media",
    )


# -- commands --------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> int:
    root = Path(args.folder).expanduser().resolve()
    if not root.exists():
        return _fail(f"{root} does not exist")
    if not root.is_dir():
        return _fail(f"{root} is a file — pass the folder that contains it")

    (root / STORE_DIRNAME).mkdir(parents=True, exist_ok=True)
    store = Store(project_db_path(root), PROJECT_SCOPE)
    graph = GraphStore(store.conn)

    missing = missing_extras()
    if missing and not args.quiet:
        print(dim(f"note: not installed — {', '.join(missing)} (see: atlas doctor)"))

    print(f"Building graph from {cyan(str(root))}…")
    try:
        stats = build(
            root,
            store,
            graph,
            options=_options(args, root),
            progress=None if args.quiet else (lambda m: print(dim(m))),
            prune=not args.no_prune,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        store.close()
        return _fail(str(exc))

    print()
    print(f"  {bold(stats.summary())}")
    if stats.by_modality:
        spread = ", ".join(f"{n} {m}" for m, n in stats.by_modality.most_common())
        print(dim(f"  {spread}"))
    if stats.entities_linked:
        print(dim(f"  {stats.entities_linked} entity mentions linked"))

    if stats.unreadable:
        # Grouped rather than one line per file: a real archive can contain
        # hundreds of these, and the useful fact is the shape of the gap.
        spread = ", ".join(f"{ext} ({n})" for ext, n in stats.unreadable.most_common(6))
        total = sum(stats.unreadable.values())
        print(yellow(f"  ! {total} files not read — {spread}"))
        print(dim("    indexed by name and metadata only; see `atlas doctor`"))

    # The per-file "cannot read this" lines are already covered by the grouped
    # count above; showing them again would bury everything else.
    shown = [
        w for w in stats.warnings
        if "no extractor handles" not in w and "legacy OLE2" not in w
    ]
    for warning in shown[:8]:
        print(yellow(f"  ! {warning}"))
    if len(shown) > 8:
        print(dim(f"    …and {len(shown) - 8} more warnings"))
    for error in stats.errors[:5]:
        print(yellow(f"  ✗ {error}"))

    print()
    _print_hubs(graph, limit=5, prefix="  ")
    print()
    print("Ask it something:")
    print(dim(f'  atlas ask "what is in this folder?" --store {root}'))
    store.close()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    opened = _open(args)
    if opened is None:
        return 1
    store, graph = opened
    kinds = [NodeKind(k) for k in args.kind] if args.kind else None
    results = graph.search(" ".join(args.query), kinds=kinds, limit=args.k)

    if args.json:
        print(json.dumps([_node_json(n, score) for n, score in results], indent=2))
    elif not results:
        print("No matches.")
    else:
        for index, (node, score) in enumerate(results, start=1):
            print(f"{index:2d}. {_label(node)}  {dim(f'{score:.2f}')}")
            print(f"    {dim(node.preview(220))}")
    store.close()
    return 0


def cmd_node(args: argparse.Namespace) -> int:
    opened = _open(args)
    if opened is None:
        return 1
    store, graph = opened
    node = graph.node(args.uid)
    if node is None:
        store.close()
        return _fail(f"no node with uid {args.uid!r}")

    if args.json:
        print(json.dumps(_node_json(node), indent=2))
    else:
        print(bold(node.name))
        print(dim(f"{node.kind}  {node.uid}"))
        if node.props:
            for key, value in sorted(node.props.items()):
                print(f"  {key}: {_short(value)}")
        if node.body:
            print()
            print(node.body if args.full else node.preview(600))
    store.close()
    return 0


def cmd_neighbors(args: argparse.Namespace) -> int:
    opened = _open(args)
    if opened is None:
        return 1
    store, graph = opened
    kinds = [EdgeKind(k) for k in args.edge] if args.edge else None
    found = graph.neighbors(args.uid, kinds=kinds, direction=args.direction, limit=args.k)

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "edge": n.edge_kind,
                        "direction": n.direction,
                        "provenance": n.provenance,
                        **_node_json(n.node),
                    }
                    for n in found
                ],
                indent=2,
            )
        )
    elif not found:
        print("No neighbours.")
    else:
        for neighbor in found:
            arrow = "→" if neighbor.direction == "out" else "←"
            tag = "" if neighbor.provenance == "extracted" else dim(f" [{neighbor.provenance}]")
            print(f"  {arrow} {yellow(neighbor.edge_kind):22s} {_label(neighbor.node)}{tag}")
    store.close()
    return 0


def cmd_path(args: argparse.Namespace) -> int:
    opened = _open(args)
    if opened is None:
        return 1
    store, graph = opened
    chain = graph.shortest_path(args.source, args.target, max_depth=args.max_depth)
    if args.json:
        print(json.dumps([_node_json(n) for n in chain], indent=2))
    elif not chain:
        print(f"No path within {args.max_depth} hops.")
    else:
        for depth, node in enumerate(chain):
            print(f"  {'  ' * depth}{'└─ ' if depth else ''}{_label(node)}")
    store.close()
    return 0


def cmd_hubs(args: argparse.Namespace) -> int:
    opened = _open(args)
    if opened is None:
        return 1
    store, graph = opened
    kind = NodeKind(args.kind) if args.kind else None
    if args.json:
        print(
            json.dumps(
                [{**_node_json(n), "degree": d} for n, d in graph.hubs(kind, args.k)], indent=2
            )
        )
    else:
        _print_hubs(graph, limit=args.k, kind=kind)
    store.close()
    return 0


def cmd_graph_status(args: argparse.Namespace) -> int:
    opened = _open(args)
    if opened is None:
        return 1
    store, graph = opened
    counts = graph.counts()

    if args.json:
        print(json.dumps({"store": str(store.path), "counts": counts}, indent=2))
        store.close()
        return 0

    print(bold(str(store.path)))
    print(f"  {counts['nodes']} nodes, {counts['edges']} edges")
    print()
    print(dim("  nodes"))
    for key, value in counts.items():
        if key.startswith("node:"):
            print(f"    {key[5:]:12s} {value:>7d}")
    print(dim("  edges"))
    for key, value in counts.items():
        if key.startswith("edge:"):
            print(f"    {key[5:]:12s} {value:>7d}")
    sources = store.sources()
    if sources:
        print()
        print(dim("  sources"))
        for row in sources:
            print(f"    {row['path']}")
    store.close()
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report which modalities this installation can actually read."""
    print(bold("Atlas capabilities"))
    print()
    for capability in capabilities():
        mark = green("✓") if capability["available"] else yellow("✗")
        name = str(capability["module"])
        print(f"  {mark} {name:16s} {capability['modality']!s:10s} {capability['description']}")
        if not capability["available"]:
            print(dim(f"      pip install 'atlas-context[{capability['extra']}]'"))

    import shutil

    print()
    print(bold("External tools"))
    for tool, why in (
        ("ffprobe", "duration and codec metadata for audio/video"),
        ("ffmpeg", "video keyframe extraction"),
        ("tesseract", "the OCR engine behind --ocr"),
    ):
        found = shutil.which(tool)
        mark = green("✓") if found else yellow("✗")
        print(f"  {mark} {tool:16s} {dim(found or why)}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    opened = _open(args)
    if opened is None:
        return 1
    store, graph = opened
    nodes = [_node_json(n) for n in graph.all_nodes(limit=args.limit)]
    uids = {n["uid"] for n in nodes}
    edges = [
        {
            "source": edge.src_uid,
            "target": edge.dst_uid,
            "kind": edge.kind,
            "weight": edge.weight,
            "provenance": edge.provenance,
            **({"label": edge.props["label"]} if "label" in edge.props else {}),
        }
        for edge in graph.all_edges()
        # A node cap can leave edges pointing at nodes that were not exported;
        # dropping them keeps the file loadable by any graph library.
        if edge.src_uid in uids and edge.dst_uid in uids
    ]

    payload = {
        "version": 1,
        "root": str(store.path.parent.parent),
        "nodes": nodes,
        "edges": edges,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {len(nodes)} nodes and {len(edges)} edges to {cyan(args.out)}")
    else:
        print(text)
    store.close()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Hand the graph to an agent and let it answer in natural language."""
    opened = _open(args)
    if opened is None:
        return 1
    store, graph = opened
    try:
        from .agent import AgentUnavailable, GraphAgent

        agent = GraphAgent(graph, model=args.model, max_steps=args.max_steps)
        if args.trace:
            print(dim(f"asking {args.model}…"))
        answer = agent.ask(" ".join(args.question), trace=args.trace)
    except AgentUnavailable as exc:
        return _fail(str(exc))
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")
    finally:
        store.close()

    print()
    print(answer)
    if args.trace:
        used = agent.trace
        print()
        print(dim(f"{len(used.calls)} tool calls · {used.input_tokens} in / {used.output_tokens} out tokens"))
    return 0


def cmd_enrich(args: argparse.Namespace) -> int:
    """Add LLM-inferred entities and relations on top of the extracted graph."""
    opened = _open(args)
    if opened is None:
        return 1
    store, graph = opened
    from . import enrich as enrich_mod

    forecast = enrich_mod.estimate(graph, args.model)
    price = bold(f"${forecast['cost']:.2f}")
    print(
        f"{forecast['chunks']} passages · ~{forecast['input_tokens']:,} input tokens · "
        f"estimated {price} on {args.model}"
    )
    if args.estimate_only:
        store.close()
        return 0
    if not args.yes:
        try:
            if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                store.close()
                print("Cancelled.")
                return 0
        except EOFError:
            store.close()
            return _fail("not a terminal — pass --yes to run non-interactively")

    try:
        stats = enrich_mod.enrich(
            graph,
            model=args.model,
            limit=args.limit,
            progress=None if args.quiet else (lambda m: print(dim(m))),
        )
    except RuntimeError as exc:
        return _fail(str(exc))
    finally:
        store.close()

    print(f"  {bold(stats.summary(args.model))}")
    for error in stats.errors[:5]:
        print(yellow(f"  ! {error}"))
    return 0


# -- formatting ------------------------------------------------------------


def _print_hubs(graph: GraphStore, limit: int, kind: Any = None, prefix: str = "") -> None:
    rows = graph.hubs(kind, limit)
    if not rows:
        return
    print(f"{prefix}{dim('most connected')}")
    for node, degree in rows:
        print(f"{prefix}  {degree:4d}  {_label(node)}")


def _label(node: Any) -> str:
    return f"{yellow(f'[{node.kind}]')} {node.name}  {dim(node.uid)}"


def _node_json(node: Any, score: float | None = None) -> dict[str, Any]:
    payload = {
        "uid": node.uid,
        "kind": node.kind,
        "name": node.name,
        "props": node.props,
        "preview": node.preview(300),
    }
    if score is not None:
        payload["score"] = round(score, 3)
    return payload


def _short(value: Any, limit: int = 120) -> str:
    text = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
    return text[: limit - 1] + "…" if len(text) > limit else text


# -- parser registration ---------------------------------------------------

GRAPH_COMMANDS = frozenset(
    {
        "build", "ask", "search", "node", "neighbors", "path",
        "hubs", "graph", "export", "enrich", "doctor",
    }
)


def register(sub: Any) -> None:
    """Add every graph command to the shared subparser."""

    def store_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--store",
            metavar="FOLDER",
            help="the built folder to query (default: found by walking up from cwd)",
        )

    p = sub.add_parser("build", help="convert a folder into a graph")
    p.add_argument("folder", help="folder to read")
    p.add_argument("--ocr", action="store_true", help="read text inside images")
    p.add_argument("--no-transcribe", action="store_true", help="skip audio/video transcription")
    p.add_argument("--whisper", default="base", help="whisper model size (default: base)")
    p.add_argument("--language", help="force a transcription language, e.g. en")
    p.add_argument("--keyframes", type=int, default=4, help="video keyframes per file")
    p.add_argument("--no-keyframes", action="store_true")
    p.add_argument("--no-prune", action="store_true", help="keep nodes for deleted files")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("ask", help="ask an AI agent about the graph")
    p.add_argument("question", nargs="+")
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--trace", action="store_true", help="show the agent's tool calls")
    store_arg(p)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("search", help="full-text search over graph nodes")
    p.add_argument("query", nargs="+")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--kind", action="append", choices=[k.value for k in NodeKind])
    p.add_argument("--json", action="store_true")
    store_arg(p)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("node", help="show one node by uid")
    p.add_argument("uid")
    p.add_argument("--full", action="store_true")
    p.add_argument("--json", action="store_true")
    store_arg(p)
    p.set_defaults(func=cmd_node)

    p = sub.add_parser("neighbors", help="one hop out from a node")
    p.add_argument("uid")
    p.add_argument("-k", type=int, default=25)
    p.add_argument("--direction", choices=["out", "in", "both"], default="both")
    p.add_argument("--edge", action="append", choices=[k.value for k in EdgeKind])
    p.add_argument("--json", action="store_true")
    store_arg(p)
    p.set_defaults(func=cmd_neighbors)

    p = sub.add_parser("path", help="shortest path between two nodes")
    p.add_argument("source")
    p.add_argument("target")
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--json", action="store_true")
    store_arg(p)
    p.set_defaults(func=cmd_path)

    p = sub.add_parser("hubs", help="the most connected nodes")
    p.add_argument("-k", type=int, default=20)
    p.add_argument("--kind", choices=[k.value for k in NodeKind])
    p.add_argument("--json", action="store_true")
    store_arg(p)
    p.set_defaults(func=cmd_hubs)

    p = sub.add_parser("graph", help="graph statistics")
    p.add_argument("--json", action="store_true")
    store_arg(p)
    p.set_defaults(func=cmd_graph_status)

    p = sub.add_parser("export", help="write the graph as JSON")
    p.add_argument("--out", metavar="FILE")
    p.add_argument("--limit", type=int, default=5000, help="max nodes per kind")
    p.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    store_arg(p)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("enrich", help="add LLM-inferred entities and relationships")
    p.add_argument("--model", default="claude-opus-5",
                   help="use claude-haiku-4-5 for bulk work at a fifth the price")
    p.add_argument("--limit", type=int, help="only enrich this many passages")
    p.add_argument("--estimate-only", action="store_true", help="print the cost and stop")
    p.add_argument("-y", "--yes", action="store_true", help="skip the cost confirmation")
    p.add_argument("-q", "--quiet", action="store_true")
    store_arg(p)
    p.set_defaults(func=cmd_enrich)

    p = sub.add_parser("doctor", help="show which modalities this install can read")
    p.set_defaults(func=cmd_doctor)
