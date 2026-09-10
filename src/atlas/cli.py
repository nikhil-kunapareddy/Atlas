"""Command line interface.

Two surfaces share one tool. `atlas <folder>` builds a graph out of a directory
and everything under `cli_graph.py` queries it; the commands in this file manage
the fact store that agents read and write. They live in the same database and
the same process, so `atlas status` can describe both.

The bare form is the headline: `atlas ~/research` is shorthand for
`atlas build ~/research`, resolved in `main()` before argparse sees it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__, cli_graph
from .config import (
    GLOBAL_SCOPE,
    PROJECT_SCOPE,
    STORE_DIRNAME,
    find_project_root,
    global_db_path,
    project_db_path,
)
from .indexer import index_path, reindex_all
from .memory import (
    export_markdown,
    forget,
    list_facts,
    remember,
    resolve_scope,
    set_pinned,
)
from .search import recall
from .store import open_store, open_stores

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def dim(t: str) -> str:
    return _c(t, "2")


def bold(t: str) -> str:
    return _c(t, "1")


def cyan(t: str) -> str:
    return _c(t, "36")


def yellow(t: str) -> str:
    return _c(t, "33")


def green(t: str) -> str:
    return _c(t, "32")


def _fail(message: str) -> int:
    print(f"atlas: {message}", file=sys.stderr)
    return 1


# -- commands --------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.path or Path.cwd()).expanduser().resolve()
    if not root.is_dir():
        return _fail(f"{root} is not a directory")

    store_dir = root / STORE_DIRNAME
    already = store_dir.exists()
    store_dir.mkdir(parents=True, exist_ok=True)

    store = open_store(PROJECT_SCOPE, root)
    assert store is not None
    try:
        print(f"{'Reusing' if already else 'Created'} {dim(str(store_dir))}")
        if args.no_index:
            print("Skipping initial index (--no-index).")
        else:
            print(f"Indexing {cyan(str(root))}…")
            stats = index_path(store, root, progress=lambda m: print(dim(m)))
            print(f"  {stats.summary()}")
            for err in stats.errors[:5]:
                print(dim(f"  ! {err}"))
    finally:
        store.close()

    print()
    print("Next: register the MCP server with your agent —")
    print(dim("  claude mcp add atlas -- atlas mcp"))
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    root = find_project_root()
    try:
        scope, resolved_root = resolve_scope(args.scope, root)
    except ValueError as exc:
        return _fail(str(exc))

    store = open_store(scope, resolved_root)
    if store is None:
        return _fail("no store available")

    try:
        for raw in args.paths:
            target = Path(raw).expanduser().resolve()
            if not target.exists():
                _fail(f"{target} does not exist")
                continue
            print(f"Indexing {cyan(str(target))} into {scope} store…")
            stats = index_path(store, target, progress=lambda m: print(dim(m)))
            print(f"  {stats.summary()}")
            for err in stats.errors[:5]:
                print(dim(f"  ! {err}"))
    finally:
        store.close()
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    total_sources = 0
    for store in open_stores(create=False):
        try:
            source_count = store.counts()["sources"]
            if source_count == 0:
                continue
            total_sources += source_count
            print(f"{bold(store.scope)} store:")
            stats = reindex_all(store, progress=lambda m: print(dim(f"  {m}")))
            print(f"  {stats.summary()}")
        finally:
            store.close()
    if total_sources == 0:
        print("Nothing indexed yet. Try `atlas init` or `atlas index <folder>`.")
    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    query = " ".join(args.query)
    kinds = ("fact",) if args.facts_only else ("fact", "chunk")
    scopes = (args.scope,) if args.scope else (PROJECT_SCOPE, GLOBAL_SCOPE)

    if args.semantic:
        from .embeddings import load_embedder

        if load_embedder() is None:
            return _fail(
                "--semantic needs the embeddings extra — "
                "`pip install 'atlas-context[embeddings]'`, then `atlas embed`"
            )

    results = recall(
        query,
        k=args.k,
        scopes=scopes,
        kinds=kinds,
        include_pinned=not args.no_pinned,
        semantic=args.semantic,
    )

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0

    if not results:
        print(dim("No matches. Index a folder, or record what you know with `atlas remember`."))
        return 0

    for i, r in enumerate(results, start=1):
        tag = "fact" if r.kind == "fact" else "file"
        marker = yellow("pinned") if r.pinned else dim(r.scope)
        label = cyan(r.ref) if r.kind == "chunk" else green(r.ref)
        print(f"{bold(f'{i:>2}.')} [{tag}] {label}  {marker}  {dim(f'{r.score:.2f}')}")
        if r.heading:
            print(f"    {dim(r.heading)}")
        body = r.text if args.full else r.snippet()
        for line in body.splitlines():
            print(f"    {line}")
        if r.tags:
            print(f"    {dim('#' + ' #'.join(r.tags))}")
        print()
    return 0


def cmd_remember(args: argparse.Namespace) -> int:
    text = " ".join(args.text).strip()
    if not text:
        return _fail("nothing to remember")
    try:
        result = remember(text, scope=args.scope, tags=args.tag or (), pinned=args.pin)
    except ValueError as exc:
        return _fail(str(exc))

    verb = "Updated" if result.action == "updated" else "Remembered"
    pin_note = yellow(" (pinned)") if args.pin else ""
    print(f"{verb} {green(f'fact#{result.fact_id}')} in {bold(result.scope)} scope{pin_note}")
    print(f"  {text}")
    if result.related:
        print(dim(f"  related: {', '.join(f'fact#{f.id}' for f in result.related[:5])}"))
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    if forget(args.id, scope=args.scope):
        print(f"Forgot fact#{args.id}")
        return 0
    return _fail(f"no fact#{args.id} found")


def cmd_pin(args: argparse.Namespace) -> int:
    pinned = args.command == "pin"
    if set_pinned(args.id, pinned, scope=args.scope):
        print(f"{'Pinned' if pinned else 'Unpinned'} fact#{args.id}")
        return 0
    return _fail(f"no fact#{args.id} found")


def cmd_facts(args: argparse.Namespace) -> int:
    scopes = (args.scope,) if args.scope else (PROJECT_SCOPE, GLOBAL_SCOPE)
    facts = list_facts(scopes=scopes, tag=args.tag)

    if args.export:
        print(export_markdown(facts))
        return 0
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": f.id,
                        "scope": f.scope,
                        "text": f.text,
                        "tags": f.tag_list,
                        "pinned": f.pinned,
                        "hits": f.hits,
                    }
                    for f in facts
                ],
                indent=2,
            )
        )
        return 0

    if not facts:
        print(dim("No facts yet. Record one with `atlas remember \"...\"`."))
        return 0

    for f in facts:
        marker = yellow("*") if f.pinned else " "
        meta = dim(f"{f.scope}  {f.hits} hits")
        print(f"{marker} {green(f'#{f.id}'):>6}  {f.text}")
        trailer = f"         {meta}"
        if f.tag_list:
            trailer += dim(f"  #{' #'.join(f.tag_list)}")
        print(trailer)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    root = find_project_root()
    print(bold("Atlas"), dim(f"v{__version__}"))
    print()

    if root:
        print(f"Project  {cyan(str(root))}")
        print(dim(f"         {project_db_path(root)}"))
    else:
        print(f"Project  {dim('none — run `atlas init` in a repo')}")
    print(f"Global   {dim(str(global_db_path()))}")
    print()

    for store in open_stores(create=False):
        try:
            counts = store.counts()
            if not any(counts.values()):
                continue
            print(bold(f"{store.scope} store"))
            print(
                f"  {counts['files']} files, {counts['chunks']} chunks, "
                f"{counts['facts']} facts"
                + (f", {counts['vectors']} vectors" if counts["vectors"] else "")
            )
            for src in store.sources():
                seen = src["last_indexed_at"]
                when = "never" if not seen else _ago(seen)
                print(dim(f"    {src['path']}  ({when})"))
        finally:
            store.close()
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    if args.remove:
        target = Path(args.remove).expanduser().resolve()
        removed = False
        for store in open_stores(create=False):
            try:
                if store.remove_source(target):
                    print(f"Removed {target} from {store.scope} store")
                    removed = True
            finally:
                store.close()
        return 0 if removed else _fail(f"{target} is not an indexed source")

    for store in open_stores(create=False):
        try:
            sources = store.sources()
            if sources:
                print(bold(f"{store.scope} store"))
                for src in sources:
                    print(f"  {src['path']}")
        finally:
            store.close()
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    from .embeddings import build_embeddings, load_embedder

    embedder = load_embedder(args.model)
    if embedder is None:
        return _fail(
            "embeddings extra not installed — `pip install 'atlas-context[embeddings]'`"
        )
    for store in open_stores(create=False):
        try:
            if not any(store.counts().values()):
                continue
            print(f"Embedding {store.scope} store with {embedder.name}…")
            written = build_embeddings(store, embedder, progress=lambda m: print(dim(f"  {m}")))
            print(f"  {written} vectors written")
        finally:
            store.close()
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    try:
        from .mcp_server import run
    except ImportError:
        return _fail("MCP extra not installed — `pip install 'atlas-context[mcp]'`")
    run()
    return 0


def _ago(ts: float) -> str:
    import time

    delta = max(0, time.time() - ts)
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if delta >= size:
            return f"{int(delta // size)}{unit} ago"
    return "just now"


# -- parser ----------------------------------------------------------------


KNOWN_COMMANDS: set[str] = set()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas",
        description="Turn a folder into a graph your agent can query.",
        epilog="Start with:  atlas <folder>",
    )
    parser.add_argument("--version", action="version", version=f"atlas {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    cli_graph.register(sub)

    def scope_arg(p: argparse.ArgumentParser, help_text: str) -> None:
        p.add_argument(
            "--scope",
            choices=[PROJECT_SCOPE, GLOBAL_SCOPE],
            default=None,
            help=help_text,
        )

    p = sub.add_parser("init", help="set up Atlas in this repo and index it")
    p.add_argument("path", nargs="?", help="directory to initialise (default: cwd)")
    p.add_argument("--no-index", action="store_true", help="create the store but don't index")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("index", help="index one or more folders")
    p.add_argument("paths", nargs="+")
    scope_arg(p, "which store to index into (default: project when in one)")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("reindex", help="refresh every folder already registered")
    p.set_defaults(func=cmd_reindex)

    p = sub.add_parser("recall", help="search facts and indexed files")
    p.add_argument("query", nargs="+")
    p.add_argument("-k", type=int, default=8, help="number of results (default: 8)")
    p.add_argument("--full", action="store_true", help="print whole chunks, not snippets")
    p.add_argument("--facts-only", action="store_true", help="skip file chunks")
    p.add_argument("--no-pinned", action="store_true", help="omit the always-on pinned facts")
    p.add_argument(
        "--semantic",
        action="store_true",
        help="blend in embedding similarity (needs the embeddings extra)",
    )
    p.add_argument("--json", action="store_true")
    scope_arg(p, "restrict to one scope")
    p.set_defaults(func=cmd_recall)

    p = sub.add_parser("remember", help="record a durable fact")
    p.add_argument("text", nargs="+")
    p.add_argument("--tag", action="append", help="repeatable")
    p.add_argument("--pin", action="store_true", help="always surface this in recall")
    scope_arg(p, "where to store it (default: project when in one)")
    p.set_defaults(func=cmd_remember)

    p = sub.add_parser("forget", help="delete a fact by id")
    p.add_argument("id", type=int)
    scope_arg(p, "restrict to one scope")
    p.set_defaults(func=cmd_forget)

    for name, help_text in (("pin", "add a fact to the always-on layer"),
                            ("unpin", "remove a fact from the always-on layer")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("id", type=int)
        scope_arg(p, "restrict to one scope")
        p.set_defaults(func=cmd_pin)

    p = sub.add_parser("facts", help="list recorded facts")
    p.add_argument("--tag")
    p.add_argument("--json", action="store_true")
    p.add_argument("--export", action="store_true", help="render as markdown")
    scope_arg(p, "restrict to one scope")
    p.set_defaults(func=cmd_facts)

    p = sub.add_parser("status", help="show stores, sources and counts")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("sources", help="list or remove indexed folders")
    p.add_argument("--remove", metavar="PATH")
    p.set_defaults(func=cmd_sources)

    p = sub.add_parser("embed", help="build embeddings (needs the embeddings extra)")
    p.add_argument("--model", default=None)
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("mcp", help="run the MCP server on stdio")
    p.set_defaults(func=cmd_mcp)

    KNOWN_COMMANDS.update(sub.choices)
    return parser


def _expand_default_command(argv: list[str]) -> list[str]:
    """Rewrite `atlas <folder>` as `atlas build <folder>`.

    Only when the first argument is not a known command and does name a real
    directory — so a typo like `atlas serach` still gets argparse's "invalid
    choice" error rather than being misread as a path.
    """
    if not argv or argv[0].startswith("-") or argv[0] in KNOWN_COMMANDS:
        return argv
    try:
        if Path(argv[0]).expanduser().is_dir():
            return ["build", *argv]
    except OSError:
        pass
    return argv


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(_expand_default_command(raw))
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
