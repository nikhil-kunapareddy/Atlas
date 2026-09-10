"""MCP server — the interface coding agents actually use.

Tool descriptions here are load-bearing. An agent decides whether to call
`atlas_recall` based on nothing but the text below, so each description says
when to reach for the tool, not just what it does.

Requires the `mcp` extra: pip install 'atlas-context[mcp]'
"""

from __future__ import annotations

from pathlib import Path

from mcp.server import MCPServer

from . import __version__
from .config import GLOBAL_SCOPE, PROJECT_SCOPE, find_project_root, project_name
from .indexer import index_path
from .mcp_graph import GRAPH_INSTRUCTIONS
from .mcp_graph import register as register_graph_tools
from .memory import forget as forget_fact
from .memory import list_facts, remember, resolve_scope, set_pinned
from .search import recall
from .store import open_store, open_stores

INSTRUCTIONS = """\
Atlas is this project's memory. It holds two things: durable facts someone \
recorded about how this codebase works, and a searchable index of its files.

Call `atlas_recall` early when you need project context — conventions, where \
something lives, why a thing is the way it is, how to run tests. It is faster \
than grepping and it surfaces knowledge that isn't written in any file.

Call `atlas_remember` when you learn something durable that a future session \
would otherwise have to rediscover: a non-obvious command, a constraint, a \
gotcha, a decision and its reason. Record the fact, not the transcript. Do not \
record transient state ("the test is failing right now") or anything already \
obvious from the code.
"""

# The graph half of the server explains itself separately; clients see both.
INSTRUCTIONS = f"{INSTRUCTIONS}\n\n{GRAPH_INSTRUCTIONS}"

server: MCPServer = MCPServer(
    name="atlas",
    version=__version__,
    instructions=INSTRUCTIONS,
)


register_graph_tools(server)


def _root() -> Path | None:
    return find_project_root()


def _scope_label(scope: str) -> str:
    return "project" if scope == PROJECT_SCOPE else "global"


@server.tool(
    name="atlas_recall",
    description=(
        "Search this project's memory for facts and relevant file excerpts. "
        "Use before assuming how the codebase works, before grepping for a "
        "concept rather than a literal string, and whenever you need "
        "conventions or setup details. Returns hand-recorded facts first, then "
        "matching file excerpts with exact line ranges you can open."
    ),
)
def atlas_recall(
    query: str,
    limit: int = 8,
    facts_only: bool = False,
    scope: str | None = None,
) -> str:
    """Recall project knowledge matching `query`."""
    scopes = (scope,) if scope in (PROJECT_SCOPE, GLOBAL_SCOPE) else (PROJECT_SCOPE, GLOBAL_SCOPE)
    kinds = ("fact",) if facts_only else ("fact", "chunk")
    results = recall(query, k=max(1, min(limit, 30)), scopes=scopes, kinds=kinds)

    if not results:
        return (
            f"No matches for {query!r}.\n"
            "Nothing is indexed for this concept yet — fall back to reading files, "
            "and consider recording what you learn with atlas_remember."
        )

    lines: list[str] = []
    facts = [r for r in results if r.kind == "fact"]
    chunks = [r for r in results if r.kind == "chunk"]

    if facts:
        lines.append("## Recorded facts")
        for r in facts:
            marker = " (always)" if r.pinned else ""
            tags = f" [{', '.join(r.tags)}]" if r.tags else ""
            lines.append(f"- {r.text}{tags}  — {r.ref}, {_scope_label(r.scope)}{marker}")
        lines.append("")

    if chunks:
        lines.append("## From indexed files")
        for r in chunks:
            heading = f" — {r.heading}" if r.heading else ""
            lines.append(f"### {r.ref}{heading}")
            lines.append(r.text)
            lines.append("")

    return "\n".join(lines).rstrip()


@server.tool(
    name="atlas_remember",
    description=(
        "Record a durable fact about this project so future sessions inherit "
        "it. Good facts: build/test commands that aren't discoverable, "
        "architectural constraints, why a decision was made, gotchas that cost "
        "you time. Write one self-contained fact per call, phrased so it makes "
        "sense with no other context. Near-identical existing facts are updated "
        "rather than duplicated. Set pin=true only for facts that should surface "
        "on every recall regardless of query."
    ),
)
def atlas_remember(
    text: str,
    tags: list[str] | None = None,
    scope: str | None = None,
    pin: bool = False,
) -> str:
    """Store a fact in project (default) or global memory."""
    try:
        result = remember(text, scope=scope, tags=tags or (), pinned=pin)
    except ValueError as exc:
        return f"Could not record fact: {exc}"

    verb = "Updated existing fact" if result.action == "updated" else "Recorded fact"
    out = [f"{verb} {result.fact_id} in {_scope_label(result.scope)} memory."]
    if result.related:
        out.append(
            "Related facts already stored: "
            + ", ".join(f"#{f.id} {f.text[:60]}" for f in result.related[:3])
        )
    return " ".join(out)


@server.tool(
    name="atlas_facts",
    description=(
        "List every fact recorded for this project. Use at the start of a "
        "session to load standing context, or when the user asks what Atlas "
        "knows. For a targeted lookup prefer atlas_recall."
    ),
)
def atlas_facts(tag: str | None = None, scope: str | None = None) -> str:
    """List stored facts, optionally filtered by tag."""
    scopes = (scope,) if scope in (PROJECT_SCOPE, GLOBAL_SCOPE) else (PROJECT_SCOPE, GLOBAL_SCOPE)
    facts = list_facts(scopes=scopes, tag=tag)
    if not facts:
        return "No facts recorded yet."

    lines = []
    for group, label in ((PROJECT_SCOPE, "Project"), (GLOBAL_SCOPE, "Global")):
        subset = [f for f in facts if f.scope == group]
        if not subset:
            continue
        lines.append(f"## {label}")
        for f in subset:
            marker = " (always)" if f.pinned else ""
            tags = f" [{', '.join(f.tag_list)}]" if f.tag_list else ""
            lines.append(f"- #{f.id} {f.text}{tags}{marker}")
        lines.append("")
    return "\n".join(lines).rstrip()


@server.tool(
    name="atlas_forget",
    description=(
        "Delete a fact by id when it has become wrong or obsolete. Prefer "
        "atlas_remember with corrected text when the fact merely changed — that "
        "updates in place and keeps the history clean."
    ),
)
def atlas_forget(fact_id: int) -> str:
    """Delete a stored fact."""
    return f"Forgot fact {fact_id}." if forget_fact(fact_id) else f"No fact {fact_id} found."


@server.tool(
    name="atlas_pin",
    description=(
        "Pin or unpin a fact. Pinned facts are returned by every atlas_recall "
        "regardless of the query — this is the always-on layer, so keep it "
        "small and reserve it for things that apply to all work in the repo."
    ),
)
def atlas_pin(fact_id: int, pinned: bool = True) -> str:
    """Toggle a fact's pinned state."""
    if set_pinned(fact_id, pinned):
        return f"{'Pinned' if pinned else 'Unpinned'} fact {fact_id}."
    return f"No fact {fact_id} found."


@server.tool(
    name="atlas_index",
    description=(
        "Index a folder so its contents become searchable via atlas_recall. "
        "Respects .gitignore. Reindexing is incremental and cheap, so call this "
        "after large changes, or to add external docs the repo doesn't contain."
    ),
)
def atlas_index(path: str, scope: str | None = None) -> str:
    """Index a folder into project or global memory."""
    target = Path(path).expanduser()
    if not target.exists():
        return f"{target} does not exist."
    try:
        resolved_scope, root = resolve_scope(scope, _root())
    except ValueError as exc:
        return str(exc)

    store = open_store(resolved_scope, root)
    if store is None:
        return "No store available to index into."
    try:
        stats = index_path(store, target)
        return f"Indexed {target}: {stats.summary()}"
    finally:
        store.close()


@server.tool(
    name="atlas_status",
    description=(
        "Report what Atlas currently holds: indexed folders, file and chunk "
        "counts, fact counts. Use to check whether memory is populated before "
        "concluding that a recall miss means the information doesn't exist."
    ),
)
def atlas_status() -> str:
    """Summarise both stores."""
    root = _root()
    lines = [f"Project: {project_name(root)}" + (f" ({root})" if root else "")]
    for store in open_stores(create=False):
        try:
            counts = store.counts()
            if not any(counts.values()):
                continue
            lines.append(
                f"{_scope_label(store.scope)}: {counts['files']} files, "
                f"{counts['chunks']} chunks, {counts['facts']} facts"
            )
            for src in store.sources():
                lines.append(f"  indexed: {src['path']}")
        finally:
            store.close()
    if len(lines) == 1:
        lines.append("Nothing indexed yet — run `atlas init` or call atlas_index.")
    return "\n".join(lines)


def run() -> None:
    """Serve on stdio. Entry point for `atlas mcp`."""
    server.run(transport="stdio")


if __name__ == "__main__":
    run()
