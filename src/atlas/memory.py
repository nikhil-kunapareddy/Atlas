"""The write path: durable facts an agent records and later recalls.

This is the half that actually replaces CLAUDE.md. The index makes a repo
searchable, but the thing worth keeping is the knowledge that isn't in any file
— "integration tests need REDIS_URL set", "don't touch the generated client".

Facts are deduplicated on write. Agents re-learn the same thing every session,
and without dedupe a memory store degenerates into fifty phrasings of one fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .config import GLOBAL_SCOPE, PROJECT_SCOPE, find_project_root
from .store import Fact, Store, open_store, open_stores

# Above this, two facts are the same fact and the new text wins.
DUPLICATE_THRESHOLD = 0.85
# Above this, they're merely related and worth reporting back to the caller.
# Jaccard on one-sentence facts is harsh, so this sits well below the duplicate
# bar — sharing a third of the content words already means "you may be about to
# contradict something".
RELATED_THRESHOLD = 0.35

TOKEN = re.compile(r"[a-z0-9_]+")
# Stopwords only for the similarity check, so "run tests with pytest" and
# "tests are run with pytest" compare as near-identical.
NOISE = frozenset(
    "a an the is are was were be to of in on for with and or not this that it its "
    "you your we our i my use uses used using do does don't should always never".split()
)


@dataclass
class RememberResult:
    action: str  # "created" | "updated"
    fact_id: int
    scope: str
    text: str
    related: list[Fact] = field(default_factory=list)


def _tokens(text: str) -> set[str]:
    return {t for t in TOKEN.findall(text.lower()) if t not in NOISE and len(t) > 1}


def similarity(a: str, b: str) -> float:
    """Jaccard overlap on content words. Cheap, stable, good enough for dedupe."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def resolve_scope(scope: str | None, root: Path | None = None) -> tuple[str, Path | None]:
    """Pick a scope, falling back to global when there's no project to write to."""
    if root is None:
        root = find_project_root()
    if scope == GLOBAL_SCOPE:
        return GLOBAL_SCOPE, root
    if scope == PROJECT_SCOPE:
        if root is None:
            raise ValueError(
                "no project found here — run `atlas init` first, or use --scope global"
            )
        return PROJECT_SCOPE, root
    # Default: project when we're in one, otherwise global.
    return (PROJECT_SCOPE, root) if root is not None else (GLOBAL_SCOPE, None)


def remember(
    text: str,
    scope: str | None = None,
    tags: Iterable[str] = (),
    pinned: bool = False,
    root: Path | None = None,
    store: Store | None = None,
) -> RememberResult:
    """Record a fact, merging it into a near-identical existing one if present."""
    text = text.strip()
    if not text:
        raise ValueError("cannot remember empty text")

    resolved_scope, resolved_root = resolve_scope(scope, root)
    owned = store is None
    if store is None:
        store = open_store(resolved_scope, resolved_root)
        if store is None:
            raise ValueError(f"could not open {resolved_scope} store")

    try:
        best: tuple[float, Fact] | None = None
        related: list[Fact] = []
        for existing in store.all_facts():
            score = similarity(text, existing.text)
            if score >= DUPLICATE_THRESHOLD and (best is None or score > best[0]):
                best = (score, existing)
            elif score >= RELATED_THRESHOLD:
                related.append(existing)

        if best is not None:
            _, existing = best
            merged_tags = sorted(set(existing.tag_list) | {t.strip() for t in tags if t.strip()})
            store.update_fact(existing.id, text, merged_tags)
            if pinned and not existing.pinned:
                set_pinned(existing.id, True, store=store)
            return RememberResult("updated", existing.id, store.scope, text, related)

        fact_id = store.add_fact(text, tags=tags, pinned=pinned)
        return RememberResult("created", fact_id, store.scope, text, related)
    finally:
        if owned:
            store.close()


def forget(fact_id: int, scope: str | None = None, root: Path | None = None) -> bool:
    """Delete a fact. Without a scope, tries project then global."""
    scopes = [scope] if scope else [PROJECT_SCOPE, GLOBAL_SCOPE]
    for candidate in scopes:
        store = open_store(candidate, root, create=False)
        if store is None:
            continue
        try:
            if store.delete_fact(fact_id):
                return True
        finally:
            store.close()
    return False


def set_pinned(fact_id: int, pinned: bool, scope: str | None = None,
               root: Path | None = None, store: Store | None = None) -> bool:
    """Pin a fact into the always-on layer (or unpin it)."""
    if store is not None:
        cur = store.conn.execute(
            "UPDATE facts SET pinned = ? WHERE id = ?", (int(pinned), fact_id)
        )
        store.conn.commit()
        return cur.rowcount > 0

    scopes = [scope] if scope else [PROJECT_SCOPE, GLOBAL_SCOPE]
    for candidate in scopes:
        opened = open_store(candidate, root, create=False)
        if opened is None:
            continue
        try:
            cur = opened.conn.execute(
                "UPDATE facts SET pinned = ? WHERE id = ?", (int(pinned), fact_id)
            )
            opened.conn.commit()
            if cur.rowcount > 0:
                return True
        finally:
            opened.close()
    return False


def list_facts(
    scopes: Iterable[str] = (PROJECT_SCOPE, GLOBAL_SCOPE),
    tag: str | None = None,
    root: Path | None = None,
) -> list[Fact]:
    wanted = set(scopes)
    facts: list[Fact] = []
    for store in open_stores(root):
        try:
            if store.scope in wanted:
                facts.extend(store.all_facts(tag=tag))
        finally:
            store.close()
    return facts


def export_markdown(facts: Sequence[Fact], title: str = "Project memory") -> str:
    """Render facts as markdown — a bridge back to CLAUDE.md when you need one."""
    if not facts:
        return f"# {title}\n\n_No facts recorded yet._\n"

    lines = [f"# {title}", ""]
    pinned = [f for f in facts if f.pinned]
    if pinned:
        lines += ["## Always", ""]
        lines += [f"- {f.text}" for f in pinned]
        lines.append("")

    by_tag: dict[str, list[Fact]] = {}
    for fact in facts:
        if fact.pinned:
            continue
        for tag in fact.tag_list or ["general"]:
            by_tag.setdefault(tag, []).append(fact)

    for tag in sorted(by_tag):
        lines += [f"## {tag}", ""]
        lines += [f"- {f.text}" for f in by_tag[tag]]
        lines.append("")

    return "\n".join(lines)
