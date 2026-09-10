"""Plain text, Markdown, code, HTML, and other decodable files.

The simplest modality, and the one that sets the pattern the others follow:
produce text, hand it to `text_chunks`, and let links become REFERENCES edges.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from ..graph import Edge, EdgeKind, Modality, Provenance, document_uid
from .base import ExtractContext, Extraction, Extractor, clean, text_chunks

TEXT_EXTENSIONS = frozenset(
    [".txt", ".text", ".md", ".mdx", ".markdown", ".rst", ".adoc", ".org", ".tex", ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", ".properties", ".html", ".htm", ".xhtml", ".xml", ".svg", ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".go", ".rs", ".java", ".kt", ".kts", ".scala", ".rb", ".php", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".swift", ".m", ".mm", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".sql", ".graphql", ".proto", ".tf", ".hcl", ".lua", ".pl", ".r", ".jl", ".dart", ".vue", ".svelte", ".ex", ".exs", ".gradle", ".cmake", ".mk", ".dockerfile", ".gitignore", ".editorconfig"]
)

MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
HTML_HREF = re.compile(r"""(?:href|src)=["']([^"'>]+)["']""", re.IGNORECASE)
TITLE_HEADING = re.compile(r"^\s{0,3}#\s+(.+?)\s*$", re.MULTILINE)
HTML_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
TAG = re.compile(r"<[^>]+>")


class TextExtractor(Extractor):
    name = "text"
    version = "1"
    modality = Modality.TEXT
    extensions = TEXT_EXTENSIONS

    def matches(self, path: Path) -> bool:
        # Extension-less files that are conventionally text (Dockerfile,
        # Makefile, LICENSE) still deserve indexing; the pipeline has already
        # confirmed the bytes decode before dispatching here.
        return path.suffix.lower() in self.extensions or not path.suffix

    def extract(self, ctx: ExtractContext) -> Extraction:
        raw = _read(ctx.path, ctx.options.max_text_bytes)
        if raw is None:
            result = Extraction(modality=Modality.TEXT)
            result.warn("not decodable as text")
            return result

        is_html = ctx.suffix in {".html", ".htm", ".xhtml"}
        title = _title(raw, ctx.path, is_html)
        body = _html_to_text(raw) if is_html else raw
        body = clean(body)

        result = Extraction(
            modality=Modality.TEXT,
            title=title,
            text=body,
            props={"chars": len(body), "lines": body.count("\n") + 1},
        )

        nodes, edges, _ = text_chunks(ctx.doc_uid, body, ctx.rel_path)
        result.nodes.extend(nodes)
        result.edges.extend(edges)
        result.edges.extend(_link_edges(raw, ctx))
        return result


def _read(path: Path, limit: int) -> str | None:
    try:
        data = path.read_bytes()[:limit]
    except OSError:
        return None
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def _title(raw: str, path: Path, is_html: bool) -> str:
    if is_html:
        match = HTML_TITLE.search(raw)
        if match:
            return html.unescape(clean(TAG.sub("", match.group(1)))).strip()[:200]
    match = TITLE_HEADING.search(raw)
    if match:
        return match.group(1).strip()[:200]
    return path.name


def _html_to_text(raw: str) -> str:
    return html.unescape(TAG.sub(" ", SCRIPT_STYLE.sub(" ", raw)))


def _link_edges(raw: str, ctx: ExtractContext) -> list[Edge]:
    """Turn relative links into REFERENCES edges between documents.

    Only local, relative targets become edges — an `http://` link is a fact
    about the outside world and belongs to the entity layer, not the document
    graph. Targets are resolved against the file's own directory and normalised,
    so `../shared/spec.md` from `docs/api/v2.md` lands on `docs/shared/spec.md`.
    The edge is emitted even if that file has not been indexed yet; the store
    parks it and resolves it once the walk finishes.
    """
    base = Path(ctx.rel_path).parent
    edges: list[Edge] = []
    seen: set[str] = set()

    for pattern in (MARKDOWN_LINK, HTML_HREF):
        for target in pattern.findall(raw):
            target = target.split("#", 1)[0].split("?", 1)[0].strip()
            if not target or target in seen:
                continue
            if "://" in target or target.startswith(("mailto:", "tel:", "data:", "/", "#")):
                continue
            seen.add(target)
            try:
                resolved = (base / target).resolve().relative_to(Path.cwd().resolve())
            except (ValueError, OSError):
                # Not resolvable against the cwd — normalise textually instead.
                resolved = Path(_normalise(str(base / target)))
            uid = document_uid(str(resolved))
            if uid != ctx.doc_uid:
                edges.append(
                    Edge(
                        ctx.doc_uid,
                        uid,
                        EdgeKind.REFERENCES,
                        provenance=Provenance.INFERRED,
                        props={"label": "link", "target": target},
                    )
                )
    return edges


def _normalise(path: str) -> str:
    parts: list[str] = []
    for part in path.replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)
