"""Word documents and PowerPoint decks.

Both formats carry structure worth keeping. Word has a heading hierarchy, which
is converted to Markdown headings so the shared chunker splits on sections
instead of on character count. PowerPoint has slides, which behave exactly like
PDF pages — including speaker notes, which are often where the actual argument
lives while the slide itself holds three words and a chart.
"""

from __future__ import annotations

from typing import Any

from ..graph import Edge, EdgeKind, Modality, Node, NodeKind, child_uid
from .base import ExtractContext, Extraction, Extractor, clean, require, text_chunks

PREVIEW_CHARS = 300


class DocxExtractor(Extractor):
    name = "docx"
    version = "1"
    modality = Modality.TEXT
    extensions = frozenset({".docx", ".docm"})

    def extract(self, ctx: ExtractContext) -> Extraction:
        docx = require("docx", "office")
        result = Extraction(modality=Modality.TEXT, title=ctx.path.stem)

        try:
            document = docx.Document(str(ctx.path))
        except Exception as exc:
            result.warn(f"unreadable Word document: {exc}")
            return result

        result.props.update(_core_properties(document))
        if result.props.get("title"):
            result.title = str(result.props["title"])

        blocks: list[str] = []
        for paragraph in document.paragraphs:
            text = (paragraph.text or "").strip()
            if not text:
                continue
            level = _heading_level(paragraph)
            blocks.append(f"{'#' * level} {text}" if level else text)

        for index, table in enumerate(document.tables, start=1):
            rendered = _render_table(table)
            if rendered:
                blocks.append(f"\n[table {index}]\n{rendered}")

        body = clean("\n\n".join(blocks))
        result.text = body
        result.props["chars"] = len(body)

        nodes, edges, _ = text_chunks(ctx.doc_uid, body, ctx.rel_path)
        result.nodes.extend(nodes)
        result.edges.extend(edges)
        if not body:
            result.warn("document contained no text")
        return result


class PptxExtractor(Extractor):
    name = "pptx"
    version = "1"
    modality = Modality.SLIDES
    extensions = frozenset({".pptx", ".pptm"})

    def extract(self, ctx: ExtractContext) -> Extraction:
        pptx = require("pptx", "office")
        result = Extraction(modality=Modality.SLIDES, title=ctx.path.stem)

        try:
            presentation = pptx.Presentation(str(ctx.path))
        except Exception as exc:
            result.warn(f"unreadable presentation: {exc}")
            return result

        result.props["slides"] = len(presentation.slides)
        previous_uid: str | None = None
        ordinal = 0
        full: list[str] = []

        for index, slide in enumerate(presentation.slides, start=1):
            title, body = _slide_text(slide)
            notes = _slide_notes(slide)
            combined = clean("\n".join(p for p in (body, notes) if p))
            if not combined and not title:
                continue
            full.append(f"{title}\n{combined}" if title else combined)

            slide_uid = child_uid(ctx.doc_uid, NodeKind.PAGE, index)
            label = f"{ctx.path.name} slide {index}"
            props: dict[str, Any] = {
                "page": index,
                "slide": index,
                "path": ctx.rel_path,
                "has_notes": bool(notes),
            }
            if title:
                props["heading"] = title

            result.add(
                Node(
                    uid=slide_uid,
                    kind=NodeKind.PAGE,
                    name=f"{label} — {title}" if title else label,
                    body=combined[:PREVIEW_CHARS],
                    props=props,
                ),
                contained_by=ctx.doc_uid,
            )
            if previous_uid:
                result.edges.append(Edge(previous_uid, slide_uid, EdgeKind.NEXT))
            previous_uid = slide_uid

            nodes, edges, ordinal = text_chunks(
                slide_uid,
                f"# {title}\n{combined}" if title else combined,
                ctx.rel_path,
                extra_props={"page": index, "slide": index, "path": ctx.rel_path},
                start_ord=ordinal,
            )
            result.nodes.extend(nodes)
            result.edges.extend(edges)

        result.text = "\n\n".join(full)
        return result


# -- helpers ---------------------------------------------------------------


def _core_properties(document: Any) -> dict[str, Any]:
    props: dict[str, Any] = {}
    try:
        core = document.core_properties
    except Exception:
        return props
    for key in ("title", "author", "subject", "keywords", "last_modified_by", "category"):
        value = getattr(core, key, None)
        if value:
            props[key] = str(value)[:200]
    for key in ("created", "modified"):
        value = getattr(core, key, None)
        if value:
            props[key] = str(value)
    return props


def _heading_level(paragraph: Any) -> int:
    """Map a Word paragraph style to a Markdown heading level, or 0."""
    try:
        style = (paragraph.style.name or "").lower()
    except Exception:
        return 0
    if style.startswith("heading"):
        tail = style.replace("heading", "").strip()
        if tail.isdigit():
            return min(int(tail), 6)
        return 1
    if style in ("title", "subtitle"):
        return 1
    return 0


def _render_table(table: Any) -> str:
    rows: list[str] = []
    try:
        for row in table.rows:
            cells = [(cell.text or "").strip().replace("\n", " ") for cell in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
    except Exception:
        return ""
    return "\n".join(rows)


def _slide_text(slide: Any) -> tuple[str, str]:
    title = ""
    parts: list[str] = []
    try:
        if slide.shapes.title is not None and slide.shapes.title.text:
            title = slide.shapes.title.text.strip()
    except Exception:
        title = ""
    for shape in slide.shapes:
        try:
            if not shape.has_text_frame:
                continue
            text = (shape.text_frame.text or "").strip()
        except Exception:
            continue
        if text and text != title:
            parts.append(text)
    return title, "\n".join(parts)


def _slide_notes(slide: Any) -> str:
    try:
        if not slide.has_notes_slide:
            return ""
        notes = (slide.notes_slide.notes_text_frame.text or "").strip()
    except Exception:
        return ""
    return f"[speaker notes]\n{notes}" if notes else ""
