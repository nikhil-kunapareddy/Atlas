"""PDF documents.

Pages are the unit that matters: a PDF's own citation system is page numbers,
so a result an agent can hand back to a human should say `q3-report.pdf p.14`.
Every page becomes a node, chunks hang off pages, and the outline (bookmarks)
is folded in as section names when the file has one.

pypdf, not PyMuPDF: PyMuPDF is AGPL, and linking it into an MIT-licensed tool
would force the whole project to relicense.
"""

from __future__ import annotations

from typing import Any

from ..graph import Edge, EdgeKind, Modality, Node, NodeKind, child_uid
from .base import ExtractContext, Extraction, Extractor, clean, require, text_chunks

PREVIEW_CHARS = 300


class PdfExtractor(Extractor):
    name = "pdf"
    version = "1"
    modality = Modality.PDF
    extensions = frozenset({".pdf"})

    def extract(self, ctx: ExtractContext) -> Extraction:
        pypdf = require("pypdf", "pdf")
        result = Extraction(modality=Modality.PDF, title=ctx.path.stem)

        # Page text extraction is the expensive part and is deterministic, so it
        # is cached by digest: reindexing an unchanged PDF costs one hash.
        payload = ctx.cached(self.name, self.version, lambda: _read_pdf(pypdf, ctx))
        if payload.get("error"):
            result.warn(payload["error"])
            return result

        result.title = payload.get("title") or ctx.path.stem
        result.props = {
            k: v
            for k, v in {
                "pages": payload.get("page_count"),
                "author": payload.get("author"),
                "subject": payload.get("subject"),
                "producer": payload.get("producer"),
                "created": payload.get("created"),
                "encrypted": payload.get("encrypted"),
            }.items()
            if v
        }
        for warning in payload.get("warnings", []):
            result.warn(warning)

        outline = {int(k): v for k, v in payload.get("outline", {}).items()}
        pages: list[str] = payload.get("pages_text", [])
        previous_page_uid: str | None = None
        ordinal = 0
        full_text: list[str] = []

        for index, page_text in enumerate(pages, start=1):
            page_text = clean(page_text)
            if not page_text:
                continue
            full_text.append(page_text)
            page_uid = child_uid(ctx.doc_uid, NodeKind.PAGE, index)
            section = outline.get(index)
            page_props: dict[str, Any] = {
                "page": index,
                "path": ctx.rel_path,
                "chars": len(page_text),
            }
            if section:
                page_props["heading"] = section

            name = f"{ctx.path.name} p.{index}"
            result.add(
                Node(
                    uid=page_uid,
                    kind=NodeKind.PAGE,
                    name=f"{name} — {section}" if section else name,
                    # Structural nodes carry a preview; the chunks under them
                    # carry the text. One retrieval unit across every modality.
                    body=page_text[:PREVIEW_CHARS],
                    props=page_props,
                ),
                contained_by=ctx.doc_uid,
            )
            if previous_page_uid:
                result.edges.append(Edge(previous_page_uid, page_uid, EdgeKind.NEXT))
            previous_page_uid = page_uid

            nodes, edges, ordinal = text_chunks(
                page_uid,
                page_text,
                ctx.rel_path,
                extra_props={"page": index, "path": ctx.rel_path},
                start_ord=ordinal,
            )
            result.nodes.extend(nodes)
            result.edges.extend(edges)

        result.text = "\n\n".join(full_text)
        if not result.text:
            result.warn(
                "no extractable text — likely a scanned PDF; "
                "install the ocr extra and rerun with --ocr"
            )
        return result


def _read_pdf(pypdf: Any, ctx: ExtractContext) -> dict[str, Any]:
    """Pull text and metadata out of a PDF, tolerating damage.

    Returned as a plain dict because it goes through the JSON derived-cache.
    """
    payload: dict[str, Any] = {"warnings": [], "pages_text": [], "outline": {}}
    try:
        reader = pypdf.PdfReader(str(ctx.path))
    except Exception as exc:  # pypdf raises a wide range on malformed files
        return {"error": f"unreadable PDF: {exc}"}

    if reader.is_encrypted:
        payload["encrypted"] = True
        try:
            # Many PDFs are "encrypted" with an empty owner password purely to
            # set permissions; those open fine and are worth indexing.
            if reader.decrypt("") == 0:
                return {"error": "password-protected PDF"}
        except Exception:
            return {"error": "password-protected PDF"}

    try:
        info = reader.metadata or {}
        payload["title"] = _meta(info, "/Title")
        payload["author"] = _meta(info, "/Author")
        payload["subject"] = _meta(info, "/Subject")
        payload["producer"] = _meta(info, "/Producer")
        payload["created"] = _meta(info, "/CreationDate")
    except Exception:
        payload["warnings"].append("metadata unreadable")

    payload["page_count"] = len(reader.pages)
    for index, page in enumerate(reader.pages, start=1):
        try:
            payload["pages_text"].append(page.extract_text() or "")
        except Exception as exc:
            payload["pages_text"].append("")
            payload["warnings"].append(f"page {index} unreadable: {exc}")

    payload["outline"] = _outline(reader)
    return payload


def _meta(info: Any, key: str) -> str | None:
    try:
        value = info.get(key)
    except Exception:
        return None
    if not value:
        return None
    text = str(value).strip()
    return text[:300] or None


def _outline(reader: Any) -> dict[str, str]:
    """Map page number -> bookmark title, for the first bookmark on each page."""
    mapping: dict[str, str] = {}

    def walk(items: Any) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item)
                continue
            try:
                page_number = reader.get_destination_page_number(item) + 1
                title = str(item.title).strip()
            except Exception:
                continue
            if title and str(page_number) not in mapping:
                mapping[str(page_number)] = title[:200]

    try:
        walk(reader.outline)
    except Exception:
        return {}
    return mapping
