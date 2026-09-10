"""Spreadsheets and delimited files: .xlsx, .xlsm, .csv, .tsv.

Sheets are the one modality with genuinely rich deterministic structure — a
workbook really does contain sheets, which really do contain named columns with
inferable types. That structure is worth modelling properly, because it answers
questions no amount of text search can: *which files have a `customer_id`
column?*, *what else joins to this table?*

Rows are deliberately **not** one node each. A 50,000-row sheet would drown the
graph in nodes that no one queries individually, so rows are batched into text
chunks that remain searchable, while columns — the part with reusable identity —
get first-class nodes.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Iterator, Sequence
from datetime import date, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from ..graph import Edge, EdgeKind, Modality, Node, NodeKind, child_uid
from .base import ExtractContext, Extraction, Extractor, clean, require

ROWS_PER_CHUNK = 40
SAMPLE_VALUES = 8
PREVIEW_ROWS = 5

EXCEL_EXTENSIONS = frozenset({".xlsx", ".xlsm"})
DELIMITED_EXTENSIONS = frozenset({".csv", ".tsv", ".psv"})

NUMBER = re.compile(r"^-?[\d,]*\.?\d+(?:[eE][-+]?\d+)?$")
CURRENCY = re.compile(r"^[$€£¥]\s?-?[\d,]*\.?\d+$|^-?[\d,]*\.?\d+\s?(?:USD|EUR|GBP)$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?$")
BOOLEAN = frozenset({"true", "false", "yes", "no", "y", "n", "0", "1"})


class SheetExtractor(Extractor):
    name = "sheet"
    version = "1"
    modality = Modality.SHEET
    extensions = EXCEL_EXTENSIONS | DELIMITED_EXTENSIONS

    def extract(self, ctx: ExtractContext) -> Extraction:
        result = Extraction(modality=Modality.SHEET, title=ctx.path.stem)
        try:
            sheets = list(_read_sheets(ctx))
        except Exception as exc:
            result.warn(f"unreadable spreadsheet: {exc}")
            return result

        if not sheets:
            result.warn("no readable sheets")
            return result

        result.props["sheets"] = len(sheets)
        previous_sheet: str | None = None
        text_parts: list[str] = []

        for sheet_name, rows in sheets:
            if not rows:
                continue
            header, body_rows = _split_header(rows)
            sheet_uid = child_uid(ctx.doc_uid, NodeKind.SHEET, _slug(sheet_name))
            preview = _render(([header] if header else []) + body_rows[:PREVIEW_ROWS])

            result.add(
                Node(
                    uid=sheet_uid,
                    kind=NodeKind.SHEET,
                    name=f"{ctx.path.name} [{sheet_name}]",
                    body=preview,
                    props={
                        "sheet": sheet_name,
                        "path": ctx.rel_path,
                        "rows": len(body_rows),
                        "columns": len(header) if header else _width(body_rows),
                        "headers": header or [],
                    },
                ),
                contained_by=ctx.doc_uid,
            )
            if previous_sheet:
                result.edges.append(Edge(previous_sheet, sheet_uid, EdgeKind.NEXT))
            previous_sheet = sheet_uid

            if header:
                columns = _column_nodes(sheet_uid, sheet_name, header, body_rows, ctx)
                result.nodes.extend(columns)
                result.edges.extend(
                    Edge(sheet_uid, node.uid, EdgeKind.CONTAINS) for node in columns
                )

            chunks = _row_chunks(sheet_uid, sheet_name, header, body_rows, ctx)
            result.nodes.extend(chunks)
            result.edges.extend(Edge(sheet_uid, node.uid, EdgeKind.CONTAINS) for node in chunks)
            result.edges.extend(
                Edge(before.uid, after.uid, EdgeKind.NEXT)
                for before, after in pairwise(chunks)
            )
            text_parts.append(f"# {sheet_name}\n{preview}")

        result.text = "\n\n".join(text_parts)
        return result


# -- reading ---------------------------------------------------------------


def _read_sheets(ctx: ExtractContext) -> Iterator[tuple[str, list[list[str]]]]:
    if ctx.suffix in EXCEL_EXTENSIONS:
        yield from _read_excel(ctx)
    else:
        yield _read_delimited(ctx)


def _read_excel(ctx: ExtractContext) -> Iterator[tuple[str, list[list[str]]]]:
    openpyxl = require("openpyxl", "sheets")
    # read_only streams rows instead of building the whole worksheet in memory;
    # data_only gives computed values rather than formula source, which is what
    # someone asking a question about the data actually means.
    workbook = openpyxl.load_workbook(
        str(ctx.path), read_only=True, data_only=True, keep_links=False
    )
    try:
        for worksheet in workbook.worksheets:
            rows: list[list[str]] = []
            for row in worksheet.iter_rows(values_only=True):
                if len(rows) >= ctx.options.max_sheet_rows:
                    break
                cells = [_cell(v) for v in row]
                if any(cells):
                    rows.append(cells)
            yield worksheet.title, _trim(rows)
    finally:
        workbook.close()


def _read_delimited(ctx: ExtractContext) -> tuple[str, list[list[str]]]:
    data = ctx.path.read_bytes()[: ctx.options.max_text_bytes]
    text = data.decode("utf-8-sig", errors="replace")
    delimiter = {".tsv": "\t", ".psv": "|"}.get(ctx.suffix) or _sniff(text)
    rows: list[list[str]] = []
    for row in csv.reader(io.StringIO(text), delimiter=delimiter):
        if len(rows) >= ctx.options.max_sheet_rows:
            break
        cells = [c.strip() for c in row]
        if any(cells):
            rows.append(cells)
    return ctx.path.stem, _trim(rows)


def _sniff(text: str) -> str:
    try:
        return csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _trim(rows: list[list[str]]) -> list[list[str]]:
    """Drop trailing all-empty columns, which Excel produces in abundance."""
    if not rows:
        return rows
    width = max((len(r) for r in rows), default=0)
    while width > 0 and all(len(r) < width or not r[width - 1] for r in rows):
        width -= 1
    return [r[:width] + [""] * (width - len(r[:width])) for r in rows]


def _split_header(rows: list[list[str]]) -> tuple[list[str] | None, list[list[str]]]:
    """Treat row 0 as a header when it looks like labels rather than data.

    The test is deliberately simple — every cell non-empty, non-numeric, and
    reasonably short. Guessing wrong costs a mislabelled column node, not a
    failed extraction, so a cheap heuristic beats a clever one.
    """
    if not rows:
        return None, []
    first = rows[0]
    if not first or not all(c for c in first):
        return None, rows
    if any(NUMBER.match(c) or ISO_DATE.match(c) for c in first):
        return None, rows
    if any(len(c) > 80 for c in first):
        return None, rows
    if len({c.casefold() for c in first}) != len(first):
        return None, rows
    return first, rows[1:]


def _width(rows: Sequence[Sequence[str]]) -> int:
    return max((len(r) for r in rows), default=0)


# -- node construction -----------------------------------------------------


def _column_nodes(
    sheet_uid: str,
    sheet_name: str,
    header: list[str],
    rows: list[list[str]],
    ctx: ExtractContext,
) -> list[Node]:
    nodes: list[Node] = []
    for index, label in enumerate(header):
        values = [r[index] for r in rows if index < len(r) and r[index]]
        samples = _distinct(values, SAMPLE_VALUES)
        nodes.append(
            Node(
                uid=child_uid(sheet_uid, NodeKind.COLUMN, _slug(label) or index),
                kind=NodeKind.COLUMN,
                name=label,
                # The body is what makes "which sheet has customer emails?"
                # answerable: the column name plus real values from it.
                body=f"{label}\n" + "\n".join(samples),
                props={
                    "sheet": sheet_name,
                    "path": ctx.rel_path,
                    "position": index,
                    "dtype": _infer_type(values),
                    "non_empty": len(values),
                    "distinct": len(set(values)),
                    "samples": samples,
                },
            )
        )
    return nodes


def _row_chunks(
    sheet_uid: str,
    sheet_name: str,
    header: list[str] | None,
    rows: list[list[str]],
    ctx: ExtractContext,
) -> list[Node]:
    """Batch rows into searchable text blocks, repeating the header in each.

    Repeating the header is what lets a chunk stand alone in an agent's context:
    a block of bare values is unreadable, but `name | email | plan` above them
    is a small table.
    """
    nodes: list[Node] = []
    for ordinal, start in enumerate(range(0, len(rows), ROWS_PER_CHUNK)):
        batch = rows[start : start + ROWS_PER_CHUNK]
        text = _render(([header] if header else []) + batch)
        nodes.append(
            Node(
                uid=child_uid(sheet_uid, NodeKind.CHUNK, ordinal),
                kind=NodeKind.CHUNK,
                name=f"{Path(ctx.rel_path).name} [{sheet_name}] rows {start + 1}-{start + len(batch)}",
                body=text,
                props={
                    "sheet": sheet_name,
                    "path": ctx.rel_path,
                    "first_row": start + 1,
                    "last_row": start + len(batch),
                    "ord": ordinal,
                },
            )
        )
    return nodes


def _render(rows: Sequence[Sequence[str]]) -> str:
    return clean("\n".join(" | ".join(str(c) for c in row) for row in rows))


def _distinct(values: Sequence[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value[:120])
        if len(out) >= limit:
            break
    return out


def _infer_type(values: Sequence[str]) -> str:
    """Label a column by what most of its values look like.

    A 90% threshold rather than 100%: real spreadsheets have a stray "N/A" in an
    otherwise numeric column, and calling that column `text` would be less
    useful than calling it `number`.
    """
    sample = [v for v in values[:200] if v]
    if not sample:
        return "empty"
    checks = (
        ("currency", lambda v: bool(CURRENCY.match(v))),
        ("date", lambda v: bool(ISO_DATE.match(v))),
        ("number", lambda v: bool(NUMBER.match(v))),
        ("boolean", lambda v: v.casefold() in BOOLEAN),
    )
    for label, test in checks:
        if sum(1 for v in sample if test(v)) >= 0.9 * len(sample):
            return label
    return "text"


_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(text: str | int) -> str:
    return _SLUG.sub("-", str(text).casefold()).strip("-")[:64]
