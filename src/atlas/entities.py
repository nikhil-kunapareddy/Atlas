"""Deterministic entity extraction.

This is the semantic floor: entities that can be found by pattern alone, with no
model, no API key, and no possibility of hallucination. Emails, URLs, dates,
amounts, phone numbers, and structured identifiers all have unambiguous surface
forms, and pulling them out is what makes a zero-cost graph genuinely useful —
`invoice INV-2024-0912` links the spreadsheet row, the PDF, and the email thread
that mention it, without anything having to *understand* any of them.

What is deliberately absent: people, organisations, places, and concepts. Those
cannot be identified reliably by regex, and a "capitalised words are probably
names" heuristic produces a graph full of confident nonsense — `Best Regards`
and `Table Of Contents` as first-class people. They are the LLM enrichment
pass's job (`enrich.py`), where the guess is at least a good one and is tagged
`Provenance.LLM` so an agent knows to hedge.

Precision is chosen over recall throughout. A missed entity costs one edge; a
wrong one corrupts every query that traverses it.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import date
from typing import NamedTuple

from .graph import EntityType

# How many entities one chunk may contribute. A minified bundle or a base64 blob
# can match thousands of "identifiers"; without a cap, one pathological file
# dominates the graph.
MAX_PER_CHUNK = 40
MIN_TEXT_FOR_ENTITIES = 20


class Mention(NamedTuple):
    """One entity found in a piece of text."""

    type: EntityType
    display: str  # what a human should see
    value: str    # normalised form, used to build the uid


EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,63}\b")
URL = re.compile(r"\bhttps?://[^\s<>\"'{}|\\^`\[\]]+", re.IGNORECASE)

# Deliberately narrow: an international prefix or a clear separator pattern.
# A bare run of ten digits is far more often an order number than a phone.
PHONE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{2,4}[\s.-])\d{3,4}[\s.-]\d{3,4}(?![\w.])"
)

MONEY = re.compile(
    r"(?<![\w.])(?:(?P<sym>[$€£¥₹])\s?(?P<amt1>\d[\d,]*(?:\.\d{1,2})?)"
    r"|(?P<amt2>\d[\d,]*(?:\.\d{1,2})?)\s?(?P<code>USD|EUR|GBP|JPY|INR|CAD|AUD|CHF))(?![\w])",
    re.IGNORECASE,
)

ISO_DATE = re.compile(r"(?<![\d-])(\d{4})-(\d{2})-(\d{2})(?![\d-])")
US_DATE = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})/(\d{4})(?![\d/])")
LONG_DATE = re.compile(
    r"\b(?P<month>January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)

# Ticket keys, invoice numbers, SKUs: an uppercase prefix bound to digits by a
# separator. The two-character minimum and required digits keep ordinary
# hyphenated words out.
#
# The trailing repeat group matters more than it looks: invoice and order
# numbers are routinely multi-segment (INV-2024-0912), and without it the match
# stops at the first boundary and yields `INV-2024` — which then merges every
# 2024 invoice in the folder into one entity, silently linking unrelated files.
IDENTIFIER = re.compile(r"\b([A-Z][A-Z0-9]{1,9})[-_](\d{1,8}(?:[-_]\d{1,8})*)\b")
UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE
)

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR"}

# Domains whose URLs are noise in almost every corpus.
BORING_HOSTS = frozenset(
    {"www.w3.org", "schemas.microsoft.com", "schemas.openxmlformats.org", "localhost"}
)


def extract(text: str, limit: int = MAX_PER_CHUNK) -> list[Mention]:
    """Find every literal entity in `text`, deduplicated, capped at `limit`.

    Order is by first appearance, so the cap keeps what the text leads with —
    which in practice is the material an author considered important.
    """
    if not text or len(text) < MIN_TEXT_FOR_ENTITIES:
        return []

    seen: set[tuple[EntityType, str]] = set()
    found: list[Mention] = []
    for mention in _scan(text):
        key = (mention.type, mention.value)
        if key in seen:
            continue
        seen.add(key)
        found.append(mention)
        if len(found) >= limit:
            break
    return found


def _scan(text: str) -> Iterator[Mention]:
    for match in EMAIL.finditer(text):
        raw = match.group(0).rstrip(".")
        yield Mention(EntityType.EMAIL, raw, raw.casefold())

    for match in URL.finditer(text):
        url = match.group(0).rstrip(".,;:)]}\"'")
        host = _host(url)
        if host and host not in BORING_HOSTS:
            yield Mention(EntityType.URL, url, url.casefold())

    for match in ISO_DATE.finditer(text):
        iso = _iso(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if iso:
            yield Mention(EntityType.DATE, match.group(0), iso)

    for match in LONG_DATE.finditer(text):
        month = MONTHS.get(match.group("month")[:4].casefold().rstrip("."))
        if month is None:
            month = MONTHS.get(match.group("month")[:3].casefold())
        if month:
            iso = _iso(int(match.group("year")), month, int(match.group("day")))
            if iso:
                yield Mention(EntityType.DATE, match.group(0), iso)

    for match in US_DATE.finditer(text):
        # Ambiguous by nature (03/04/2024). Read as US month/day, and skip
        # anything where that reading is impossible rather than guessing the
        # other convention — a wrong date silently merges unrelated documents.
        iso = _iso(int(match.group(3)), int(match.group(1)), int(match.group(2)))
        if iso:
            yield Mention(EntityType.DATE, match.group(0), iso)

    for match in MONEY.finditer(text):
        currency = (
            CURRENCY_SYMBOLS.get(match.group("sym") or "", "")
            or (match.group("code") or "").upper()
        )
        amount = (match.group("amt1") or match.group("amt2") or "").replace(",", "")
        if amount and currency:
            yield Mention(EntityType.MONEY, match.group(0).strip(), f"{currency} {amount}")

    for match in PHONE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if 7 <= len(digits) <= 15:
            yield Mention(EntityType.PHONE, match.group(0).strip(), digits)

    for match in UUID.finditer(text):
        yield Mention(EntityType.IDENTIFIER, match.group(0), match.group(0).casefold())

    for match in IDENTIFIER.finditer(text):
        raw = match.group(0)
        yield Mention(EntityType.IDENTIFIER, raw, raw.casefold())


def _iso(year: int, month: int, day: int) -> str | None:
    """Validate and normalise a date, or None if it is not a real one."""
    if not (1900 <= year <= 2200):
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _host(url: str) -> str:
    without_scheme = url.split("://", 1)[-1]
    return without_scheme.split("/", 1)[0].split(":", 1)[0].casefold()
