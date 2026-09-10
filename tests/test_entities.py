"""Literal entity extraction.

The contract here is precision. A missed entity costs one edge; a wrong one
corrupts every query that traverses it, and a *merged* one silently links files
that have nothing to do with each other. Most of these tests are therefore
negative — they assert what must NOT be extracted.
"""

from __future__ import annotations

import pytest

from atlas.entities import extract
from atlas.graph import EntityType

PAD = " Some surrounding prose so the length floor is comfortably cleared. "


def types_and_values(text):
    return {(m.type, m.value) for m in extract(PAD + text + PAD)}


def values_of(text, kind):
    return {m.value for m in extract(PAD + text + PAD) if m.type == kind}


# -- what must be found ----------------------------------------------------


def test_emails_and_urls():
    found = types_and_values("write to Ana.B@Acme.example or see https://acme.example/docs.")
    assert (EntityType.EMAIL, "ana.b@acme.example") in found
    assert (EntityType.URL, "https://acme.example/docs") in found, "trailing period trimmed"


def test_multi_segment_identifiers_stay_whole():
    """The regression that mattered: INV-2024-0912 must not become INV-2024.

    Truncating would merge every 2024 invoice into one entity and link unrelated
    files through it.
    """
    assert values_of("invoice INV-2024-0912 today", EntityType.IDENTIFIER) == {"inv-2024-0912"}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("dated 2024-09-30 exactly", "2024-09-30"),
        ("dated September 30, 2024 exactly", "2024-09-30"),
        ("dated Sept 30 2024 exactly", "2024-09-30"),
        ("dated 9/30/2024 exactly", "2024-09-30"),
    ],
)
def test_date_formats_normalise_to_one_entity(text, expected):
    """Four spellings, one node — otherwise the join never happens."""
    assert expected in values_of(text, EntityType.DATE)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("costs $42,500.00 total", "USD 42500.00"),
        ("costs 42,500.00 USD total", "USD 42500.00"),
        ("costs £99.50 total", "GBP 99.50"),
    ],
)
def test_money_normalises_currency_and_separators(text, expected):
    assert expected in values_of(text, EntityType.MONEY)


def test_uuids_are_identifiers():
    uid = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    assert uid in values_of(f"trace {uid} here", EntityType.IDENTIFIER)


# -- what must NOT be found ------------------------------------------------


def test_impossible_dates_are_rejected():
    assert values_of("released 2024-13-45 supposedly", EntityType.DATE) == set()
    assert values_of("released 2024-02-30 supposedly", EntityType.DATE) == set()


def test_ordinary_hyphenated_words_are_not_identifiers():
    for phrase in ("a well-known result", "state-of-the-art tooling", "e-mail us"):
        assert values_of(phrase, EntityType.IDENTIFIER) == set(), phrase


def test_bare_digit_runs_are_not_phone_numbers():
    """An order number is far more common than a bare unformatted phone."""
    assert values_of("order 4155550142 shipped", EntityType.PHONE) == set()
    assert values_of("call +1 415-555-0142 now", EntityType.PHONE) == {"14155550142"}


def test_boilerplate_urls_are_skipped():
    assert values_of("xmlns http://www.w3.org/1999/xhtml here", EntityType.URL) == set()


# -- limits ----------------------------------------------------------------


def test_results_are_deduplicated():
    text = "ana@acme.example " * 20
    assert len([m for m in extract(PAD + text) if m.type == EntityType.EMAIL]) == 1


def test_extraction_is_capped():
    """One minified blob must not be allowed to dominate the whole graph."""
    text = " ".join(f"user{i}@acme.example" for i in range(500))
    assert len(extract(text)) <= 40


def test_short_text_is_skipped():
    assert extract("hi@a.co") == []
