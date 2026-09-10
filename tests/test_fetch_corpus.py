"""Selection logic in the corpus fetcher.

The fetcher itself needs the network, but the decisions it makes — what fits in
a budget, which of several encodings of the same recording to keep — are pure
and are exactly where silent mistakes live. A duplicate-bitrate bug would
inflate the corpus and make a transcription benchmark measure the same audio
three times.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
fetch_corpus = pytest.importorskip("fetch_corpus")


# -- budget ----------------------------------------------------------------


def test_a_file_that_would_overflow_the_budget_is_skipped():
    fits = fetch_corpus.fits
    assert fits(written=0, size=50, budget=100)
    assert fits(written=40, size=60, budget=100)
    assert not fits(written=41, size=60, budget=100)


def test_the_first_file_is_always_allowed():
    """Otherwise a source whose smallest file exceeds its share contributes
    nothing at all, and that modality silently vanishes from the corpus."""
    assert fetch_corpus.fits(written=0, size=10_000, budget=10)


# -- duplicate encodings ---------------------------------------------------


def entry(name, size):
    return {"name": name, "size": str(size)}


def test_bitrate_variants_of_one_recording_collapse_to_one():
    """archive.org lists the same chapter three times at different bitrates."""
    files = [
        entry("chapter01.mp3", 9_000_000),
        entry("chapter01_64kb.mp3", 3_000_000),
        entry("chapter01_128kb.mp3", 6_000_000),
        entry("chapter02.mp3", 8_000_000),
        entry("chapter02_64kb.mp3", 2_500_000),
    ]
    kept = fetch_corpus._one_per_chapter(files)
    assert [f["name"] for f in kept] == ["chapter01_64kb.mp3", "chapter02_64kb.mp3"]


def test_distinct_recordings_are_all_kept():
    files = [entry("intro.mp3", 100), entry("outro.mp3", 200)]
    assert len(fetch_corpus._one_per_chapter(files)) == 2


def test_collapsing_handles_an_empty_list():
    assert fetch_corpus._one_per_chapter([]) == []


# -- attribution -----------------------------------------------------------


def test_html_credit_markup_is_stripped():
    """Commons returns the artist field as HTML; it goes into a text file."""
    raw = '<a href="//commons.wikimedia.org/wiki/User:Someone">Someone</a>'
    assert fetch_corpus._strip_tags(raw) == "Someone"


def test_signatures_cover_what_the_sources_actually_serve():
    """Each configured source must have a matching entry in the mix."""
    assert set(fetch_corpus.MIX) == set(fetch_corpus.SOURCES)
    assert abs(sum(fetch_corpus.MIX.values()) - 1.0) < 0.01, "the mix should spend the budget"
