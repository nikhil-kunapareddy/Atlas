# Contributing

## Setup

```bash
uv venv && uv pip install -e ".[dev,all]"
uv run pytest
uv run ruff check src tests
```

## Adding a modality

This is the most common change, and it is meant to be small. Everything about
reading a file type lives in one class.

1. Add `src/atlas/extract/<name>.py` with a subclass of `Extractor`:

   ```python
   class ThingExtractor(Extractor):
       name = "thing"          # identifies its entries in the derived cache
       version = "1"           # bump to invalidate just this extractor's cache
       modality = Modality.THING
       extensions = frozenset({".thing"})

       def extract(self, ctx: ExtractContext) -> Extraction:
           ...
   ```

2. Register it in `src/atlas/extract/__init__.py` (`REGISTRY`), keeping
   `TextExtractor` last — it claims extension-less files.

3. Declare the dependency as an extra in `pyproject.toml` and add a row to
   `CAPABILITIES` so `atlas doctor` reports it.

4. Add a fixture builder to `tests/fixtures.py` that writes a **real** file of
   that type, and test against it. Do not mock the parser — parsing is where the
   bugs are.

### What an extractor must honour

- **Never raise on a bad file.** Catch what the underlying library throws and
  return an `Extraction` with `warnings`. A build over a thousand files cannot
  be ended by one of them.
- **Require optional dependencies through `require()`**, so a missing package
  produces an install hint rather than an ImportError.
- **Put text in chunks.** Call `text_chunks()` once you have text; structural
  nodes (pages, sheets, segments) carry a short preview and act as the citable
  anchor. This is what keeps every modality comparable in one ranking.
- **Route expensive work through `ctx.cached`**, so re-indexing an unchanged
  file re-runs nothing.

## Conventions

- Comments explain *why*, not *what*. If a line needs a comment to say what it
  does, rewrite the line. If a decision would look arbitrary to the next reader,
  say what the alternative was and why it lost.
- Prefer precision over recall in anything that writes entities. A missing edge
  costs one query; a wrong one corrupts every query that traverses it.
- Tests describe promises, not implementations. `test_deleting_a_file_prunes_it_and_collects_its_private_entities`
  survives a refactor; `test_clear_file_calls_gc` does not.

## Performance testing

The test suite uses tiny fixtures so it stays fast. To see how Atlas behaves on
real documents, fetch a corpus from public archives:

```bash
python scripts/fetch_corpus.py --out ./benchmark-corpus --size-mb 100
python scripts/benchmark.py --corpus ./benchmark-corpus --json before.json
```

This downloads real files, which is the point — synthetic documents parse
cleanly and therefore test almost nothing. Real ones are where the failures
live: PDFs with no text layer, spreadsheets whose header is three rows down,
mojibake from a 1998 word processor, files whose extension contradicts their
bytes. The `mislabelled`/`detected_format` handling in `sniff.py` exists because
a real corpus surfaced legacy OLE2 workbooks named `.xlsx`.

Four archives are used, each for what it is good at:

| Source | Contributes | Licence |
|---|---|---|
| [Govdocs1](https://digitalcorpora.org/corpora/file-corpora/files/) | ~1M messy real documents from the `.gov` domain: PDF, DOC, XLS, PPT, HTML, TXT, CSV, JPG | Public domain (US Government works) |
| Same, `by_type/` | Modern OOXML: docx, xlsx, pptx | Public domain |
| [Wikimedia Commons](https://commons.wikimedia.org) | Photographs with genuine EXIF, camera models and GPS | Per-file; recorded in the manifest |
| [LibriVox](https://librivox.org) via the Internet Archive | Real human speech, so transcription has something to transcribe | Public domain |
| [Prelinger Archives](https://archive.org/details/prelinger) | Public-domain film with real muxed audio | Public domain |

Govdocs1 archives are 300–500 MB each, so the fetcher does **not** download
them. It reads each zip's central directory over HTTP range requests and pulls
only the members it keeps — indexing a 464 MB archive costs about 80 KB of
transfer.

Every file's URL, size, SHA-256 and licence go into `CORPUS.json`, and
`SOURCES.md` records attribution. Because upstream archives change,
`--from-manifest CORPUS.json` refetches exactly the previous set and reports any
file whose hash has drifted — so two benchmark runs stay comparable.

```bash
python scripts/fetch_corpus.py --from-manifest ./benchmark-corpus/CORPUS.json
```

### Offline fallback

`scripts/make_corpus.py` generates a synthetic corpus with the same shape. It
needs no network, so it is what CI and air-gapped work use, but prefer the real
one when you can reach the internet — it finds bugs the generator cannot.

The corpus folder is gitignored. Regenerate or refetch rather than committing
100 MB of someone else's documents.

Both scripts are dev tools, not part of the shipped package.
