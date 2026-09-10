<p align="center">
  <img src="assets/banner.png" alt="Atlas" width="640">
</p>

<p align="center">
  <strong>Turn a folder into a graph your agent can query.</strong>
</p>

<p align="center">
  <a href="https://github.com/nikhil-kunapareddy/Atlas/actions/workflows/ci.yml"><img src="https://github.com/nikhil-kunapareddy/Atlas/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT">
</p>

---

Point Atlas at a directory of PDFs, spreadsheets, images, recordings and notes.
It reads each file according to what it actually is — a PDF by page, a workbook
by sheet and column, a recording by timestamp — and links them through the
things they mention. Then you talk to the result.

```bash
atlas ~/research          # build the graph
atlas ask "which documents discuss the Q3 migration, and what did they conclude?"
```

The graph is a single SQLite file inside the folder. No server, no cloud, no API
key needed to build one.

---

## Why a graph

Search over a folder answers *"which file contains these words?"*. That is the
wrong question for most of what people keep in folders. The useful questions are
relational:

- *Which files reference invoice `INV-2024-0912`?* — the number appears in a PDF,
  a spreadsheet row, and an email export. Three formats, one answer.
- *Which spreadsheets have a `customer_id` column?* — a question about schema,
  not text.
- *What was said about the migration, and where?* — a transcript at `14:32`, a
  slide's speaker notes, a paragraph on page 4.

Atlas answers those by making the join explicit. Files become nodes, their
internal structure becomes nodes, and shared entities become the edges between
them.

```
dir:.
└── doc:docs/report.pdf ──contains──> doc:docs/report.pdf#page:1
                                          └──contains──> …#chunk:0
                                                             │
                                                          mentions
                                                             ↓
                                          ent:identifier:inv-2024-0912
                                                             ↑
                                                          mentions
                                                             │
      doc:README.md ──contains──> doc:README.md#chunk:0 ─────┘
```

Nothing above required a language model. `INV-2024-0912` is a string with an
unambiguous shape, so the link between the PDF and the Markdown file is exact,
free, and reproducible.

---

## Install

```bash
pip install "atlas-context[all]"     # every modality
```

The core has **no dependencies** — stdlib `sqlite3` stores and queries the
graph. Each modality is an extra, so you can install only what you need:

| Extra | Unlocks |
|---|---|
| `pdf` | PDF text, pages, outlines |
| `sheets` | Excel workbooks (`.csv` needs nothing) |
| `images` | dimensions, EXIF, GPS |
| `office` | Word documents, PowerPoint decks |
| `media` | speech transcription for audio and video |
| `ocr` | text inside images and scans |
| `llm` | the `ask` agent and semantic enrichment |
| `mcp` | the MCP server |

`atlas doctor` tells you what the current installation can read, and prints the
exact `pip install` line for anything missing.

Two external tools are used when present: **ffmpeg/ffprobe** for media duration
and video keyframes, **tesseract** for OCR. Neither is required.

---

## Quickstart

```bash
$ atlas ~/research

Building graph from /Users/you/research…

  6 files, 31 nodes, 39 edges in 0.3s
  2 text, 1 sheet, 1 pdf, 1 image, 1 audio
  15 entity mentions linked

  most connected
       8  [sheet] customers.csv [customers]
       5  [chunk] README.md:1-4 — Northwind Q3
       4  [folder] docs
```

Then either talk to it:

```bash
$ atlas ask "who owns the Q3 migration and what is it costing?"
```

or query it directly:

```bash
$ atlas search "migration"
$ atlas neighbors ent:identifier:inv-2024-0912 --direction in
  ← mentions   [chunk] README.md:1-4 — Northwind Q3
  ← mentions   [chunk] brief.md:1-4 — Brief
  ← mentions   [chunk] report.pdf p.1
```

Re-running `atlas ~/research` is nearly free: unchanged files are recognised by
`(mtime, size)` and never opened, and transcription and OCR results are cached
by content hash. Edit one file and only that file is re-read.

---

## What gets extracted

Every modality produces the same kind of output — a document node, its internal
structure, and text chunks — so results from a spreadsheet and a video rank
against each other in one search.

| Format | Becomes |
|---|---|
| PDF | pages (with numbers and bookmark titles), text chunks |
| `.xlsx` `.csv` `.tsv` | sheets, **columns** with inferred type and sample values, row batches |
| `.docx` | heading hierarchy preserved as structure, tables |
| `.pptx` | slides, **including speaker notes** |
| Images | dimensions, format, camera, capture time, GPS; text via `--ocr` |
| Audio | timestamped transcript segments |
| Video | transcript segments plus sampled keyframes, each attached to its moment |
| Text, code, Markdown, HTML | headed chunks; relative links become edges |

**A folder always produces a graph.** A corrupt PDF, a password-protected
workbook, or a video on a machine without ffmpeg becomes a node carrying a
warning explaining what was missing. One bad file never fails a run over a
thousand good ones.

---

## Talking to it

`atlas ask` gives Claude five graph primitives — `overview`, `search_graph`,
`get_node`, `get_neighbors`, `find_path` — and lets it investigate. The graph is
never pasted into the prompt, so folders far larger than a context window stay
answerable.

For agents that speak MCP, the same tools are available to Claude Code, Codex,
and anything else:

```bash
claude mcp add atlas -- atlas mcp
```

```json
{ "mcpServers": { "atlas": { "command": "atlas", "args": ["mcp"] } } }
```

---

## Two layers of meaning, and telling them apart

Atlas extracts entities in two passes, and **never confuses one for the other**.

**Deterministic** (always on, free, offline). Emails, URLs, dates, amounts,
phone numbers, and identifiers like `INV-2024-0912` or `PROJ-441`. These have
unambiguous surface forms, so they are found by pattern and are exact. Dates
normalise across spellings, so `September 30, 2024`, `9/30/2024` and
`2024-09-30` all resolve to one node — which is what makes the join work.

People, organisations and concepts are deliberately *not* in this pass. A
"capitalised words are probably names" heuristic yields a graph confidently
asserting that `Best Regards` is a person, and a wrong entity is worse than a
missing one because every query that traverses it inherits the error.

**Semantic** (opt-in, costs money). `atlas enrich` reads passages with Claude and
adds the people, organisations, places and concepts a document is *about*, plus
labelled relationships between them.

Everything from this pass is tagged `provenance: llm`. The CLI marks it, the MCP
tools report it, and the agent's system prompt instructs it to hedge on those
claims. You can always tell what was read off the bytes from what was guessed.

### Cost

Building a graph costs **nothing**. Enrichment is the only part that bills, it
is opt-in per run, and it prints an estimate before spending anything:

```bash
$ atlas enrich --estimate-only
482 passages · ~361,000 input tokens · estimated $2.61 on claude-opus-5
```

| Model | ~1,000-file folder |
|---|---|
| `claude-haiku-4-5` | ~$8 |
| `claude-sonnet-5` | ~$16 |
| `claude-opus-5` (default) | ~$40 |

Results are cached against the hash of the passage text, so re-running after
editing one file re-bills only that file.

---

## Performance

Measured on a **115 MB corpus of real documents** — 301 files pulled from
Govdocs1 (`.gov` archives), Wikimedia Commons, LibriVox and the Prelinger
Archives — on an M2 Pro laptop. Reproduce with:

```bash
python scripts/fetch_corpus.py --out ./benchmark-corpus --size-mb 100
python scripts/benchmark.py --corpus ./benchmark-corpus
```

**Build.** 18.6s cold for 301 files → 19,319 nodes and 36,202 edges (6.2 MB/s).

| | |
|---|---|
| cold build (empty database) | 18.6 s |
| rebuild, nothing changed | **0.01 s** |
| speech transcription (35 min of audio, Whisper base, CPU) | +82 s |
| store on disk | 86 MB — **74%** of the corpus |

The warm number is the one that matters: re-running over an unchanged folder
costs 10ms, because files are recognised from `(mtime, size)` and never opened.

**Queries**, p50, with search terms sampled from the corpus itself:

| | |
|---|---|
| shortest path between two documents | 0.05 ms |
| neighbours of the busiest entity | 0.6 ms |
| `counts` | 1.8 ms |
| full-text search | 0.3–13 ms |
| `hubs` (degree across every node) | 20 ms |

### What real data changed

An earlier version of this benchmark used a *generated* corpus. Swapping it for
real documents moved three numbers enough to invalidate the conclusions drawn
from it:

| | generated | real |
|---|---|---|
| store size vs. corpus | 191% | **74%** |
| mentions per entity | 7.5 | **1.8** |
| files no extractor can read | 0.03% | **17%** |

The storage figure was inflated because generated files are all text; a real
archive is mostly media bytes that produce few nodes. The entity figure was
inflated because the generator seeded shared invoice IDs across files —
real documents share literals far less often, so cross-document joins are
scarcer and harder-won than the synthetic corpus implied.

The third row is the one that mattered most: **17% of a real `.gov` archive is
legacy `.doc`/`.xls`/`.ppt`**, which Atlas cannot read. It now says so, grouped,
instead of silently indexing them by filename alone.

**Entity density.** 3,076 entities, of which 151 appear in more than one file.
The rest occur once — still searchable, but contributing nothing to
cross-document joins while carrying most of the `mentions` edges.

## Command reference

```
atlas <folder>                    build a graph (shorthand for `atlas build`)
atlas build <folder>              --ocr, --no-transcribe, --whisper MODEL,
                                  --keyframes N, --language, --no-prune

atlas ask <question>              ask an agent; --trace, --model, --max-steps
atlas search <words>              full-text search; --kind, -k, --json
atlas node <uid>                  one node in full; --full, --json
atlas neighbors <uid>             one hop; --direction, --edge, --json
atlas path <uid> <uid>            shortest connection; --max-depth
atlas hubs                        most-connected nodes; --kind
atlas graph                       node and edge counts
atlas export --out graph.json     node-link JSON for any graph library
atlas enrich                      semantic pass; --estimate-only, --model, --yes
atlas doctor                      what this installation can read
atlas mcp                         run the MCP server on stdio
```

Query commands find the graph by walking up from the working directory, like
git. `--store <folder>` targets one explicitly.

Atlas also keeps the fact store it began as — `atlas remember`, `atlas recall`,
`atlas facts` — for durable notes an agent writes and reads back. Both live in
the same database.

---

## How it works

```
atlas <folder>
   │
   ├─ walk ─────────► skip ignored dirs, oversized files
   │
   ├─ per file ─────► digest → unchanged? stop.
   │                  extractor for this type → pages / sheets / segments / chunks
   │                  regex pass over chunks → entity nodes + MENTIONS edges
   │
   └─ finish ───────► resolve forward edges, collect orphans, commit
```

Four design decisions carry most of the weight:

**Node identity comes from what a thing is, not when it was written.** A uid is
`doc:reports/q3.pdf#page:4`, so a rebuild upserts in place. Idempotency and
incremental rebuilds both fall out of this.

**Structural nodes are owned by a file; entity nodes are not.** Deleting a PDF
must delete its pages but must not delete `Acme Corp`, which two hundred other
files also mention. Ownership is recorded per row, and entities left with no
edges are swept afterwards.

**Chunks are the one retrieval unit.** Pages, sheets and segments carry a short
preview and act as citable anchors; the text itself lives in chunks. That is why
a transcript window and a spreadsheet row block are comparable in one ranking.

**Anything expensive is content-addressed.** Whisper transcripts, OCR output and
LLM enrichment are keyed by the digest of their input plus the producer's
version — so improving an extractor invalidates only its own cache.

Storage is SQLite: `nodes`, `edges`, and an FTS5 index, in one file at
`<folder>/.atlas/store.db`. It is transactional and incrementally writable,
which a JSON graph file is not. `atlas export` emits portable node-link JSON for
anything that wants it.

---

## Development

```bash
uv venv && uv pip install -e ".[dev,all]"
uv run pytest
uv run ruff check src tests
```

221 tests cover the graph store, every extractor, the build pipeline's
incremental and deletion behaviour, entity precision, content sniffing, the
agent's tool loop, and the CLI. Fixtures build genuine PDFs, workbooks, images
and WAVs rather than mocking parsers, since parsing is where the bugs are.

For anything performance- or robustness-related, run against real documents
rather than the fixtures — `scripts/fetch_corpus.py` assembles a corpus from
public archives, and it is what surfaced the mislabelled-file handling, the
parser-noise suppression, and the unreadable-format reporting. See
[CONTRIBUTING.md](CONTRIBUTING.md#performance-testing).

New modalities are additive: implement `Extractor`, add it to the registry in
`src/atlas/extract/__init__.py`. Nothing else needs to know the list grew.

## Status

Working and tested, and benchmarked on a 100 MB corpus (see above). Build and
traversal are comfortable at ~80k nodes and ~420k edges.

Known and deliberate, in rough priority order:

- **No support for legacy Office formats** (`.doc`, `.xls`, `.ppt`), which are
  17% of a real government archive. They are indexed by name and metadata and
  reported as unread.
- **The FTS index stores a second copy of every string**, because it is a plain
  rather than external-content FTS5 table. Less costly on a real corpus (74%
  overhead overall) than on an all-text one, but still the first thing to fix.
- **`hubs` is 72 ms**, because degree is computed across the whole edge table on
  every call. A maintained degree column would make it constant-time.
- **Search is lexical only.** BM25 over FTS5, with brute-force cosine available
  behind the `embeddings` extra. `GraphStore` is the seam an ANN index slots
  into.
- **Semantic entity extraction requires an LLM.** There is no local NER pass, so
  people and organisations are absent from a zero-cost graph by design.

## License

MIT
