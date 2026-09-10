#!/usr/bin/env python3
"""Download a real, diverse benchmark corpus from public archives.

    python scripts/fetch_corpus.py --out ./benchmark-corpus --size-mb 100

Synthetic files test the plumbing and nothing else. Real documents are where the
interesting failures live: PDFs with no text layer, spreadsheets whose header row
is three rows down, mojibake from a 1998 word processor, JPEGs with truncated
EXIF, MP3s of actual human speech. This pulls such files from four public
archives, each chosen for what it is good at:

  govdocs   ~1M real documents harvested from the .gov domain (Digital Corpora).
            Public domain, deliberately messy, and spanning PDF/DOC/XLS/PPT/HTML/
            TXT/CSV/JPG. This is the bulk of the corpus.
  ooxml     Modern Office formats (docx/xlsx/pptx) from the same corpus, because
            the mixed archives are dominated by their legacy predecessors.
  commons   Wikimedia Commons photographs — genuine EXIF, camera models, and GPS,
            which no generator produces convincingly.
  librivox  Public-domain audiobook recordings from the Internet Archive: real
            human speech, so a transcription pass has something to transcribe.
  video     Prelinger Archives public-domain film, for real audio/video muxing.

Nothing is downloaded twice: files already present with the right size are kept,
so the fetch resumes. Every file's URL, size, SHA-256 and licence are written to
`CORPUS.json`, and `--from-manifest` re-fetches exactly that set — so a benchmark
result stays reproducible even though the upstream archives keep changing.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

MB = 1024 * 1024

# A contactable User-Agent is required by several of these archives and is
# simply good manners for the rest.
USER_AGENT = (
    "atlas-benchmark/0.2 (+https://github.com/nikhil-kunapareddy/Atlas) "
    "python-urllib"
)
GOVDOCS = "https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
ARCHIVE_SEARCH = "https://archive.org/advancedsearch.php"

# Per-file bounds. Tiny files add count without exercising anything; enormous
# ones eat the budget and skew every average.
MIN_BYTES = 4 * 1024
MAX_BYTES = 4 * MB


@dataclass
class Manifest:
    """What was fetched, so a run can be reproduced and attributed."""

    entries: list[dict] = field(default_factory=list)

    def add(self, path: str, url: str, size: int, digest: str, source: str,
            licence: str, credit: str = "") -> None:
        self.entries.append({
            "path": path, "url": url, "bytes": size, "sha256": digest,
            "source": source, "license": licence, "credit": credit,
        })

    @property
    def total(self) -> int:
        return sum(e["bytes"] for e in self.entries)


# -- HTTP ------------------------------------------------------------------


def _request(url: str, headers: dict[str, str] | None = None, method: str = "GET"):
    return urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, **(headers or {})}, method=method
    )


def fetch(url: str, headers: dict[str, str] | None = None, timeout: int = 90,
          retries: int = 3) -> bytes:
    """GET with retries. Archives rate-limit; backing off is expected, not exceptional."""
    delay = 2.0
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(_request(url, headers), timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"failed after {retries} attempts: {url} ({last})")


def fetch_json(url: str) -> dict:
    return json.loads(fetch(url).decode("utf-8", "replace"))


class HttpRangeReader(io.RawIOBase):
    """A seekable file over HTTP range requests.

    This is what makes the Govdocs1 archives usable: each is 300-500MB, but
    `zipfile` only needs the central directory at the tail plus the byte ranges
    of the members actually wanted. Indexing a 464MB archive costs about 80KB.
    """

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        self.requests = 0
        self.downloaded = 0
        with urllib.request.urlopen(_request(url, method="HEAD"), timeout=60) as response:
            self.size = int(response.headers["Content-Length"])

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.pos
        if size <= 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + size, self.size) - 1
        data = fetch(self.url, {"Range": f"bytes={self.pos}-{end}"})
        self.requests += 1
        self.downloaded += len(data)
        self.pos += len(data)
        return data


# -- writing ---------------------------------------------------------------


def save(out: Path, rel: str, blob: bytes, url: str, source: str, licence: str,
         manifest: Manifest, credit: str = "") -> int:
    path = out / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    manifest.add(rel, url, len(blob), hashlib.sha256(blob).hexdigest(),
                 source, licence, credit)
    return len(blob)


def fits(written: int, size: int, budget: int) -> bool:
    """Whether a file of `size` still belongs in this allocation.

    Without this a single 25MB film silently triples a 10MB video budget. The
    `written == 0` case keeps a source from contributing nothing at all when
    every candidate happens to be larger than its share.
    """
    return written == 0 or written + size <= budget


def already_have(out: Path, rel: str, size: int | None = None) -> bool:
    path = out / rel
    if not path.exists():
        return False
    return size is None or path.stat().st_size == size


# -- sources ---------------------------------------------------------------


def source_govdocs(out: Path, budget: int, rng: random.Random, log, manifest: Manifest) -> int:
    """Mixed real documents from the .gov domain.

    Each `zipfiles/NNN.zip` holds ~1000 files: roughly 200 PDF, 180 HTML, 150
    TXT, 110 DOC, 90 JPG, 90 PPT, 60 XLS. Several archives are sampled so the
    corpus is not dominated by one crawl's quirks.
    """
    wanted = {".pdf", ".html", ".htm", ".txt", ".csv", ".xml", ".doc", ".xls", ".ppt", ".jpg"}
    written = 0
    archives = rng.sample(range(0, 200), 6)

    for number in archives:
        if written >= budget:
            break
        url = f"{GOVDOCS}/zipfiles/{number:03d}.zip"
        try:
            reader = HttpRangeReader(url)
            archive = zipfile.ZipFile(reader)
        except Exception as exc:
            log(f"    skipping {number:03d}.zip ({exc})")
            continue

        members = [
            info for info in archive.infolist()
            if Path(info.filename).suffix.lower() in wanted
            and MIN_BYTES <= info.file_size <= MAX_BYTES
        ]
        rng.shuffle(members)

        for info in members:
            if written >= budget:
                break
            if not fits(written, info.file_size, budget):
                continue
            suffix = Path(info.filename).suffix.lower()
            rel = f"govdocs/{suffix.lstrip('.')}/{Path(info.filename).name}"
            if already_have(out, rel, info.file_size):
                written += info.file_size
                continue
            try:
                blob = archive.read(info)
            except Exception:
                continue  # a corrupt member in a corpus of corrupt members
            written += save(out, rel, blob, f"{url}#{info.filename}", "govdocs1",
                            "Public domain (US Government works)", manifest)
        log(f"    {number:03d}.zip → {written / MB:5.1f} MB cumulative "
            f"({reader.downloaded / MB:.1f} MB transferred)")
    return written


def source_ooxml(out: Path, budget: int, rng: random.Random, log, manifest: Manifest) -> int:
    """Modern Office formats, which the mixed archives barely contain."""
    written = 0
    for kind, archive_name in (("xlsx", "xlsx.zip"), ("docx", "docx.zip"), ("pptx", "pptx.zip")):
        if written >= budget:
            break
        share = budget // 3
        url = f"{GOVDOCS}/by_type/{archive_name}"
        try:
            reader = HttpRangeReader(url)
            archive = zipfile.ZipFile(reader)
        except Exception as exc:
            log(f"    skipping {archive_name} ({exc})")
            continue

        members = [
            info for info in archive.infolist()
            if info.filename.lower().endswith(f".{kind}")
            and MIN_BYTES <= info.file_size <= MAX_BYTES
        ]
        rng.shuffle(members)
        taken = 0
        for info in members:
            if taken >= share:
                break
            if not fits(taken, info.file_size, share):
                continue
            rel = f"ooxml/{kind}/{Path(info.filename).name}"
            if already_have(out, rel, info.file_size):
                taken += info.file_size
                continue
            try:
                blob = archive.read(info)
            except Exception:
                continue
            taken += save(out, rel, blob, f"{url}#{info.filename}", "govdocs1",
                          "Public domain (US Government works)", manifest)
        written += taken
        log(f"    {kind:5s} → {taken / MB:5.1f} MB")
    return written


def source_commons(out: Path, budget: int, rng: random.Random, log, manifest: Manifest) -> int:
    """Photographs from Wikimedia Commons, for genuine EXIF and GPS."""
    categories = [
        "Category:Quality images of landscapes",
        "Category:Quality images of architecture",
        "Category:Quality images of vehicles",
        "Category:Featured pictures of animals",
    ]
    candidates: list[dict] = []
    for category in categories:
        params = urllib.parse.urlencode({
            "action": "query", "format": "json", "generator": "categorymembers",
            "gcmtitle": category, "gcmtype": "file", "gcmlimit": "120",
            "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
        })
        try:
            payload = fetch_json(f"{COMMONS_API}?{params}")
        except Exception as exc:
            log(f"    {category}: {exc}")
            continue
        for page in payload.get("query", {}).get("pages", {}).values():
            for info in page.get("imageinfo", []):
                if info.get("mime") != "image/jpeg":
                    continue
                if not (200 * 1024 <= info.get("size", 0) <= 3 * MB):
                    continue
                meta = info.get("extmetadata", {})
                candidates.append({
                    "title": page.get("title", ""),
                    "url": info["url"].split("?")[0],
                    "size": info["size"],
                    "licence": meta.get("LicenseShortName", {}).get("value", "see Commons"),
                    "credit": _strip_tags(meta.get("Artist", {}).get("value", "")),
                })

    rng.shuffle(candidates)
    written = 0
    for item in candidates:
        if written >= budget:
            break
        if not fits(written, item["size"], budget):
            continue
        rel = f"images/{Path(urllib.parse.unquote(item['url'])).name}"
        if already_have(out, rel, item["size"]):
            written += item["size"]
            continue
        try:
            blob = fetch(item["url"])
        except Exception:
            continue
        written += save(out, rel, blob, item["url"], "wikimedia-commons",
                        item["licence"], manifest, item["credit"])
    log(f"    {len(candidates)} candidates → {written / MB:5.1f} MB")
    return written


def _archive_items(query: str, rows: int, rng: random.Random) -> list[str]:
    params = urllib.parse.urlencode({
        "q": query, "fl[]": "identifier", "rows": str(rows),
        "page": str(rng.randrange(1, 6)), "output": "json",
    })
    payload = fetch_json(f"{ARCHIVE_SEARCH}?{params}")
    return [d["identifier"] for d in payload.get("response", {}).get("docs", [])]


def source_librivox(out: Path, budget: int, rng: random.Random, log, manifest: Manifest) -> int:
    """Public-domain audiobook chapters: real speech for a real transcription test."""
    written = 0
    try:
        items = _archive_items("collection:librivoxaudio", 40, rng)
    except Exception as exc:
        log(f"    archive.org search failed: {exc}")
        return 0
    rng.shuffle(items)

    for identifier in items:
        if written >= budget:
            break
        try:
            meta = fetch_json(f"https://archive.org/metadata/{identifier}")
        except Exception:
            continue
        files = _one_per_chapter([
            f for f in meta.get("files", [])
            if f.get("name", "").endswith(".mp3")
            and MIN_BYTES <= int(f.get("size", 0) or 0) <= max(2 * MB, min(12 * MB, budget))
        ])
        # One chapter per book keeps voices and recording conditions varied.
        for entry in files[:2]:
            if written >= budget:
                break
            if not fits(written, int(entry["size"]), budget):
                continue
            rel = f"audio/{identifier}-{entry['name']}"
            size = int(entry["size"])
            if already_have(out, rel, size):
                written += size
                continue
            url = f"https://archive.org/download/{identifier}/{urllib.parse.quote(entry['name'])}"
            try:
                blob = fetch(url, timeout=180)
            except Exception:
                continue
            written += save(out, rel, blob, url, "librivox",
                            "Public domain", manifest)
        time.sleep(0.5)  # be a good citizen with a shared archive
    log(f"    {written / MB:5.1f} MB of speech")
    return written


def source_video(out: Path, budget: int, rng: random.Random, log, manifest: Manifest) -> int:
    """Prelinger Archives film — real muxed audio/video, public domain."""
    written = 0
    try:
        items = _archive_items("collection:prelinger AND mediatype:movies", 40, rng)
    except Exception as exc:
        log(f"    archive.org search failed: {exc}")
        return 0
    rng.shuffle(items)

    for identifier in items:
        if written >= budget:
            break
        try:
            meta = fetch_json(f"https://archive.org/metadata/{identifier}")
        except Exception:
            continue
        # Archive.org keeps a large master plus smaller derivatives; take the
        # smallest usable derivative rather than a 128MB master.
        # Cap per-file size by the allocation itself. The "first file is
        # always allowed" rule exists so a source is never empty, but combined
        # with a 25MB ceiling it let one film triple the video budget.
        ceiling = max(4 * MB, min(25 * MB, budget))
        videos = sorted(
            (f for f in meta.get("files", [])
             if f.get("name", "").lower().endswith(".mp4")
             and MIN_BYTES <= int(f.get("size", 0) or 0) <= ceiling),
            key=lambda f: int(f["size"]),
        )
        if not videos:
            continue
        entry = videos[0]
        if not fits(written, int(entry["size"]), budget):
            continue
        rel = f"video/{identifier}-{entry['name']}"
        size = int(entry["size"])
        if already_have(out, rel, size):
            written += size
            continue
        url = f"https://archive.org/download/{identifier}/{urllib.parse.quote(entry['name'])}"
        try:
            blob = fetch(url, timeout=300)
        except Exception:
            continue
        written += save(out, rel, blob, url, "prelinger",
                        meta.get("metadata", {}).get("licenseurl", "Public domain"), manifest)
        time.sleep(0.5)
    log(f"    {written / MB:5.1f} MB of film")
    return written


def _one_per_chapter(files: list[dict]) -> list[dict]:
    """Collapse archive.org's bitrate variants of the same recording.

    An item lists `chapter.mp3`, `chapter_64kb.mp3` and `chapter_128kb.mp3` —
    the same audio three times. Indexing all three would inflate the corpus and
    make the transcription benchmark measure duplicate work.
    """
    best: dict[str, dict] = {}
    for entry in files:
        stem = entry["name"].rsplit(".", 1)[0]
        for suffix in ("_64kb", "_128kb", "_vbr", "_hifi"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        current = best.get(stem)
        if current is None or int(entry["size"]) < int(current["size"]):
            best[stem] = entry
    return sorted(best.values(), key=lambda f: f["name"])


def _strip_tags(html: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", html).strip()[:200]


# -- driver ----------------------------------------------------------------

# Fractions of the byte budget. Documents dominate node count; audio and video
# dominate bytes while producing few nodes — both axes matter.
MIX = {
    "govdocs": 0.50,
    "ooxml": 0.16,
    "audio": 0.14,
    "images": 0.12,
    "video": 0.08,
}

SOURCES = {
    "govdocs": source_govdocs,
    "ooxml": source_ooxml,
    "images": source_commons,
    "audio": source_librivox,
    "video": source_video,
}


def write_attribution(out: Path, manifest: Manifest) -> None:
    by_source: dict[str, list[dict]] = {}
    for entry in manifest.entries:
        by_source.setdefault(entry["source"], []).append(entry)

    lines = [
        "# Corpus sources",
        "",
        "This folder was assembled by `scripts/fetch_corpus.py` from public",
        "archives. It is benchmark input, not part of Atlas, and is gitignored.",
        "",
    ]
    descriptions = {
        "govdocs1": ("Govdocs1 / Digital Corpora — https://digitalcorpora.org/corpora/file-corpora/files/",
                     "Documents harvested from the US .gov domain. US Government works are public domain."),
        "wikimedia-commons": ("Wikimedia Commons — https://commons.wikimedia.org",
                              "Individual files carry their own licences; see per-file entries in CORPUS.json."),
        "librivox": ("LibriVox via the Internet Archive — https://librivox.org",
                     "LibriVox recordings are released into the public domain."),
        "prelinger": ("Prelinger Archives via the Internet Archive — https://archive.org/details/prelinger",
                      "Public domain ephemeral film."),
    }
    for source, entries in sorted(by_source.items()):
        title, note = descriptions.get(source, (source, ""))
        total = sum(e["bytes"] for e in entries) / MB
        lines += [f"## {title}", "", note, "",
                  f"{len(entries)} files, {total:.1f} MB.", ""]
        licences = sorted({e["license"] for e in entries})
        if len(licences) > 1 or (licences and licences[0] != "Public domain"):
            lines.append("Licences present: " + ", ".join(licences[:12]))
            lines.append("")
    (out / "SOURCES.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="./benchmark-corpus")
    parser.add_argument("--size-mb", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true", help="delete and refetch")
    parser.add_argument("--only", action="append", choices=sorted(SOURCES))
    parser.add_argument("--from-manifest", metavar="FILE",
                        help="refetch exactly the files listed in a CORPUS.json")
    args = parser.parse_args()

    out = Path(args.out).expanduser().resolve()
    if out.exists() and args.force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = Manifest()
    rng = random.Random(args.seed)
    started = time.time()

    if args.from_manifest:
        return _replay(Path(args.from_manifest), out, manifest, started)

    print(f"Fetching ~{args.size_mb:.0f} MB of real documents into {out}")
    print("Sources: Govdocs1 (.gov), Wikimedia Commons, LibriVox, Prelinger Archives\n")

    total_budget = args.size_mb * MB
    for name, share in MIX.items():
        if args.only and name not in args.only:
            continue
        print(f"  {name}")
        try:
            SOURCES[name](out, int(total_budget * share), rng, print, manifest)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"    {name} failed: {exc}")

    _finish(out, manifest, started)
    return 0


def _replay(manifest_path: Path, out: Path, manifest: Manifest, started: float) -> int:
    """Refetch exactly the files a previous run recorded.

    Upstream archives change: files are withdrawn, re-encoded, or replaced. A
    benchmark comparison across weeks is only meaningful if the inputs are the
    same bytes, so this verifies each SHA-256 and says plainly when one has
    drifted rather than quietly benchmarking different data.
    """
    previous = json.loads(manifest_path.read_text())
    entries = previous["files"]
    print(f"Replaying {len(entries)} files from {manifest_path}")

    # One ZipFile per archive: each construction re-reads a central directory,
    # and a replay typically pulls many members from the same few archives.
    archives: dict[str, zipfile.ZipFile] = {}
    restored = missing = changed = 0

    for entry in entries:
        if already_have(out, entry["path"], entry["bytes"]):
            manifest.entries.append(entry)
            restored += 1
            continue
        url = entry["url"]
        try:
            if "#" in url:
                archive_url, member = url.split("#", 1)
                if archive_url not in archives:
                    archives[archive_url] = zipfile.ZipFile(HttpRangeReader(archive_url))
                blob = archives[archive_url].read(member)
            else:
                blob = fetch(url, timeout=300)
        except Exception as exc:
            print(f"  gone: {entry['path']} ({type(exc).__name__})")
            missing += 1
            continue

        if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
            print(f"  CHANGED upstream: {entry['path']}")
            changed += 1
        path = out / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        manifest.entries.append(entry)
        restored += 1

    print(f"  {restored} restored, {missing} no longer available, {changed} changed upstream")
    _finish(out, manifest, started)
    return 0


def _finish(out: Path, manifest: Manifest, started: float) -> None:
    files = [p for p in out.rglob("*") if p.is_file() and p.name not in
             ("CORPUS.json", "SOURCES.md")]
    actual = sum(p.stat().st_size for p in files)
    (out / "CORPUS.json").write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": manifest.entries,
        "total_bytes": manifest.total,
    }, indent=2))
    write_attribution(out, manifest)
    print(f"\n  {actual / MB:.1f} MB across {len(files)} files "
          f"in {time.time() - started:.0f}s")
    print(f"  provenance: {out / 'CORPUS.json'} · attribution: {out / 'SOURCES.md'}")


if __name__ == "__main__":
    sys.exit(main())
