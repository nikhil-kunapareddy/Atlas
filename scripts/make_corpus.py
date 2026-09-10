#!/usr/bin/env python3
"""Generate a large, realistic multimodal corpus for local performance testing.

The point is not to make 100MB of noise. Random bytes would exercise the
walker and nothing else. This builds a plausible company archive — invoices,
ledgers, specs, research papers, photos, recordings, screencasts — in which the
same people, invoice numbers, tickets, amounts and dates **recur across files
and across formats**. That is what actually stresses Atlas: entity resolution,
edge fan-out on hub nodes, and the FTS index, none of which random data touches.

    python scripts/make_corpus.py --out ./benchmark-corpus --size-mb 100

Deterministic for a given `--seed`, so two runs produce byte-identical trees and
benchmark numbers are comparable. The generated folder is gitignored.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# The file builders are already written and tested for the suite; a dev script
# reusing them beats a second, untested PDF writer living here.
sys.path.insert(0, str(REPO / "tests"))
import fixtures  # noqa: E402

MB = 1024 * 1024

# -- the fictional world ---------------------------------------------------
#
# A small closed vocabulary is the whole trick: with ~20 companies and ~20
# people spread over thousands of files, entity nodes accumulate real degree,
# so hub queries and MENTIONS fan-out get exercised at a scale that matters.

COMPANIES = [
    "Acme Robotics", "Globex Shipping", "Initech Systems", "Umbrella Freight",
    "Soylent Foods", "Vandelay Industries", "Hooli Cloud", "Massive Dynamic",
    "Wonka Logistics", "Stark Materials", "Tyrell Analytics", "Cyberdyne Rail",
]
PEOPLE = [
    "Ana Rodriguez", "Ben Okafor", "Chen Wei", "Dara Ellis", "Emil Novak",
    "Farah Haddad", "Grace Lindqvist", "Hugo Marchetti", "Iris Nakamura",
    "Jonas Weber", "Kiara Patel", "Luca Moreau",
]
CITIES = [
    "Lisbon", "Rotterdam", "Singapore", "Chicago", "Hamburg", "Osaka",
    "São Paulo", "Vancouver", "Nairobi", "Gdansk",
]
TEAMS = ["Platform", "Logistics", "Finance", "Research", "Field Ops"]

SENTENCES = [
    "{person} confirmed that {company} would take delivery in {city} on {d}.",
    "Invoice {inv} was raised against {company} for {amount} and settled on {d}.",
    "Ticket {tick} tracks the {team} migration; {person} is the owner.",
    "The {team} team reported a {pct}% reduction in transit time out of {city}.",
    "Escalation from {email} regarding {inv}: the {company} shipment was short by {n} units.",
    "Following the {city} audit, {person} recommended renegotiating the {company} rate card.",
    "Budget for {tick} is {amount}, approved {d} by {person}.",
    "{company} and {company2} both route through {city}, which creates a single point of failure.",
    "Quarterly spend with {company} reached {amount}, up from the prior period.",
    "See {inv} and {tick} for the reconciliation {person} prepared on {d}.",
    "The {team} runbook was updated after the {d} incident affecting {city}.",
    "{person} met {person2} to review the {company} contract renewal at {amount} per year.",
]


class World:
    """Deterministic generator of prose full of recurring, linkable entities."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.invoices = [f"INV-2024-{n:04d}" for n in range(1, 260)]
        self.tickets = [f"NW-{n:04d}" for n in range(1, 200)]

    def email(self, person: str) -> str:
        first, last = person.split(" ", 1)
        domain = self.rng.choice(COMPANIES).split(" ")[0].lower()
        return f"{first.lower()}.{last.lower().replace(' ', '')}@{domain}.example"

    def day(self) -> str:
        return (date(2024, 1, 1) + timedelta(days=self.rng.randrange(0, 365))).isoformat()

    def amount(self) -> str:
        return f"${self.rng.randrange(500, 900000):,}.{self.rng.randrange(0, 100):02d}"

    def sentence(self) -> str:
        person, person2 = self.rng.sample(PEOPLE, 2)
        company, company2 = self.rng.sample(COMPANIES, 2)
        return self.rng.choice(SENTENCES).format(
            person=person, person2=person2,
            company=company, company2=company2,
            city=self.rng.choice(CITIES), team=self.rng.choice(TEAMS),
            inv=self.rng.choice(self.invoices), tick=self.rng.choice(self.tickets),
            amount=self.amount(), email=self.email(person), d=self.day(),
            pct=self.rng.randrange(2, 40), n=self.rng.randrange(2, 500),
        )

    def paragraph(self, sentences: int = 5) -> str:
        return " ".join(self.sentence() for _ in range(sentences))

    def section(self, level: int = 2) -> str:
        heading = self.rng.choice(
            ["Background", "Findings", "Method", "Risks", "Next steps", "Costs",
             "Timeline", "Open questions", "Appendix", "Summary"]
        )
        return f"{'#' * level} {heading}\n\n{self.paragraph(self.rng.randrange(3, 8))}\n"


# -- per-modality builders -------------------------------------------------


def build_text(root: Path, world: World, budget: int, log) -> int:
    """Markdown, HTML and source files — cheap bytes, many chunks."""
    written = 0
    docs = root / "engineering" / "docs"
    src = root / "engineering" / "src"
    docs.mkdir(parents=True, exist_ok=True)
    src.mkdir(parents=True, exist_ok=True)

    index = 0
    while written < budget:
        index += 1
        team = world.rng.choice(TEAMS).lower().replace(" ", "-")
        body = f"# {world.rng.choice(TEAMS)} note {index}\n\n" + "\n".join(
            world.section() for _ in range(world.rng.randrange(3, 10))
        )
        # A relative link every few files, so REFERENCES edges get built and the
        # forward-reference resolution path is exercised at scale.
        if index > 1 and index % 3 == 0:
            body += f"\n\nSee [the previous note](note-{index - 1}-{team}.md).\n"
        path = docs / f"note-{index}-{team}.md"
        path.write_text(body)
        written += path.stat().st_size

        if index % 4 == 0:
            code = (
                f'"""Module {index} — {world.sentence()}"""\n\n'
                + "\n\n".join(
                    f"def handler_{i}(payload):\n"
                    f'    """{world.sentence()}"""\n'
                    f"    return payload[{i}]\n"
                    for i in range(world.rng.randrange(4, 15))
                )
            )
            p = src / f"service_{index}.py"
            p.write_text(code)
            written += p.stat().st_size

        if index % 7 == 0:
            html = (
                f"<html><head><title>Report {index}</title></head><body>\n"
                + "\n".join(f"<p>{world.sentence()}</p>" for _ in range(30))
                + "\n</body></html>"
            )
            p = docs / f"report-{index}.html"
            p.write_text(html)
            written += p.stat().st_size

    log(f"  text     {written / MB:6.1f} MB  {index} notes")
    return written


def build_pdfs(root: Path, world: World, budget: int, log) -> int:
    written = count = 0
    invoices = root / "finance" / "invoices"
    papers = root / "research" / "papers"
    invoices.mkdir(parents=True, exist_ok=True)
    papers.mkdir(parents=True, exist_ok=True)

    while written < budget:
        count += 1
        if count % 4 == 0:
            # A long document: many pages, exercising page/chunk fan-out.
            pages = [world.paragraph(12) for _ in range(world.rng.randrange(8, 30))]
            path = papers / f"paper-{count:04d}.pdf"
            title = f"Study {count}: {world.rng.choice(CITIES)} corridor analysis"
        else:
            inv = world.rng.choice(world.invoices)
            pages = [
                f"INVOICE {inv}\n{world.rng.choice(COMPANIES)}\n{world.paragraph(6)}",
                world.paragraph(8),
            ]
            path = invoices / f"{inv}-{count:04d}.pdf"
            title = f"Invoice {inv}"
        fixtures.make_pdf(path, pages, title=title)
        written += path.stat().st_size

    log(f"  pdf      {written / MB:6.1f} MB  {count} files")
    return written


def build_sheets(root: Path, world: World, budget: int, log) -> int:
    written = count = 0
    ledgers = root / "finance" / "ledgers"
    data = root / "research" / "data"
    ledgers.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    header = ["invoice_id", "customer", "owner_email", "city", "amount_usd", "booked_on", "ticket"]
    while written < budget:
        count += 1
        rows = world.rng.randrange(400, 4000)
        if count % 5 == 0:
            # A real workbook with several sheets.
            sheets = {}
            for name in ("Ledger", "Disputes", "Forecast"):
                sheets[name] = [header] + [
                    [
                        world.rng.choice(world.invoices),
                        world.rng.choice(COMPANIES),
                        world.email(world.rng.choice(PEOPLE)),
                        world.rng.choice(CITIES),
                        round(world.rng.uniform(100, 90000), 2),
                        world.day(),
                        world.rng.choice(world.tickets),
                    ]
                    for _ in range(rows // 3)
                ]
            path = ledgers / f"ledger-{count:03d}.xlsx"
            fixtures.make_xlsx(path, sheets)
        else:
            path = data / f"shipments-{count:03d}.csv"
            lines = [",".join(header)]
            for _ in range(rows):
                lines.append(
                    ",".join(
                        [
                            world.rng.choice(world.invoices),
                            world.rng.choice(COMPANIES).replace(",", ""),
                            world.email(world.rng.choice(PEOPLE)),
                            world.rng.choice(CITIES),
                            f"{world.rng.uniform(100, 90000):.2f}",
                            world.day(),
                            world.rng.choice(world.tickets),
                        ]
                    )
                )
            path.write_text("\n".join(lines) + "\n")
        written += path.stat().st_size

    log(f"  sheets   {written / MB:6.1f} MB  {count} files")
    return written


def build_docx(root: Path, world: World, budget: int, log) -> int:
    try:
        import docx
    except ImportError:
        log("  docx        skipped (python-docx not installed)")
        return 0

    specs = root / "engineering" / "specs"
    specs.mkdir(parents=True, exist_ok=True)
    written = count = 0
    while written < budget:
        count += 1
        document = docx.Document()
        document.add_heading(f"Specification {count}", level=1)
        for _ in range(world.rng.randrange(4, 12)):
            document.add_heading(
                world.rng.choice(["Scope", "Interfaces", "Constraints", "Rollout"]), level=2
            )
            document.add_paragraph(world.paragraph(6))
        table = document.add_table(rows=1, cols=3)
        for cell, text in zip(table.rows[0].cells, ("invoice", "owner", "amount"), strict=True):
            cell.text = text
        for _ in range(world.rng.randrange(5, 25)):
            cells = table.add_row().cells
            cells[0].text = world.rng.choice(world.invoices)
            cells[1].text = world.rng.choice(PEOPLE)
            cells[2].text = world.amount()
        path = specs / f"spec-{count:03d}.docx"
        document.save(str(path))
        written += path.stat().st_size

    log(f"  docx     {written / MB:6.1f} MB  {count} files")
    return written


def build_images(root: Path, world: World, budget: int, log) -> int:
    import numpy as np
    from PIL import Image

    photos = root / "media" / "photos"
    photos.mkdir(parents=True, exist_ok=True)
    written = count = 0
    rng = np.random.default_rng(world.rng.randrange(1 << 30))

    while written < budget:
        count += 1
        w, h = world.rng.choice([(1280, 960), (1600, 1200), (1920, 1080), (2400, 1600)])
        # Smooth gradients plus noise: compresses like a photograph rather than
        # to a few hundred bytes, so the sizes here are representative.
        base = np.linspace(0, 255, w, dtype=np.uint8)
        frame = np.repeat(base[None, :], h, axis=0)
        rgb = np.stack([frame, np.roll(frame, 60, axis=1), np.roll(frame, 120, axis=1)], axis=2)
        rgb = np.clip(rgb.astype(np.int16) + rng.integers(-40, 40, (h, w, 3)), 0, 255)
        image = Image.fromarray(rgb.astype(np.uint8))

        exif = image.getexif()
        exif[271] = world.rng.choice(["Canon", "Nikon", "Apple", "Fujifilm"])
        exif[272] = f"Model {world.rng.randrange(1, 40)}"
        exif[306] = f"{world.day().replace('-', ':')} 12:{world.rng.randrange(10, 59)}:00"
        exif[270] = world.sentence()[:120]
        path = photos / f"{world.rng.choice(CITIES).lower()}-{count:04d}.jpg"
        image.save(str(path), quality=80, exif=exif)
        written += path.stat().st_size

    log(f"  images   {written / MB:6.1f} MB  {count} files")
    return written


def _ffmpeg(args: list[str]) -> bool:
    try:
        done = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args],
            capture_output=True, timeout=300, check=False,
        )
        return done.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def build_audio(root: Path, world: World, budget: int, log) -> int:
    if not shutil.which("ffmpeg"):
        log("  audio       skipped (ffmpeg not found)")
        return 0
    out = root / "media" / "recordings"
    out.mkdir(parents=True, exist_ok=True)
    written = count = 0
    while written < budget:
        count += 1
        # Kept short and mostly compressed: a few enormous WAVs would eat the
        # whole audio budget and leave the extractor barely exercised.
        as_wav = count % 6 == 0
        seconds = world.rng.randrange(15, 35) if as_wav else world.rng.randrange(40, 150)
        freq = world.rng.randrange(180, 900)
        path = out / f"standup-{count:03d}.{'wav' if as_wav else 'mp3'}"
        codec = ["-c:a", "pcm_s16le"] if as_wav else ["-c:a", "libmp3lame", "-b:a", "128k"]
        if not _ffmpeg([
            "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
            "-ar", "44100", "-ac", "2", *codec, str(path),
        ]):
            log("  audio       ffmpeg failed; stopping audio generation")
            break
        written += path.stat().st_size
    log(f"  audio    {written / MB:6.1f} MB  {count} files")
    return written


def build_video(root: Path, world: World, budget: int, log) -> int:
    if not shutil.which("ffmpeg"):
        log("  video       skipped (ffmpeg not found)")
        return 0
    out = root / "media" / "screencasts"
    out.mkdir(parents=True, exist_ok=True)
    written = count = 0
    while written < budget:
        count += 1
        seconds = world.rng.randrange(15, 45)
        path = out / f"screencast-{count:03d}.mp4"
        if not _ffmpeg([
            "-f", "lavfi", "-i", f"testsrc=size=1280x720:rate=24:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ]):
            log("  video       ffmpeg failed; stopping video generation")
            break
        written += path.stat().st_size
    log(f"  video    {written / MB:6.1f} MB  {count} files")
    return written


def build_edge_cases(root: Path, world: World, log) -> int:
    """Files chosen to break things, not to add bytes.

    Every one of these has a defined correct behaviour — a node with a warning,
    or a clean skip — and none of them may end a build.
    """
    out = root / "edge-cases"
    out.mkdir(parents=True, exist_ok=True)
    written = 0

    cases: dict[str, bytes] = {
        "truncated.pdf": b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer",
        "not-really-a.pdf": b"just text wearing a pdf extension\n",
        "empty.txt": b"",
        "empty.csv": b"",
        "headers-only.csv": b"id,name,email\n",
        "no-extension": b"# Notes\n\nInvoice INV-2024-0001 for $1,000.00 on 2024-06-01.\n",
        "binary.dat": bytes(range(256)) * 40,
        "not-really-an.jpg": b"definitely not a jpeg",
        "unicode-文書-ملف.md": (
            "# Unicode title éàü\n\nInvoice INV-2024-0002 for €5.000,00.\n"
        ).encode(),
        "one-huge-line.txt": (b"word " * 200_000) + b"\n",
        "many-blank-lines.md": b"# Title\n" + b"\n" * 5000 + b"Body after the void.\n",
        "control-chars.txt": b"before\x00\x0cafter\x07 INV-2024-0003\n",
        "sparse.csv": b"a,b,c\n1,,\n,,\n,,3\n",
        "mixed-types.csv": b"id,value\n1,10\n2,hello\n3,2024-01-01\n4,$5.00\n",
    }
    for name, blob in cases.items():
        path = out / name
        path.write_bytes(blob)
        written += len(blob)

    # A path deep enough to catch anything assuming shallow trees.
    deep = root / "archive"
    for level in range(12):
        deep = deep / f"level-{level:02d}"
    deep.mkdir(parents=True, exist_ok=True)
    buried = deep / "buried.md"
    buried.write_text(f"# Buried\n\n{world.paragraph(4)}\n")
    written += buried.stat().st_size

    # A file whose name collides with a directory name elsewhere in the tree.
    (root / "archive" / "media").mkdir(parents=True, exist_ok=True)
    twin = root / "archive" / "media" / "photos.md"
    twin.write_text(f"# Not the photos folder\n\n{world.paragraph(3)}\n")
    written += twin.stat().st_size

    log(f"  edge     {written / MB:6.1f} MB  {len(cases) + 2} files")
    return written


# -- driver ----------------------------------------------------------------

# Fractions of the total size budget. Video and audio dominate bytes; text and
# sheets dominate *nodes*, which is the more interesting axis.
MIX = {
    "video": 0.30,
    "audio": 0.18,
    "images": 0.20,
    "sheets": 0.14,
    "pdf": 0.10,
    "text": 0.05,
    "docx": 0.03,
}

BUILDERS = {
    "text": build_text,
    "pdf": build_pdfs,
    "sheets": build_sheets,
    "docx": build_docx,
    "images": build_images,
    "audio": build_audio,
    "video": build_video,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="./benchmark-corpus", help="where to write it")
    parser.add_argument("--size-mb", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true", help="overwrite an existing corpus")
    parser.add_argument("--only", action="append", choices=sorted(BUILDERS),
                        help="generate only these kinds (repeatable)")
    args = parser.parse_args()

    root = Path(args.out).expanduser().resolve()
    if root.exists():
        if not args.force:
            print(f"{root} exists — pass --force to regenerate", file=sys.stderr)
            return 1
        shutil.rmtree(root)
    root.mkdir(parents=True)

    world = World(random.Random(args.seed))
    total_budget = args.size_mb * MB
    started = time.time()
    print(f"Generating ~{args.size_mb:.0f} MB into {root}")

    written = 0
    for kind, share in MIX.items():
        if args.only and kind not in args.only:
            continue
        written += BUILDERS[kind](root, world, int(total_budget * share), print)
    written += build_edge_cases(root, world, print)

    files = sum(1 for p in root.rglob("*") if p.is_file())
    actual = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    (root / "CORPUS.json").write_text(
        json.dumps(
            {"seed": args.seed, "target_mb": args.size_mb,
             "actual_mb": round(actual / MB, 2), "files": files},
            indent=2,
        )
    )
    print(f"\n  {actual / MB:.1f} MB across {files} files in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
