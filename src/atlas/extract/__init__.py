"""Extractor registry: file → the thing that knows how to read it.

Dispatch is by extension, in registration order, with `TextExtractor` last
because it is the only one that claims extension-less files. Adding a modality
means adding a class and one line here — nothing else in Atlas needs to know
the list grew.
"""

from __future__ import annotations

from pathlib import Path

from ..graph import Modality
from .base import (
    ExtractContext,
    Extraction,
    ExtractionError,
    ExtractOptions,
    Extractor,
    MissingDependency,
    clean,
    have,
    require,
    text_chunks,
)
from .image import ImageExtractor
from .media import AudioExtractor, VideoExtractor
from .office import DocxExtractor, PptxExtractor
from .pdf import PdfExtractor
from .sheet import SheetExtractor
from .text import TextExtractor

REGISTRY: tuple[Extractor, ...] = (
    PdfExtractor(),
    SheetExtractor(),
    DocxExtractor(),
    PptxExtractor(),
    ImageExtractor(),
    AudioExtractor(),
    VideoExtractor(),
    TextExtractor(),  # last: claims files with no extension
)


def for_path(path: Path) -> Extractor | None:
    """The extractor that should read `path`, or None if nothing handles it."""
    for extractor in REGISTRY:
        if extractor.matches(path):
            return extractor
    return None


def modality_for(path: Path) -> Modality:
    extractor = for_path(path)
    return extractor.modality if extractor else Modality.UNKNOWN


# Optional dependencies, the modality each unlocks, and how to install it. The
# CLI prints this so a user can see what their graph is missing before they run
# a build over ten thousand files and wonder why the videos are empty.
CAPABILITIES: tuple[tuple[str, str, str, str], ...] = (
    ("pypdf", "pdf", "pdf", "PDF text and outlines"),
    ("openpyxl", "sheets", "sheet", "Excel workbooks"),
    ("PIL", "images", "image", "image dimensions and EXIF"),
    ("docx", "office", "text", "Word documents"),
    ("pptx", "office", "slides", "PowerPoint decks"),
    ("faster_whisper", "media", "audio/video", "speech transcription"),
    ("pytesseract", "ocr", "image", "text inside images (--ocr)"),
    ("anthropic", "llm", "all", "semantic entities and relationships (--enrich)"),
)


def capabilities() -> list[dict[str, object]]:
    """What this installation can currently read."""
    return [
        {
            "module": module,
            "extra": extra,
            "modality": modality,
            "description": description,
            "available": have(module),
        }
        for module, extra, modality, description in CAPABILITIES
    ]


def missing_extras() -> list[str]:
    """Extras that are not installed, deduplicated and ordered for display."""
    seen: list[str] = []
    for capability in capabilities():
        extra = str(capability["extra"])
        if not capability["available"] and extra not in seen:
            seen.append(extra)
    return seen


__all__ = [
    "CAPABILITIES",
    "REGISTRY",
    "AudioExtractor",
    "DocxExtractor",
    "ExtractContext",
    "ExtractOptions",
    "Extraction",
    "ExtractionError",
    "Extractor",
    "ImageExtractor",
    "MissingDependency",
    "PdfExtractor",
    "PptxExtractor",
    "SheetExtractor",
    "TextExtractor",
    "VideoExtractor",
    "capabilities",
    "clean",
    "for_path",
    "have",
    "missing_extras",
    "modality_for",
    "require",
    "text_chunks",
]
