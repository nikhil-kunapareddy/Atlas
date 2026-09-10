"""Images.

An image has no text to index, so without a model it contributes structure and
metadata only — dimensions, format, camera, capture time, GPS. That is less than
a PDF gives, but it is not nothing: "photos taken in Lisbon in March" is a real
question answerable from EXIF alone, with no inference and no cost.

Two optional layers add content. OCR (`--ocr`) reads text out of screenshots,
scans, and diagrams — which is the common case in a work folder. Vision
enrichment (`--vision`, in enrich.py) describes what is actually depicted.
Neither is required for the file to land in the graph.
"""

from __future__ import annotations

from typing import Any

from ..graph import Modality
from .base import ExtractContext, Extraction, Extractor, clean, require, text_chunks

IMAGE_EXTENSIONS = frozenset(
    [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif", ".avif", ".ico", ".ppm", ".pgm"]
)

# EXIF tags worth keeping. The full table is ~250 entries of lens minutiae that
# would bloat every node's props without answering any plausible question.
EXIF_TAGS = {
    271: "camera_make",
    272: "camera_model",
    274: "orientation",
    306: "captured_at",
    36867: "captured_at",
    37377: "shutter_speed",
    33434: "exposure_time",
    33437: "f_number",
    34855: "iso",
    42036: "lens",
    270: "description",
    315: "artist",
}


class ImageExtractor(Extractor):
    name = "image"
    version = "1"
    modality = Modality.IMAGE
    extensions = IMAGE_EXTENSIONS

    def extract(self, ctx: ExtractContext) -> Extraction:
        pil = require("PIL.Image", "images")
        result = Extraction(modality=Modality.IMAGE, title=ctx.path.stem)

        try:
            with pil.open(str(ctx.path)) as image:
                result.props.update(
                    {
                        "width": image.width,
                        "height": image.height,
                        "format": image.format,
                        "mode": image.mode,
                        "megapixels": round(image.width * image.height / 1e6, 2),
                    }
                )
                result.props.update(_exif(image))
        except Exception as exc:
            result.warn(f"unreadable image: {exc}")
            return result

        if ctx.options.ocr:
            text = ctx.cached(f"{self.name}-ocr", self.version, lambda: _ocr(ctx))
            if text:
                result.text = text
                result.props["ocr_chars"] = len(text)
                nodes, edges, _ = text_chunks(
                    ctx.doc_uid, text, ctx.rel_path, extra_props={"source": "ocr"}
                )
                result.nodes.extend(nodes)
                result.edges.extend(edges)

        # Even with no text, the node needs something searchable, or the image
        # is invisible to every query. The caption below is what makes
        # "screenshots from my phone" work with no model in the loop.
        if not result.text:
            result.text = _caption(result.props, ctx.rel_path)
        return result


def _exif(image: Any) -> dict[str, Any]:
    props: dict[str, Any] = {}
    try:
        exif = image.getexif()
    except Exception:
        return props
    if not exif:
        return props

    for tag, key in EXIF_TAGS.items():
        value = exif.get(tag)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text and key not in props:
            props[key] = text[:200]

    gps = _gps(exif)
    if gps:
        props["latitude"], props["longitude"] = gps
    return props


def _gps(exif: Any) -> tuple[float, float] | None:
    """Decode EXIF GPS into signed decimal degrees."""
    try:
        gps = exif.get_ifd(0x8825)
    except Exception:
        return None
    if not gps:
        return None
    try:
        lat = _dms(gps[2])
        lon = _dms(gps[4])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if gps.get(1) == "S":
        lat = -lat
    if gps.get(3) == "W":
        lon = -lon
    return round(lat, 6), round(lon, 6)


def _dms(values: Any) -> float:
    degrees, minutes, seconds = (float(v) for v in values)
    return degrees + minutes / 60 + seconds / 3600


def _ocr(ctx: ExtractContext) -> str:
    """Read text out of an image, returning "" when OCR is unavailable.

    OCR failure is not extraction failure: the image still belongs in the graph
    with its metadata, so this swallows errors rather than raising.
    """
    try:
        pytesseract = require("pytesseract", "ocr")
        pil = require("PIL.Image", "images")
        with pil.open(str(ctx.path)) as image:
            return clean(pytesseract.image_to_string(image), limit=200_000)
    except Exception:
        return ""


def _caption(props: dict[str, Any], rel_path: str) -> str:
    """A deterministic, searchable sentence describing the file."""
    bits = [rel_path.replace("/", " ").replace("_", " ").replace("-", " ")]
    if props.get("format"):
        bits.append(f"{props['format']} image")
    if props.get("width"):
        bits.append(f"{props['width']}x{props['height']} pixels")
    for key in ("camera_make", "camera_model", "captured_at", "description", "artist"):
        if props.get(key):
            bits.append(str(props[key]))
    if "latitude" in props:
        bits.append(f"located at {props['latitude']}, {props['longitude']}")
    return " · ".join(bits)
