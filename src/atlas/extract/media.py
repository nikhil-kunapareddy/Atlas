"""Audio and video.

Time is to a recording what the page is to a PDF: the anchor a citation needs.
So a transcript is not stored as one wall of text — it is cut into SEGMENT nodes
carrying `start_time`/`end_time`, and an answer can say *"at 14:32 in
standup.m4a"*. Video adds sampled keyframes, so an agent has something to look
at as well as read.

Everything here degrades rather than fails. No ffmpeg means no duration and no
keyframes, but the file still enters the graph. No faster-whisper means no
transcript, and the node carries its metadata and a warning explaining what to
install. A folder of videos always produces a graph.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..graph import Edge, EdgeKind, Modality, Node, NodeKind, child_uid
from .base import (
    ExtractContext,
    Extraction,
    Extractor,
    clean,
    have,
    require,
    text_chunks,
)

AUDIO_EXTENSIONS = frozenset(
    [".mp3", ".wav", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".wma", ".opus", ".aiff", ".aif", ".amr"]
)
VIDEO_EXTENSIONS = frozenset(
    [".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".m4v", ".mpg", ".mpeg", ".3gp", ".ogv"]
)

# A transcript window long enough to be a coherent thought, short enough to
# point at precisely.
WINDOW_SECONDS = 45.0
WINDOW_CHARS = 900
PREVIEW_CHARS = 300

# One model instance per size, reused across every file in a build. Loading
# Whisper takes seconds and allocates hundreds of megabytes; doing it per file
# would dominate the runtime of any real folder.
_WHISPER_CACHE: dict[str, Any] = {}


class MediaExtractor(Extractor):
    """Shared implementation for both time-based modalities."""

    name = "media"
    version = "1"
    extensions = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

    def extract(self, ctx: ExtractContext) -> Extraction:
        is_video = ctx.suffix in VIDEO_EXTENSIONS
        result = Extraction(
            modality=Modality.VIDEO if is_video else Modality.AUDIO,
            title=ctx.path.stem,
        )

        probe = ctx.cached(f"{self.name}-probe", self.version, lambda: _probe(ctx.path))
        result.props.update(probe.get("props", {}))
        for warning in probe.get("warnings", []):
            result.warn(warning)

        if not ctx.options.transcribe:
            result.text = _caption(result.props, ctx.rel_path, is_video)
            return result

        if not have("faster_whisper"):
            result.warn(
                "no transcript — install the media extra: "
                "pip install 'atlas-context[media]'"
            )
            result.text = _caption(result.props, ctx.rel_path, is_video)
            return result

        model_size = ctx.options.whisper_model
        transcript = ctx.cached(
            f"{self.name}-transcript-{model_size}",
            self.version,
            lambda: _transcribe(ctx, model_size),
        )
        if transcript.get("error"):
            result.warn(transcript["error"])
            result.text = _caption(result.props, ctx.rel_path, is_video)
            return result

        if transcript.get("language"):
            result.props["language"] = transcript["language"]

        segments = list(_windows(transcript.get("segments", [])))
        if not segments:
            result.warn("no speech detected")
            result.text = _caption(result.props, ctx.rel_path, is_video)
            return result

        previous_uid: str | None = None
        ordinal = 0
        full: list[str] = []

        for index, window in enumerate(segments):
            text = clean(window["text"])
            if not text:
                continue
            full.append(text)
            segment_uid = child_uid(ctx.doc_uid, NodeKind.SEGMENT, index)
            stamp = _timestamp(window["start"])
            result.add(
                Node(
                    uid=segment_uid,
                    kind=NodeKind.SEGMENT,
                    name=f"{ctx.path.name} @ {stamp}",
                    body=text[:PREVIEW_CHARS],
                    props={
                        "path": ctx.rel_path,
                        "start_time": round(window["start"], 2),
                        "end_time": round(window["end"], 2),
                        "timestamp": stamp,
                        "ord": index,
                    },
                ),
                contained_by=ctx.doc_uid,
            )
            if previous_uid:
                result.edges.append(Edge(previous_uid, segment_uid, EdgeKind.NEXT))
            previous_uid = segment_uid

            nodes, edges, ordinal = text_chunks(
                segment_uid,
                text,
                ctx.rel_path,
                extra_props={
                    "path": ctx.rel_path,
                    "start_time": round(window["start"], 2),
                    "end_time": round(window["end"], 2),
                },
                start_ord=ordinal,
            )
            result.nodes.extend(nodes)
            result.edges.extend(edges)

        result.text = " ".join(full)
        result.props["transcript_chars"] = len(result.text)

        if is_video and ctx.options.keyframes > 0:
            self._add_keyframes(ctx, result, segments)
        return result

    def _add_keyframes(
        self, ctx: ExtractContext, result: Extraction, segments: list[dict[str, Any]]
    ) -> None:
        """Sample frames and attach each to the segment it falls inside.

        Attaching to segments rather than to the video is what makes the frames
        useful: a frame is evidence for what was said at that moment, so a
        question answered from a transcript window can show the matching image.
        """
        duration = float(result.props.get("duration") or 0)
        if duration <= 0 or not shutil.which("ffmpeg"):
            return
        media_dir = ctx.options.media_dir
        if media_dir is None:
            return

        target = Path(media_dir) / ctx.digest[:16]
        count = ctx.options.keyframes
        times = [duration * (i + 0.5) / count for i in range(count)]
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            result.warn(f"cannot write keyframes: {exc}")
            return

        for index, at in enumerate(times):
            out = target / f"frame-{index:02d}.jpg"
            if not out.exists() and not _grab_frame(ctx.path, at, out):
                continue
            frame_uid = child_uid(ctx.doc_uid, NodeKind.FRAME, index)
            stamp = _timestamp(at)
            result.nodes.append(
                Node(
                    uid=frame_uid,
                    kind=NodeKind.FRAME,
                    name=f"{ctx.path.name} frame @ {stamp}",
                    body=f"video frame from {ctx.rel_path} at {stamp}",
                    props={
                        "path": ctx.rel_path,
                        "time": round(at, 2),
                        "timestamp": stamp,
                        "image": str(out),
                    },
                )
            )
            result.edges.append(Edge(ctx.doc_uid, frame_uid, EdgeKind.CONTAINS))
            owner = _segment_at(segments, at)
            if owner is not None:
                result.edges.append(
                    Edge(
                        child_uid(ctx.doc_uid, NodeKind.SEGMENT, owner),
                        frame_uid,
                        EdgeKind.HAS_FRAME,
                    )
                )


class AudioExtractor(MediaExtractor):
    name = "audio"
    modality = Modality.AUDIO
    extensions = AUDIO_EXTENSIONS


class VideoExtractor(MediaExtractor):
    name = "video"
    modality = Modality.VIDEO
    extensions = VIDEO_EXTENSIONS


# -- ffprobe ---------------------------------------------------------------


def _probe(path: Path) -> dict[str, Any]:
    if not shutil.which("ffprobe"):
        return {
            "props": {},
            "warnings": ["ffprobe not found — install ffmpeg for duration and codec metadata"],
        }
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(path),
            ],
            capture_output=True,
            timeout=60,
            check=False,
        )
        data = json.loads(completed.stdout or b"{}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {"props": {}, "warnings": [f"ffprobe failed: {exc}"]}

    fmt = data.get("format", {})
    props: dict[str, Any] = {}
    if fmt.get("duration"):
        try:
            props["duration"] = round(float(fmt["duration"]), 2)
            props["duration_human"] = _timestamp(props["duration"])
        except (TypeError, ValueError):
            pass
    if fmt.get("bit_rate"):
        props["bitrate"] = fmt["bit_rate"]
    for key in ("title", "artist", "album", "date", "comment"):
        value = (fmt.get("tags") or {}).get(key)
        if value:
            props[key] = str(value)[:200]

    for stream in data.get("streams", []):
        kind = stream.get("codec_type")
        if kind == "video" and "video_codec" not in props:
            props["video_codec"] = stream.get("codec_name")
            props["width"] = stream.get("width")
            props["height"] = stream.get("height")
        elif kind == "audio" and "audio_codec" not in props:
            props["audio_codec"] = stream.get("codec_name")
            props["sample_rate"] = stream.get("sample_rate")
            props["channels"] = stream.get("channels")
    return {"props": props, "warnings": []}


def _grab_frame(source: Path, at: float, out: Path) -> bool:
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{at:.2f}",
                "-i", str(source), "-frames:v", "1", "-q:v", "4", "-y", str(out),
            ],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and out.exists()


# -- transcription ---------------------------------------------------------


def _transcribe(ctx: ExtractContext, model_size: str) -> dict[str, Any]:
    """Run faster-whisper, returning a JSON-serialisable transcript."""
    try:
        model = _whisper(model_size)
    except Exception as exc:
        return {"error": f"could not load whisper model {model_size!r}: {exc}"}

    try:
        segments, info = model.transcribe(
            str(ctx.path),
            language=ctx.options.language,
            # VAD keeps Whisper from hallucinating dialogue over silence, which
            # it does readily on long recordings with quiet stretches.
            vad_filter=True,
            beam_size=1,
        )
        rows = [
            {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
            for s in segments
            if s.text and s.text.strip()
        ]
    except Exception as exc:
        return {"error": f"transcription failed: {exc}"}

    return {"segments": rows, "language": getattr(info, "language", None)}


def _whisper(model_size: str) -> Any:
    if model_size not in _WHISPER_CACHE:
        module = require("faster_whisper", "media")
        # int8 on CPU is the only combination that is fast enough to be usable
        # on a laptop without a GPU, which is the assumed environment.
        _WHISPER_CACHE[model_size] = module.WhisperModel(
            model_size, device="cpu", compute_type="int8"
        )
    return _WHISPER_CACHE[model_size]


def _windows(segments: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Merge Whisper's short segments into citable windows."""
    buffer: list[str] = []
    start = end = 0.0
    chars = 0

    for segment in segments:
        if not buffer:
            start = segment["start"]
        buffer.append(segment["text"])
        chars += len(segment["text"])
        end = segment["end"]
        if end - start >= WINDOW_SECONDS or chars >= WINDOW_CHARS:
            yield {"start": start, "end": end, "text": " ".join(buffer)}
            buffer, chars = [], 0

    if buffer:
        yield {"start": start, "end": end, "text": " ".join(buffer)}


def _segment_at(segments: list[dict[str, Any]], at: float) -> int | None:
    for index, window in enumerate(segments):
        if window["start"] <= at <= window["end"]:
            return index
    return None


def _timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _caption(props: dict[str, Any], rel_path: str, is_video: bool) -> str:
    bits = [rel_path.replace("/", " ").replace("_", " ").replace("-", " ")]
    bits.append("video recording" if is_video else "audio recording")
    if props.get("duration_human"):
        bits.append(f"{props['duration_human']} long")
    for key in ("title", "artist", "album", "video_codec", "audio_codec"):
        if props.get(key):
            bits.append(str(props[key]))
    return " · ".join(bits)
