"""The graph vocabulary: what a node can be and what an edge can mean.

Atlas models a folder as a property graph. Two rules make the rest of the
system work, and both are enforced here rather than at the call sites:

**Node identity is derived from what a thing *is*, never from when it was
written.** A node's `uid` is a deterministic string (`doc:reports/q3.pdf`,
`doc:reports/q3.pdf#page:4`), so re-running a build upserts in place instead of
appending a second copy. Incremental rebuilds fall out of this for free.

**Every node and edge records who claimed it.** A `CONTAINS` edge from a folder
to a file is a fact about the filesystem; a `RELATED_TO` edge between two people
is a language model's opinion. Both belong in the graph, but an agent answering
a question needs to tell them apart, so `Provenance` travels with the edge.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeKind(str, Enum):
    """What a node represents.

    The first group is structural — extracted deterministically from bytes on
    disk, and identical on every machine. The second group is semantic: it only
    appears once something has interpreted the content.
    """

    # Structural
    FOLDER = "folder"
    DOCUMENT = "document"
    PAGE = "page"          # one page of a PDF or slide deck
    SHEET = "sheet"        # one worksheet of a workbook
    COLUMN = "column"      # one column of a sheet or CSV
    ROW = "row"            # a named record, only when a sheet has a key column
    CHUNK = "chunk"        # a span of text, line- or offset-anchored
    SEGMENT = "segment"    # a time range of audio or video
    FRAME = "frame"        # a keyframe sampled from a video

    # Semantic
    ENTITY = "entity"      # a person, org, place, concept, or literal
    TOPIC = "topic"        # a cluster label over entities
    FACT = "fact"          # hand- or agent-written knowledge

    def __str__(self) -> str:  # so f-strings and SQL params carry the value
        return self.value


class EdgeKind(str, Enum):
    """How two nodes relate.

    Kept deliberately small. Graphify carries ~40 edge types because code has
    that many genuinely distinct relationships (`calls`, `inherits`, `mixes_in`).
    Documents do not: almost everything is containment, ordering, derivation, or
    reference. A large vocabulary here would be mostly aspirational, and an
    agent has to reason over whatever we invent — so each kind below earns its
    place by supporting a query the others cannot answer.
    """

    CONTAINS = "contains"          # folder→file, file→page, sheet→column
    NEXT = "next"                  # page→page, chunk→chunk, segment→segment
    DERIVED_FROM = "derived_from"  # chunk→page, transcript chunk→segment
    MENTIONS = "mentions"          # chunk→entity
    REFERENCES = "references"      # document→document (a link or a citation)
    RELATED_TO = "related_to"      # entity→entity, typed by `props["label"]`
    HAS_FRAME = "has_frame"        # segment→frame
    SIMILAR_TO = "similar_to"      # reserved for the embedding pass

    def __str__(self) -> str:
        return self.value


class Provenance(str, Enum):
    """How much to trust an edge.

    Mirrors graphify's EXTRACTED / INFERRED / AMBIGUOUS split, with the LLM case
    called out explicitly because it is the one an agent should hedge about.
    """

    EXTRACTED = "extracted"  # read directly off the bytes; deterministic
    INFERRED = "inferred"    # resolved by a rule (a filename reference matched)
    LLM = "llm"              # a model proposed it
    ASSERTED = "asserted"    # a human wrote it down

    def __str__(self) -> str:
        return self.value


class Modality(str, Enum):
    """The broad content class of a document, chosen by the extractor registry."""

    TEXT = "text"
    PDF = "pdf"
    SHEET = "sheet"
    SLIDES = "slides"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    ARCHIVE = "archive"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


class EntityType(str, Enum):
    """What kind of thing an entity node is.

    The `literal` group (email through identifier) is extracted with regexes and
    is therefore exact. The `semantic` group needs a model and is only populated
    when enrichment runs.
    """

    # Literal — deterministic, no model required
    EMAIL = "email"
    URL = "url"
    PATH = "path"
    DATE = "date"
    MONEY = "money"
    PHONE = "phone"
    IDENTIFIER = "identifier"

    # Semantic — requires enrichment
    PERSON = "person"
    ORG = "org"
    PLACE = "place"
    PRODUCT = "product"
    EVENT = "event"
    CONCEPT = "concept"

    def __str__(self) -> str:
        return self.value


LITERAL_ENTITY_TYPES = frozenset(
    {
        EntityType.EMAIL,
        EntityType.URL,
        EntityType.PATH,
        EntityType.DATE,
        EntityType.MONEY,
        EntityType.PHONE,
        EntityType.IDENTIFIER,
    }
)


@dataclass
class Node:
    """One vertex.

    `body` is what search reads; `name` is what a human sees in a result. They
    differ on purpose — a page node is named `q3.pdf p.4` but its body is the
    full page text.
    """

    uid: str
    kind: NodeKind
    name: str
    body: str = ""
    props: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.uid:
            raise ValueError("node uid must be non-empty")
        self.kind = NodeKind(self.kind)


@dataclass
class Edge:
    """One directed, typed relationship between two `uid`s.

    Edges are addressed by uid rather than row id so an extractor can emit an
    edge to a node another extractor will create, in either order. The store
    resolves uids to ids at write time.
    """

    src: str
    dst: str
    kind: EdgeKind
    weight: float = 1.0
    provenance: Provenance = Provenance.EXTRACTED
    props: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.kind = EdgeKind(self.kind)
        self.provenance = Provenance(self.provenance)
        if self.src == self.dst:
            raise ValueError(f"self-loop on {self.src} ({self.kind})")


# -- uid construction ------------------------------------------------------
#
# Every uid is `<prefix>:<path>` with optional `#<kind>:<key>` suffixes for
# things that live inside a file. Paths are always relative to the indexed root
# and always use forward slashes, so a graph built on macOS and one built on
# Windows agree — which matters the moment anyone commits a graph or shares one.


def folder_uid(rel_path: str) -> str:
    return f"dir:{_norm_path(rel_path) or '.'}"


def document_uid(rel_path: str) -> str:
    return f"doc:{_norm_path(rel_path)}"


def child_uid(parent: str, kind: NodeKind, key: str | int) -> str:
    """A uid for something contained in `parent` — a page, sheet, chunk, segment."""
    return f"{parent}#{kind}:{key}"


def entity_uid(etype: EntityType, name: str) -> str:
    """A uid for an entity, shared across every file that mentions it.

    Normalisation is what makes `Acme Corp.`, `ACME CORP`, and `Acme  Corp`
    collapse to one node — which is the whole point of the entity layer, since
    the interesting query is "what else mentions this?".
    """
    return f"ent:{etype}:{normalize_entity(name)}"


_WS = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"^[\s\"'(\[]+|[\s\"'),.\];:!?]+$")


def normalize_entity(name: str) -> str:
    """Casefold, strip surrounding punctuation, collapse whitespace, drop accents.

    Deliberately conservative: it does not stem, singularise, or strip corporate
    suffixes. Over-normalising silently merges distinct entities, and a merge is
    much harder to notice — and to undo — than a duplicate.
    """
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _TRAILING_PUNCT.sub("", text)
    text = _WS.sub(" ", text).strip().casefold()
    return text


def _norm_path(rel_path: str) -> str:
    return str(rel_path).replace("\\", "/").strip("/")
