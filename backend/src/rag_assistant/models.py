"""The data that crosses module boundaries.

Everything here is a frozen dataclass with a `to_dict`. The API serialises by
calling that, so the wire format is defined in one place and a field added to
a dataclass cannot silently fail to reach the browser.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Literal

DocStatus = Literal["empty", "indexing", "ready", "failed", "deleting"]
"""The whole application state machine.

`deleting` exists because deletion touches three stores and can fail halfway.
Without it, a failed delete leaves a document that looks `ready` while its
vectors are already gone — retrieval then returns ids that resolve to nothing.
"""

Role = Literal["user", "assistant"]


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> float:
    return time.time()


@dataclass(frozen=True)
class Page:
    """One page of extracted text. `number` is 1-based, as a reader counts."""

    number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A passage, its page, and its position in that page.

    `id` is the FAISS row. It is assigned by the store on insert and is the
    only key shared between the vector index and MongoDB, so there is exactly
    one mapping and nothing to keep in sync.
    """

    id: int
    text: str
    page: int
    ordinal: int

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "page": self.page, "ordinal": self.ordinal}


@dataclass(frozen=True)
class Retrieved:
    """A chunk with the similarity that retrieved it."""

    chunk: Chunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {**self.chunk.to_dict(), "score": round(self.score, 4)}


@dataclass(frozen=True)
class Progress:
    """What indexing is doing right now.

    A spinner cannot distinguish "embedding chunk 3 of 400" from "hung", and a
    1B model on CPU makes that distinction take minutes.
    """

    stage: str = ""
    done: int = 0
    total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "done": self.done, "total": self.total}


@dataclass(frozen=True)
class Document:
    """The single indexed PDF. There is at most one at a time, by design."""

    id: str
    filename: str
    status: DocStatus
    pages: int = 0
    chunks: int = 0
    bytes: int = 0
    created_at: float = field(default_factory=now)
    error: str | None = None
    progress: Progress = field(default_factory=Progress)

    def with_progress(self, stage: str, done: int = 0, total: int = 0) -> Document:
        return replace(self, progress=Progress(stage, done, total))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "pages": self.pages,
            "chunks": self.chunks,
            "bytes": self.bytes,
            "created_at": self.created_at,
            "error": self.error,
            "progress": self.progress.to_dict(),
        }


EMPTY_DOCUMENT_STATE: dict[str, Any] = {
    "id": None,
    "filename": None,
    "status": "empty",
    "pages": 0,
    "chunks": 0,
    "bytes": 0,
    "created_at": None,
    "error": None,
    "progress": Progress().to_dict(),
}
"""What `/document` returns when there is nothing indexed.

Deliberately the same shape as a real document rather than `{}` or a 404, so
the frontend has one branch on `status` instead of two on whether the body
parsed.
"""


@dataclass(frozen=True)
class Citation:
    """The passage an answer was built from, kept with the answer forever.

    Citations are stored on the message, not recomputed at read time: the
    index they came from may have been deleted, and an old answer showing
    passages from a *different* document would be worse than showing none.
    """

    chunk_id: int
    page: int
    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "page": self.page,
            "text": self.text,
            "score": round(self.score, 4),
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> Citation:
        return Citation(
            chunk_id=int(raw["chunk_id"]),
            page=int(raw["page"]),
            text=str(raw["text"]),
            score=float(raw["score"]),
        )

    @staticmethod
    def from_retrieved(hit: Retrieved) -> Citation:
        return Citation(
            chunk_id=hit.chunk.id, page=hit.chunk.page, text=hit.chunk.text, score=hit.score
        )


@dataclass(frozen=True)
class Message:
    id: str
    role: Role
    content: str
    citations: tuple[Citation, ...] = ()
    partial: bool = False
    created_at: float = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "citations": [c.to_dict() for c in self.citations],
            "partial": self.partial,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> Message:
        return Message(
            id=str(raw["id"]),
            role=raw["role"],
            content=str(raw["content"]),
            citations=tuple(Citation.from_dict(c) for c in raw.get("citations", ())),
            partial=bool(raw.get("partial", False)),
            created_at=float(raw.get("created_at", 0.0)),
        )


@dataclass(frozen=True)
class Conversation:
    """A thread, plus the document it was about.

    `document_filename` is copied in rather than joined, because the whole
    point of keeping conversations past a delete is that they stay readable
    once the document row is gone.
    """

    id: str
    title: str
    document_id: str | None
    document_filename: str | None
    created_at: float = field(default_factory=now)
    updated_at: float = field(default_factory=now)
    messages: tuple[Message, ...] = ()
    loaded_count: int | None = None
    """Set when the thread was loaded as a summary, with `messages` left empty.

    The sidebar needs a count and nothing else, and a message carries its
    citations, which carry whole passages of the PDF. Loading every thread in
    full to call `len()` on it shipped the document back several times over on
    a request that runs after every answer.
    """

    def to_dict(self, *, with_messages: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "document_id": self.document_id,
            "document_filename": self.document_filename,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": (
                self.loaded_count if self.loaded_count is not None else len(self.messages)
            ),
        }
        if with_messages:
            out["messages"] = [m.to_dict() for m in self.messages]
        return out


def title_from(question: str, *, limit: int = 60) -> str:
    """A thread title taken from its first question.

    The model is not asked to name the thread: that is one model call per new
    conversation, on the latency path, for a string the user can already read
    off their own first message.
    """

    collapsed = " ".join(question.split())
    if not collapsed:
        return "New conversation"
    if len(collapsed) <= limit:
        return collapsed
    cut = collapsed[:limit].rsplit(" ", 1)[0]
    return (cut or collapsed[:limit]).rstrip() + "…"
