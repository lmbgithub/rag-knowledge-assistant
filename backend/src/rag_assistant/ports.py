"""The four things the pipeline needs, as protocols.

`ingest`, `retrieve` and `chat` import only from here. Every adapter — FAISS,
MongoDB, Ollama — is behind one of these, which is why the pipeline tests run
with no database, no model and no sockets.

These are `Protocol`s and not base classes on purpose: the in-memory fakes in
`fakes.py` are not subclasses of anything, so a test double cannot inherit a
default implementation and quietly pass a test that the real adapter fails.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol, runtime_checkable

from .models import Chunk, Conversation, Document, Message

Vector = Sequence[float]


@runtime_checkable
class Embedder(Protocol):
    """Text to vectors. `dim` must be constant for the life of an index."""

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class ChatModel(Protocol):
    """A streaming chat completion.

    Streaming is the only mode. A non-streaming variant would be a second code
    path to keep correct for the sake of a call that, on a 1B model on CPU,
    looks identical to a hang.
    """

    def stream(self, system: str, user: str) -> Iterator[str]: ...


@runtime_checkable
class VectorIndex(Protocol):
    """Approximate nearest neighbours over the chunk vectors.

    `add` returns the ids it assigned, in the order given. Those ids are the
    store's primary key for the chunks, so the index is the id authority and
    there is no second numbering to reconcile.
    """

    @property
    def size(self) -> int: ...

    def add(self, vectors: Sequence[Vector]) -> list[int]: ...

    def search(self, vector: Vector, k: int) -> list[tuple[int, float]]: ...

    def reset(self) -> None: ...


@runtime_checkable
class Store(Protocol):
    """Durable state: the document record, the chunk text, the conversations."""

    # --- document ---------------------------------------------------------
    def get_document(self) -> Document | None: ...

    def put_document(self, document: Document) -> None: ...

    def delete_document(self) -> None: ...

    # --- chunks -----------------------------------------------------------
    def put_chunks(self, chunks: Iterable[Chunk]) -> None: ...

    def get_chunks(self, ids: Sequence[int]) -> list[Chunk]: ...

    def count_chunks(self) -> int: ...

    def delete_chunks(self) -> None: ...

    # --- conversations ----------------------------------------------------
    def create_conversation(self, conversation: Conversation) -> Conversation: ...

    def list_conversations(self) -> list[Conversation]: ...

    def get_conversation(self, conversation_id: str) -> Conversation | None: ...

    def delete_conversation(self, conversation_id: str) -> bool: ...

    def append_message(self, conversation_id: str, message: Message) -> None: ...
