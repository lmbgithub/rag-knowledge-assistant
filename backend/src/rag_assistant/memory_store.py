"""An in-memory `Store`. Used by the tests, the examples and `OFFLINE=true`.

It is written to the same protocol as the MongoDB one and returns the same
immutable dataclasses, so a test that passes here is a test of the pipeline
rather than of dictionary juggling. It is deliberately *not* a subclass of
anything the Mongo store also inherits: shared code between a fake and the
real thing is shared bugs, and a fake that cannot fail hides them.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence

from .models import Chunk, Conversation, Document, Message


class MemoryStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._document: Document | None = None
        self._chunks: dict[int, Chunk] = {}
        self._conversations: dict[str, Conversation] = {}

    # --- document ---------------------------------------------------------
    def get_document(self) -> Document | None:
        with self._lock:
            return self._document

    def put_document(self, document: Document) -> None:
        with self._lock:
            self._document = document

    def delete_document(self) -> None:
        with self._lock:
            self._document = None

    # --- chunks -----------------------------------------------------------
    def put_chunks(self, chunks: Iterable[Chunk]) -> None:
        with self._lock:
            for chunk in chunks:
                self._chunks[chunk.id] = chunk

    def get_chunks(self, ids: Sequence[int]) -> list[Chunk]:
        # Missing ids are skipped, not faked. A retrieved id with no chunk
        # behind it means index and store disagree, and inventing a blank
        # passage would feed the model an empty citation.
        with self._lock:
            return [self._chunks[i] for i in ids if i in self._chunks]

    def count_chunks(self) -> int:
        with self._lock:
            return len(self._chunks)

    def delete_chunks(self) -> None:
        with self._lock:
            self._chunks = {}

    # --- conversations ----------------------------------------------------
    def create_conversation(self, conversation: Conversation) -> Conversation:
        with self._lock:
            self._conversations[conversation.id] = conversation
            return conversation

    def list_conversations(self) -> list[Conversation]:
        with self._lock:
            return sorted(self._conversations.values(), key=lambda c: c.updated_at, reverse=True)

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._lock:
            return self._conversations.get(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._lock:
            return self._conversations.pop(conversation_id, None) is not None

    def append_message(self, conversation_id: str, message: Message) -> None:
        from dataclasses import replace

        with self._lock:
            existing = self._conversations.get(conversation_id)
            if existing is None:
                raise KeyError(conversation_id)
            self._conversations[conversation_id] = replace(
                existing,
                messages=(*existing.messages, message),
                updated_at=message.created_at,
            )
