"""The MongoDB `Store`.

Three collections in one database:

  document       exactly one row, pinned to `_id="current"`
  chunks         `_id` is the FAISS row id — the only key shared between them
  conversations  one document per thread, messages embedded

Messages are embedded in the conversation rather than kept in a fourth
collection. A thread is read whole and written by appending to its end, which
is precisely what `$push` does in one round trip; splitting them out would buy
a join for every read and nothing else. The 16MB document ceiling is thousands
of messages, and is named in the README rather than pretended away.

`document` is a single pinned row and not a collection with one member,
because "there is at most one document" is the application's core invariant
and a fixed `_id` makes it impossible to violate by inserting a second.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .errors import BackendUnavailable
from .models import Chunk, Conversation, Document, Message, Progress

DOCUMENT_ID = "current"


class MongoStore:
    def __init__(
        self, uri: str, database: str = "rag_assistant", *, timeout_ms: int = 5000
    ) -> None:
        from pymongo import MongoClient

        # Constructing a client is lazy and never fails; the connection error
        # surfaces on the first operation, which without `connect()` below is
        # somewhere deep in a request. Failing here instead turns a stopped
        # database into one clear line rather than a pymongo traceback.
        self._client: Any = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        self._uri = uri
        self._db = self._client[database]
        self.documents = self._db["document"]
        self.chunks = self._db["chunks"]
        self.conversations = self._db["conversations"]

    def ensure_indexes(self) -> None:
        """Called at startup, not at import. Idempotent on MongoDB's side."""

        self.conversations.create_index("updated_at")

    def connect(self) -> None:
        """Fail now, with a message, rather than mid-request with a traceback."""

        from pymongo.errors import PyMongoError

        try:
            self._client.admin.command("ping")
        except PyMongoError as exc:
            raise BackendUnavailable(
                f"cannot reach MongoDB at {self._uri}: {type(exc).__name__}. "
                "Start it (docker compose up -d mongo), or run with OFFLINE=true."
            ) from exc

    def close(self) -> None:
        self._client.close()

    # --- document ---------------------------------------------------------
    def get_document(self) -> Document | None:
        raw = self.documents.find_one({"_id": DOCUMENT_ID})
        return None if raw is None else _document_from(raw)

    def put_document(self, document: Document) -> None:
        self.documents.replace_one({"_id": DOCUMENT_ID}, _document_to(document), upsert=True)

    def delete_document(self) -> None:
        self.documents.delete_one({"_id": DOCUMENT_ID})

    # --- chunks -----------------------------------------------------------
    def put_chunks(self, chunks: Iterable[Chunk]) -> None:
        docs = [_chunk_to(c) for c in chunks]
        if docs:
            self.chunks.insert_many(docs, ordered=False)

    def get_chunks(self, ids: Sequence[int]) -> list[Chunk]:
        if not ids:
            return []
        rows = self.chunks.find({"_id": {"$in": list(ids)}})
        found = {raw["_id"]: _chunk_from(raw) for raw in rows}
        # Preserve the caller's order: it is retrieval rank, and Mongo's is not.
        return [found[i] for i in ids if i in found]

    def count_chunks(self) -> int:
        return int(self.chunks.count_documents({}))

    def delete_chunks(self) -> None:
        self.chunks.delete_many({})

    # --- conversations ----------------------------------------------------
    def create_conversation(self, conversation: Conversation) -> Conversation:
        self.conversations.insert_one(_conversation_to(conversation))
        return conversation

    def list_conversations(self) -> list[Conversation]:
        """Summaries only: the count is computed server-side and the bodies stay put.

        `find()` with no projection pulled every message of every thread —
        including each citation's full passage text — and rebuilt them all as
        dataclasses, for a route that uses nothing but the length. The frontend
        refreshes this list after every answer, so the cost grew with both the
        number of threads and their length.
        """

        rows = self.conversations.aggregate(
            [
                {"$sort": {"updated_at": -1}},
                {
                    "$project": {
                        "title": 1,
                        "document_id": 1,
                        "document_filename": 1,
                        "created_at": 1,
                        "updated_at": 1,
                        "loaded_count": {"$size": {"$ifNull": ["$messages", []]}},
                    }
                },
            ]
        )
        return [_conversation_from(raw) for raw in rows]

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        raw = self.conversations.find_one({"_id": conversation_id})
        return None if raw is None else _conversation_from(raw)

    def delete_conversation(self, conversation_id: str) -> bool:
        return self.conversations.delete_one({"_id": conversation_id}).deleted_count > 0

    def append_message(self, conversation_id: str, message: Message) -> None:
        result = self.conversations.update_one(
            {"_id": conversation_id},
            {"$push": {"messages": message.to_dict()}, "$set": {"updated_at": message.created_at}},
        )
        if result.matched_count == 0:
            raise KeyError(conversation_id)


# --- mapping, kept in functions so it is testable without a server ---------


def _chunk_to(chunk: Chunk) -> dict[str, Any]:
    raw = chunk.to_dict()
    raw["_id"] = raw.pop("id")
    return raw


def _chunk_from(raw: dict[str, Any]) -> Chunk:
    return Chunk(
        id=int(raw["_id"]),
        text=raw["text"],
        page=int(raw["page"]),
        ordinal=int(raw["ordinal"]),
    )


def _document_to(document: Document) -> dict[str, Any]:
    raw = document.to_dict()
    raw["_id"] = DOCUMENT_ID
    return raw


def _document_from(raw: dict[str, Any]) -> Document:
    progress = raw.get("progress") or {}
    return Document(
        id=raw["id"],
        filename=raw["filename"],
        status=raw["status"],
        pages=int(raw.get("pages", 0)),
        chunks=int(raw.get("chunks", 0)),
        bytes=int(raw.get("bytes", 0)),
        created_at=float(raw.get("created_at", 0.0)),
        error=raw.get("error"),
        progress=Progress(
            stage=progress.get("stage", ""),
            done=int(progress.get("done", 0)),
            total=int(progress.get("total", 0)),
        ),
    )


def _conversation_to(conversation: Conversation) -> dict[str, Any]:
    raw = conversation.to_dict()
    raw["_id"] = raw.pop("id")
    raw.pop("message_count", None)
    return raw


def _conversation_from(raw: dict[str, Any]) -> Conversation:
    return Conversation(
        id=str(raw["_id"]),
        title=raw.get("title", ""),
        document_id=raw.get("document_id"),
        document_filename=raw.get("document_filename"),
        created_at=float(raw.get("created_at", 0.0)),
        updated_at=float(raw.get("updated_at", 0.0)),
        messages=tuple(Message.from_dict(m) for m in raw.get("messages", ())),
        # Present only on the summary projection, which leaves `messages` out.
        loaded_count=raw.get("loaded_count"),
    )
