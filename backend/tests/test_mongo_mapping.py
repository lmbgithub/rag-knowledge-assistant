"""The MongoDB mapping functions, with no server.

These are the part of the adapter that can be wrong silently — a dropped field
reads back as a default rather than as an error — so they are tested directly.
Whether pymongo can reach a server is not this suite's business.
"""

from __future__ import annotations

from rag_assistant.models import Chunk, Citation, Conversation, Document, Message, Progress
from rag_assistant.mongo_store import (
    DOCUMENT_ID,
    _chunk_from,
    _chunk_to,
    _conversation_from,
    _conversation_to,
    _document_from,
    _document_to,
)


def test_document_round_trips_through_mongo_shape():
    document = Document(
        id="d",
        filename="p.pdf",
        status="ready",
        pages=4,
        chunks=11,
        bytes=999,
        created_at=1.5,
        progress=Progress("ready", 11, 11),
    )
    assert _document_from(_document_to(document)) == document


def test_the_document_row_is_pinned_to_one_id():
    # "There is at most one document" is the core invariant; a fixed _id makes
    # inserting a second one impossible rather than merely unlikely.
    assert _document_to(Document(id="d", filename="p.pdf", status="ready"))["_id"] == DOCUMENT_ID


def test_a_document_row_missing_optional_fields_still_loads():
    raw = {"id": "d", "filename": "p.pdf", "status": "ready"}
    document = _document_from(raw)
    assert (document.pages, document.chunks, document.progress.stage) == (0, 0, "")


def test_conversation_round_trips_with_messages_and_citations():
    conversation = Conversation(
        id="c",
        title="t",
        document_id="d",
        document_filename="p.pdf",
        created_at=1.0,
        updated_at=2.0,
        messages=(
            Message(id="m1", role="user", content="q", created_at=1.0),
            Message(
                id="m2",
                role="assistant",
                content="a",
                citations=(Citation(1, 2, "passage", 0.5),),
                created_at=2.0,
            ),
        ),
    )
    assert _conversation_from(_conversation_to(conversation)) == conversation


def test_the_conversation_id_becomes_mongos_primary_key():
    raw = _conversation_to(
        Conversation(id="c", title="t", document_id=None, document_filename=None)
    )
    assert raw["_id"] == "c"
    assert "id" not in raw
    assert "message_count" not in raw


def test_a_conversation_whose_document_is_gone_still_loads():
    # The whole reason history outlives the document.
    raw = _conversation_to(
        Conversation(id="c", title="t", document_id="d", document_filename="deleted.pdf")
    )
    assert _conversation_from(raw).document_filename == "deleted.pdf"


def test_chunk_round_trips_through_mongo_shape():
    # It used to be hand-built here rather than going through Chunk.to_dict, so
    # a field added to Chunk reached Mongo for documents and conversations but
    # not for chunks.
    chunk = Chunk(id=7, text="a passage", page=3, ordinal=1)
    assert _chunk_from(_chunk_to(chunk)) == chunk


def test_the_chunk_id_is_mongos_primary_key():
    raw = _chunk_to(Chunk(id=7, text="t", page=1, ordinal=0))
    assert raw["_id"] == 7
    assert "id" not in raw


def test_a_summary_row_reports_its_count_without_carrying_the_messages():
    # The sidebar projection leaves `messages` out; the count must still be
    # right, or the list silently reads "0 messages" for every thread.
    summary = _conversation_from(
        {
            "_id": "c",
            "title": "t",
            "document_id": "d",
            "document_filename": "p.pdf",
            "created_at": 1.0,
            "updated_at": 2.0,
            "loaded_count": 4,
        }
    )
    assert summary.messages == ()
    assert summary.to_dict(with_messages=False)["message_count"] == 4


def test_a_fully_loaded_thread_counts_its_own_messages():
    conversation = Conversation(
        id="c",
        title="t",
        document_id=None,
        document_filename=None,
        messages=(Message(id="m", role="user", content="q"),),
    )
    assert conversation.to_dict(with_messages=False)["message_count"] == 1
