from __future__ import annotations

import pytest

from rag_assistant.memory_store import MemoryStore
from rag_assistant.models import Chunk, Conversation, Document, Message


def conversation(cid="c1", **kwargs):
    return Conversation(
        id=cid,
        title=kwargs.get("title", "t"),
        document_id=kwargs.get("document_id", "d1"),
        document_filename=kwargs.get("document_filename", "p.pdf"),
        created_at=kwargs.get("created_at", 1.0),
        updated_at=kwargs.get("updated_at", 1.0),
    )


def test_document_starts_absent(store: MemoryStore):
    assert store.get_document() is None


def test_put_then_get_document(store: MemoryStore):
    document = Document(id="d", filename="p.pdf", status="ready")
    store.put_document(document)
    assert store.get_document() == document


def test_delete_document_is_idempotent(store: MemoryStore):
    store.delete_document()
    store.put_document(Document(id="d", filename="p.pdf", status="ready"))
    store.delete_document()
    store.delete_document()
    assert store.get_document() is None


def test_chunks_are_returned_in_the_order_asked_for(store: MemoryStore):
    store.put_chunks([Chunk(i, f"text {i}", 1, i) for i in range(3)])
    assert [c.id for c in store.get_chunks([2, 0])] == [2, 0]


def test_a_missing_chunk_id_is_skipped_not_faked(store: MemoryStore):
    # Index and store disagreeing must not become a blank passage in a prompt.
    store.put_chunks([Chunk(0, "text", 1, 0)])
    assert [c.id for c in store.get_chunks([0, 99])] == [0]


def test_get_chunks_of_nothing(store: MemoryStore):
    assert store.get_chunks([]) == []


def test_delete_chunks_empties_the_collection(store: MemoryStore):
    store.put_chunks([Chunk(0, "t", 1, 0)])
    store.delete_chunks()
    assert store.count_chunks() == 0


def test_conversations_are_listed_newest_first(store: MemoryStore):
    store.create_conversation(conversation("old", updated_at=1.0))
    store.create_conversation(conversation("new", updated_at=2.0))
    assert [c.id for c in store.list_conversations()] == ["new", "old"]


def test_appending_a_message_moves_the_conversation_to_the_top(store: MemoryStore):
    store.create_conversation(conversation("a", updated_at=1.0))
    store.create_conversation(conversation("b", updated_at=2.0))
    store.append_message("a", Message(id="m", role="user", content="q", created_at=9.0))
    assert [c.id for c in store.list_conversations()] == ["a", "b"]


def test_messages_keep_their_order(store: MemoryStore):
    store.create_conversation(conversation())
    for i in range(3):
        store.append_message("c1", Message(id=str(i), role="user", content=str(i), created_at=i))
    assert [m.id for m in store.get_conversation("c1").messages] == ["0", "1", "2"]


def test_appending_to_a_missing_conversation_raises(store: MemoryStore):
    with pytest.raises(KeyError):
        store.append_message("nope", Message(id="m", role="user", content="q"))


def test_delete_conversation_reports_whether_it_existed(store: MemoryStore):
    store.create_conversation(conversation())
    assert store.delete_conversation("c1") is True
    assert store.delete_conversation("c1") is False


def test_deleting_a_document_leaves_conversations_alone(store: MemoryStore):
    # Replacing a PDF must not silently destroy the user's chat history.
    store.put_document(Document(id="d", filename="p.pdf", status="ready"))
    store.create_conversation(conversation())
    store.delete_document()
    store.delete_chunks()
    assert store.get_conversation("c1") is not None
