from __future__ import annotations

from typing import get_args

import pytest

from rag_assistant.models import (
    EMPTY_DOCUMENT_STATE,
    Chunk,
    Citation,
    Conversation,
    Document,
    Message,
    Retrieved,
    title_from,
)


def test_document_round_trips_through_its_dict():
    document = Document(id="a", filename="paper.pdf", status="ready", pages=3, chunks=9)
    raw = document.to_dict()
    assert raw["status"] == "ready"
    assert raw["progress"] == {"stage": "", "done": 0, "total": 0}


def test_with_progress_does_not_mutate():
    document = Document(id="a", filename="p.pdf", status="indexing")
    later = document.with_progress("embedding", 5, 10)
    assert document.progress.stage == ""
    assert later.progress.done == 5


def test_empty_state_has_the_same_keys_as_a_real_document():
    real = Document(id="a", filename="p.pdf", status="ready").to_dict()
    assert set(EMPTY_DOCUMENT_STATE) == set(real)


def test_every_status_is_declared():
    from rag_assistant.models import DocStatus

    assert set(get_args(DocStatus)) == {"empty", "indexing", "ready", "failed", "deleting"}


def test_citation_round_trip():
    citation = Citation(chunk_id=3, page=2, text="hello", score=0.5)
    assert Citation.from_dict(citation.to_dict()) == citation


def test_citation_from_retrieved_keeps_the_score():
    hit = Retrieved(chunk=Chunk(id=7, text="t", page=4, ordinal=0), score=0.812)
    citation = Citation.from_retrieved(hit)
    assert (citation.chunk_id, citation.page, citation.score) == (7, 4, 0.812)


def test_message_round_trip_preserves_citations_and_partial():
    message = Message(
        id="m",
        role="assistant",
        content="hi",
        citations=(Citation(1, 1, "t", 0.9),),
        partial=True,
    )
    restored = Message.from_dict(message.to_dict())
    assert restored == message


def test_message_from_dict_tolerates_a_missing_partial_flag():
    restored = Message.from_dict({"id": "m", "role": "user", "content": "q"})
    assert restored.partial is False
    assert restored.citations == ()


def test_conversation_can_omit_messages_but_keeps_the_count():
    conversation = Conversation(
        id="c",
        title="t",
        document_id="d",
        document_filename="p.pdf",
        messages=(Message(id="m", role="user", content="q"),),
    )
    summary = conversation.to_dict(with_messages=False)
    assert "messages" not in summary
    assert summary["message_count"] == 1


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the deadline?", "What is the deadline?"),
        ("   spaced    out   ", "spaced out"),
        ("", "New conversation"),
        ("   ", "New conversation"),
    ],
)
def test_title_from(question, expected):
    assert title_from(question) == expected


def test_title_is_truncated_on_a_word_boundary():
    title = title_from("alpha " * 40, limit=20)
    assert title.endswith("…")
    assert len(title) <= 21
    assert "alph…" not in title


def test_title_truncates_a_single_long_word_anyway():
    title = title_from("x" * 100, limit=10)
    assert title == "x" * 10 + "…"
