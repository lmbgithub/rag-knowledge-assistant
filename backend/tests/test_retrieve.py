from __future__ import annotations

import pytest

from rag_assistant.errors import BadRequest
from rag_assistant.models import Chunk
from rag_assistant.retrieve import retrieve

PASSAGES = [
    "The migration to PostgreSQL completed in March.",
    "Caching is served by Redis with a sixty second ttl.",
    "Deployment runs on Kubernetes with three replicas.",
]


@pytest.fixture
def indexed(store, index, embedder):
    chunks = [Chunk(id=i, text=text, page=1, ordinal=i) for i, text in enumerate(PASSAGES)]
    index.add(embedder.embed_documents([c.text for c in chunks]))
    store.put_chunks(chunks)
    return store, index, embedder


def ask(indexed, question, **kwargs):
    store, index, embedder = indexed
    return retrieve(question, store=store, index=index, embedder=embedder, **kwargs)


def test_the_relevant_passage_ranks_first(indexed):
    hits = ask(indexed, "postgresql migration", min_score=0.0)
    assert hits[0].chunk.id == 0


def test_top_k_is_honoured(indexed):
    assert len(ask(indexed, "postgresql", top_k=2, min_score=0.0)) == 2


def test_scores_are_returned_in_descending_order(indexed):
    scores = [hit.score for hit in ask(indexed, "redis caching", min_score=0.0)]
    assert scores == sorted(scores, reverse=True)


def test_an_unrelated_question_returns_nothing_above_the_floor(indexed):
    # The index always returns its k nearest neighbours; without the floor the
    # model is handed the three least-unrelated passages and writes an answer
    # out of them.
    assert ask(indexed, "photosynthesis in tropical ferns", min_score=0.2) == []


def test_the_same_unrelated_question_does_return_hits_with_no_floor(indexed):
    assert ask(indexed, "photosynthesis in tropical ferns", min_score=-1.0) != []


def test_the_floor_is_inclusive(indexed):
    hits = ask(indexed, "postgresql migration", min_score=0.0)
    exact = hits[0].score
    assert ask(indexed, "postgresql migration", min_score=exact)


def test_an_empty_question_is_rejected(indexed):
    with pytest.raises(BadRequest, match="empty"):
        ask(indexed, "   ")


def test_top_k_below_one_is_rejected(indexed):
    with pytest.raises(BadRequest, match="top_k"):
        ask(indexed, "postgres", top_k=0)


def test_searching_an_empty_index_returns_nothing(store, index, embedder):
    assert retrieve("anything", store=store, index=index, embedder=embedder) == []


def test_an_id_with_no_chunk_behind_it_is_dropped(indexed):
    # Index and store out of step. Padding with a placeholder would put an
    # empty passage in the prompt for the model to fill in itself.
    store, index, embedder = indexed
    store.delete_chunks()
    assert ask(indexed, "postgresql migration", min_score=0.0) == []


def test_retrieval_is_reproducible(indexed):
    first = ask(indexed, "redis caching", min_score=0.0)
    second = ask(indexed, "redis caching", min_score=0.0)
    assert [(h.chunk.id, h.score) for h in first] == [(h.chunk.id, h.score) for h in second]
