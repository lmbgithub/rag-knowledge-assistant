from __future__ import annotations

import pytest

from rag_assistant.fakes import EchoChat, HashEmbedder, ScriptedChat
from rag_assistant.index import MemoryIndex


def test_the_hash_embedder_is_deterministic():
    embedder = HashEmbedder(dim=32)
    assert embedder.embed_query("hello world") == embedder.embed_query("hello world")


def test_documents_and_queries_use_the_same_space():
    embedder = HashEmbedder(dim=32)
    assert embedder.embed_documents(["hello"])[0] == embedder.embed_query("hello")


def test_shared_words_score_higher_than_none():
    # Asserting a known-exact value from the fake, not a plausible one: a fake
    # nobody checks is a fake that hides the bug it was meant to expose.
    embedder = HashEmbedder(dim=256)
    index = MemoryIndex(embedder.dim)
    index.add(embedder.embed_documents(["postgres migration", "unrelated cooking recipe"]))
    hits = dict(index.search(embedder.embed_query("postgres migration"), 2))
    assert hits[0] == pytest.approx(1.0)
    assert hits[1] == pytest.approx(0.0)


def test_case_and_punctuation_are_ignored():
    embedder = HashEmbedder(dim=64)
    assert embedder.embed_query("Postgres!") == embedder.embed_query("postgres")


def test_dimension_must_be_positive():
    with pytest.raises(ValueError):
        HashEmbedder(dim=0)


def test_echo_chat_records_its_prompts():
    model = EchoChat()
    assert "".join(model.stream("sys", "user question")).startswith("From the document:")
    assert model.calls == [("sys", "user question")]


def test_scripted_chat_advances_through_its_script():
    model = ScriptedChat("first", "second")
    assert "".join(model.stream("s", "a")).strip() == "first"
    assert "".join(model.stream("s", "b")).strip() == "second"


def test_scripted_chat_repeats_its_last_answer_when_the_script_runs_out():
    model = ScriptedChat("only")
    "".join(model.stream("s", "a"))
    assert "".join(model.stream("s", "b")).strip() == "only"
