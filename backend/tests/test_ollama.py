from __future__ import annotations

import json

import pytest
from fake_ollama import FakeOllama

from rag_assistant.errors import BackendUnavailable
from rag_assistant.ollama import OllamaChat, OllamaEmbedder, version

DEAD = "http://127.0.0.1:1"


def test_the_dimension_is_discovered_not_assumed():
    # Ollama does not report it, and hard-coding 768 breaks silently the day
    # EMBED_MODEL changes — silently, because FAISS is built at the wrong width.
    with FakeOllama(dim=5) as server:
        assert OllamaEmbedder(server.url, "m").dim == 5


def test_the_dimension_is_probed_once():
    with FakeOllama() as server:
        embedder = OllamaEmbedder(server.url, "m")
        assert embedder.dim == embedder.dim == embedder.dim
        assert len(server.requests) == 1


def test_documents_are_embedded_in_batches():
    with FakeOllama() as server:
        embedder = OllamaEmbedder(server.url, "m", batch=2)
        vectors = embedder.embed_documents(["a", "bb", "ccc", "dddd", "e"])
        assert len(vectors) == 5
        embed_calls = [p for path, p in server.requests if path.endswith("/api/embed")]
        assert [len(call["input"]) for call in embed_calls] == [2, 2, 1]


def test_a_query_is_one_call():
    with FakeOllama() as server:
        assert len(OllamaEmbedder(server.url, "m").embed_query("hello")) == 4


def test_a_short_embedding_response_is_an_error_not_a_silent_truncation():
    with FakeOllama(embeddings=[[1.0, 2.0]]) as server:
        with pytest.raises(BackendUnavailable, match="1 embeddings for 2"):
            OllamaEmbedder(server.url, "m").embed_documents(["a", "b"])


def test_a_response_with_no_embeddings_key_is_an_error():
    with FakeOllama(body=b'{"error":"model not found"}') as server:
        with pytest.raises(BackendUnavailable):
            OllamaEmbedder(server.url, "m").embed_query("a")


def test_an_unreachable_host_is_a_503_not_a_crash():
    with pytest.raises(BackendUnavailable, match="cannot reach ollama"):
        OllamaEmbedder(DEAD, "m", timeout=1).embed_query("a")


def test_an_http_error_carries_the_upstream_status():
    with FakeOllama(status=500) as server:
        with pytest.raises(BackendUnavailable, match="returned 500"):
            OllamaEmbedder(server.url, "m").embed_query("a")


def test_chat_streams_fragments_in_order():
    with FakeOllama(tokens=["The ", "answer ", "is ", "March."]) as server:
        assert "".join(OllamaChat(server.url, "m").stream("s", "u")) == "The answer is March."


def test_chat_sends_the_system_and_user_messages_and_temperature_zero():
    with FakeOllama() as server:
        list(OllamaChat(server.url, "m").stream("SYSTEM", "USER"))
        _, payload = server.requests[-1]
        assert [m["role"] for m in payload["messages"]] == ["system", "user"]
        assert payload["messages"][1]["content"] == "USER"
        assert payload["options"]["temperature"] == 0.0
        assert payload["stream"] is True


def test_an_unparseable_frame_does_not_discard_the_rest_of_the_answer():
    body = (
        json.dumps({"message": {"content": "before "}}) + "\n"
        "{ this is not json\n" + json.dumps({"message": {"content": "after"}, "done": True}) + "\n"
    ).encode()
    with FakeOllama(body=body) as server:
        assert "".join(OllamaChat(server.url, "m").stream("s", "u")) == "before after"


def test_an_error_frame_mid_stream_raises():
    body = (
        json.dumps({"message": {"content": "partial"}})
        + "\n"
        + json.dumps({"error": "out of memory"})
        + "\n"
    ).encode()
    with FakeOllama(body=body) as server:
        stream = OllamaChat(server.url, "m").stream("s", "u")
        assert next(stream) == "partial"
        with pytest.raises(BackendUnavailable, match="out of memory"):
            next(stream)


def test_the_stream_stops_at_done_without_waiting_for_the_socket_to_close():
    body = (
        json.dumps({"message": {"content": "a"}, "done": True})
        + "\n"
        + json.dumps({"message": {"content": "never"}})
        + "\n"
    ).encode()
    with FakeOllama(body=body) as server:
        assert "".join(OllamaChat(server.url, "m").stream("s", "u")) == "a"


def test_empty_fragments_are_not_yielded():
    with FakeOllama(tokens=["", "real", ""]) as server:
        assert list(OllamaChat(server.url, "m").stream("s", "u")) == ["real"]


def test_version_reports_the_running_daemon():
    with FakeOllama() as server:
        assert version(server.url) == "0.0.0-test"


def test_version_of_a_dead_host_is_a_backend_error():
    with pytest.raises(BackendUnavailable):
        version(DEAD, timeout=1)
