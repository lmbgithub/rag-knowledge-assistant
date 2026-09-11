from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pdfs import simple_pdf

from rag_assistant.api import create_app
from rag_assistant.config import Config
from rag_assistant.errors import BackendUnavailable
from rag_assistant.fakes import FailingChat, ScriptedChat
from rag_assistant.index import MemoryIndex
from rag_assistant.memory_store import MemoryStore
from rag_assistant.service import Assistant

TEXT = "The migration to PostgreSQL completed in March after two false starts. " * 6
SECOND = "Caching is served by Redis with a sixty second time to live. " * 6


@pytest.fixture
def model():
    return ScriptedChat("The migration completed in March [1].")


@pytest.fixture
def app_assistant(config, model):
    from rag_assistant.fakes import HashEmbedder

    embedder = HashEmbedder(dim=64)
    return Assistant(
        config,
        store=MemoryStore(),
        index=MemoryIndex(embedder.dim),
        embedder=embedder,
        model=model,
    )


@pytest.fixture
def client(app_assistant):
    with TestClient(create_app(app_assistant)) as client:
        yield client


@pytest.fixture
def pdf_bytes():
    return simple_pdf(TEXT, SECOND)


def upload(client, data, name="paper.pdf"):
    return client.post("/document", files={"file": (name, data, "application/pdf")})


def index_it(client, app_assistant, data):
    response = upload(client, data)
    app_assistant.wait_for_indexing(timeout=30)
    return response


def stream(client, **body):
    response = client.post("/chat", json=body)
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    return response, events


# --- health ---------------------------------------------------------------


def test_health_reports_the_offline_backend_loudly(client):
    body = client.get("/health").json()
    assert body["offline"] is True
    assert body["chat_model"].startswith("fake:")


# --- document -------------------------------------------------------------


def test_document_is_not_a_404_when_nothing_is_indexed(client):
    response = client.get("/document")
    assert response.status_code == 200
    assert response.json()["status"] == "empty"


def test_the_empty_state_has_the_same_shape_as_a_real_document(client, app_assistant, pdf_bytes):
    empty = client.get("/document").json()
    index_it(client, app_assistant, pdf_bytes)
    assert set(client.get("/document").json()) == set(empty)


def test_upload_returns_202_and_the_indexing_state(client, pdf_bytes):
    response = upload(client, pdf_bytes)
    assert response.status_code == 202
    assert response.json()["status"] in {"indexing", "ready"}


def test_indexing_finishes_ready_with_real_counts(client, app_assistant, pdf_bytes):
    index_it(client, app_assistant, pdf_bytes)
    body = client.get("/document").json()
    assert body["status"] == "ready"
    assert body["pages"] == 2
    assert body["chunks"] >= 2
    assert body["progress"]["done"] == body["progress"]["total"] == body["chunks"]


def test_a_non_pdf_is_rejected_synchronously(client):
    response = upload(client, b"PK\x03\x04 not a pdf", name="cv.docx")
    assert response.status_code == 400
    assert "not a PDF" in response.json()["detail"]
    assert client.get("/document").json()["status"] == "empty"


def test_a_scan_ends_as_a_failed_document_with_a_reason(client, app_assistant):
    index_it(client, app_assistant, simple_pdf("x"))
    body = client.get("/document").json()
    assert body["status"] == "failed"
    assert "OCR" in body["error"]


def test_a_second_upload_while_one_is_indexed_is_a_409(client, app_assistant, pdf_bytes):
    index_it(client, app_assistant, pdf_bytes)
    assert upload(client, pdf_bytes).status_code == 409


def test_an_oversized_upload_is_a_413(app_assistant, pdf_bytes):
    app_assistant.config = Config(offline=True, max_upload_bytes=1024)
    with TestClient(create_app(app_assistant, config=app_assistant.config)) as client:
        assert upload(client, pdf_bytes).status_code == 413


def test_an_empty_upload_is_a_400_not_an_accepted_failure(client):
    # It used to be accepted with 202 and surface as a `failed` record.
    response = upload(client, b"")
    assert response.status_code == 400
    assert client.get("/document").json()["status"] == "empty"


# --- delete ---------------------------------------------------------------


def test_delete_without_the_filename_deletes_nothing(client, app_assistant, pdf_bytes):
    index_it(client, app_assistant, pdf_bytes)
    response = client.delete("/document")
    assert response.status_code == 400
    assert client.get("/document").json()["status"] == "ready"


def test_delete_with_the_wrong_filename_deletes_nothing(client, app_assistant, pdf_bytes):
    index_it(client, app_assistant, pdf_bytes)
    assert client.delete("/document", params={"confirm": "other.pdf"}).status_code == 400
    assert client.get("/document").json()["status"] == "ready"


def test_delete_with_the_filename_removes_everything(client, app_assistant, pdf_bytes):
    index_it(client, app_assistant, pdf_bytes)
    assert client.delete("/document", params={"confirm": "paper.pdf"}).status_code == 200
    assert client.get("/document").json()["status"] == "empty"
    assert app_assistant.index.size == 0
    assert app_assistant.store.count_chunks() == 0


def test_deleting_nothing_is_a_success_not_a_404(client):
    assert client.delete("/document", params={"confirm": "anything"}).status_code == 200


def test_a_new_document_can_be_indexed_after_a_delete(client, app_assistant, pdf_bytes):
    index_it(client, app_assistant, pdf_bytes)
    client.delete("/document", params={"confirm": "paper.pdf"})
    index_it(client, app_assistant, pdf_bytes)
    assert client.get("/document").json()["status"] == "ready"


# --- chat -----------------------------------------------------------------


def test_asking_before_a_document_exists_is_a_409(client):
    response = client.post("/chat", json={"question": "when?"})
    assert response.status_code == 409
    assert "Upload a PDF" in response.json()["detail"]


def test_an_empty_question_is_a_422_from_the_schema(client):
    assert client.post("/chat", json={"question": ""}).status_code == 422


def test_a_missing_question_is_a_422(client):
    assert client.post("/chat", json={}).status_code == 422


def test_a_whitespace_question_is_a_real_400_not_an_in_stream_error(
    client, app_assistant, pdf_bytes
):
    # It used to pass the schema, reach retrieval after the stream had opened,
    # and arrive as an `error` event carrying "status": 400 inside a 200.
    index_it(client, app_assistant, pdf_bytes)
    assert client.post("/chat", json={"question": "   "}).status_code == 400


def test_chat_streams_ndjson(client, app_assistant, pdf_bytes):
    index_it(client, app_assistant, pdf_bytes)
    response, events = stream(client, question="when did the migration finish?")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert [e["type"] for e in events][:3] == ["conversation", "message", "citations"]
    assert events[-1]["type"] == "done"


def test_the_answer_cites_the_page_it_came_from(client, app_assistant, pdf_bytes):
    index_it(client, app_assistant, pdf_bytes)
    _, events = stream(client, question="postgresql migration")
    citations = next(e for e in events if e["type"] == "citations")["citations"]
    assert citations
    assert citations[0]["page"] == 1
    assert "PostgreSQL" in citations[0]["text"]


def test_a_conversation_is_created_and_listed(client, app_assistant, pdf_bytes):
    index_it(client, app_assistant, pdf_bytes)
    _, events = stream(client, question="postgresql migration")
    conversation_id = events[0]["conversation"]["id"]
    listed = client.get("/conversations").json()["conversations"]
    assert [c["id"] for c in listed] == [conversation_id]
    assert listed[0]["message_count"] == 2
    assert "messages" not in listed[0]


def test_a_conversation_can_be_read_back_in_full(client, app_assistant, pdf_bytes):
    index_it(client, app_assistant, pdf_bytes)
    _, events = stream(client, question="postgresql migration")
    body = client.get(f"/conversations/{events[0]['conversation']['id']}").json()
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert body["messages"][1]["citations"]


def test_a_follow_up_lands_in_the_same_conversation(client, app_assistant, pdf_bytes):
    index_it(client, app_assistant, pdf_bytes)
    _, first = stream(client, question="postgresql migration")
    conversation_id = first[0]["conversation"]["id"]
    stream(client, question="and redis?", conversation_id=conversation_id)
    assert len(client.get("/conversations").json()["conversations"]) == 1
    assert client.get(f"/conversations/{conversation_id}").json()["message_count"] == 4


def test_an_unknown_conversation_is_a_404(client):
    assert client.get("/conversations/nope").status_code == 404
    assert client.delete("/conversations/nope").status_code == 404


def test_a_conversation_can_be_deleted(client, app_assistant, pdf_bytes):
    index_it(client, app_assistant, pdf_bytes)
    _, events = stream(client, question="postgresql migration")
    conversation_id = events[0]["conversation"]["id"]
    assert client.delete(f"/conversations/{conversation_id}").status_code == 200
    assert client.get("/conversations").json()["conversations"] == []


def test_conversations_survive_deleting_the_document(client, app_assistant, pdf_bytes):
    index_it(client, app_assistant, pdf_bytes)
    _, events = stream(client, question="postgresql migration")
    client.delete("/document", params={"confirm": "paper.pdf"})
    body = client.get(f"/conversations/{events[0]['conversation']['id']}").json()
    assert body["document_filename"] == "paper.pdf"
    assert body["messages"][1]["citations"]


def test_a_model_failure_mid_stream_is_an_event_not_a_status(client, app_assistant, pdf_bytes):
    index_it(client, app_assistant, pdf_bytes)
    app_assistant.model = FailingChat(BackendUnavailable("ollama is down"))
    response, events = stream(client, question="postgresql migration")
    # The status line is long gone by the time the model is called.
    assert response.status_code == 200
    assert events[-1] == {"type": "error", "status": 503, "detail": "ollama is down"}


def test_an_unanswerable_question_is_refused_with_no_citations(app_assistant, pdf_bytes):
    app_assistant.config = Config(offline=True, top_k=3, min_score=0.5)
    with TestClient(create_app(app_assistant, config=app_assistant.config)) as client:
        index_it(client, app_assistant, pdf_bytes)
        _, events = stream(client, question="photosynthesis in tropical ferns")
        done = events[-1]
        assert done["grounded"] is False
        assert done["message"]["citations"] == []
        assert "could not find" in done["message"]["content"]


def test_the_202_body_carries_the_real_record_not_a_placeholder(client, pdf_bytes):
    # It used to return whatever `get_document()` found in the instant after
    # the thread started, which was usually nothing — so the body carried an
    # empty id and stage that never matched the real record.
    body = upload(client, pdf_bytes).json()
    assert body["id"]
    assert body["progress"]["stage"]
    assert client.get("/document").json()["id"] == body["id"]


def test_an_upload_declaring_too_much_is_refused_before_it_is_buffered(app_assistant):
    app_assistant.config = Config(offline=True, max_upload_bytes=1024)
    with TestClient(create_app(app_assistant, config=app_assistant.config)) as client:
        # A declared length over the limit is refused before the multipart
        # parser reads a byte, which is why this is middleware and not a check
        # inside the route.
        response = client.post("/document", content=b"x" * 4096)
        assert response.status_code == 413
        assert "the limit is" in response.json()["detail"]


def test_health_is_200_even_when_the_model_host_is_unreachable(config):
    # The container healthcheck calls this. Answering 503 because ollama is
    # still pulling models made the api container unhealthy, so the web
    # container — which waits on it — never started at all.
    from rag_assistant.errors import BackendUnavailable

    broken = create_app(None, config=config)

    def explode(_: object) -> None:
        raise BackendUnavailable("cannot reach ollama")

    with TestClient(broken) as client:
        broken.state.assistant = None
        import rag_assistant.api as api_module

        original = api_module.Assistant.build
        api_module.Assistant.build = staticmethod(explode)  # type: ignore[assignment]
        try:
            response = client.get("/health")
        finally:
            api_module.Assistant.build = original  # type: ignore[assignment]

    assert response.status_code == 200
    assert response.json()["status"] == "starting"
    assert "cannot reach ollama" in response.json()["detail"]


def test_a_failure_mid_stream_still_terminates_the_stream(client, app_assistant, pdf_bytes):
    # Deleting the conversation mid-answer makes append_message raise KeyError.
    # The stream used to end with no `done` and no `error`, leaving a
    # half-written bubble on screen with nothing to explain it.
    index_it(client, app_assistant, pdf_bytes)

    original = app_assistant.store.append_message
    calls: list[int] = []

    def boom(*args: object, **kwargs: object) -> None:
        calls.append(1)
        if len(calls) > 1:  # the user's question lands; the answer does not
            raise KeyError("conversation vanished")
        original(*args, **kwargs)  # type: ignore[arg-type]

    app_assistant.store.append_message = boom  # type: ignore[method-assign]
    response, events = stream(client, question="postgresql migration")
    assert response.status_code == 200
    assert events[-1]["type"] == "error"
    assert events[-1]["status"] == 500
