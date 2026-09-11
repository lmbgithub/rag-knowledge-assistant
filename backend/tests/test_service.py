from __future__ import annotations

from dataclasses import replace

import pytest
from pdfs import simple_pdf

from rag_assistant.config import Config
from rag_assistant.errors import BadRequest, Conflict
from rag_assistant.service import Assistant

TEXT = "The migration to PostgreSQL completed in March after two false starts. " * 6


@pytest.fixture
def pdf():
    return simple_pdf(TEXT)


def test_build_offline_needs_no_model_or_database():
    assistant = Assistant.build(Config(offline=True))
    assert assistant.health()["offline"] is True
    assert assistant.document() is None


def test_build_without_persistence_uses_no_database_and_no_index_file(monkeypatch, tmp_path):
    # The CLI path: a one-shot run has no history to keep, and needing a
    # database for it would make the quickest reproduction the slowest setup.
    from rag_assistant.index import MemoryIndex
    from rag_assistant.memory_store import MemoryStore

    monkeypatch.setattr(
        "rag_assistant.ollama.OllamaEmbedder.dim", property(lambda self: 8), raising=False
    )
    assistant = Assistant.build(
        Config(offline=False, index_path=str(tmp_path / "unused.faiss")), persist=False
    )
    assert isinstance(assistant.store, MemoryStore)
    assert isinstance(assistant.index, MemoryIndex)
    assert not (tmp_path / "unused.faiss").exists()


def test_an_unreachable_mongo_is_one_line_not_a_traceback():
    from rag_assistant.errors import BackendUnavailable
    from rag_assistant.mongo_store import MongoStore

    store = MongoStore("mongodb://127.0.0.1:1", "rag_assistant", timeout_ms=200)
    with pytest.raises(BackendUnavailable, match="cannot reach MongoDB"):
        store.connect()
    store.close()


def test_indexing_happens_on_a_thread_and_the_record_exists_immediately(assistant, pdf):
    document = assistant.start_indexing(pdf, "paper.pdf")
    assert document.status in {"indexing", "ready"}
    assistant.wait_for_indexing(timeout=30)
    assert assistant.document().status == "ready"


def test_a_non_pdf_is_rejected_before_the_thread_starts(assistant):
    # A bad upload should be a 4xx, not a `failed` record to clear by hand.
    with pytest.raises(BadRequest):
        assistant.start_indexing(b"PK\x03\x04", "cv.docx")
    assert assistant.document() is None


def test_a_second_upload_is_refused(assistant, pdf):
    assistant.start_indexing(pdf, "paper.pdf")
    assistant.wait_for_indexing(timeout=30)
    with pytest.raises(Conflict):
        assistant.start_indexing(pdf, "other.pdf")


def test_a_failed_document_is_replaced_rather_than_blocking_forever(assistant, pdf):
    assistant.start_indexing(simple_pdf("x"), "scan.pdf")
    assistant.wait_for_indexing(timeout=30)
    assert assistant.document().status == "failed"
    assistant.start_indexing(pdf, "paper.pdf")
    assistant.wait_for_indexing(timeout=30)
    assert assistant.document().status == "ready"
    assert assistant.document().filename == "paper.pdf"


def test_the_lock_is_released_after_a_rejected_upload(assistant, pdf):
    with pytest.raises(BadRequest):
        assistant.start_indexing(b"nope", "x.pdf")
    assistant.start_indexing(pdf, "paper.pdf")
    assistant.wait_for_indexing(timeout=30)
    assert assistant.document().status == "ready"


def test_the_indexing_thread_does_not_pin_the_uploaded_bytes(assistant, pdf):
    # A closure over `data` kept the whole PDF — up to max_upload_bytes —
    # reachable for the life of the process. Nothing may hold it once the
    # thread is done.
    import gc

    assistant.start_indexing(pdf, "paper.pdf")
    assistant.wait_for_indexing(timeout=30)
    gc.collect()
    assert not [
        referrer
        for referrer in gc.get_referrers(pdf)
        if isinstance(referrer, dict) and referrer.get("data") is pdf
    ]


def test_an_empty_upload_is_refused_before_the_thread_starts(assistant):
    # It used to return 202 and land as a `failed` record to clear by hand.
    with pytest.raises(BadRequest, match="empty"):
        assistant.start_indexing(b"", "empty.pdf")
    assert assistant.document() is None


def test_an_oversized_upload_is_refused_before_the_thread_starts(assistant, pdf):
    from rag_assistant.errors import PayloadTooLarge

    assistant.config = replace(assistant.config, max_upload_bytes=1024)
    with pytest.raises(PayloadTooLarge):
        assistant.start_indexing(pdf, "paper.pdf")
    assert assistant.document() is None


def test_delete_clears_the_index_and_the_store(assistant, pdf):
    assistant.start_indexing(pdf, "paper.pdf")
    assistant.wait_for_indexing(timeout=30)
    assistant.delete_document()
    assert assistant.document() is None
    assert assistant.index.size == 0
    assert assistant.store.count_chunks() == 0


def test_health_reports_the_document_state(assistant, pdf):
    assert assistant.health()["document"] == "empty"
    assistant.start_indexing(pdf, "paper.pdf")
    assistant.wait_for_indexing(timeout=30)
    assert assistant.health()["document"] == "ready"
    assert assistant.health()["index_size"] > 0


def test_ask_goes_through_the_pipeline(assistant, pdf):
    assistant.start_indexing(pdf, "paper.pdf")
    assistant.wait_for_indexing(timeout=30)
    events = list(assistant.ask("postgresql migration", None))
    assert events[-1]["type"] == "done"
    assert events[-1]["message"]["citations"]
