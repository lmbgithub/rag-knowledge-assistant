from __future__ import annotations

import pytest
from pdfs import simple_pdf

from rag_assistant.chunking import ChunkingConfig
from rag_assistant.errors import (
    BadRequest,
    Conflict,
    PayloadTooLarge,
    RagError,
    UnprocessableDocument,
)
from rag_assistant.index import MemoryIndex
from rag_assistant.ingest import delete_everything, ingest
from rag_assistant.models import Document

TEXT = "The migration to PostgreSQL completed in March after two false starts. " * 8
SECOND = "Caching is served by Redis with a sixty second time to live. " * 8


@pytest.fixture
def pdf():
    return simple_pdf(TEXT, SECOND)


def run(pdf, store, index, embedder, **kwargs):
    return ingest(pdf, "paper.pdf", store=store, index=index, embedder=embedder, **kwargs)


def test_a_successful_ingest_ends_ready(pdf, store, index, embedder):
    document = run(pdf, store, index, embedder)
    assert document.status == "ready"
    assert document.pages == 2
    assert document.chunks == index.size == store.count_chunks()
    assert document.error is None


def test_progress_is_reported_with_real_counts(pdf, store, index, embedder):
    seen = []
    run(pdf, store, index, embedder, on_progress=lambda *a: seen.append(a))
    stages = [stage for stage, _, _ in seen]
    assert stages[0] == "reading the PDF"
    assert "embedding" in stages
    assert stages[-1] == "ready"
    embedding = [(d, t) for s, d, t in seen if s == "embedding"]
    assert embedding[-1][0] == embedding[-1][1] > 0


def test_progress_is_visible_in_the_store_while_indexing(pdf, store, index, embedder):
    stored = []
    original = store.put_document

    def spy(document: Document) -> None:
        stored.append((document.status, document.progress.stage))
        original(document)

    store.put_document = spy  # type: ignore[method-assign]
    run(pdf, store, index, embedder)
    assert ("indexing", "embedding") in stored


def test_the_chunk_ids_are_the_index_rows(pdf, store, index, embedder):
    run(pdf, store, index, embedder)
    ids = [c.id for c in store.get_chunks(list(range(index.size)))]
    assert ids == list(range(index.size))


def test_a_second_upload_is_refused_while_one_is_indexed(pdf, store, index, embedder):
    run(pdf, store, index, embedder)
    with pytest.raises(Conflict, match="already indexed"):
        run(pdf, store, index, embedder)


def test_a_failed_document_may_be_replaced(pdf, store, index, embedder):
    with pytest.raises(UnprocessableDocument):
        run(b"%PDF-1.4 truncated", store, index, embedder)
    assert store.get_document().status == "failed"
    assert run(pdf, store, index, embedder).status == "ready"


def test_an_empty_upload_is_a_400(store, index, embedder):
    with pytest.raises(BadRequest, match="empty"):
        run(b"", store, index, embedder)


def test_an_oversized_upload_is_refused_before_parsing(pdf, store, index, embedder):
    # 413, not 400: the request is well-formed, the payload is too big. It used
    # to be both, depending on which layer caught it first.
    with pytest.raises(PayloadTooLarge, match="limit"):
        run(pdf, store, index, embedder, max_bytes=10)


def test_a_failure_leaves_no_vectors_and_no_chunks(store, index, embedder):
    with pytest.raises(UnprocessableDocument):
        run(simple_pdf("x"), store, index, embedder)
    assert index.size == 0
    assert store.count_chunks() == 0


def test_a_failure_records_why_rather_than_vanishing(store, index, embedder):
    with pytest.raises(UnprocessableDocument):
        run(simple_pdf("x"), store, index, embedder)
    document = store.get_document()
    assert document.status == "failed"
    assert "no text layer" in document.error


def test_a_non_empty_index_is_detected_rather_than_trusted(pdf, store, index, embedder):
    # Chunk ids are handed out assuming an empty index. If FAISS disagrees,
    # every citation points at the wrong passage and the answers still look
    # plausible — which is why this is checked.
    index.add([[0.0] * embedder.dim])
    with pytest.raises(RagError, match="not empty"):
        run(pdf, store, index, embedder)
    assert store.get_document().status == "failed"


def test_an_embedder_that_raises_leaves_a_failed_document(pdf, store, index):
    class Broken:
        dim = 64

        def embed_documents(self, texts):
            raise RuntimeError("model gone")

        def embed_query(self, text):
            raise RuntimeError("model gone")

    with pytest.raises(RuntimeError):
        run(pdf, store, index, Broken())
    assert store.get_document().error == "RuntimeError: model gone"
    assert index.size == 0


def test_chunk_size_is_honoured(pdf, store, index, embedder):
    small = run(pdf, store, index, embedder, chunking=ChunkingConfig(32, 4))
    store.delete_document()
    store.delete_chunks()
    index.reset()
    large = run(pdf, store, MemoryIndex(embedder.dim), embedder, chunking=ChunkingConfig(2048, 64))
    assert small.chunks > large.chunks


def test_delete_removes_all_three_stores(pdf, store, index, embedder):
    run(pdf, store, index, embedder)
    delete_everything(store=store, index=index)
    assert store.get_document() is None
    assert store.count_chunks() == 0
    assert index.size == 0


def test_delete_with_nothing_indexed_is_a_no_op(store, index):
    delete_everything(store=store, index=index)
    assert store.get_document() is None


def test_a_failed_delete_leaves_the_document_in_deleting(pdf, store, index, embedder):
    # A half-deleted index must not go on to accept new data.
    run(pdf, store, index, embedder)

    def boom() -> None:
        raise RuntimeError("mongo down")

    store.delete_chunks = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        delete_everything(store=store, index=index)
    assert store.get_document().status == "deleting"


def test_uploading_after_a_half_delete_is_refused(pdf, store, index, embedder):
    run(pdf, store, index, embedder)
    store.delete_chunks = lambda: (_ for _ in ()).throw(RuntimeError("mongo down"))  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        delete_everything(store=store, index=index)
    with pytest.raises(Conflict, match="deleting"):
        run(pdf, store, index, embedder)


def test_delete_then_reindex_restarts_the_ids(pdf, store, index, embedder):
    run(pdf, store, index, embedder)
    delete_everything(store=store, index=index)
    document = run(pdf, store, index, embedder)
    assert document.status == "ready"
    assert [c.id for c in store.get_chunks([0])] == [0]


def test_a_failure_records_why_before_attempting_cleanup(pdf, store, index, embedder):
    """The order matters: cleanup can fail too.

    Marking the document `failed` after `index.reset()` meant a reset that
    raised left the record on `indexing` forever — the UI polls it at 1 Hz with
    the composer disabled, every new upload 409s, and delete runs the same
    failing cleanup again.
    """

    def boom() -> None:
        raise RuntimeError("faiss file is read-only")

    index.reset = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        run(simple_pdf("x"), store, index, embedder)

    document = store.get_document()
    assert document.status == "failed"
    assert "no text layer" in document.error


def test_the_indexing_record_exists_before_any_work_starts(pdf, store, index, embedder):
    from rag_assistant.ingest import begin

    document = begin(pdf, "paper.pdf", store=store)
    assert document.id
    assert store.get_document() == document
    assert document.status == "indexing"
