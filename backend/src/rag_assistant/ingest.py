"""PDF bytes to a searchable index, and the deletion that undoes it.

Both directions are here because they are the same invariant read forwards and
backwards: the FAISS index, the chunk rows and the document record are only
ever all present or all absent. Splitting them across two modules is how one
of them ends up updated and the other not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from .chunking import ChunkingConfig, chunk_pages
from .errors import BadRequest, Conflict, PayloadTooLarge, RagError
from .models import Document, new_id
from .pdf import extract, looks_like_pdf
from .ports import Embedder, Store, VectorIndex

Progress = Callable[[str, int, int], None]

EMBED_BATCH = 16


def _noop(stage: str, done: int, total: int) -> None:
    pass


DEFAULT_MAX_BYTES = 32 * 1024 * 1024


def validate_upload(
    data: bytes, filename: str, *, store: Store, max_bytes: int = DEFAULT_MAX_BYTES
) -> Document | None:
    """Every reason an upload can be refused, in one place. Returns what it replaces.

    All four checks used to live at three different altitudes — the size limit
    in the route, the conflict and the magic bytes in the service, all four
    again in `ingest` — which drifted in both wording and status code (the same
    oversized file was a 413 from one layer and a 400 from the other), and left
    a real gap: an empty upload passed the service's checks, returned 202, and
    surfaced as a `failed` record the user had to clear by hand.

    Being one function makes "rejected before any work starts" true by
    construction rather than by three layers agreeing.
    """

    if not data:
        raise BadRequest("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise PayloadTooLarge(
            f"{filename} is {len(data) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.0f} MB."
        )
    if not looks_like_pdf(data):
        raise BadRequest(f"{filename} is not a PDF — its first bytes are not %PDF-.")

    existing = store.get_document()
    if existing is not None and existing.status != "failed":
        # Overwriting on upload is the tempting convenience, and it makes the
        # destructive path silent: an index lost by picking the wrong file in a
        # dialog, with no confirmation anywhere. Replacing means deleting first.
        raise Conflict(
            f"'{existing.filename}' is already indexed ({existing.status}). "
            "Delete it before uploading another."
        )
    return existing


def begin(data: bytes, filename: str, *, store: Store) -> Document:
    """Write the `indexing` record. Separate so a caller can do it before a thread.

    `start_indexing` returns this record to the browser as its 202 body. When
    the thread wrote it instead, the response raced it and usually carried an
    empty id and stage — a document object whose id never matched the real one.
    """

    document = Document(
        id=new_id(), filename=filename, status="indexing", bytes=len(data)
    ).with_progress("reading the PDF")
    store.put_document(document)
    return document


def ingest(
    data: bytes,
    filename: str,
    *,
    store: Store,
    index: VectorIndex,
    embedder: Embedder,
    chunking: ChunkingConfig | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    on_progress: Progress | None = None,
    record: Document | None = None,
) -> Document:
    """Index one PDF. Validates first, so it is safe to call directly."""

    report = on_progress or _noop
    if record is None:
        validate_upload(data, filename, store=store, max_bytes=max_bytes)
        record = begin(data, filename, store=store)
    document = record

    try:
        report("reading the PDF", 0, 0)
        pages = extract(data, filename=filename)
        document = replace(document, pages=len(pages)).with_progress("splitting", 0, len(pages))
        store.put_document(document)

        report("splitting", 0, len(pages))
        chunks = chunk_pages(pages, chunking)
        if not chunks:
            # Reachable when every page survives the character floor in
            # aggregate but no page has anything the splitter keeps.
            raise BadRequest(f"{filename} produced no text to index.")

        total = len(chunks)
        report("embedding", 0, total)
        vectors: list[list[float]] = []
        for start in range(0, total, EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            vectors.extend(embedder.embed_documents([c.text for c in batch]))
            done = min(start + EMBED_BATCH, total)
            store.put_document(document.with_progress("embedding", done, total))
            report("embedding", done, total)

        assigned = index.add(vectors)
        expected = [c.id for c in chunks]
        if assigned != expected:
            # The chunk ids were handed out assuming an empty index. If FAISS
            # disagrees, every citation would point at the wrong passage —
            # answers would still look plausible, which is why this is checked
            # rather than trusted.
            index.reset()
            raise RagError(
                f"index assigned ids {assigned[:3]}… but chunks are {expected[:3]}…; "
                "the index was not empty."
            )

        store.put_chunks(chunks)
        document = replace(document, status="ready", chunks=total, error=None).with_progress(
            "ready", total, total
        )
        store.put_document(document)
        report("ready", total, total)
        return document

    except Exception as exc:
        # The record is marked `failed` *first*, then the partial work is thrown
        # away. The other order is the tempting one and it strands the app: if
        # `index.reset()` or `delete_chunks()` raises, the record is never
        # written, `/document` reports `indexing` forever, the composer stays
        # disabled and every new upload 409s — with no way out, because delete
        # runs the same failing cleanup again.
        message = exc.message if isinstance(exc, RagError) else f"{type(exc).__name__}: {exc}"
        store.put_document(
            replace(document, status="failed", chunks=0, error=message).with_progress("failed")
        )
        # A half-built index produces confident wrong answers, so it goes even
        # if saying so fails.
        index.reset()
        store.delete_chunks()
        raise


def delete_everything(*, store: Store, index: VectorIndex) -> None:
    """Remove the index, the chunks and the record — in that order.

    The order is the point. Vectors first: an index with no chunks behind it
    retrieves ids that resolve to nothing and the answer degrades to "not in
    the document", which is wrong but honest. The reverse order leaves live
    vectors with no record, and the next upload's ids collide with them —
    retrieval then returns passages from a document the user deleted.

    If a step raises, the record is left in `deleting`, which the upload route
    refuses. A half-deleted index must not accept new data.
    """

    document = store.get_document()
    if document is None:
        return  # idempotent: deleting nothing is a success, not a 404
    store.put_document(replace(document, status="deleting").with_progress("deleting"))
    try:
        index.reset()
        store.delete_chunks()
    except Exception as exc:
        # The record stays in `deleting` — upload refuses that state, which is
        # the point. But a state with no exit is a state machine stopping one
        # branch short, so the reason is recorded on the record and the UI
        # offers the delete again. Retrying is the only safe recovery: the
        # half-deleted index must never accept new data.
        message = exc.message if isinstance(exc, RagError) else f"{type(exc).__name__}: {exc}"
        store.put_document(
            replace(document, status="deleting", error=message).with_progress("deleting")
        )
        raise
    store.delete_document()
