"""The wiring: one object that owns the four adapters and the indexing thread.

Everything above this line is pure pipeline; everything below is FastAPI. This
is where the choice of FAISS-or-memory and Mongo-or-memory is made, which is
why `OFFLINE=true` is a single branch here rather than a condition threaded
through every module.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Iterator
from typing import Any

from .chunking import ChunkingConfig
from .config import Config
from .errors import RagError
from .ingest import begin, delete_everything, ingest, validate_upload
from .models import Document
from .ports import ChatModel, Embedder, Store, VectorIndex


class Assistant:
    def __init__(
        self,
        config: Config,
        *,
        store: Store,
        index: VectorIndex,
        embedder: Embedder,
        model: ChatModel,
    ) -> None:
        self.config = config
        self.store = store
        self.index = index
        self.embedder = embedder
        self.model = model
        self._indexing = threading.Lock()
        self._finished = threading.Event()
        self._finished.set()

    # --- construction -----------------------------------------------------
    @staticmethod
    def build(config: Config, *, persist: bool = True) -> Assistant:
        """`persist=False` keeps nothing: no MongoDB and no index file.

        That is what the CLI wants. A one-shot terminal run has no history to
        keep, and requiring a database for it would make the quickest way to
        reproduce a retrieval bug the slowest one to set up.
        """

        from .fakes import EchoChat, HashEmbedder
        from .index import MemoryIndex
        from .memory_store import MemoryStore

        if config.offline:
            embedder: Embedder = HashEmbedder(dim=64)
            return Assistant(
                config,
                store=MemoryStore(),
                index=MemoryIndex(embedder.dim),
                embedder=embedder,
                model=EchoChat(),
            )

        from .ollama import OllamaChat, OllamaEmbedder

        embedder = OllamaEmbedder(config.ollama_host, config.embed_model, timeout=config.timeout)
        # Probe the model host *before* opening MongoDB. The other order left a
        # connected MongoClient — monitor threads and sockets — unreferenced
        # every time the probe failed, and the API rebuilds on every request,
        # so a model host that was merely slow to start leaked one per attempt.
        dim = embedder.dim
        model = OllamaChat(
            config.ollama_host,
            config.chat_model,
            timeout=config.timeout,
            temperature=config.temperature,
            num_ctx=config.num_ctx,
        )

        if not persist:
            return Assistant(
                config,
                store=MemoryStore(),
                index=MemoryIndex(embedder.dim),
                embedder=embedder,
                model=model,
            )

        from .index import FaissIndex
        from .mongo_store import MongoStore

        store = MongoStore(config.mongo_uri, config.mongo_db)
        try:
            store.connect()
            store.ensure_indexes()
            index = FaissIndex(dim, config.index_path)
        except Exception:
            store.close()
            raise
        return Assistant(config, store=store, index=index, embedder=embedder, model=model)

    # --- document ---------------------------------------------------------
    def document(self) -> Document | None:
        return self.store.get_document()

    def start_indexing(self, data: bytes, filename: str) -> Document:
        """Validate synchronously, then index on a thread.

        The upload request returns as soon as the document record exists, so
        the browser has something to poll. Everything that can be rejected is
        rejected by `validate_upload` *before* the thread starts, so a bad
        upload is a 4xx and never a `failed` record to clear by hand.
        """

        if not self._indexing.acquire(blocking=False):
            from .errors import Conflict

            raise Conflict("Another document is being indexed.")
        try:
            if (
                validate_upload(
                    data, filename, store=self.store, max_bytes=self.config.max_upload_bytes
                )
                is not None
            ):
                # The only survivor of validation is a `failed` record; clear it
                # so the new upload starts from an empty index.
                delete_everything(store=self.store, index=self.index)
        except Exception:
            self._indexing.release()
            raise

        # The record is written here, synchronously, so the 202 body carries
        # the real id and stage. Returning `store.get_document()` after
        # starting the thread raced it and usually returned neither.
        document = begin(data, filename, store=self.store)

        self._finished.clear()
        try:
            # `args` rather than a closure: a closure over `data` is reachable
            # from the Thread object for as long as it is referenced, which
            # pinned the whole uploaded PDF — up to max_upload_bytes — for the
            # process's lifetime. The reference is dropped in the `finally`.
            threading.Thread(
                target=self._index, args=(data, document), name="indexing", daemon=True
            ).start()
        except Exception:
            # `thread.start()` can raise (no more OS threads). Outside this
            # guard the lock stayed held and `_finished` stayed clear, so every
            # later upload 409d and `wait_for_indexing` blocked to its timeout.
            self._finished.set()
            self._indexing.release()
            raise
        return document

    def _index(self, data: bytes, document: Document) -> None:
        try:
            ingest(
                data,
                document.filename,
                store=self.store,
                index=self.index,
                embedder=self.embedder,
                chunking=ChunkingConfig(self.config.chunk_size, self.config.chunk_overlap),
                max_bytes=self.config.max_upload_bytes,
                record=document,
            )
        except Exception as exc:
            # `ingest` has already recorded the reason on the document record,
            # which is where the browser looks for it, and the route that
            # started this returned long ago — there is nobody to raise to. An
            # *unexpected* failure is printed anyway rather than vanishing into
            # a dead thread.
            if not isinstance(exc, RagError):  # pragma: no cover - defensive
                traceback.print_exc()
        finally:
            self._indexing.release()
            self._finished.set()

    def wait_for_indexing(self, timeout: float | None = None) -> None:
        """Used by tests, the CLI and `examples/`. Never called from a route."""

        self._finished.wait(timeout)

    def delete_document(self) -> None:
        delete_everything(store=self.store, index=self.index)

    # --- chat -------------------------------------------------------------
    def prepare(self, question: str, conversation_id: str | None) -> Any:
        """Every check that maps to a status code, run eagerly. See `chat.prepare`."""

        from .chat import prepare

        return prepare(
            question,
            conversation_id=conversation_id,
            store=self.store,
            index=self.index,
            embedder=self.embedder,
            model=self.model,
            top_k=self.config.top_k,
            min_score=self.config.min_score,
        )

    def ask(self, question: str, conversation_id: str | None) -> Iterator[dict[str, Any]]:
        return self.prepare(question, conversation_id).events()

    # --- health -----------------------------------------------------------
    def health(self) -> dict[str, Any]:
        document = self.store.get_document()
        report: dict[str, Any] = {
            "status": "ok",
            "offline": self.config.offline,
            "chat_model": "fake:echo" if self.config.offline else self.config.chat_model,
            "embed_model": "fake:hash" if self.config.offline else self.config.embed_model,
            "index_size": self.index.size,
            "document": document.status if document else "empty",
        }
        if not self.config.offline:
            from .errors import BackendUnavailable
            from .ollama import version

            try:
                report["ollama"] = version(self.config.ollama_host)
            except BackendUnavailable as exc:
                report["status"] = "degraded"
                report["ollama"] = f"unreachable: {exc.message}"
        return report
