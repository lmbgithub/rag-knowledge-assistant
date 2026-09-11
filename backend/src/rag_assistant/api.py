"""The HTTP layer. Thin on purpose: it translates, it does not decide.

Every route is a few lines that call into the pipeline and map a `RagError`
onto its `status`. No retrieval logic, no prompt assembly and no state machine
lives here, which is why the pipeline tests need no client and the client tests
need no model.

There is no authentication, no authorisation, no rate limiting and CORS is open
to `*`. That is deliberate for a local single-user tool and it is stated here
so it cannot be mistaken for an oversight. Do not put this on a network you do
not own.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import Config
from .errors import BadRequest, Conflict, NotFound, RagError
from .models import EMPTY_DOCUMENT_STATE
from .service import Assistant


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


def create_app(assistant: Assistant | None = None, config: Config | None = None) -> FastAPI:
    config = config or (assistant.config if assistant else Config.from_env())
    app = FastAPI(title="rag-knowledge-assistant", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.assistant = assistant
    build_lock = threading.Lock()

    def service() -> Assistant:
        # Sync routes run concurrently in the threadpool, so an unguarded
        # check-then-act let two first requests each build an Assistant —
        # each opening a MongoClient and probing Ollama, with the loser's
        # client orphaned and never closed.
        if app.state.assistant is None:  # pragma: no cover - exercised at startup
            with build_lock:
                if app.state.assistant is None:
                    app.state.assistant = Assistant.build(config)
        return app.state.assistant

    @app.middleware("http")
    async def reject_declared_length(request: Request, call_next: Any) -> Any:
        """Refuse an oversized body before anything reads it.

        This cannot live in the route: FastAPI resolves `UploadFile` — which
        means parsing the whole multipart body — before the handler runs, so a
        check there happens after the cost it is meant to avoid.

        Advisory only. A client can omit or lie about `Content-Length`, and the
        real limit is still `validate_upload` on the bytes actually received.
        """

        raw = request.headers.get("content-length")
        if raw and raw.isdigit() and int(raw) > config.max_upload_bytes:
            return JSONResponse(
                {
                    "detail": (
                        f"The upload declares {int(raw) / 1e6:.1f} MB; "
                        f"the limit is {config.max_upload_bytes / 1e6:.0f} MB."
                    )
                },
                status_code=413,
            )
        return await call_next(request)

    @app.exception_handler(RagError)
    async def _rag_error(_: Request, exc: RagError) -> JSONResponse:
        return JSONResponse({"detail": exc.message}, status_code=exc.status)

    # --- health -----------------------------------------------------------
    @app.get("/health")
    def health() -> dict[str, Any]:
        """Answers 200 even when the model host does not.

        This is what the container healthcheck calls. Reporting the API as
        unhealthy because *ollama* is unreachable meant that on a cold start —
        where the bundled ollama is still pulling gigabytes of model — the api
        container went unhealthy and the web container, which waits on it,
        never started at all. The API process is up and serving; that a
        question cannot be answered yet is what the body says, and `/chat`
        still returns a 503.
        """

        try:
            return service().health()
        except RagError as exc:
            return {
                "status": "starting",
                "offline": config.offline,
                "detail": exc.message,
                "document": "unknown",
            }

    # --- document ---------------------------------------------------------
    @app.get("/document")
    def get_document() -> dict[str, Any]:
        """Never 404s.

        "There is no document" is the app's normal opening state, not an
        error, and the shape is identical to a real document so the frontend
        branches on `status` alone.
        """

        document = service().document()
        return document.to_dict() if document else dict(EMPTY_DOCUMENT_STATE)

    @app.post("/document", status_code=202)
    async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        # Every rejection — empty, oversized, not a PDF, one already indexed —
        # is `ingest.validate_upload`, reached synchronously through
        # `start_indexing`. The route used to re-check the size here and
        # answered 413 where the pipeline answered 400 for the same file. The
        # declared-length guard is middleware, because by the time this
        # function runs the body has already been read.
        document = service().start_indexing(await file.read(), file.filename or "document.pdf")
        return document.to_dict()

    @app.delete("/document")
    def delete_document(confirm: str = Query(default="")) -> dict[str, Any]:
        """`confirm` must be the document's filename.

        This destroys the index, the chunks and the record, and there is no
        undo. A dialog with an OK button is one mis-click; typing the filename
        is a deliberate act, and the server enforces it so the guard is not
        merely a frontend courtesy.
        """

        assistant = service()
        document = assistant.document()
        if document is None:
            return dict(EMPTY_DOCUMENT_STATE)  # idempotent
        if confirm != document.filename:
            raise BadRequest(f"To delete, pass confirm={document.filename!r}. Nothing was deleted.")
        assistant.delete_document()
        return dict(EMPTY_DOCUMENT_STATE)

    # --- conversations ----------------------------------------------------
    @app.get("/conversations")
    def list_conversations() -> dict[str, Any]:
        rows = service().store.list_conversations()
        return {"conversations": [c.to_dict(with_messages=False) for c in rows]}

    @app.get("/conversations/{conversation_id}")
    def get_conversation(conversation_id: str) -> dict[str, Any]:
        conversation = service().store.get_conversation(conversation_id)
        if conversation is None:
            raise NotFound(f"No conversation {conversation_id}.")
        return conversation.to_dict()

    @app.delete("/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str) -> dict[str, Any]:
        if not service().store.delete_conversation(conversation_id):
            raise NotFound(f"No conversation {conversation_id}.")
        return {"deleted": conversation_id}

    # --- chat -------------------------------------------------------------
    @app.post("/chat")
    def chat(body: AskRequest) -> StreamingResponse:
        """NDJSON, one event per line.

        Not Server-Sent Events: SSE would buy reconnection semantics this app
        has no use for — a resumed stream would need the model call to be
        replayable, and it is not — in exchange for a framing that has to be
        unescaped on the client. One JSON object per line is `json.loads` per
        chunk and nothing else.

        `prepare` runs every check that maps to a status code *before* the
        response starts, so those are real 4xx responses. Once the stream is
        open the status line is gone, so the one thing that can still fail —
        the model call — travels as an `error` event instead.
        """

        prepared = service().prepare(body.question, body.conversation_id)

        def body_stream() -> Iterator[bytes]:
            try:
                for event in prepared.events():
                    yield _line(event)
            except RagError as exc:  # pragma: no cover - the model path handles its own
                yield _line({"type": "error", "status": exc.status, "detail": exc.message})
            except Exception as exc:
                # Anything else — the conversation deleted mid-answer makes
                # `append_message` raise KeyError — used to end the stream with
                # no `done` and no `error`. The browser saw a clean EOF and left
                # a half-written bubble on screen with nothing to explain it.
                yield _line(
                    {"type": "error", "status": 500, "detail": f"{type(exc).__name__}: {exc}"}
                )

        return StreamingResponse(
            body_stream(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    return app


def _line(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode()


__all__ = ["create_app", "AskRequest", "BadRequest", "Conflict"]
