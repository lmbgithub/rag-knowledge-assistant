"""One question, one streamed answer, persisted when the stream finishes.

The module is split into two phases on purpose, and that split is the whole
design:

  `prepare()`  everything that can fail with a status code — the document must
               be ready, the conversation must exist, the question must be
               non-empty, retrieval must run. All of it eager.
  `.events()`  the stream. The only thing that can go wrong inside it is the
               model call.

An earlier version was one generator doing both, and the route had to pull the
first event by hand to turn a `Conflict` into a real 409. That worked for the
checks *before* the first yield and silently failed for the ones after: an
empty question became an in-stream `error` event carrying `"status": 400`,
while every structurally identical validation error was a real 400 response.
Splitting the phases removes the workaround and the inconsistency together.

Events rather than a string, because three different things reach the browser
over one connection: the passages (immediately, so they can be read while the
model is still thinking), the tokens, and the final persisted message.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .errors import Conflict, NotFound, RagError
from .models import Citation, Conversation, Document, Message, Retrieved, new_id, now, title_from
from .ports import ChatModel, Embedder, Store, VectorIndex
from .prompt import SYSTEM, Answer, build_user_prompt, interpret, no_context
from .retrieve import DEFAULT_MIN_SCORE, DEFAULT_TOP_K, retrieve

Event = dict[str, Any]


def require_ready(store: Store) -> Document:
    """The rule the whole app is built around.

    No document, or a document still indexing, means no question is answerable.
    Answering anyway is a small model writing fluent prose from its own weights
    with no passages and no citations — indistinguishable, to the reader, from
    a grounded answer. It is a 409 and not a 400: the request is fine, the
    application is in the wrong state for it.
    """

    document = store.get_document()
    if document is None:
        raise Conflict("No document is indexed. Upload a PDF first.")
    if document.status != "ready":
        raise Conflict(
            f"'{document.filename}' is {document.status}, not ready. "
            + ("Indexing is still running." if document.status == "indexing" else "")
        )
    return document


def start_conversation(store: Store, document: Document, question: str) -> Conversation:
    """The document is passed in, not re-read: the caller has already checked it."""

    return store.create_conversation(
        Conversation(
            id=new_id(),
            title=title_from(question),
            document_id=document.id,
            document_filename=document.filename,
        )
    )


@dataclass(frozen=True)
class Prepared:
    """A question that has passed every check and is ready to be answered.

    `history` is the thread as it stood *before* this question was appended,
    captured here rather than re-read and sliced. The earlier version read the
    conversation back from the store and dropped the last element, which was
    correct only under an invariant nothing enforced.
    """

    store: Store
    model: ChatModel
    document: Document
    conversation: Conversation
    user_message: Message
    hits: tuple[Retrieved, ...]
    citations: tuple[Citation, ...]
    history: tuple[Message, ...]

    def events(self) -> Iterator[Event]:
        yield {
            "type": "conversation",
            "conversation": self.conversation.to_dict(with_messages=False),
        }
        yield {"type": "message", "message": self.user_message.to_dict()}
        yield {"type": "citations", "citations": [c.to_dict() for c in self.citations]}

        if not self.hits:
            # No passages cleared the floor. The model is not called at all —
            # it has nothing to ground an answer in, and asking a model this
            # size to say so is a coin flip.
            yield from self._finish(no_context(), ())
            return

        yield from self._generate()

    def _generate(self) -> Iterator[Event]:
        prompt = build_user_prompt(self.user_message.content, self.hits, self.history)
        parts: list[str] = []
        try:
            for fragment in self.model.stream(SYSTEM, prompt):
                parts.append(fragment)
                yield {"type": "token", "text": fragment}
        except GeneratorExit:
            # The client went away mid-answer. Keep what arrived, marked
            # partial: a stored half-sentence that looks whole gets fed back as
            # history and the model takes it as its own prior answer.
            self._persist("".join(parts), self.citations, partial=True)
            raise
        except RagError as exc:
            # The model host failed after some tokens were already on the wire.
            # Storing the fragment as if it were the answer would be a lie.
            if parts:
                self._persist("".join(parts), self.citations, partial=True)
            yield {"type": "error", "status": exc.status, "detail": exc.message}
            return

        yield from self._finish(interpret("".join(parts)), self.citations)

    def _finish(self, answer: Answer, citations: tuple[Citation, ...]) -> Iterator[Event]:
        # A refusal carries no citations: passages listed under "I could not
        # find anything" would contradict the answer above them.
        message = self._persist(answer.text, () if answer.refused else citations, partial=False)
        # `content` is authoritative and may differ from the tokens already
        # streamed — the refusal token is rewritten into a readable sentence.
        # The browser renders this, not its own accumulation.
        yield {
            "type": "done",
            "message": message.to_dict(),
            "grounded": not answer.refused,
            "document": self.document.filename,
        }

    def _persist(self, content: str, citations: tuple[Citation, ...], *, partial: bool) -> Message:
        message = Message(
            id=new_id(),
            role="assistant",
            content=content,
            citations=citations,
            partial=partial,
            created_at=now(),
        )
        self.store.append_message(self.conversation.id, message)
        return message


def prepare(
    question: str,
    *,
    conversation_id: str | None,
    store: Store,
    index: VectorIndex,
    embedder: Embedder,
    model: ChatModel,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> Prepared:
    """Run every check that maps to a status code, then stop short of the model."""

    document = require_ready(store)

    if conversation_id is None:
        conversation = start_conversation(store, document, question)
    else:
        found = store.get_conversation(conversation_id)
        if found is None:
            raise NotFound(f"No conversation {conversation_id}.")
        conversation = found

    hits = retrieve(
        question,
        store=store,
        index=index,
        embedder=embedder,
        top_k=top_k,
        min_score=min_score,
    )

    history = conversation.messages
    user_message = Message(id=new_id(), role="user", content=question.strip(), created_at=now())
    store.append_message(conversation.id, user_message)

    return Prepared(
        store=store,
        model=model,
        document=document,
        conversation=conversation,
        user_message=user_message,
        hits=tuple(hits),
        citations=tuple(Citation.from_retrieved(hit) for hit in hits),
        history=history,
    )


def ask(question: str, *, conversation_id: str | None, **kwargs: Any) -> Iterator[Event]:
    """`prepare` then stream. Convenience for the CLI, the examples and tests."""

    return prepare(question, conversation_id=conversation_id, **kwargs).events()
