from __future__ import annotations

import pytest

from rag_assistant.chat import ask, require_ready, start_conversation
from rag_assistant.errors import BackendUnavailable, Conflict, NotFound
from rag_assistant.fakes import FailingChat, ScriptedChat
from rag_assistant.models import Chunk, Document
from rag_assistant.prompt import NO_CONTEXT_ANSWER, NOT_IN_DOCUMENT

PASSAGES = [
    "The migration to PostgreSQL completed in March.",
    "Caching is served by Redis with a sixty second ttl.",
]


@pytest.fixture
def ready(store, index, embedder):
    chunks = [Chunk(id=i, text=t, page=i + 1, ordinal=0) for i, t in enumerate(PASSAGES)]
    index.add(embedder.embed_documents([c.text for c in chunks]))
    store.put_chunks(chunks)
    store.put_document(
        Document(id="d1", filename="paper.pdf", status="ready", pages=2, chunks=len(chunks))
    )
    return store, index, embedder


def run(ready, question, model, conversation_id=None, **kwargs):
    store, index, embedder = ready
    return list(
        ask(
            question,
            conversation_id=conversation_id,
            store=store,
            index=index,
            embedder=embedder,
            model=model,
            min_score=kwargs.pop("min_score", 0.0),
            **kwargs,
        )
    )


def types(events):
    return [e["type"] for e in events]


def test_asking_with_no_document_is_a_conflict(store, index, embedder):
    with pytest.raises(Conflict, match="Upload a PDF"):
        require_ready(store)


def test_asking_while_indexing_is_a_conflict(store):
    store.put_document(Document(id="d", filename="p.pdf", status="indexing"))
    with pytest.raises(Conflict, match="still running"):
        require_ready(store)


@pytest.mark.parametrize("status", ["failed", "deleting"])
def test_every_non_ready_status_is_refused(store, status):
    store.put_document(Document(id="d", filename="p.pdf", status=status))
    with pytest.raises(Conflict):
        require_ready(store)


def test_a_ready_document_passes(ready):
    assert require_ready(ready[0]).filename == "paper.pdf"


def test_the_event_order_is_fixed(ready, scripted):
    order = types(run(ready, "when did the migration finish?", scripted))
    assert order[:3] == ["conversation", "message", "citations"]
    assert set(order[3:-1]) == {"token"}
    assert order[-1] == "done"


def test_citations_arrive_before_the_first_token(ready, scripted):
    order = types(run(ready, "postgresql migration", scripted))
    assert order.index("citations") < order.index("token")


def test_the_answer_is_persisted_with_its_citations(ready, scripted):
    store = ready[0]
    events = run(ready, "postgresql migration", scripted)
    conversation_id = events[0]["conversation"]["id"]
    messages = store.get_conversation(conversation_id).messages
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].citations
    assert messages[1].partial is False


def test_a_new_conversation_is_titled_from_the_question(ready, scripted):
    events = run(ready, "When did the migration finish?", scripted)
    assert events[0]["conversation"]["title"] == "When did the migration finish?"


def test_a_second_question_continues_the_same_conversation(ready, scripted):
    store = ready[0]
    first = run(ready, "postgresql migration", scripted)
    conversation_id = first[0]["conversation"]["id"]
    run(ready, "and the caching?", scripted, conversation_id=conversation_id)
    assert len(store.get_conversation(conversation_id).messages) == 4
    assert len(store.list_conversations()) == 1


def test_an_unknown_conversation_id_is_a_404(ready, scripted):
    with pytest.raises(NotFound):
        run(ready, "q", scripted, conversation_id="nope")


def test_history_reaches_the_prompt_without_repeating_the_new_question(ready):
    model = ScriptedChat("first answer", "second answer")
    first = run(ready, "postgresql migration", model)
    run(ready, "and redis?", model, conversation_id=first[0]["conversation"]["id"])
    _, second_prompt = model.calls[1]
    assert "first answer" in second_prompt
    assert second_prompt.count("and redis?") == 1


def test_an_unanswerable_question_never_calls_the_model(ready, scripted):
    events = run(ready, "photosynthesis in tropical ferns", scripted, min_score=0.2)
    assert scripted.calls == []
    assert events[-1]["message"]["content"] == NO_CONTEXT_ANSWER
    assert events[-1]["grounded"] is False
    assert events[-1]["message"]["citations"] == []


def test_the_refusal_token_is_rewritten_in_the_persisted_message(ready):
    events = run(ready, "postgresql migration", ScriptedChat(NOT_IN_DOCUMENT))
    done = events[-1]
    assert done["message"]["content"] == NO_CONTEXT_ANSWER
    assert done["grounded"] is False


def test_a_refused_answer_carries_no_citations(ready):
    # Citations under "I could not find anything" would contradict the answer.
    events = run(ready, "postgresql migration", ScriptedChat(NOT_IN_DOCUMENT))
    assert events[-1]["message"]["citations"] == []


def test_a_grounded_answer_is_marked_grounded(ready, scripted):
    assert run(ready, "postgresql migration", scripted)[-1]["grounded"] is True


def test_a_model_failure_becomes_an_error_event_not_an_exception(ready):
    events = run(ready, "postgresql migration", FailingChat(BackendUnavailable("ollama is down")))
    assert types(events)[-1] == "error"
    assert events[-1]["status"] == 503
    assert "ollama is down" in events[-1]["detail"]


def test_a_model_failure_before_any_token_stores_no_assistant_message(ready):
    store = ready[0]
    events = run(ready, "postgresql migration", FailingChat(BackendUnavailable("down")))
    conversation_id = events[0]["conversation"]["id"]
    roles = [m.role for m in store.get_conversation(conversation_id).messages]
    assert roles == ["user"]


def test_an_interrupted_stream_is_stored_as_partial(ready, scripted):
    store, index, embedder = ready
    events = ask(
        "postgresql migration",
        conversation_id=None,
        store=store,
        index=index,
        embedder=embedder,
        model=scripted,
        min_score=0.0,
    )
    conversation_id = None
    for event in events:
        if event["type"] == "conversation":
            conversation_id = event["conversation"]["id"]
        if event["type"] == "token":
            break
    events.close()  # the browser went away mid-answer

    messages = store.get_conversation(conversation_id).messages
    assert messages[-1].role == "assistant"
    assert messages[-1].partial is True
    assert messages[-1].content


def test_start_conversation_records_the_document_it_was_about(ready):
    store = ready[0]
    conversation = start_conversation(store, require_ready(store), "a question")
    assert conversation.document_filename == "paper.pdf"
    assert conversation.document_id == "d1"


def test_conversations_outlive_the_document(ready, scripted):
    from rag_assistant.ingest import delete_everything

    store, index, _ = ready
    events = run(ready, "postgresql migration", scripted)
    delete_everything(store=store, index=index)
    conversation = store.get_conversation(events[0]["conversation"]["id"])
    assert conversation is not None
    assert conversation.document_filename == "paper.pdf"
