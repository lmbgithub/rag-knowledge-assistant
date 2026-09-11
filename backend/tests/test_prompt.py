from __future__ import annotations

import pytest

from rag_assistant.models import Chunk, Message, Retrieved
from rag_assistant.prompt import (
    MAX_HISTORY_TURNS,
    NO_CONTEXT_ANSWER,
    NOT_IN_DOCUMENT,
    SYSTEM,
    build_user_prompt,
    format_history,
    format_passages,
    interpret,
    no_context,
)


def hit(chunk_id: int, text: str, page: int = 1, score: float = 0.5) -> Retrieved:
    return Retrieved(chunk=Chunk(id=chunk_id, text=text, page=page, ordinal=0), score=score)


def test_passages_are_numbered_from_one_and_carry_their_page():
    formatted = format_passages([hit(0, "alpha", page=3), hit(1, "beta", page=7)])
    assert "[1] (page 3)" in formatted
    assert "[2] (page 7)" in formatted


def test_the_system_prompt_names_the_refusal_token_and_forbids_outside_knowledge():
    assert NOT_IN_DOCUMENT in SYSTEM
    assert "Do not use anything you know from training" in SYSTEM


def test_the_user_prompt_contains_the_question_and_the_passages():
    prompt = build_user_prompt("When did it finish?", [hit(0, "It finished in March.")])
    assert "Question: When did it finish?" in prompt
    assert "It finished in March." in prompt


def test_history_is_omitted_entirely_when_there_is_none():
    assert "Earlier in this conversation" not in build_user_prompt("q", [hit(0, "t")])


def test_history_is_included_oldest_first():
    history = [
        Message(id="1", role="user", content="first question"),
        Message(id="2", role="assistant", content="first answer"),
    ]
    formatted = format_history(history)
    assert formatted.index("first question") < formatted.index("first answer")
    assert formatted.startswith("User: ")


def test_history_is_truncated_to_the_recent_turns():
    messages = []
    for i in range(10):
        messages.append(Message(id=f"u{i}", role="user", content=f"q{i}"))
        messages.append(Message(id=f"a{i}", role="assistant", content=f"a{i}"))
    formatted = format_history(messages, turns=MAX_HISTORY_TURNS)
    assert "q9" in formatted
    assert "q0" not in formatted
    assert len(formatted.splitlines()) == MAX_HISTORY_TURNS * 2


def test_partial_messages_are_kept_out_of_history():
    # A half-sentence fed back as the model's own prior answer teaches it to
    # produce another one.
    messages = [
        Message(id="1", role="user", content="q"),
        Message(id="2", role="assistant", content="the answer is th", partial=True),
    ]
    assert "the answer is th" not in format_history(messages)


def test_empty_messages_are_kept_out_of_history():
    assert format_history([Message(id="1", role="user", content="   ")]) == ""


def test_the_refusal_token_becomes_a_readable_sentence():
    answer = interpret(NOT_IN_DOCUMENT)
    assert answer.text == NO_CONTEXT_ANSWER
    assert answer.refused is True


@pytest.mark.parametrize(
    "raw",
    [
        NOT_IN_DOCUMENT,
        f"Sorry, {NOT_IN_DOCUMENT}.",
        f"**{NOT_IN_DOCUMENT}**",
        f"Answer: {NOT_IN_DOCUMENT}",
    ],
)
def test_the_refusal_token_is_caught_even_when_the_model_pads_it(raw):
    assert interpret(raw).refused is True


@pytest.mark.parametrize(
    "raw",
    [
        # A model this size restates its own instructions constantly. Treating
        # that as a refusal threw away a correct answer and its citations.
        f"The migration finished in March [1]. Otherwise I would reply {NOT_IN_DOCUMENT}.",
        f"The rule says to answer {NOT_IN_DOCUMENT} when the passages do not cover it,"
        " but page 2 does [1].",
    ],
)
def test_an_answer_that_merely_mentions_the_token_is_still_an_answer(raw):
    answer = interpret(raw)
    assert answer.refused is False
    assert answer.text == raw.strip()


def test_an_empty_answer_becomes_the_refusal_rather_than_a_blank_bubble():
    assert interpret("   ") == no_context()


def test_a_real_answer_is_passed_through_stripped():
    answer = interpret("  It finished in March [1].  ")
    assert answer.text == "It finished in March [1]."
    assert answer.refused is False


def test_refusal_is_carried_not_re_derived_from_the_text():
    # The caller must never have to compare against NO_CONTEXT_ANSWER by
    # identity: a stray .strip() downstream would then mark every refusal
    # grounded. The function that took the branch reports it.
    assert interpret(NOT_IN_DOCUMENT).refused is True
    assert no_context().refused is True
