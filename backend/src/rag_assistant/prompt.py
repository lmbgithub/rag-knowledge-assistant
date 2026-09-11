"""Prompt assembly. Pure string work, no model, no I/O.

Separate from `chat.py` so the exact bytes sent to the model are asserted in
tests. A prompt built inline inside the streaming call is a prompt nobody ever
reads again.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .models import Message, Retrieved

SYSTEM = """You answer questions using only the numbered passages below.

Rules:
- Use only the passages. Do not use anything you know from training.
- If the passages do not answer the question, reply exactly: NOT_IN_DOCUMENT
- Cite the passages you used as [1], [2] and so on.
- Be brief. Two or three sentences unless asked for more."""

NOT_IN_DOCUMENT = "NOT_IN_DOCUMENT"

NO_CONTEXT_ANSWER = (
    "I could not find anything in this document that answers that. "
    "Try rephrasing, or asking about something the document covers."
)
"""Returned without calling the model at all when retrieval finds nothing.

Sending an empty context and asking the model to refuse works about as often
as a 1B model follows any instruction. Not calling it is free, instant and
correct every time.
"""

MAX_HISTORY_TURNS = 3
"""How many previous exchanges are carried into the prompt.

The context here is a 1B model's, and it is spent on passages first: history
competes with the retrieved text for the same budget. Three turns is enough
for "and what about the second one?" to resolve, and short enough that the
passages still dominate.
"""


def format_passages(hits: Sequence[Retrieved]) -> str:
    return "\n\n".join(
        f"[{i}] (page {hit.chunk.page})\n{hit.chunk.text}" for i, hit in enumerate(hits, start=1)
    )


def format_history(messages: Sequence[Message], *, turns: int = MAX_HISTORY_TURNS) -> str:
    """Recent turns, oldest first, truncated to whole exchanges.

    Partial assistant messages — a stream the user interrupted — are excluded.
    Feeding a sentence that stops mid-word back to the model as its own prior
    answer teaches it to produce another one.
    """

    if turns <= 0:
        # `usable[-0:]` is `usable[:]` — the whole history. Asking for none and
        # getting all of it is the one wrong answer this function can give.
        return ""
    usable = [m for m in messages if not m.partial and m.content.strip()]
    recent = usable[-(turns * 2) :]
    if not recent:
        return ""
    lines = [f"{'User' if m.role == 'user' else 'Assistant'}: {m.content.strip()}" for m in recent]
    return "\n".join(lines)


def build_user_prompt(
    question: str, hits: Sequence[Retrieved], history: Sequence[Message] = ()
) -> str:
    sections = [f"Passages:\n{format_passages(hits)}"]
    conversation = format_history(history)
    if conversation:
        sections.append(f"Earlier in this conversation:\n{conversation}")
    sections.append(f"Question: {question.strip()}")
    return "\n\n".join(sections)


@dataclass(frozen=True)
class Answer:
    """What the model said, and whether it refused.

    `refused` is carried rather than re-derived. The obvious alternative is to
    return only the text and let the caller compare it against
    `NO_CONTEXT_ANSWER` — but that makes a module constant's *identity* load
    bearing, so a stray `.strip()` anywhere downstream silently reclassifies
    every refusal as a grounded answer. The function that takes the branch is
    the only one that knows; it says so here instead of throwing it away.
    """

    text: str
    refused: bool


def interpret(text: str) -> Answer:
    """The model's refusal token becomes the sentence a person should read.

    The token exists because `NOT_IN_DOCUMENT` is a string a test can assert
    on and "I'm sorry, I don't have enough information" is not. It is
    translated here, once, at the boundary — never shown raw.
    """

    stripped = text.strip()
    if not stripped or _is_refusal(stripped):
        return Answer(NO_CONTEXT_ANSWER, refused=True)
    return Answer(stripped, refused=False)


REFUSAL_NOISE = 24
"""How much text may surround the token and still count as a refusal.

Enough for "Sorry, " or "Answer: "; far short of a sentence carrying an actual
claim.
"""


def _is_refusal(text: str) -> bool:
    """The token has to *be* the answer, not merely appear somewhere in it.

    A plain substring test was the first version, and small instruct models
    restate their instructions constantly — a correct, cited answer ending
    "...otherwise I would reply NOT_IN_DOCUMENT" was discarded and its
    citations stripped with it. Requiring the token to be the *entire* answer
    is the opposite error: models wrap it in "Sorry," often enough that real
    refusals would be shown as answers, which is the more dangerous direction.

    So: the token must be on the first line, and what remains of that line once
    the token is removed must be too short to carry a claim.
    """

    head = text.splitlines()[0].upper()
    if NOT_IN_DOCUMENT not in head:
        return False
    remainder = re.sub(r"[^A-Z0-9]+", " ", head.replace(NOT_IN_DOCUMENT, " ")).strip()
    return len(remainder) <= REFUSAL_NOISE


def no_context() -> Answer:
    """The refusal used when retrieval found nothing and the model was never called."""

    return Answer(NO_CONTEXT_ANSWER, refused=True)
