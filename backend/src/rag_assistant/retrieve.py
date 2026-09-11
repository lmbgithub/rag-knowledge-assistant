"""Query to passages.

Two rules that the naive version gets wrong, both about *not* returning things.
"""

from __future__ import annotations

from .errors import BadRequest
from .models import Retrieved
from .ports import Embedder, Store, VectorIndex

DEFAULT_TOP_K = 4
"""Small, because the model is small.

Retrieval quality is not the binding constraint at 1B parameters — attention
over the prompt is. Ten passages measurably produce a worse answer than four,
by giving the model more to average over.
"""

DEFAULT_MIN_SCORE = 0.45
"""Below this, a passage is dropped rather than ranked.

A vector index always returns its k nearest neighbours. Asked a question the
document has nothing to say about, it returns the k *least unrelated*
passages, and a model handed passages will write an answer out of them. The
floor is what turns that into "not in the document".

The value is measured, not guessed, and the first guess was wrong. Against the
three-page sample document with `nomic-embed-text`, the best-matching passage
for a question the document plainly does not answer scores:

    0.346  What is the capital of Peru?
    0.347  Who wrote Middlemarch?
    0.359  How do ferns photosynthesise?
    0.364  What is the boiling point of ethanol?
    0.369  Explain the offside rule in football.
    0.394  Describe the plot of Hamlet.

while five questions the document does answer score 0.595 to 0.840. A floor of
0.20 — the intuitive "well below a half" — admits every one of those. The gap
between 0.394 and 0.595 is wide and empty, so 0.45 sits in the middle of it.

Two caveats that keep this honest. It is eleven questions over one document,
which is enough to show that 0.20 is wrong and not enough to call 0.45 optimal.
And it is a property of *this embedding model's* geometry: change EMBED_MODEL
and the whole distribution moves, which is why it is configuration rather than
a literal buried in the retrieval code. `examples/similarity_floor.py`
reproduces the table.
"""


def retrieve(
    question: str,
    *,
    store: Store,
    index: VectorIndex,
    embedder: Embedder,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[Retrieved]:
    question = question.strip()
    if not question:
        raise BadRequest("The question is empty.")
    if top_k <= 0:
        raise BadRequest("top_k must be at least 1.")

    hits = index.search(embedder.embed_query(question), top_k)
    kept = [(chunk_id, score) for chunk_id, score in hits if score >= min_score]
    if not kept:
        return []

    chunks = {c.id: c for c in store.get_chunks([chunk_id for chunk_id, _ in kept])}
    # An id with no chunk behind it means the index and the store disagree.
    # It is dropped, never padded with a placeholder: an empty passage in the
    # prompt is an invitation for the model to fill the gap itself.
    return [Retrieved(chunk=chunks[i], score=score) for i, score in kept if i in chunks]
