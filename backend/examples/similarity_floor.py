#!/usr/bin/env python3
"""Where to put MIN_SCORE, measured rather than guessed.

Embeds a small document and two sets of questions — ones it answers and ones
it plainly does not — and prints the best-matching passage's cosine for each.
The gap between the two sets is where the floor belongs.

This one needs a real embedding model, because the number it produces is a
property of that model's geometry and of nothing else:

    OLLAMA_HOST=http://localhost:11434 python examples/similarity_floor.py

Run it again after changing EMBED_MODEL. The floor does not transfer.
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_assistant.errors import BackendUnavailable  # noqa: E402
from rag_assistant.index import MemoryIndex  # noqa: E402
from rag_assistant.ollama import OllamaEmbedder  # noqa: E402

PASSAGES = [
    "The migration from MySQL to PostgreSQL completed in March 2026 after two false "
    "starts. The cutover took forty minutes and no rows were lost. Replication lag "
    "peaked at eleven seconds during the switch. The rollback plan was never used.",
    "Caching is served by Redis with a sixty second time to live. Cache hits account "
    "for roughly eighty percent of read traffic at peak. The eviction policy is "
    "allkeys-lru and the working set fits in four gigabytes.",
    "Deployment runs on three Kubernetes replicas behind an nginx ingress. Rolling "
    "updates take about ninety seconds. The readiness probe checks the database "
    "connection, not just the process.",
]

ANSWERED = [
    "When did the PostgreSQL migration finish?",
    "What is the cache eviction policy?",
    "How many Kubernetes replicas are there?",
    "How long did the cutover take?",
    "What percentage of reads hit the cache?",
]

UNANSWERED = [
    "How do ferns photosynthesise?",
    "What is the capital of Peru?",
    "Explain the offside rule in football.",
    "Who wrote Middlemarch?",
    "What is the boiling point of ethanol?",
    "Describe the plot of Hamlet.",
]


def main() -> int:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("EMBED_MODEL", "nomic-embed-text")
    embedder = OllamaEmbedder(host, model)

    # The same index the application searches, not a second hand-rolled cosine.
    # A measurement produced by a different implementation than the one serving
    # queries stops describing the system the moment either one changes.
    index = MemoryIndex(embedder.dim)
    try:
        index.add(embedder.embed_documents(PASSAGES))
    except BackendUnavailable as exc:
        print(f"{exc.message}\nStart ollama, or run examples/offline_pipeline.py instead.")
        return 1

    def best(question: str) -> float:
        return index.search(embedder.embed_query(question), 1)[0][1]

    print(f"model: {model}   passages: {len(PASSAGES)}\n")
    answered = [best(q) for q in ANSWERED]
    unanswered = [best(q) for q in UNANSWERED]

    for label, questions, scores in [
        ("answered by the document", ANSWERED, answered),
        ("not answered by it", UNANSWERED, unanswered),
    ]:
        print(f"{label}:")
        for question, score in sorted(zip(questions, scores, strict=True), key=lambda p: p[1]):
            print(f"  {score:.3f}  {question}")
        print(f"  mean {statistics.mean(scores):.3f}\n")

    low, high = min(answered), max(unanswered)
    print(f"lowest answered   {low:.3f}")
    print(f"highest unanswered {high:.3f}")
    if low > high:
        print(f"→ the sets are separable; a floor at {(low + high) / 2:.2f} sits in the gap")
    else:
        # Worth saying out loud: with some embedders they overlap, and then no
        # single threshold separates them and the refusal has to come from
        # somewhere else.
        print("→ the sets OVERLAP: no single floor separates them for this model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
