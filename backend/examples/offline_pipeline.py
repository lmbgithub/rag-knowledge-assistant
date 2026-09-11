#!/usr/bin/env python3
"""The whole pipeline, end to end, with no model, no database and no network.

Builds a PDF in memory, indexes it, and asks two questions: one the document
answers and one it does not. The second is the interesting one — a retrieval
system that cannot say "no" is a system whose "yes" means nothing.

    python examples/offline_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from pdfs import simple_pdf  # noqa: E402

from rag_assistant.config import Config  # noqa: E402
from rag_assistant.service import Assistant  # noqa: E402

PAGES = [
    "The migration from MySQL to PostgreSQL completed in March 2026 after two "
    "false starts. The cutover took forty minutes and no rows were lost. "
    "Replication lag peaked at eleven seconds during the switch.",
    "Caching is served by Redis with a sixty second time to live. Cache hits "
    "account for roughly eighty percent of read traffic at peak. The eviction "
    "policy is allkeys-lru and the working set fits in four gigabytes.",
]


def main() -> int:
    assistant = Assistant.build(Config(offline=True, top_k=2, min_score=0.25))

    print("indexing a two-page PDF built in memory…")
    assistant.start_indexing(simple_pdf(*PAGES), "engineering-notes.pdf")
    assistant.wait_for_indexing(timeout=60)

    document = assistant.document()
    assert document is not None
    print(f"  {document.status}: {document.pages} pages, {document.chunks} chunks\n")

    conversation_id = None
    for question in ["When did the PostgreSQL migration finish?", "How do ferns photosynthesise?"]:
        print(f"? {question}")
        for event in assistant.ask(question, conversation_id):
            if event["type"] == "conversation":
                conversation_id = event["conversation"]["id"]
            elif event["type"] == "citations":
                if not event["citations"]:
                    print("  (nothing in the document cleared the similarity floor)")
                for i, citation in enumerate(event["citations"], start=1):
                    head = " ".join(citation["text"].split())[:72]
                    print(f"  [{i}] page {citation['page']}  {citation['score']:.3f}  {head}…")
            elif event["type"] == "done":
                print(f"> {event['message']['content']}")
                print(f"  grounded={event['grounded']}\n")

    conversation = assistant.store.get_conversation(conversation_id)
    assert conversation is not None
    print(f"conversation '{conversation.title}' has {len(conversation.messages)} messages")

    assistant.delete_document()
    print(f"after delete: document={assistant.document()}, vectors={assistant.index.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
