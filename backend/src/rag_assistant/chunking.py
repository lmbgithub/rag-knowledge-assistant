"""Pages to chunks, using LlamaIndex's sentence splitter — one page at a time.

**Chunks never span a page boundary.** The obvious build concatenates the whole
document and splits the result, which is better for continuity and fatal for
citation: a chunk straddling pages 4 and 5 has no page number, so either the
citation is wrong or it is dropped. Since every answer here shows the passage
it came from, an unciteable chunk is not a chunk.

The cost is real and worth stating: a sentence carried across a page break is
cut in two, and a paragraph split that way retrieves less well than it would
whole. That is the trade, taken deliberately — a slightly worse retrieval that
can be checked beats a slightly better one that cannot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from .models import Chunk, Page

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
"""Small on purpose.

The chat model here is under 2B parameters. Its effective context is far
shorter than its advertised one, and filling it with five 1,500-token chunks
reliably produces an answer that ignores most of them. Small chunks with a
small top-k give the model less to lose track of.
"""


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if self.chunk_overlap >= self.chunk_size:
            # Overlap >= size makes the splitter's window fail to advance.
            raise ValueError("chunk_overlap must be smaller than chunk_size")


@lru_cache(maxsize=8)
def _splitter(chunk_size: int, chunk_overlap: int):  # noqa: ANN202 - LlamaIndex type
    from llama_index.core.node_parser import SentenceSplitter

    return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def split_text(text: str, config: ChunkingConfig | None = None) -> list[str]:
    """One page of text into passages. The only place LlamaIndex is called."""

    config = config or ChunkingConfig()
    cleaned = text.strip()
    if not cleaned:
        return []
    parts = _splitter(config.chunk_size, config.chunk_overlap).split_text(cleaned)
    return [p.strip() for p in parts if p.strip()]


def chunk_pages(pages: Sequence[Page], config: ChunkingConfig | None = None) -> list[Chunk]:
    """Every page's passages, numbered consecutively across the document.

    Ids start at 0 and there is no offset parameter: `ingest` requires an empty
    index and checks that FAISS agreed, so an incremental append is a thing the
    design forbids rather than a thing this function should offer.

    `ordinal` restarts per page — it is the passage's position *on that page*,
    which is what makes "page 7, passage 2" a thing a reader can find.
    """

    config = config or ChunkingConfig()
    chunks: list[Chunk] = []
    next_id = 0
    for page in pages:
        for ordinal, piece in enumerate(split_text(page.text, config)):
            chunks.append(Chunk(id=next_id, text=piece, page=page.number, ordinal=ordinal))
            next_id += 1
    return chunks
