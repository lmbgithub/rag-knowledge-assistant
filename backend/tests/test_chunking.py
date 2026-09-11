from __future__ import annotations

import pytest

from rag_assistant.chunking import ChunkingConfig, chunk_pages, split_text
from rag_assistant.models import Page

PARAGRAPH = "The migration to PostgreSQL completed in March. " * 60


def test_a_short_page_is_one_chunk():
    assert split_text("A short sentence.") == ["A short sentence."]


def test_blank_text_produces_nothing():
    assert split_text("") == []
    assert split_text("   \n  ") == []


def test_a_long_page_is_split():
    parts = split_text(PARAGRAPH, ChunkingConfig(chunk_size=128, chunk_overlap=16))
    assert len(parts) > 1
    assert all(part.strip() == part for part in parts)


def test_chunks_never_span_a_page_boundary():
    pages = [Page(1, PARAGRAPH), Page(2, "Caching is handled by Redis.")]
    chunks = chunk_pages(pages, ChunkingConfig(chunk_size=128, chunk_overlap=16))
    page_two = [c for c in chunks if c.page == 2]
    assert len(page_two) == 1
    assert page_two[0].text == "Caching is handled by Redis."
    assert "Redis" not in " ".join(c.text for c in chunks if c.page == 1)


def test_ids_are_consecutive_across_pages():
    chunks = chunk_pages(
        [Page(1, PARAGRAPH), Page(2, PARAGRAPH)], ChunkingConfig(chunk_size=128, chunk_overlap=16)
    )
    assert [c.id for c in chunks] == list(range(len(chunks)))


def test_ordinals_restart_on_each_page():
    chunks = chunk_pages(
        [Page(1, PARAGRAPH), Page(2, PARAGRAPH)], ChunkingConfig(chunk_size=128, chunk_overlap=16)
    )
    assert min(c.ordinal for c in chunks if c.page == 2) == 0


def test_empty_pages_contribute_no_chunks():
    assert chunk_pages([Page(1, "   ")]) == []


def test_no_pages_is_not_an_error():
    assert chunk_pages([]) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_size": -1},
        {"chunk_overlap": -1},
        {"chunk_size": 64, "chunk_overlap": 64},
        {"chunk_size": 64, "chunk_overlap": 100},
    ],
)
def test_degenerate_configuration_is_rejected_at_construction(kwargs):
    # Overlap >= size makes the splitter's window fail to advance, which shows
    # up as a hang rather than an error if it is not caught here.
    with pytest.raises(ValueError):
        ChunkingConfig(**kwargs)


def test_chunking_is_deterministic():
    pages = [Page(1, PARAGRAPH)]
    config = ChunkingConfig(chunk_size=128, chunk_overlap=16)
    assert chunk_pages(pages, config) == chunk_pages(pages, config)
