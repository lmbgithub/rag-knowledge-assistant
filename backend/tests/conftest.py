from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from rag_assistant.config import Config  # noqa: E402
from rag_assistant.fakes import EchoChat, HashEmbedder, ScriptedChat  # noqa: E402
from rag_assistant.index import MemoryIndex  # noqa: E402
from rag_assistant.memory_store import MemoryStore  # noqa: E402
from rag_assistant.service import Assistant  # noqa: E402


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=64)


@pytest.fixture
def index(embedder: HashEmbedder) -> MemoryIndex:
    return MemoryIndex(embedder.dim)


@pytest.fixture
def config() -> Config:
    # min_score=0.0 unless a test is specifically about the floor: the hashing
    # fake's geometry is not the real embedder's, and pinning tests to its
    # exact similarities would make them tests of the fake.
    return Config(offline=True, top_k=3, min_score=0.0)


@pytest.fixture
def assistant(config, store, index, embedder) -> Assistant:
    return Assistant(config, store=store, index=index, embedder=embedder, model=EchoChat())


@pytest.fixture
def scripted() -> ScriptedChat:
    return ScriptedChat("The migration finished in March [1].")
