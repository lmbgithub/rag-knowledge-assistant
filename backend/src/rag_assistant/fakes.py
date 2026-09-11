"""Deterministic stand-ins for the embedder and the chat model.

These are shipped in `src/`, not in `tests/`, for two reasons: `OFFLINE=true`
serves them so the whole stack can be started and clicked through with no model
installed, and `examples/` runs against them with no network.

`HashEmbedder` is a hashing vectoriser — a real, if crude, embedding: tokens
are hashed into buckets and the vector is the bucket counts. Two texts sharing
words score high, two sharing none score zero, and the same text always
produces the same vector. That last property is the point. A measurement tool
checked only against a random fake is measuring its own bugs, so the retrieval
tests assert exact known values from this.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence

TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in TOKEN.findall(text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self._dim] += 1.0
        return vector

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class EchoChat:
    """Answers with the passages it was given, one token at a time.

    It never adds information, so an answer containing a fact that is not in
    the retrieved passages is proof that retrieval, not the model, is at fault
    — which is exactly what you want a fake to be able to prove.
    """

    def __init__(self, prefix: str = "From the document:") -> None:
        self.prefix = prefix
        self.calls: list[tuple[str, str]] = []

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.calls.append((system, user))
        yield self.prefix
        for word in user.split()[:40]:
            yield " " + word


class ScriptedChat:
    """Yields a fixed script. For asserting on assembly, not on generation."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses) or [""]
        self.calls: list[tuple[str, str]] = []

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.calls.append((system, user))
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        for word in self._responses[index].split(" "):
            yield word + " "


class FailingChat:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def stream(self, system: str, user: str) -> Iterator[str]:
        raise self.error
        yield ""  # pragma: no cover - unreachable, keeps this a generator
