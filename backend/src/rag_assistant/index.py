"""Vector indexes: an in-memory one for tests, and FAISS for real use.

Both normalise vectors and use inner product, so `search` returns cosine
similarity in [-1, 1] and the two are interchangeable in a test. The
brute-force one is not a toy: with one PDF's worth of chunks it is exact and
fast, and it is what `OFFLINE=true` serves so the whole stack can come up with
no native wheel installed.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Sequence
from pathlib import Path

from .ports import Vector


def check_dim(expected: int, vector: Vector) -> None:
    """One message for both implementations — `test_index.py` asserts it for each."""

    if len(vector) != expected:
        raise ValueError(f"expected dim {expected}, got {len(vector)}")


def normalise(vector: Vector) -> list[float]:
    """Unit-length, with the zero vector returned unchanged rather than NaN.

    A zero vector is what a broken embedder returns, and dividing by its norm
    turns that into NaN scores that sort arbitrarily — a silent wrong answer
    instead of a visible zero one.
    """

    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    if norm == 0.0:
        return [0.0] * len(vector)
    return [float(x) / norm for x in vector]


class MemoryIndex:
    """Exact brute-force cosine search. The reference implementation."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors: list[list[float]] = []
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._vectors)

    def add(self, vectors: Sequence[Vector]) -> list[int]:
        with self._lock:
            start = len(self._vectors)
            for vector in vectors:
                check_dim(self.dim, vector)
                self._vectors.append(normalise(vector))
            return list(range(start, len(self._vectors)))

    def search(self, vector: Vector, k: int) -> list[tuple[int, float]]:
        if k <= 0 or not self._vectors:
            return []
        query = normalise(vector)
        scored = [
            (i, sum(a * b for a, b in zip(query, row, strict=True)))
            for i, row in enumerate(self._vectors)
        ]
        # Ties break on the lower id so the ordering is reproducible; an
        # unstable order here would make a retrieval regression untestable.
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]

    def reset(self) -> None:
        with self._lock:
            self._vectors = []


class FaissIndex:
    """`IndexFlatIP` over normalised vectors, persisted to a single file.

    Flat and not HNSW or IVF: one PDF is thousands of vectors, where an
    approximate index costs recall and saves nothing measurable. An index that
    is *approximate* when it did not need to be is a source of wrong answers
    that no amount of prompt work recovers.

    One process, one writer. Every mutation is under a lock and followed by a
    write, because the alternative — persisting on shutdown — loses the index
    on the one exit path nobody tests, `kill -9`.
    """

    def __init__(self, dim: int, path: str | os.PathLike[str]) -> None:
        import faiss

        self.dim = dim
        self.path = Path(path)
        self._faiss = faiss
        self._lock = threading.Lock()
        self._index = self._load_or_create()

    def _load_or_create(self):  # noqa: ANN202 - faiss type
        if self.path.exists():
            index = self._faiss.read_index(str(self.path))
            if index.d != self.dim:
                # The embedding model changed under a stored index. Silently
                # searching it would raise deep inside FAISS at query time, or
                # worse, compare vectors from two different models.
                raise ValueError(
                    f"index at {self.path} has dim {index.d}, embedder reports {self.dim}. "
                    "Delete the document and re-index."
                )
            return index
        return self._faiss.IndexFlatIP(self.dim)

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self._faiss.write_index(self._index, str(tmp))
        tmp.replace(self.path)  # atomic: a crash mid-write leaves the old index

    @property
    def size(self) -> int:
        # Under the lock like the writers: `reset()` rebinds `_index`, and
        # `/health` reads this from a different thread than the one indexing.
        with self._lock:
            return int(self._index.ntotal)

    def add(self, vectors: Sequence[Vector]) -> list[int]:
        with self._lock:
            for vector in vectors:
                check_dim(self.dim, vector)
            rows = [normalise(v) for v in vectors]
            start = int(self._index.ntotal)
            if rows:
                self._index.add(_as_array(rows))
                self._write()
            return list(range(start, start + len(rows)))

    def search(self, vector: Vector, k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        # faiss's IndexFlatIP is not safe for a concurrent add/search, and
        # indexing runs on its own thread. Reading `_index` outside the lock
        # also let a `reset()` swap it mid-query.
        with self._lock:
            total = int(self._index.ntotal)
            if total == 0:
                return []
            scores, ids = self._index.search(_as_array([normalise(vector)]), min(k, total))
        # FAISS pads with -1 when it has fewer than k vectors.
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if int(i) >= 0]

    def reset(self) -> None:
        with self._lock:
            self._index = self._faiss.IndexFlatIP(self.dim)
            self.path.unlink(missing_ok=True)


def _as_array(rows: Sequence[Sequence[float]]):  # noqa: ANN202 - numpy type
    import numpy as np

    return np.asarray(rows, dtype="float32")
