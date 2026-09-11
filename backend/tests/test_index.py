from __future__ import annotations

import math

import pytest

from rag_assistant.index import FaissIndex, MemoryIndex, normalise

faiss = pytest.importorskip("faiss")


@pytest.fixture(params=["memory", "faiss"])
def index(request, tmp_path):
    if request.param == "memory":
        return MemoryIndex(3)
    return FaissIndex(3, tmp_path / "index.faiss")


def test_normalise_returns_a_unit_vector():
    assert math.isclose(sum(x * x for x in normalise([3.0, 4.0])), 1.0)


def test_normalise_leaves_the_zero_vector_alone_rather_than_producing_nan():
    # A broken embedder returns zeros. NaN scores sort arbitrarily, which is a
    # silently wrong answer; a zero score is a visibly wrong one.
    assert normalise([0.0, 0.0]) == [0.0, 0.0]


def test_add_returns_consecutive_ids_from_zero(index):
    assert index.add([[1, 0, 0], [0, 1, 0]]) == [0, 1]
    assert index.add([[0, 0, 1]]) == [2]
    assert index.size == 3


def test_search_ranks_by_cosine(index):
    index.add([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    hits = index.search([1, 0, 0], 3)
    assert hits[0][0] == 0
    assert math.isclose(hits[0][1], 1.0, abs_tol=1e-5)
    assert math.isclose(hits[1][1], 1 / math.sqrt(2), abs_tol=1e-5)


def test_search_is_scale_invariant(index):
    index.add([[1, 0, 0]])
    assert math.isclose(index.search([50, 0, 0], 1)[0][1], 1.0, abs_tol=1e-5)


def test_search_on_an_empty_index_returns_nothing(index):
    assert index.search([1, 0, 0], 5) == []


def test_k_larger_than_the_index_is_not_padded(index):
    index.add([[1, 0, 0]])
    assert len(index.search([1, 0, 0], 10)) == 1


def test_k_of_zero_returns_nothing(index):
    index.add([[1, 0, 0]])
    assert index.search([1, 0, 0], 0) == []


def test_wrong_dimension_is_rejected(index):
    with pytest.raises(ValueError):
        index.add([[1, 0]])


def test_reset_empties_the_index(index):
    index.add([[1, 0, 0]])
    index.reset()
    assert index.size == 0
    assert index.search([1, 0, 0], 1) == []


def test_ids_restart_from_zero_after_a_reset(index):
    index.add([[1, 0, 0], [0, 1, 0]])
    index.reset()
    assert index.add([[0, 0, 1]]) == [0]


def test_memory_index_breaks_ties_on_the_lower_id():
    index = MemoryIndex(3)
    index.add([[1, 0, 0], [1, 0, 0]])
    assert [i for i, _ in index.search([1, 0, 0], 2)] == [0, 1]


def test_faiss_index_survives_a_restart(tmp_path):
    path = tmp_path / "index.faiss"
    FaissIndex(3, path).add([[1, 0, 0], [0, 1, 0]])
    assert FaissIndex(3, path).size == 2


def test_faiss_reset_removes_the_file(tmp_path):
    path = tmp_path / "index.faiss"
    index = FaissIndex(3, path)
    index.add([[1, 0, 0]])
    assert path.exists()
    index.reset()
    assert not path.exists()


def test_a_stored_index_of_the_wrong_width_is_refused(tmp_path):
    # The embedding model changed under a stored index. Searching it anyway
    # compares vectors from two different models.
    path = tmp_path / "index.faiss"
    FaissIndex(3, path).add([[1, 0, 0]])
    with pytest.raises(ValueError, match="dim"):
        FaissIndex(8, path)


def test_adding_nothing_does_not_create_a_file(tmp_path):
    path = tmp_path / "index.faiss"
    assert FaissIndex(3, path).add([]) == []
    assert not path.exists()


def test_faiss_search_and_size_are_safe_against_a_concurrent_reset(tmp_path):
    # Indexing runs on its own thread while /health reads `size` and /chat
    # searches. faiss's IndexFlatIP tolerates neither unsynchronised.
    import threading

    index = FaissIndex(3, tmp_path / "index.faiss")
    index.add([[1, 0, 0]] * 200)
    errors: list[Exception] = []

    def churn() -> None:
        try:
            for _ in range(50):
                index.reset()
                index.add([[1, 0, 0]] * 20)
        except Exception as exc:  # pragma: no cover - the failure under test
            errors.append(exc)

    def read() -> None:
        try:
            for _ in range(200):
                index.search([1, 0, 0], 3)
                _ = index.size
        except Exception as exc:  # pragma: no cover - the failure under test
            errors.append(exc)

    threads = [threading.Thread(target=churn), threading.Thread(target=read)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert errors == []
