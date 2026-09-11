from __future__ import annotations

import pytest

from rag_assistant.config import Config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in list(__import__("os").environ):
        if name.isupper() and name in {
            "OLLAMA_HOST",
            "CHAT_MODEL",
            "EMBED_MODEL",
            "MONGO_URI",
            "TOP_K",
            "MIN_SCORE",
            "CHUNK_SIZE",
            "CHUNK_OVERLAP",
            "OFFLINE",
            "CORS_ORIGINS",
            "MAX_UPLOAD_BYTES",
            "NUM_CTX",
            "TEMPERATURE",
        }:
            monkeypatch.delenv(name, raising=False)


def test_defaults_are_usable_with_no_environment():
    config = Config.from_env()
    assert config.top_k == 4
    assert config.offline is False
    assert config.chat_model.endswith("b")


def test_the_default_floor_sits_in_the_measured_gap():
    # Measured with nomic-embed-text: unrelated questions top out at 0.394 and
    # answerable ones start at 0.595. The intuitive 0.20 admits every one of
    # the unrelated ones. See examples/similarity_floor.py.
    from rag_assistant.retrieve import DEFAULT_MIN_SCORE

    assert Config().min_score == DEFAULT_MIN_SCORE
    assert 0.394 < DEFAULT_MIN_SCORE < 0.595


def test_the_default_chat_model_is_under_two_billion_parameters():
    # The premise of the project. A regression here changes every latency and
    # quality claim in the README.
    size = float(Config().chat_model.rsplit(":", 1)[1].rstrip("b"))
    assert size < 2.0


def test_a_trailing_slash_on_the_host_is_removed(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434/")
    assert Config.from_env().ollama_host == "http://ollama:11434"


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
def test_offline_is_truthy(monkeypatch, raw):
    monkeypatch.setenv("OFFLINE", raw)
    assert Config.from_env().offline is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "", "maybe"])
def test_everything_else_is_falsy(monkeypatch, raw):
    monkeypatch.setenv("OFFLINE", raw)
    assert Config.from_env().offline is False


def test_a_non_numeric_top_k_is_rejected_at_startup(monkeypatch):
    monkeypatch.setenv("TOP_K", "lots")
    with pytest.raises(ValueError, match="TOP_K"):
        Config.from_env()


def test_an_empty_value_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("TOP_K", "")
    assert Config.from_env().top_k == 4


def test_top_k_below_one_is_rejected(monkeypatch):
    monkeypatch.setenv("TOP_K", "0")
    with pytest.raises(ValueError, match="at least"):
        Config.from_env()


def test_overlap_not_smaller_than_chunk_size_is_rejected():
    # It makes the splitter's window fail to advance — a hang, not an error.
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Config(chunk_size=256, chunk_overlap=256)


@pytest.mark.parametrize("score", [-1.5, 1.5])
def test_min_score_outside_the_cosine_range_is_rejected(score):
    with pytest.raises(ValueError, match="cosine"):
        Config(min_score=score)


def test_cors_origins_are_split_and_stripped(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test ,")
    assert Config.from_env().cors_origins == ("http://a.test", "http://b.test")
