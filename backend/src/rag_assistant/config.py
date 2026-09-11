"""Configuration from the environment, validated once at startup.

Reading `os.environ` at the point of use spreads defaults across the codebase
and turns a typo in a variable name into a silent fallback. Everything is read
here, coerced here, and rejected here if it is nonsense.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any

from .chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from .ingest import DEFAULT_MAX_BYTES
from .retrieve import DEFAULT_MIN_SCORE, DEFAULT_TOP_K

DEFAULT_CHAT_MODEL = "qwen3:1.7b"
"""Under 2B parameters, and the smallest that follows the grounding rule.

Everything in this class of model is a trade between refusing when it should
and answering when it can. The alternatives worth trying — `llama3.2:1b`,
`gemma3:1b`, `smollm2:1.7b` — are all a CHAT_MODEL away, which is the point of
making it configuration.
"""

DEFAULT_EMBED_MODEL = "nomic-embed-text"


def _default(name: str) -> Any:
    """This class's declared default for one field. The single source."""

    return _FIELD_DEFAULTS[name]


def _int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    ollama_host: str = "http://localhost:11434"
    chat_model: str = DEFAULT_CHAT_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "rag_assistant"
    index_path: str = "data/index.faiss"
    # Imported, not restated. Each number is documented where it is justified —
    # the measured similarity floor in `retrieve`, the chunk geometry in
    # `chunking` — and a second literal here is a value that silently wins over
    # the documented one for everyone running through `from_env`.
    top_k: int = DEFAULT_TOP_K
    min_score: float = DEFAULT_MIN_SCORE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    max_upload_bytes: int = DEFAULT_MAX_BYTES
    timeout: float = 600.0
    temperature: float = 0.0
    num_ctx: int = 4096
    offline: bool = False
    """Serve the deterministic fakes and an in-memory store.

    The whole stack then starts with no model and no database, which is how
    the wiring is exercised in CI and how a reviewer sees the UI in one
    command. It is loud in `/health`, never a silent fallback.
    """

    cors_origins: tuple[str, ...] = ("*",)

    def __post_init__(self) -> None:
        # Validated on the class, not only in `from_env`, so a value that
        # arrives through `dataclasses.replace` (the CLI's flags) fails here
        # rather than deep inside retrieval.
        if self.top_k < 1:
            raise ValueError("TOP_K must be at least 1")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if not -1.0 <= self.min_score <= 1.0:
            raise ValueError("MIN_SCORE is a cosine similarity and must be in [-1, 1]")

    @staticmethod
    def from_env() -> Config:
        """Read the environment, falling back to this class's own defaults.

        `_default("top_k")` rather than a literal: every default previously
        appeared twice — once as the field default a test asserted on, once as
        a literal here, which is what the application actually ran on. They
        could disagree with nothing failing.
        """

        origins = os.environ.get("CORS_ORIGINS", "*")
        return Config(
            ollama_host=os.environ.get("OLLAMA_HOST", _default("ollama_host")).rstrip("/"),
            chat_model=os.environ.get("CHAT_MODEL", _default("chat_model")),
            embed_model=os.environ.get("EMBED_MODEL", _default("embed_model")),
            mongo_uri=os.environ.get("MONGO_URI", _default("mongo_uri")),
            mongo_db=os.environ.get("MONGO_DB", _default("mongo_db")),
            index_path=os.environ.get("INDEX_PATH", _default("index_path")),
            top_k=_int("TOP_K", _default("top_k")),
            min_score=_float("MIN_SCORE", _default("min_score")),
            chunk_size=_int("CHUNK_SIZE", _default("chunk_size"), minimum=64),
            chunk_overlap=_int("CHUNK_OVERLAP", _default("chunk_overlap"), minimum=0),
            max_upload_bytes=_int("MAX_UPLOAD_BYTES", _default("max_upload_bytes"), minimum=1024),
            timeout=_float("OLLAMA_TIMEOUT", _default("timeout")),
            temperature=_float("TEMPERATURE", _default("temperature")),
            num_ctx=_int("NUM_CTX", _default("num_ctx"), minimum=512),
            offline=_bool("OFFLINE"),
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        )


_FIELD_DEFAULTS: dict[str, Any] = {f.name: f.default for f in fields(Config)}
