"""Ollama over `urllib`. No client library.

Two endpoints, one of which streams NDJSON. An SDK for that is a dependency,
a version to track and a layer between this code and the wire format it has to
understand anyway when a response arrives malformed.

Every failure that is *not* this code's fault — connection refused, model not
pulled, timeout — becomes `BackendUnavailable`, which the API answers 503. A
model host that is down must never look like a bug in retrieval.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from typing import Any

from .errors import BackendUnavailable

PROBE = "dimension probe"
"""Embedded once at startup to learn the vector width.

Ollama does not report an embedding model's dimension anywhere, and hard-coding
768 for `nomic-embed-text` breaks silently the day someone sets EMBED_MODEL to
something else — silently, because FAISS would be built at the wrong width and
only fail once there was data in it.
"""


def _post(url: str, payload: dict, timeout: float, *, stream: bool = False) -> Any:
    request = urllib.request.Request(  # noqa: S310 - the scheme is fixed by construction
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        # The scheme is fixed by construction: the host comes from configuration
        # and every path here is an http(s) URL built above.
        response = urllib.request.urlopen(request, timeout=timeout)  # noqa: S310
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise BackendUnavailable(f"ollama returned {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BackendUnavailable(f"cannot reach ollama at {url}: {exc}") from exc
    if stream:
        return response
    with response:
        return json.loads(response.read().decode())


class OllamaEmbedder:
    """`/api/embed`, batched, with the dimension discovered once and cached."""

    def __init__(self, host: str, model: str, *, timeout: float = 600.0, batch: int = 32) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.batch = batch
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self._embed([PROBE])[0])
        return self._dim

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        body = _post(
            f"{self.host}/api/embed",
            {"model": self.model, "input": list(texts)},
            self.timeout,
        )
        vectors = body.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise BackendUnavailable(
                f"ollama returned {len(vectors) if isinstance(vectors, list) else 'no'} "
                f"embeddings for {len(texts)} inputs"
            )
        return [[float(x) for x in vector] for vector in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch):
            out.extend(self._embed(texts[start : start + self.batch]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class OllamaChat:
    """`/api/chat` with `stream: true`, yielding content fragments.

    `temperature` defaults to 0. That is reproducibility on one machine, not
    determinism across machines — the same model at 0 answers differently on
    CPU and on Metal, and anything pinned to an exact string will break the
    first time it moves.
    """

    def __init__(
        self,
        host: str,
        model: str,
        *,
        timeout: float = 600.0,
        temperature: float = 0.0,
        num_ctx: int = 4096,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx

    def stream(self, system: str, user: str) -> Iterator[str]:
        response = _post(
            f"{self.host}/api/chat",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": True,
                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
            },
            self.timeout,
            stream=True,
        )
        with response:
            for line in response:
                text = line.decode(errors="replace").strip()
                if not text:
                    continue
                try:
                    event = json.loads(text)
                except json.JSONDecodeError:
                    # One unparseable frame is not a reason to discard a
                    # half-written answer; the stream continues.
                    continue
                if event.get("error"):
                    raise BackendUnavailable(f"ollama: {event['error']}")
                fragment = (event.get("message") or {}).get("content", "")
                if fragment:
                    yield fragment
                if event.get("done"):
                    return


def version(host: str, timeout: float = 3.0) -> str:
    """Used by the health check, and by nothing on the request path."""

    url = f"{host.rstrip('/')}/api/version"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return str(json.loads(response.read().decode()).get("version", "unknown"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise BackendUnavailable(f"cannot reach ollama at {host}: {exc}") from exc
