"""A single-PDF retrieval-augmented chat assistant.

The package is arranged as ports and adapters. `ports.py` declares the four
things the pipeline needs — an embedder, a chat model, a vector index and a
store — as protocols; `ingest`, `retrieve` and `chat` are written against
those protocols and never import FAISS, MongoDB, LlamaIndex or HTTP. That is
what makes the whole pipeline testable with no model, no database, no network
and no ports bound.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
