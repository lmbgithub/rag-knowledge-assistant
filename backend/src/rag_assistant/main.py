"""`uvicorn rag_assistant.main:app`."""

from __future__ import annotations

from .api import create_app
from .config import Config

app = create_app(config=Config.from_env())
