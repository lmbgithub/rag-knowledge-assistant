FROM python:3.12-slim

WORKDIR /app

# Dependencies are installed from the metadata alone, before the source is
# copied, so editing a module does not reinstall faiss and torch-sized wheels.
COPY pyproject.toml ./
RUN mkdir -p src/rag_assistant && touch src/rag_assistant/__init__.py README.md \
    && pip install --no-cache-dir . \
    && rm -rf src README.md

COPY src ./src
COPY examples ./examples

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    INDEX_PATH=/data/index.faiss

# The FAISS index is the one piece of state not in MongoDB, so it needs a
# volume of its own or a restart silently loses the document's vectors while
# MongoDB still reports it as ready.
VOLUME ["/data"]
EXPOSE 8000

# /health answers 200 while the model host is still starting, so this reports
# on the API process alone — which is what the dependency in compose needs it
# to mean.
HEALTHCHECK --interval=10s --timeout=5s --retries=12 --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "rag_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]
