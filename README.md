# rag-knowledge-assistant

Upload one PDF, wait for it to be indexed, then ask questions about it, with
the passages each answer came from shown underneath it. It runs on a **local
chat model under 2B parameters**, a FAISS index on disk, and MongoDB for the
document record and the conversation history. No login, no accounts, no API key.

The intention is a complete, self-hosted RAG application: small enough to run on
a laptop, and built so every answer can be traced back to the text it came
from.

FastAPI, LlamaIndex, FAISS, pypdf and pymongo on the backend; Next.js 15,
Tailwind v4 and hand-rolled shadcn-style components on the front. Inference
through Ollama, with the backend picked automatically — Metal on macOS, CUDA
where there is an NVIDIA GPU, CPU otherwise.

**269 pytest tests and 59 Jest tests**, none of which need a model, a database,
a network or a bound port. Both suites run in CI on every push alongside `ruff
check`, `ruff format --check`, `eslint`, `prettier --check`, `tsc --noEmit` and
a Next.js production build. The same checks run locally through `pre-commit`.

```bash
./start.sh          # picks the fastest inference backend on this machine
```

```
PDF ─► parse ─► chunk (within a page) ─► embed ─► FAISS index ─► on disk
                          │                          │
                          └──── chunk text ──────────┴──► MongoDB
                                                            │
question ─► embed ─► search ─► floor ─► passages ─► prompt ─► answer + citations
                                 │                                 │
                            nothing above it                  MongoDB (history)
                                 │
                        "not in the document" — no model call
```

## A real run

`qwen3:1.7b` and `nomic-embed-text` on host Ollama (Metal), against the bundled
three-page sample, at the shipped defaults:

```
$ python -m rag_assistant.cli data/engineering-notes.pdf \
    "When did the PostgreSQL migration finish and how long was the cutover?"
indexed engineering-notes.pdf: 3 pages, 3 chunks
  [1] page 1  (0.850)  The migration from MySQL to PostgreSQL completed in March 2026 after two false starts. The
  [2] page 3  (0.500)  Deployment runs on three Kubernetes replicas behind an nginx ingress. Rolling updates take
  [3] page 2  (0.492)  Caching is served by Redis with a sixty second time to live. Cache hits account for roughl

The PostgreSQL migration completed in March 2026, and the cutover took forty minutes. [1]

$ python -m rag_assistant.cli data/engineering-notes.pdf "How do ferns photosynthesise?"; echo "exit=$?"
indexed engineering-notes.pdf: 3 pages, 3 chunks
  (nothing cleared the 0.45 similarity floor)

I could not find anything in this document that answers that. Try rephrasing,
or asking about something the document covers.
exit=2
```

The second run is the one worth looking at. The model was never called —
nothing in the document cleared the similarity floor, and there is nothing for
a 1.7B model to do with an empty context except invent something. It exits
**2**, so a calling script cannot read a refusal as an answer.

## The eight decisions worth discussing

### 1. The similarity floor was set by measurement, and the intuitive value was wrong

A vector index always returns its k nearest neighbours. Ask it about ferns and
it returns the four _least unrelated_ passages in the document, with no signal
anywhere that they are useless — and a model handed four passages writes an
answer out of them. So retrieval needs a floor below which a passage is dropped
rather than ranked.

The first version set it at 0.20, on the reasoning that a cosine "well below a
half" is obviously unrelated. That is wrong, and `examples/similarity_floor.py`
shows it. Against the sample document with `nomic-embed-text`, the best-matching
passage for a question the document plainly does not answer scores:

| best cosine | question                              |
| ----------- | ------------------------------------- |
| 0.346       | What is the capital of Peru?          |
| 0.347       | Who wrote Middlemarch?                |
| 0.359       | How do ferns photosynthesise?         |
| 0.364       | What is the boiling point of ethanol? |
| 0.369       | Explain the offside rule in football. |
| 0.394       | Describe the plot of Hamlet.          |

while five questions the document _does_ answer score **0.595 to 0.840**. A
floor at 0.20 admits every single one of the unrelated ones. The two sets are
cleanly separable with a wide empty gap between 0.394 and 0.595, so the default
is **0.45**, and a test asserts it stays inside that gap.

Two things keep this honest. It is eleven questions over one document: enough
to prove 0.20 is wrong, nowhere near enough to call 0.45 optimal. And it is a
property of _this embedding model's_ geometry — the entire distribution moves
when `EMBED_MODEL` changes, which is why it is configuration and why the
example script exists to re-measure it rather than a number to copy.

### 2. Chunks never span a page boundary

The obvious build concatenates the document and splits the result, which
retrieves better and cites worse: a chunk straddling pages 4 and 5 has no page
number, so the citation is either wrong or dropped. Since every answer here
shows the passage it came from, an uncitable chunk is not a chunk.

The cost is real and taken deliberately — a sentence carried across a page
break is cut in two, and that paragraph retrieves worse than it would whole. A
slightly worse retrieval that can be checked beats a slightly better one that
cannot.

### 3. Deletion removes three things, in an order chosen for how it fails

"Start over" has to drop the FAISS index file, the chunk rows and the document
record. Delete two of the three and the next upload silently searches stale
vectors whose ids no longer resolve — the worst kind of bug, because retrieval
still _returns something_.

The order is vectors, then chunks, then the record:

- **Vectors first**: an index with no chunks behind it retrieves ids that
  resolve to nothing, and the answer degrades to "not in the document" — wrong,
  but honest and visible.
- **The reverse** leaves live vectors with no record, and the next upload's
  chunk ids collide with them. Retrieval then returns passages from a document
  the user deleted, attributed to the one they just uploaded.

If a step raises, the record is left in `deleting`, and upload refuses that
state. A half-deleted index must not accept new data. Both paths are tested by
making the middle step throw.

The confirmation is typing the filename, not clicking OK, and **the server
enforces it** (`DELETE /document?confirm=<filename>`) so the guard is not
merely a frontend courtesy.

### 4. Conversations outlive the document, on purpose

Deleting the PDF does not delete the threads. Silently destroying someone's
chat history as a side effect of replacing a file is a surprise nobody asked
for, and the delete dialog says so before you confirm.

That has a consequence the code has to handle: an old thread's citations refer
to chunks that no longer exist. So **citations are stored on the message**, not
recomputed at read time. Recomputing them would show passages from a _different_
document beside an old answer, which is worse than showing none. The sidebar
labels any thread whose document is not the current one.

### 5. The chat route is 409 until a document is ready

The whole app is three states — `empty`, `indexing`, `ready` (plus `failed` and
`deleting`, which exist because both are real) — and the naive build shows the
chat immediately and answers from an empty index. That produces a fluent,
sourceless hallucination on the very first question, indistinguishable to the
reader from a grounded answer.

`require_ready` is one function, called before anything else in `ask`, and it is
a **409 rather than a 400**: the request is fine, the application is in the
wrong state for it. The frontend disables the composer for the same reason, and
the server does not rely on it having done so.

### 6. Progress is a count, because a spinner cannot be told apart from a hang

Indexing on CPU takes minutes. `ingest` takes an `on_progress` callback, writes
the stage and the running count into the document record, and the UI polls it:

```
reading the PDF → splitting 0/3 → embedding 16/48 → embedding 48/48 → ready
```

The pipeline never imports a logger, so the library stays quiet and the caller
decides what to print. Polling stops the moment the status leaves `indexing` —
polling a ready document forever is a request per second for a value that
cannot change.

### 7. The answer that reaches the browser is the server's, not the one it streamed

The model is told to reply `NOT_IN_DOCUMENT` when the passages do not answer the
question, because that is a string a test can assert on and _"I'm sorry, I don't
have enough information"_ is not. It is translated into a readable sentence at
one boundary — but by then the raw token has already been streamed.

So the `done` event carries the persisted message and the browser renders
**that**, not its own accumulation. Otherwise `NOT_IN_DOCUMENT` sits on screen.
The same event carries `grounded`, and a refusal is stored with **no citations**:
passages listed under "I could not find anything" would contradict the answer
above them.

Three more things the streaming path has to get right, each tested:

- **An interrupted stream is stored as `partial=True`**, never dropped and never
  silently completed. A stored half-sentence that looks whole gets fed back as
  history on the next turn and the model takes it as its own prior answer — so
  partial messages are excluded from history, and the UI labels them.
- **A model failure after the first token** is an `error` event inside a 200
  response, because the status line left long ago. The fragment already
  received is stored as partial rather than passed off as the answer.
- **NDJSON, not SSE.** SSE buys reconnection semantics this app cannot use — a
  resumed stream would need the model call to be replayable, and it is not — in
  exchange for framing the client has to unescape. One JSON object per line is
  `json.loads` and nothing else. The client carries an incomplete tail across
  chunk boundaries, which is the bug that otherwise drops a word with no error
  anywhere.

### 8. The container is the slowest place to run the model, and the fallback says so

Docker Desktop on macOS runs a Linux VM with **no Metal passthrough**, so an
ollama container on a Mac has no GPU at all, however much GPU the machine has.
So the bundled ollama sits behind a Compose profile and **the default is the
host's**, via `host.docker.internal`. `./start.sh` picks:

| Detected                                 | Backend                                                  |
| ---------------------------------------- | -------------------------------------------------------- |
| ollama answering on the host             | the host's — Metal on macOS, CUDA on Linux               |
| else an NVIDIA GPU and container runtime | bundled ollama, GPU passed through                       |
| else                                     | bundled ollama on CPU, with the slowness stated up front |

The fallback is loud rather than silent, because a silent fallback is exactly
how a large slowdown goes unnoticed. `decide()` takes the probe results as
arguments instead of running them, so every branch is tested without a Mac, an
NVIDIA card or a running daemon.

`./start.sh offline` runs the whole stack on deterministic fakes — a hashing
embedder and an echoing model, in memory — with no model and no MongoDB at all.
It is reported by `/health`, never silent.

## Why one vector store and not two

The common shape is "FAISS for embeddings, plus Chroma or Qdrant for storage",
which is two systems holding the same vectors and two places for them to
disagree. FAISS _is_ an open-source vector index and it is sufficient here. The
split is by what the data is:

| Where                 | Holds                                             | Why there                  |
| --------------------- | ------------------------------------------------- | -------------------------- |
| FAISS (`index.faiss`) | the float vectors, nothing else                   | it is what does ANN search |
| MongoDB               | chunk text, pages, document status, conversations | it is what does documents  |

FAISS returns integer ids, and those ids **are** the chunk keys in MongoDB.
Nothing is duplicated, so nothing can drift. Ingest checks it rather than
trusting it: the chunk ids are handed out assuming an empty index, and if FAISS
assigns anything else the ingest fails loudly instead of producing citations
that point at the wrong passages.

The index is `IndexFlatIP` over normalised vectors — exact, not approximate.
One PDF is thousands of vectors, where HNSW or IVF costs recall and saves
nothing measurable. Being approximate when you did not need to be is a source
of wrong answers no amount of prompt work recovers.

## Layout

```
backend/
  src/rag_assistant/
    ports.py        the four protocols the pipeline is written against
    models.py       the dataclasses that cross module boundaries
    pdf.py          bytes → pages, with encrypted / scanned / not-a-PDF named
    chunking.py     pages → chunks, one page at a time (LlamaIndex splitter)
    index.py        MemoryIndex (exact, reference) and FaissIndex (persisted)
    memory_store.py in-memory Store — tests, examples, OFFLINE
    mongo_store.py  the MongoDB Store, mapping functions kept pure
    ollama.py       embeddings and streaming chat over urllib, no SDK
    fakes.py        HashEmbedder and EchoChat — deterministic, shipped in src
    ingest.py       PDF → index, and the deletion that undoes it
    retrieve.py     query → passages, with the floor
    prompt.py       prompt assembly, pure strings
    chat.py         prepare (every status-code check) then the event stream
    service.py      wiring, the indexing thread, health
    api.py          FastAPI: translation only
    cli.py          the same pipeline with no browser in the way; keeps nothing
  tests/            269 pytest tests
  examples/         offline_pipeline.py (no network), similarity_floor.py
frontend/           Next.js 15, Tailwind v4, shadcn-style components
  __tests__/        59 Jest tests
docker-compose.yml  mongo + api + web; ollama behind the `bundled` profile
start.sh            picks metal / cuda / cpu / offline, then brings it up
.pre-commit-config.yaml
```

## Design

| Module                       | Knows about                                             |
| ---------------------------- | ------------------------------------------------------- |
| `ingest`, `retrieve`, `chat` | the four protocols in `ports.py` — and nothing else     |
| `index`, `*_store`, `ollama` | one external system each, behind a protocol             |
| `service`                    | which adapters exist; the only place `OFFLINE` branches |
| `api`                        | HTTP. No retrieval logic, no prompts, no state machine  |

`ingest`, `retrieve` and `chat` import no HTTP client, no database driver and no
FAISS. That is why the pipeline tests need no model, no database and no port,
and why the API tests need no model.

The protocols are `Protocol`s rather than base classes deliberately: the fakes
subclass nothing, so a test double cannot inherit a default implementation and
quietly pass a test the real adapter would fail.

## Tests

```
$ cd backend && pytest -q
269 passed in 9.48s

$ cd frontend && npm test
Tests: 59 passed, 59 total
```

Every test runs with no network, no model, no database and no bound port —
except the Ollama client tests, which bind a loopback port and serve real
NDJSON, because monkeypatching `urlopen` would test the mock rather than the
parsing, the streaming and the error translation.

The PDFs are **built byte by byte in the tests** (`tests/pdfs.py`) rather than
checked in. A committed fixture PDF is opaque: when a test fails inside one,
nobody can see why. These have the expected text visible in the test that
builds them, and the encrypted and no-text-layer cases are constructed rather
than found.

`HashEmbedder` is the deterministic fake the retrieval tests assert exact known
values against — two texts sharing every word score exactly 1.0, two sharing
none exactly 0.0. A measurement tool never checked against known truth is
measuring its own bugs.

What is tested is mostly the failure paths: the empty upload, the single-chunk
index, a stored FAISS index whose width no longer matches the embedder, an
overlap larger than the chunk size (which makes the splitter hang rather than
raise), an id in the index with no chunk behind it, a delete that fails halfway,
a stream the client abandons, a model that dies after three tokens, and an
unparseable NDJSON frame arriving mid-answer.

## Running it

```bash
./start.sh                 # auto: host ollama → NVIDIA container → CPU
./start.sh offline         # no model, no MongoDB — fakes, for the UI
./start.sh --down

# web  http://localhost:3000
# api  http://localhost:8000/health
```

Without Docker:

```bash
cd backend && pip install -e ".[dev]"
uvicorn rag_assistant.main:app --reload           # needs MongoDB and Ollama
OFFLINE=true uvicorn rag_assistant.main:app       # needs neither

python examples/offline_pipeline.py               # no network at all
python examples/similarity_floor.py               # needs an embedding model
python -m rag_assistant.cli paper.pdf "what does it say about latency?"
python -m rag_assistant.cli --persist paper.pdf "…"   # use MongoDB and the index file
```

```bash
cd frontend && npm install && npm run dev
```

Hooks:

```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files      # 12 hooks; the same checks CI runs
```

The `pytest` hook needs the backend's dependencies in the environment
pre-commit runs in (`pip install -e "backend[dev]"`).

## Configuration

| Variable                       | Default                     | Note                                                                                            |
| ------------------------------ | --------------------------- | ----------------------------------------------------------------------------------------------- |
| `CHAT_MODEL`                   | `qwen3:1.7b`                | anything under 2B; `llama3.2:1b`, `gemma3:1b`, `smollm2:1.7b` are the alternatives worth trying |
| `EMBED_MODEL`                  | `nomic-embed-text`          | changing it invalidates `MIN_SCORE` — re-measure                                                |
| `MIN_SCORE`                    | `0.45`                      | measured; see decision 1                                                                        |
| `TOP_K`                        | `4`                         | small because the model is small                                                                |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64`                | tokens                                                                                          |
| `OFFLINE`                      | `false`                     | fakes and in-memory store; reported by `/health`                                                |
| `MONGO_URI`                    | `mongodb://localhost:27017` |                                                                                                 |
| `INDEX_PATH`                   | `data/index.faiss`          |                                                                                                 |

## Not included

- **Authentication, authorisation and rate limiting.** There are none, CORS is
  open to `*`, and the API will index and answer anything anyone posts it. That
  is deliberate for a local single-user tool and it is stated in `api.py` so it
  cannot be mistaken for an oversight. Do not put this on a network you do not
  own.
- **More than one document at a time.** One PDF is the whole model, and upload
  refuses while one is indexed rather than overwriting — losing an index by
  picking the wrong file in a dialog is not a convenience. Multiple documents
  need per-document filtering in the index and a corpus picker in the UI.
- **Concurrent writers.** One process, one FAISS index file, one lock. Two API
  replicas against one index file would corrupt it. That is the honest limit of
  a flat file, and the fix is a vector database with a server, not a mutex.
- **OCR.** A scanned PDF is rejected with a message saying it is a scan, rather
  than indexed as an empty string. Reporting a scan as "unreadable" sends people
  hunting for a corruption problem they do not have.
- **Layout-aware PDF extraction.** A two-column PDF will interleave. pypdf reads
  glyph placement, not reading order.
- **Reranking, query rewriting, HyDE, multi-hop.** All would help. None are in
  scope until the plain pipeline is measured, and measuring them properly needs
  the labelled question set below.
- **Retrieval evaluation.** There is no labelled question set here, so no
  recall@k or nDCG is claimed anywhere. The similarity-floor measurement is
  eleven questions over one document and is described as exactly that.
  `08.25.rag-faiss` in this portfolio is where retrieval metrics are done
  properly, with bootstrap intervals.
- **Answer-quality evaluation.** Whether the model's answer is _correct_ given
  correct passages is not measured. `grounded` says a passage cleared the floor
  and the model did not refuse; it does not say the answer is right.
- **Server-side rendering.** The browser calls the API directly, so
  `NEXT_PUBLIC_API_URL` is baked into the client bundle at build time. A real
  deployment would proxy through a route handler.
- **Retries and backpressure against Ollama.** A local single-user daemon needs
  neither; both would be required against a hosted API.

## License

MIT — see [LICENSE](LICENSE).
