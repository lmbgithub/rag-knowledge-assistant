"""A terminal client for the same pipeline the API serves.

Its purpose is that the pipeline can be exercised, and a bug bisected, without
a browser, a database or a container in the way. `--offline` runs the whole
thing on the deterministic fakes with no model at all.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .config import Config
from .errors import RagError
from .service import Assistant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-assistant", description=__doc__)
    parser.add_argument("pdf", type=Path, help="the PDF to index")
    parser.add_argument("question", nargs="*", help="the question to ask")
    parser.add_argument("--offline", action="store_true", help="fake model and in-memory store")
    parser.add_argument(
        "--persist",
        action="store_true",
        help="use the configured MongoDB and index file instead of keeping nothing",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    base = Config.from_env()
    overrides: dict[str, object] = {}
    if args.offline:
        overrides["offline"] = True
    if args.top_k is not None:
        # `is not None`, not truthiness: `--top-k 0` is a mistake worth
        # reporting, and `or base.top_k` silently ignored it.
        overrides["top_k"] = args.top_k
    if args.min_score is not None:
        overrides["min_score"] = args.min_score

    try:
        config = replace(base, **overrides)
    except ValueError as exc:
        print(f"bad options: {exc}", file=sys.stderr)
        return 1

    try:
        assistant = Assistant.build(config, persist=args.persist)
        assistant.start_indexing(args.pdf.read_bytes(), args.pdf.name)
        assistant.wait_for_indexing(timeout=config.timeout)
        document = assistant.document()
        if document is None or document.status != "ready":
            detail = document.error if document else "no document"
            print(f"indexing failed: {detail}", file=sys.stderr)
            return 1
        print(f"indexed {document.filename}: {document.pages} pages, {document.chunks} chunks")

        if not args.question:
            return 0

        conversation_id = None
        streamed: list[str] = []
        for event in assistant.ask(" ".join(args.question), conversation_id):
            kind = event["type"]
            if kind == "citations":
                if not event["citations"]:
                    print(f"  (nothing cleared the {config.min_score:.2f} similarity floor)")
                for i, citation in enumerate(event["citations"], start=1):
                    head = " ".join(citation["text"].split())[:90]
                    print(f"  [{i}] page {citation['page']}  ({citation['score']:.3f})  {head}")
                print()
            elif kind == "token":
                streamed.append(event["text"])
                sys.stdout.write(event["text"])
                sys.stdout.flush()
            elif kind == "done":
                # The stored message is authoritative and usually identical to
                # what was streamed. It differs when the refusal token was
                # rewritten, and only then is it worth printing again.
                content = event["message"]["content"]
                if content != "".join(streamed).strip():
                    print(content)
                else:
                    print()
                # An ungrounded answer exits non-zero so a script cannot read
                # a refusal as a successful answer.
                return 0 if event.get("grounded") else 2
            elif kind == "error":
                print(f"\nerror: {event['detail']}", file=sys.stderr)
                return 1
        return 0
    except RagError as exc:
        print(f"{type(exc).__name__}: {exc.message}", file=sys.stderr)
        return 1
    except FileNotFoundError:
        print(f"no such file: {args.pdf}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
