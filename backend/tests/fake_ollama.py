"""A stand-in ollama, on a real socket.

The client under test speaks HTTP and NDJSON. Monkeypatching `urlopen` would
test the mock; binding a port tests the parsing, the streaming and the error
translation, and it costs a few milliseconds.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOllama:
    def __init__(self, *, embeddings=None, tokens=None, status=200, body=None, dim=4) -> None:
        self.embeddings = embeddings
        self.tokens = tokens if tokens is not None else ["hello", " world"]
        self.status = status
        self.body = body
        self.dim = dim
        self.requests: list[tuple[str, dict]] = []
        self._server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> FakeOllama:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # keep the test output clean
                pass

            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"version": "0.0.0-test"}).encode())

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append((self.path, payload))

                if outer.status != 200:
                    self.send_response(outer.status)
                    self.end_headers()
                    self.wfile.write(b"upstream said no")
                    return

                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()

                if outer.body is not None:
                    self.wfile.write(outer.body)
                elif self.path.endswith("/api/embed"):
                    inputs = payload.get("input", [])
                    vectors = outer.embeddings or [
                        [float(len(text) % 7)] * outer.dim for text in inputs
                    ]
                    self.wfile.write(json.dumps({"embeddings": vectors}).encode())
                else:
                    for token in outer.tokens:
                        frame = {"message": {"content": token}, "done": False}
                        self.wfile.write((json.dumps(frame) + "\n").encode())
                        self.wfile.flush()
                    self.wfile.write((json.dumps({"done": True, "message": {}}) + "\n").encode())

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        assert self._server is not None
        self._server.shutdown()
        self._server.server_close()
