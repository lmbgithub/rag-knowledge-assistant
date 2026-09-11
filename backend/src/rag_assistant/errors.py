"""Errors that map onto HTTP status codes, and nothing else.

Each carries the status the API should return, so `api.py` translates by
reading an attribute rather than by maintaining a second table of exception
types that drifts out of step with this one.
"""

from __future__ import annotations


class RagError(Exception):
    """Base class. `status` is the HTTP code the API answers with."""

    status = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class BadRequest(RagError):
    status = 400


class NotFound(RagError):
    status = 404


class Conflict(RagError):
    """The request is well-formed but the app is in the wrong state for it.

    Asking a question before a document is indexed is the case that matters:
    answering it anyway would produce a fluent, sourceless answer, which is
    the single worst failure a retrieval system has.
    """

    status = 409


class PayloadTooLarge(RagError):
    status = 413


class UnprocessableDocument(RagError):
    """The upload is a PDF but nothing can be retrieved from it.

    An encrypted file or a scan with no text layer lands here. It is 422 and
    not 400: the request was fine, the document is not usable.
    """

    status = 422


class BackendUnavailable(RagError):
    """The model host is not answering. Distinct from every bug in this code."""

    status = 503
