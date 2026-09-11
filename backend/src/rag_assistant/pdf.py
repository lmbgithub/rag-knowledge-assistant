"""PDF bytes to pages of text, with the unusable cases named.

The three ways an upload fails are genuinely different problems for the person
holding the file, so they are three different messages and not one "could not
read the PDF":

  encrypted        re-export it without a password
  no text layer    it is a scan; this app has no OCR
  not a PDF        the magic bytes say otherwise

Reporting a scan as "unreadable" sends people looking for a corruption problem
they do not have, which is the specific failure this module exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import BadRequest, UnprocessableDocument
from .models import Page

PDF_MAGIC = b"%PDF-"

MIN_CHARS_PER_DOC = 32
"""Below this, treat the whole document as having no text layer.

A scanned PDF is not empty — page furniture, a stamped page number or a stray
ligature routinely yields a handful of characters. Indexing those produces an
index that answers every question with the same meaningless chunk, which is
worse than refusing the upload.
"""


@dataclass(frozen=True)
class Extraction:
    pages: tuple[Page, ...]
    total_chars: int


def looks_like_pdf(data: bytes) -> bool:
    """Sniff the magic bytes rather than trust the filename.

    A DOCX renamed to `.pdf` would otherwise reach the parser as gibberish and
    be reported as a corrupt PDF.
    """

    return data[:5] == PDF_MAGIC


def pages_from_texts(texts: list[str]) -> Extraction:
    """Assemble the result from already-extracted text. Pure; no pypdf.

    Split out so every rule above — the character floor, blank-page dropping,
    1-based numbering — is tested without constructing a PDF.
    """

    pages = tuple(
        Page(number=i, text=text.strip()) for i, text in enumerate(texts, start=1) if text.strip()
    )
    return Extraction(pages=pages, total_chars=sum(len(p.text) for p in pages))


def check_extraction(extraction: Extraction, *, filename: str) -> tuple[Page, ...]:
    if extraction.total_chars < MIN_CHARS_PER_DOC:
        raise UnprocessableDocument(
            f"{filename} has no text layer — it looks like a scan. "
            "This app does not do OCR; run the file through an OCR tool first."
        )
    return extraction.pages


def extract(data: bytes, *, filename: str = "document.pdf") -> tuple[Page, ...]:
    """The adapter. Imports pypdf lazily so the pure helpers stay importable."""

    if not looks_like_pdf(data):
        raise BadRequest(f"{filename} is not a PDF — its first bytes are not %PDF-.")

    import io

    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # An empty user password is common and decrypts silently; a real
            # one cannot be guessed, and saying so beats a stack trace.
            if reader.decrypt("") == 0:
                raise UnprocessableDocument(
                    f"{filename} is password-protected. Re-export it without encryption."
                )
        texts = [page.extract_text() or "" for page in reader.pages]
    except UnprocessableDocument:
        raise
    except Exception as exc:
        # Deliberately every exception, not a list of pypdf's. The list was
        # `(PdfReadError, ValueError, OSError)` and missed `DependencyError`,
        # which pypdf raises for an AES-encrypted file when `cryptography` is
        # not installed — a case that then escaped this module's whole "three
        # named failure modes" contract as an uncaught traceback. A parser is
        # exactly the place where the set of things that can go wrong is not
        # knowable in advance; what matters is that none of them reach the
        # caller as anything but "this document is unusable".
        raise UnprocessableDocument(f"{filename} could not be parsed as a PDF: {exc}") from exc

    return check_extraction(pages_from_texts(texts), filename=filename)
