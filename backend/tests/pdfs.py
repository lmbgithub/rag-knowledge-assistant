"""Build small, valid PDFs in memory, so no fixture file is ever needed.

A checked-in PDF is opaque: nobody can see why a test fails inside it. These
are assembled byte by byte from uncompressed content streams, so the text the
extractor is expected to find is visible in the test that builds it.
"""

from __future__ import annotations


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content(lines: list[str]) -> bytes:
    body = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
    for line in lines:
        body.append(f"({_escape(line)}) Tj")
        body.append("T*")
    body.append("ET")
    return "\n".join(body).encode("latin-1")


def make_pdf(pages: list[list[str]], *, encrypted: bool = False) -> bytes:
    """One list of lines per page. `encrypted` fakes only the trailer flag."""

    objects: list[bytes] = []

    def add(raw: bytes) -> int:
        objects.append(raw)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []
    pages_id = len(pages) * 2 + 2  # font + one content and one page object each

    content_ids: list[int] = []
    for lines in pages:
        stream = _content(lines)
        content_ids.append(
            add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        )

    for content_id in content_ids:
        page_ids.append(
            add(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
                % (pages_id, font, content_id)
            )
        )

    kids = b" ".join(b"%d 0 R" % pid for pid in page_ids)
    pages_obj = add(b"<< /Type /Pages /Count %d /Kids [%s] >>" % (len(page_ids), kids))
    assert pages_obj == pages_id, (pages_obj, pages_id)
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id)
    # Added before the byte layout is computed: an object appended afterwards
    # would not appear in the xref table, and pypdf reports that as a missing
    # object rather than as the encryption this test is about.
    encrypt_id = add(b"<< /Filter /Standard /V 1 /R 2 /O <00> /U <00> /P -1 >>") if encrypted else 0

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, raw in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + raw + b"\nendobj\n"

    xref_at = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset

    trailer = b"<< /Size %d /Root %d 0 R" % (len(objects) + 1, catalog)
    if encrypted:
        # Enough for `PdfReader.is_encrypted`; the file is not really encrypted,
        # which is all this test needs — the branch under test is the check.
        trailer += b" /Encrypt %d 0 R" % encrypt_id
    trailer += b" >>"

    out += b"trailer\n" + trailer + b"\nstartxref\n%d\n%%%%EOF\n" % xref_at
    return bytes(out)


def simple_pdf(*paragraphs: str) -> bytes:
    """One page per argument, each wrapped at a readable width."""

    pages = []
    for paragraph in paragraphs:
        words = paragraph.split()
        lines: list[str] = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 > 70:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines.append(current)
        pages.append(lines)
    return make_pdf(pages)
