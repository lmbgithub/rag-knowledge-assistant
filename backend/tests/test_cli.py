from __future__ import annotations

import pytest
from pdfs import simple_pdf

from rag_assistant.cli import main

TEXT = "The migration to PostgreSQL completed in March after two false starts. " * 6


@pytest.fixture
def pdf_path(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(simple_pdf(TEXT))
    return path


def test_indexing_only_reports_the_counts(pdf_path, capsys):
    assert main([str(pdf_path), "--offline"]) == 0
    assert "1 pages" in capsys.readouterr().out


def test_a_question_prints_the_passages_and_the_answer(pdf_path, capsys):
    code = main([str(pdf_path), "--offline", "--min-score", "0", "postgresql", "migration"])
    out = capsys.readouterr().out
    assert code == 0
    assert "page 1" in out
    assert "PostgreSQL" in out


def test_a_refused_answer_exits_non_zero(pdf_path, capsys):
    # A script must not read "I could not find anything" as a successful answer.
    code = main([str(pdf_path), "--offline", "--min-score", "0.9", "photosynthesis in ferns"])
    out = capsys.readouterr().out
    assert code == 2
    assert "could not find" in out
    assert "nothing cleared the 0.90 similarity floor" in out


def test_the_answer_is_not_printed_twice(pdf_path, capsys):
    # It is streamed as tokens and then stored; reprinting the stored copy is
    # only useful when the two differ, which is the refusal rewrite.
    main([str(pdf_path), "--offline", "--min-score", "0", "postgresql migration"])
    out = capsys.readouterr().out
    assert out.count("From the document:") == 1


def test_a_scan_fails_with_a_reason(tmp_path, capsys):
    path = tmp_path / "scan.pdf"
    path.write_bytes(simple_pdf("x"))
    assert main([str(path), "--offline"]) == 1
    assert "no text layer" in capsys.readouterr().err


def test_a_missing_file_is_reported_not_traced(tmp_path, capsys):
    assert main([str(tmp_path / "absent.pdf"), "--offline"]) == 1
    assert "no such file" in capsys.readouterr().err
