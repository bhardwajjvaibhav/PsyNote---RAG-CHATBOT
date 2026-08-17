"""
services/parser.py

Step 1 of the ingestion flow (architecture doc, Section 3): turn an
uploaded journal file (.txt / .pdf / .csv) into plain text, before it
ever reaches the chunker.

Design note -- dispatch by extension, not by sniffing content: the
frontend upload form (Section 2.5) already restricts the file input to
these three extensions, and doc_registry.create_note stores the
original filename. Trusting the extension keeps this function simple
and predictable; if a mismatched/corrupt file slips through, the
underlying library raises and the caller (ingestion pipeline) catches
it and calls doc_registry.mark_failed, same pattern as every other
write in this codebase.

Design note -- lazy imports: same shape as llm_client.py's `requests`
import and reranker.py's `sentence_transformers` import. pypdf is only
imported the first time a PDF is actually parsed, so importing this
module (e.g. from a test that only exercises .txt/.csv) never requires
pypdf to be installed.

Output contract: every parser function returns a single plain-text
string for the whole file. Page/row boundaries are preserved as blank
lines so the chunker's overlap logic still reads naturally across them,
but no page/row metadata survives this step -- if that's ever needed
downstream, it should be added deliberately, not smuggled in here.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".csv"}


class ParseError(Exception):
    """Raised when a file can't be parsed into text at all."""


def _parse_txt(raw_bytes: bytes) -> str:
    # Journal exports are assumed UTF-8; fall back to latin-1 (never
    # raises) rather than crash ingestion over an encoding quirk.
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1")


def _parse_pdf(raw_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ParseError(
            "pypdf is not installed -- required to parse .pdf uploads"
        ) from e

    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as e:
        raise ParseError(f"could not read PDF: {e}") from e

    return "\n\n".join(pages)


def _parse_csv(raw_bytes: bytes) -> str:
    try:
        text = _parse_txt(raw_bytes)
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
    except Exception as e:
        raise ParseError(f"could not read CSV: {e}") from e

    if not rows:
        return ""

    # Render each row as "header: value, header: value" when a header
    # row is present, so structured session-log CSVs (date, mood score,
    # note text columns, etc.) stay readable as prose for the chunker
    # and, eventually, the LLM -- a raw comma-joined row loses that.
    header, *body = rows
    if not body:
        return ", ".join(header)

    lines = []
    for row in body:
        if len(row) == len(header):
            lines.append(", ".join(f"{h.strip()}: {v.strip()}" for h, v in zip(header, row)))
        else:
            # Malformed row (wrong column count) -- keep it, raw, rather
            # than silently dropping a clinician's data.
            lines.append(", ".join(row))
    return "\n".join(lines)


_PARSERS = {
    ".txt": _parse_txt,
    ".pdf": _parse_pdf,
    ".csv": _parse_csv,
}


def parse_file(filename: str, raw_bytes: bytes) -> str:
    """
    Parse an uploaded file's raw bytes into plain text, dispatching on
    the file extension.

    Raises ParseError for unsupported extensions or unparseable content
    -- the caller (ingestion pipeline) is expected to catch this and
    call doc_registry.mark_failed(note_id, str(err)).
    """
    ext = Path(filename).suffix.lower()
    parser_fn = _PARSERS.get(ext)
    if parser_fn is None:
        raise ParseError(
            f"unsupported file type {ext!r} -- expected one of {sorted(SUPPORTED_EXTENSIONS)}"
        )

    text = parser_fn(raw_bytes)
    text = text.strip()
    if not text:
        raise ParseError(f"{filename} parsed to empty text -- nothing to ingest")
    return text


# --- Quick self-test ----------------------------------------------------------
# Run this file directly: python parser.py

if __name__ == "__main__":
    print("=== .txt parsing ===")
    txt_bytes = "Patient reports improved mood this week.\nSlept 7 hours nightly.".encode("utf-8")
    result = parse_file("session_1.txt", txt_bytes)
    print(result)
    assert "improved mood" in result

    print("\n=== .txt latin-1 fallback ===")
    latin1_bytes = "Café session notes, mood improving".encode("latin-1")
    result_latin = parse_file("session_2.txt", latin1_bytes)
    print(result_latin)
    assert "session notes" in result_latin

    print("\n=== .csv with header row ===")
    csv_text = "date,mood_score,note\n2026-01-10,7,Slept well\n2026-01-17,4,Poor sleep this week"
    result_csv = parse_file("mood_log.csv", csv_text.encode("utf-8"))
    print(result_csv)
    assert "mood_score: 7" in result_csv
    assert "note: Poor sleep this week" in result_csv

    print("\n=== .csv single-row (no body) falls back to raw join ===")
    result_csv_single = parse_file("header_only.csv", b"date,mood_score,note")
    print(result_csv_single)

    print("\n=== unsupported extension raises ParseError ===")
    try:
        parse_file("notes.docx", b"whatever")
        print("FAILED: should have raised ParseError")
    except ParseError as e:
        print(f"OK, raised: {e}")

    print("\n=== empty file raises ParseError ===")
    try:
        parse_file("empty.txt", b"   \n  ")
        print("FAILED: should have raised ParseError")
    except ParseError as e:
        print(f"OK, raised: {e}")

    print("\nSelf-test passed (pdf parsing requires pypdf + a real PDF, not exercised here).")