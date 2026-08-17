"""
services/chunker.py

Step 2 of the ingestion flow (architecture doc, Section 3): split
parsed note text into overlapping chunks -- 500 tokens, 50 token
overlap (fix carried from v2: this is a TOKEN unit, not characters --
see doc Section 3).

Design note -- what counts as a "token" here: the embedding model
(all-MiniLM-L6-v2) and the generation models (gpt-4o-mini /
claude-3.5-sonnet, Section 6) each have their own subword tokenizers,
and none of them is loaded by this module. Rather than pull in a
model-specific tokenizer here (which would quietly couple the chunker
to one specific downstream model), this module uses a simple,
dependency-free whitespace tokenizer as a deterministic proxy for
"token" -- same pragmatic choice bm25_index.py makes for its own
tokenizer. Whitespace-token counts and subword-token counts aren't
identical, but they track closely enough for chunk-sizing purposes,
and the `tokenize`/`detokenize` pair is injectable so a real
model-specific tokenizer can be swapped in later without touching the
windowing logic below.

Design note -- overlap direction: overlap is measured in tokens carried
over from the END of chunk N into the START of chunk N+1, so a sentence
split across a chunk boundary still has its context on both sides
without the reader falling out of view.
"""

from __future__ import annotations

import re
import uuid

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

_WHITESPACE_TOKEN_RE = re.compile(r"\S+")


def _default_tokenize(text: str) -> list[str]:
    """Whitespace tokenizer -- see module docstring for why this is the default."""
    return _WHITESPACE_TOKEN_RE.findall(text)


def _default_detokenize(tokens: list[str]) -> str:
    return " ".join(tokens)


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    tokenize=None,
    detokenize=None,
) -> list[str]:
    """
    Split `text` into overlapping chunks of `chunk_size` tokens, each
    consecutive pair sharing `chunk_overlap` tokens.

    Pure function: no chunk_ids, no note_id, no I/O -- just text in,
    list of chunk texts out. Step 2 (below) adds the identifiers that
    doc_registry/vector_store/bm25_index all need.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size, or chunks never advance")

    tokenize_fn = tokenize or _default_tokenize
    detokenize_fn = detokenize or _default_detokenize

    tokens = tokenize_fn(text)
    if not tokens:
        return []

    stride = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        window = tokens[start : start + chunk_size]
        chunks.append(detokenize_fn(window))
        if start + chunk_size >= len(tokens):
            break
        start += stride
    return chunks


def chunk_note(
    note_id: str,
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    tokenize=None,
    detokenize=None,
) -> list[dict]:
    """
    Chunk a note's text and attach the identifiers every downstream
    write needs (vector_store.upsert_chunks, bm25_index.upsert_chunks
    both take parallel chunk_ids/chunk_texts lists keyed this way).

    chunk_id format: "{note_id}::chunk{n}" -- deterministic and
    human-debuggable rather than a fresh uuid per chunk, so re-chunking
    the same note_id with the same parameters reliably produces the
    same ids (upsert overwrites cleanly instead of accumulating stale
    chunks from a previous chunking pass under different ids).
    """
    if not note_id or not note_id.strip():
        raise ValueError("note_id cannot be empty")

    texts = chunk_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenize=tokenize,
        detokenize=detokenize,
    )
    return [
        {"chunk_id": f"{note_id}::chunk{i}", "note_id": note_id, "text": chunk}
        for i, chunk in enumerate(texts)
    ]


# --- Quick self-test ----------------------------------------------------------
# Run this file directly: python chunker.py

if __name__ == "__main__":
    print("=== short text -> single chunk ===")
    short = "Patient reports improved mood this week."
    result = chunk_text(short, chunk_size=500, chunk_overlap=50)
    print(result)
    assert len(result) == 1
    assert result[0] == short

    print("\n=== long text -> multiple overlapping chunks ===")
    words = [f"word{i}" for i in range(1200)]
    long_text = " ".join(words)
    chunks = chunk_text(long_text, chunk_size=500, chunk_overlap=50)
    print(f"{len(chunks)} chunks produced")
    assert len(chunks) == 3  # tokens 0-499, 450-949, 900-1199
    # verify overlap: last 50 tokens of chunk 0 == first 50 tokens of chunk 1
    chunk0_tokens = chunks[0].split()
    chunk1_tokens = chunks[1].split()
    assert chunk0_tokens[-50:] == chunk1_tokens[:50]
    print("OK, 50-token overlap verified between consecutive chunks.")

    print("\n=== full coverage: every token appears in at least one chunk ===")
    covered = set()
    for c in chunks:
        covered.update(c.split())
    assert covered == set(words)
    print("OK, no token dropped.")

    print("\n=== chunk_note attaches note_id-based chunk_ids ===")
    note_chunks = chunk_note("note_abc123", long_text, chunk_size=500, chunk_overlap=50)
    for nc in note_chunks:
        print(nc["chunk_id"], "-", len(nc["text"].split()), "tokens")
    assert [nc["chunk_id"] for nc in note_chunks] == [
        "note_abc123::chunk0", "note_abc123::chunk1", "note_abc123::chunk2"
    ]
    assert all(nc["note_id"] == "note_abc123" for nc in note_chunks)

    print("\n=== empty text -> no chunks ===")
    assert chunk_text("", chunk_size=500, chunk_overlap=50) == []
    assert chunk_note("note_x", "   ") == []
    print("OK, empty in -> empty out.")

    print("\n=== chunk_overlap >= chunk_size rejected ===")
    try:
        chunk_text("some text", chunk_size=10, chunk_overlap=10)
        print("FAILED: should have raised ValueError")
    except ValueError as e:
        print(f"OK, raised: {e}")

    print("\n=== injected tokenizer (char-based, for testing DI) ===")
    char_tokens = chunk_text(
        "abcdefghij",
        chunk_size=4,
        chunk_overlap=1,
        tokenize=list,
        detokenize=lambda toks: "".join(toks),
    )
    print(char_tokens)
    assert char_tokens == ["abcd", "defg", "ghij"]

    print("\nSelf-test passed.")