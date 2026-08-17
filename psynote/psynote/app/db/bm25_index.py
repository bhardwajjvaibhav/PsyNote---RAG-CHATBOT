"""
db/bm25_index.py

Lexical (exact keyword) retrieval index using rank-bm25, filtered by
patient_id (architecture doc, Section 1 and 6).

Why this exists alongside Chroma, not instead of it: dense embeddings
are good at semantic similarity but can miss exact terms a clinician
searches for (a specific medication name, a specific date, a specific
phrase from a note). BM25 catches those exact-match cases that
embedding similarity alone can blur past. Fusion (fusion.py, Phase 5)
combines both signals via RRF.

Persistence design note: rank-bm25's BM25Okapi is an in-memory,
non-incremental structure -- it computes IDF over the whole corpus at
construction time, so it can't be "updated" in place the way Chroma
handles upsert. To still get real persistence and patient isolation
(matching vector_store.py's per-patient-collection approach, Section 1),
this module persists each patient's raw chunk corpus to its own JSON
file on disk, and rebuilds a fresh BM25Okapi index from that corpus at
query time. For per-patient note volumes this is cheap; if that stops
being true, Phase 11 (compressor.py) or a caching layer would be the
place to revisit it -- not this module today.

Same isolation guarantee as vector_store.py: one corpus file per
patient means there is no shared-index "forgot the filter" failure
mode. A query against patient A's index can only ever return patient
A's chunks.
"""

from __future__ import annotations

from pathlib import Path
import json
import re

from rank_bm25 import BM25Okapi

# Parallel to db/chroma_store/ and db/psynote.db -- same open question
# on backup/ownership (architecture doc, Section 9).
PERSIST_DIR = Path(__file__).parent / "bm25_store"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Simple, deterministic tokenizer: lowercase alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def _corpus_path(patient_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", patient_id)
    return PERSIST_DIR / f"{safe}.json"


def _load_corpus(patient_id: str) -> dict[str, dict]:
    """
    Returns {chunk_id: {"text": ..., "note_id": ...}} for this patient.
    Empty dict if the patient has no corpus file yet (no chunks indexed).
    """
    path = _corpus_path(patient_id)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_corpus(patient_id: str, corpus: dict[str, dict]) -> None:
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    path = _corpus_path(patient_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(corpus, f)


# --- Step 1: write path (called from ingestion, alongside vector_store.upsert_chunks) -

def upsert_chunks(
    patient_id: str,
    note_id: str,
    chunk_ids: list[str],
    chunk_texts: list[str],
) -> None:
    """
    Upsert a note's chunks into this patient's lexical corpus.
    Same upsert-by-id semantics as vector_store.py: re-ingesting the
    same chunk_id overwrites rather than duplicates.
    """
    if not patient_id or not patient_id.strip():
        raise ValueError("patient_id cannot be empty")
    if not (len(chunk_ids) == len(chunk_texts)):
        raise ValueError("chunk_ids and chunk_texts must be the same length")
    if not chunk_ids:
        raise ValueError("no chunks to write")

    corpus = _load_corpus(patient_id)
    for chunk_id, text in zip(chunk_ids, chunk_texts):
        corpus[chunk_id] = {"text": text, "note_id": note_id}
    _save_corpus(patient_id, corpus)


def _floor_idf(bm25: BM25Okapi, floor: float = 0.10) -> None:
    """
    rank-bm25's standard IDF, log((N - n + 0.5) / (n + 0.5)), hits exactly
    zero for any term appearing in precisely half the corpus -- and goes
    negative for terms in more than half. BM25Okapi already patches
    negative IDFs with an epsilon; it does NOT patch exact zero.

    This matters here specifically because our corpus is per-patient
    (Section 1's isolation design), so corpora are often tiny -- a new
    patient's first note might be their whole corpus of 2 chunks. A
    clinician searching an exact term that happens to land at N=2, n=1
    would silently get zero hits despite an exact keyword match sitting
    right there. That's a correctness bug, not a ranking nuance -- floor
    every non-positive IDF so an exact match always contributes a
    positive score, regardless of corpus size.
    """
    for term, value in bm25.idf.items():
        if value <= 0:
            bm25.idf[term] = floor


# --- Step 2: read path (called from query flow, Section 4) -------------------

def search(patient_id: str, query_text: str, top_k: int = 20) -> list[dict]:
    """
    BM25 search within ONLY this patient's corpus, top_k by score.

    Structurally the same isolation guarantee as vector_store.query():
    there's no cross-patient corpus to accidentally search, because each
    patient's chunks live in a separate file and a separate in-memory
    index is built fresh per call.
    """
    if not patient_id or not patient_id.strip():
        raise ValueError("patient_id cannot be empty")

    corpus = _load_corpus(patient_id)
    if not corpus:
        return []

    chunk_ids = list(corpus.keys())
    tokenized_docs = [_tokenize(corpus[cid]["text"]) for cid in chunk_ids]
    bm25 = BM25Okapi(tokenized_docs)
    _floor_idf(bm25)

    tokenized_query = _tokenize(query_text)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
    top = ranked[:top_k]

    return [
        {
            "chunk_id": cid,
            "text": corpus[cid]["text"],
            "note_id": corpus[cid]["note_id"],
            "patient_id": patient_id,
            "score": float(score),
        }
        for cid, score in top
        if score > 0  # BM25 gives 0 to docs with no term overlap; don't return noise
    ]


# --- Step 3: delete path -----------------------------------------------------

def delete_note_chunks(patient_id: str, note_id: str) -> None:
    """Remove all chunks belonging to one note from this patient's corpus."""
    corpus = _load_corpus(patient_id)
    remaining = {cid: v for cid, v in corpus.items() if v["note_id"] != note_id}
    _save_corpus(patient_id, remaining)


def delete_patient_index(patient_id: str) -> None:
    """
    Drop a patient's entire lexical corpus outright. Same use case as
    vector_store.delete_patient_collection: test teardown or a
    deliberate erasure request, not a normal soft-delete side effect.
    """
    path = _corpus_path(patient_id)
    if path.exists():
        path.unlink()


# --- Step 4: quick self-test -------------------------------------------------
# Run this file directly: python bm25_index.py
# Uses a throwaway persist dir so it never touches real data.

if __name__ == "__main__":
    import shutil

    TEST_DIR = Path(__file__).parent / "_bm25_index_selftest"
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    PERSIST_DIR = TEST_DIR

    print("=== upsert chunks for patient_a and patient_b ===")
    upsert_chunks(
        "patient_a", "note_1",
        chunk_ids=["a_note1_chunk0", "a_note1_chunk1"],
        chunk_texts=[
            "Patient reports taking sertraline 50mg daily.",
            "Patient discussed work stress and poor sleep this week.",
        ],
    )
    upsert_chunks(
        "patient_b", "note_1",
        chunk_ids=["b_note1_chunk0"],
        chunk_texts=["Patient discussed a family conflict over the holidays."],
    )
    print("OK, upserted.")

    print("\n=== exact-keyword search patient_a: 'sertraline' ===")
    hits_a = search("patient_a", "sertraline", top_k=20)
    for h in hits_a:
        print(f"{h['score']:.3f}", h["patient_id"], "-", h["text"])
    assert len(hits_a) == 1
    assert all(h["patient_id"] == "patient_a" for h in hits_a)
    assert "sertraline" in hits_a[0]["text"].lower()

    print("\n=== search patient_b: must never see patient_a's terms ===")
    hits_b = search("patient_b", "sertraline", top_k=20)
    print(hits_b if hits_b else "No hits (expected -- term only exists in patient_a's corpus).")
    assert hits_b == []

    print("\n=== search patient_b for its own term: 'family conflict' ===")
    hits_b2 = search("patient_b", "family conflict", top_k=20)
    for h in hits_b2:
        print(f"{h['score']:.3f}", h["patient_id"], "-", h["text"])
    assert len(hits_b2) == 1
    assert all(h["patient_id"] == "patient_b" for h in hits_b2)

    print("\n=== upsert (not duplicate) on same chunk id ===")
    upsert_chunks(
        "patient_a", "note_1",
        chunk_ids=["a_note1_chunk0"],
        chunk_texts=["Patient reports taking sertraline 100mg daily, dose increased."],
    )
    hits_a_updated = search("patient_a", "sertraline", top_k=20)
    assert len(hits_a_updated) == 1  # still 1, overwrote not duplicated
    assert "100mg" in hits_a_updated[0]["text"]
    print("OK, chunk count unchanged after upsert, text updated.")

    print("\n=== delete_note_chunks removes only that note's chunks ===")
    upsert_chunks(
        "patient_a", "note_2",
        chunk_ids=["a_note2_chunk0"],
        chunk_texts=["Patient A note 2 mentions sertraline again."],
    )
    assert len(search("patient_a", "sertraline", top_k=20)) == 2
    delete_note_chunks("patient_a", "note_2")
    remaining = search("patient_a", "sertraline", top_k=20)
    assert len(remaining) == 1
    assert all(h["note_id"] != "note_2" for h in remaining)
    print("OK, note_2's chunk gone, note_1's chunk remains.")

    print("\n=== search on empty/never-indexed patient ===")
    assert search("patient_c_never_indexed", "anything", top_k=20) == []
    print("OK, empty as expected (no corpus file for this patient).")

    print("\n=== delete_patient_index wipes everything for that patient ===")
    delete_patient_index("patient_a")
    assert search("patient_a", "sertraline", top_k=20) == []
    # patient_b untouched
    assert len(search("patient_b", "family conflict", top_k=20)) == 1
    print("OK, patient_a's corpus gone, patient_b unaffected.")

    shutil.rmtree(TEST_DIR)
    print("\nSelf-test passed.")