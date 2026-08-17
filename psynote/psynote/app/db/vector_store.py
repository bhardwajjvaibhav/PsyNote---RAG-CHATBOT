"""
db/vector_store.py

Chroma client wrapper (embedded, persistent mode). RENAMED from the v2
plan's db/faiss_db.py -- see architecture doc, Section 1 and 6.

Design decision: one Chroma COLLECTION PER PATIENT, not a single shared
collection with a metadata filter applied at query time.

Why per-collection instead of "single collection + where filter":
a where={'patient_id': patient_id} filter on a shared collection is only
as safe as every single call site remembering to pass it. A missing
filter on one query path is a cross-patient data leak in a clinical
product -- exactly the class of bug this module exists to make
structurally impossible. Per-patient collections mean there is no
"forgot the filter" failure mode: querying patient A's collection can
only ever return patient A's vectors, full stop. We additionally stamp
patient_id into each chunk's metadata anyway, as a second, redundant
check (see query()), so isolation doesn't rely on collection naming
alone either.

This module does NOT decide when a write is "confirmed" -- that's
doc_registry.py's job (mark_indexed / mark_failed). This module just
does the Chroma I/O and raises on failure so the caller (ingestion
pipeline, not yet built) can catch it and call mark_failed.
"""

from __future__ import annotations

from pathlib import Path
import re

import chromadb
from chromadb.api.models.Collection import Collection

# Defined on-disk location, parallel to db/psynote.db (SQLite).
# Known gap carried forward (architecture doc, Section 9): this path's
# backup/ownership story isn't decided yet -- same open question as the
# SQLite file.
PERSIST_DIR = Path(__file__).parent / "chroma_store"

_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    """
    Lazily create the embedded persistent Chroma client, once per process.
    Embedded mode = SQLite + hnswlib under the hood, no separate server
    to run or manage (architecture doc, Section 1).
    """
    global _client
    if _client is None:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return _client


def _collection_name(patient_id: str) -> str:
    """
    Chroma collection names are restricted (alnum, ., _, -, 3-63 chars).
    patient_id is already a uuid4 hex string (see db/patients.py), which
    is safe as-is, but we sanitize defensively in case that assumption
    ever changes upstream.
    """
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", patient_id)
    name = f"patient_{safe}"
    return name[:63]


def get_patient_collection(patient_id: str) -> Collection:
    """
    Get-or-create this patient's collection. Safe to call repeatedly --
    Chroma no-ops if the collection already exists.
    """
    if not patient_id or not patient_id.strip():
        raise ValueError("patient_id cannot be empty")
    client = get_client()
    return client.get_or_create_collection(
        name=_collection_name(patient_id),
        metadata={"patient_id": patient_id},
    )


# --- Step 1: write path (called from ingestion, once chunker/embedder exist) -

def upsert_chunks(
    patient_id: str,
    note_id: str,
    chunk_ids: list[str],
    chunk_texts: list[str],
    chunk_embeddings: list[list[float]],
) -> None:
    """
    Upsert a note's chunks into this patient's collection.

    Upsert (not add) so re-ingesting/re-processing the same note_id's
    chunks overwrites cleanly rather than duplicating -- real
    update-by-id semantics, which is one of the reasons Chroma replaced
    FAISS (architecture doc, Section 1).

    Every chunk is tagged with both note_id and patient_id in its
    metadata. patient_id here is redundant with which collection it's
    in, by design (see module docstring) -- defense in depth, not decoration.

    Raises on failure; caller is responsible for calling
    doc_registry.mark_failed(note_id, str(err)) if this raises, per the
    write-ordering fix in Section 3.
    """
    if not (len(chunk_ids) == len(chunk_texts) == len(chunk_embeddings)):
        raise ValueError("chunk_ids, chunk_texts, and chunk_embeddings must be the same length")
    if not chunk_ids:
        raise ValueError("no chunks to write")

    collection = get_patient_collection(patient_id)
    metadatas = [{"patient_id": patient_id, "note_id": note_id} for _ in chunk_ids]
    collection.upsert(
        ids=chunk_ids,
        documents=chunk_texts,
        embeddings=chunk_embeddings,
        metadatas=metadatas,
    )


# --- Step 2: read path (called from query flow, Section 4) -------------------

def query(
    patient_id: str,
    query_embedding: list[float],
    top_k: int = 20,
) -> list[dict]:
    """
    Query ONLY this patient's collection, top_k nearest neighbors.

    Belt-and-braces isolation: even though the collection itself is
    already scoped to one patient, we also assert every returned hit's
    metadata.patient_id matches the requested patient_id before
    returning. If that assertion ever fails, that's a bug worth crashing
    loudly on rather than silently leaking a hit -- this is exactly the
    guarantee (architecture doc, Section 4) that "must never surface
    another patient's notes."
    """
    if not patient_id or not patient_id.strip():
        raise ValueError("patient_id cannot be empty")

    collection = get_patient_collection(patient_id)
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    hits: list[dict] = []
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        assert meta.get("patient_id") == patient_id, (
            f"Isolation violation: got chunk tagged patient_id={meta.get('patient_id')!r} "
            f"from a query scoped to patient_id={patient_id!r}"
        )
        hits.append({
            "chunk_id": chunk_id,
            "text": text,
            "note_id": meta.get("note_id"),
            "patient_id": meta.get("patient_id"),
            "distance": distance,
        })
    return hits


# --- Step 3: delete path -----------------------------------------------------

def delete_note_chunks(patient_id: str, note_id: str) -> None:
    """
    Delete all chunks belonging to one note (e.g. note soft-deleted, or
    re-ingestion needs a clean slate first). Real delete-by-id, unlike
    FAISS -- another reason for the Chroma switch (Section 1).
    """
    collection = get_patient_collection(patient_id)
    collection.delete(where={"note_id": note_id})


def delete_patient_collection(patient_id: str) -> None:
    """
    Drop an entire patient's collection outright. Not used by normal
    soft-delete flows (patients.delete_patient is soft-delete only) --
    this exists for hard cleanup paths like test teardown or a future
    GDPR-style erasure request, called deliberately, not as a side effect.
    """
    client = get_client()
    try:
        client.delete_collection(name=_collection_name(patient_id))
    except Exception:
        # Collection may not exist -- deleting a non-existent collection
        # is a no-op from the caller's point of view.
        pass


# --- Step 4: quick self-test -------------------------------------------------
# Run this file directly: python vector_store.py
# Uses a throwaway persist dir so it never touches real data. Uses tiny
# fake embeddings (embedder.py isn't built yet) -- just enough dimensionality
# to exercise nearest-neighbor behavior deterministically.

if __name__ == "__main__":
    import shutil

    TEST_DIR = Path(__file__).parent / "_vector_store_selftest"
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    PERSIST_DIR = TEST_DIR
    _client = None  # force re-init against the test dir

    def fake_embed(text: str) -> list[float]:
        # Deterministic 4-dim "embedding" from a hash, good enough to test
        # upsert/query/delete mechanics without a real model.
        h = abs(hash(text))
        return [((h >> (8 * i)) % 100) / 100.0 for i in range(4)]

    print("=== upsert chunks for patient_a and patient_b ===")
    upsert_chunks(
        "patient_a", "note_1",
        chunk_ids=["a_note1_chunk0", "a_note1_chunk1"],
        chunk_texts=["Patient A reports improved sleep.", "Patient A discussed work stress."],
        chunk_embeddings=[fake_embed("A improved sleep"), fake_embed("A work stress")],
    )
    upsert_chunks(
        "patient_b", "note_1",
        chunk_ids=["b_note1_chunk0"],
        chunk_texts=["Patient B disclosed a family conflict."],
        chunk_embeddings=[fake_embed("B family conflict")],
    )
    print("OK, upserted.")

    print("\n=== query patient_a: must only see patient_a's chunks ===")
    hits_a = query("patient_a", fake_embed("A improved sleep"), top_k=20)
    for h in hits_a:
        print(h["patient_id"], "-", h["text"])
    assert all(h["patient_id"] == "patient_a" for h in hits_a)
    assert len(hits_a) == 2

    print("\n=== query patient_b: must only see patient_b's chunks ===")
    hits_b = query("patient_b", fake_embed("B family conflict"), top_k=20)
    for h in hits_b:
        print(h["patient_id"], "-", h["text"])
    assert all(h["patient_id"] == "patient_b" for h in hits_b)
    assert len(hits_b) == 1

    print("\n=== upsert (not duplicate) on same chunk id ===")
    upsert_chunks(
        "patient_a", "note_1",
        chunk_ids=["a_note1_chunk0"],  # same id as before, updated text
        chunk_texts=["Patient A reports significantly improved sleep this week."],
        chunk_embeddings=[fake_embed("A improved sleep updated")],
    )
    hits_a_after_update = query("patient_a", fake_embed("A improved sleep"), top_k=20)
    assert len(hits_a_after_update) == 2  # still 2, not 3 -- upsert overwrote
    updated_texts = [h["text"] for h in hits_a_after_update]
    assert "significantly improved sleep this week" in " ".join(updated_texts)
    print("OK, chunk count unchanged after upsert (overwrote, didn't duplicate).")

    print("\n=== delete_note_chunks removes only that note's chunks ===")
    upsert_chunks(
        "patient_a", "note_2",
        chunk_ids=["a_note2_chunk0"],
        chunk_texts=["Patient A note 2 content."],
        chunk_embeddings=[fake_embed("A note2")],
    )
    assert len(query("patient_a", fake_embed("A improved sleep"), top_k=20)) == 3
    delete_note_chunks("patient_a", "note_2")
    remaining = query("patient_a", fake_embed("A improved sleep"), top_k=20)
    assert len(remaining) == 2
    assert all(h["note_id"] != "note_2" for h in remaining)
    print("OK, note_2's chunk gone, note_1's chunks remain.")

    print("\n=== delete_patient_collection wipes everything for that patient ===")
    delete_patient_collection("patient_a")
    assert query("patient_a", fake_embed("A improved sleep"), top_k=20) == []
    # patient_b untouched
    assert len(query("patient_b", fake_embed("B family conflict"), top_k=20)) == 1
    print("OK, patient_a's collection gone, patient_b unaffected.")

    # Cleanup: on Windows, Chroma's sqlite backend can hold the file
    # handle open briefly after the last call returns, which makes an
    # immediate shutil.rmtree fail with PermissionError (WinError 32).
    # Not a functional bug -- release our reference and retry with a
    # short backoff rather than crash the self-test on its last line.
    _client = None
    import gc
    import time
    gc.collect()
    for attempt in range(5):
        try:
            shutil.rmtree(TEST_DIR)
            break
        except PermissionError:
            time.sleep(0.5)
    else:
        print(f"(Note: could not remove {TEST_DIR} -- Windows file lock; safe to delete by hand.)")

    print("\nSelf-test passed.")