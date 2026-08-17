"""
db/doc_registry.py

Journal note metadata registry, backed by SQLite.

Owns the source-of-truth record for every ingested note: which patient
it belongs to, where it came from, and whether it's actually retrievable
yet. This table is written to BEFORE the Chroma/BM25 writes (Section 3),
so it can track a `status` field through the ingestion lifecycle:

    pending  -> row created, Chroma/BM25 writes not yet confirmed
    indexed  -> Chroma + BM25 writes both confirmed, note is retrievable
    failed   -> a write failed; note exists in SQL but NOT in the vector/
                lexical indexes, so it must never be treated as retrievable

Why this status field exists (fix carried from architecture doc, Section 3):
if a note is marked "ready" in doc_registry before its Chroma write is
confirmed, and that write then fails, you get a "ghost" note -- SQL knows
it exists, but it can never be retrieved, and nothing surfaces the failure.
Tracking status here makes ingestion failures visible and retryable
instead of silent.

Same soft-delete pattern as db/patients.py: notes are never hard-deleted,
only flagged inactive, since audit trails (security/audit_log.py) may
reference a note_id after the note itself is withdrawn.
"""

import sqlite3
import uuid
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "psynote.db"

VALID_STATUSES = {"pending", "indexed", "failed"}


# --- Step 1: schema ---------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS doc_registry (
    id TEXT PRIMARY KEY,                  -- uuid4 hex string (the note_id)
    patient_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | indexed | failed
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    is_active INTEGER NOT NULL DEFAULT 1,    -- soft-delete flag
    error_detail TEXT                        -- populated only when status='failed'
);

CREATE INDEX IF NOT EXISTS idx_doc_registry_patient_id
    ON doc_registry (patient_id);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# --- Step 2: CRUD + status transition functions ------------------------------
# Plain functions, no HTTP here, same separation as db/patients.py.
# The ingestion pipeline (services/parser.py -> chunker.py -> embedder.py,
# once built) is the caller that drives these status transitions.

def create_note(patient_id: str, filename: str) -> dict:
    """
    Register a new note as 'pending' BEFORE any Chroma/BM25 write is
    attempted. This is step 1 of the write-ordering fix: the row exists
    and is visible (as pending, not retrievable) the instant ingestion
    starts, not after it succeeds.
    """
    if not patient_id or not patient_id.strip():
        raise ValueError("patient_id cannot be empty")
    if not filename or not filename.strip():
        raise ValueError("filename cannot be empty")

    note_id = uuid.uuid4().hex
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO doc_registry (id, patient_id, filename, status) "
            "VALUES (?, ?, ?, 'pending')",
            (note_id, patient_id.strip(), filename.strip()),
        )
    return get_note(note_id)


def get_note(note_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM doc_registry WHERE id = ? AND is_active = 1",
            (note_id,),
        ).fetchone()
    return dict(row) if row else None


def list_notes_for_patient(patient_id: str, status: str | None = None) -> list[dict]:
    """
    List a single patient's notes, optionally filtered by status.
    Every caller in the query flow (Section 4) must go through something
    like this rather than a global note list -- there is no "list all
    notes across patients" function here on purpose.
    """
    with get_conn() as conn:
        if status is not None:
            rows = conn.execute(
                "SELECT * FROM doc_registry "
                "WHERE patient_id = ? AND status = ? AND is_active = 1 "
                "ORDER BY created_at DESC",
                (patient_id, status),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM doc_registry "
                "WHERE patient_id = ? AND is_active = 1 "
                "ORDER BY created_at DESC",
                (patient_id,),
            ).fetchall()
    return [dict(r) for r in rows]


def mark_indexed(note_id: str) -> dict | None:
    """
    Call this only after BOTH the Chroma write and the BM25 write for
    this note are confirmed. This is the note becoming retrievable.
    """
    with get_conn() as conn:
        conn.execute(
            "UPDATE doc_registry SET status = 'indexed', error_detail = NULL, "
            "updated_at = datetime('now') WHERE id = ? AND is_active = 1",
            (note_id,),
        )
    return get_note(note_id)


def mark_failed(note_id: str, error_detail: str) -> dict | None:
    """
    Call this when a Chroma or BM25 write fails. Keeps the note visible
    (not retrievable) with the failure reason attached, so it shows up
    in retry tooling instead of silently vanishing.
    """
    with get_conn() as conn:
        conn.execute(
            "UPDATE doc_registry SET status = 'failed', error_detail = ?, "
            "updated_at = datetime('now') WHERE id = ? AND is_active = 1",
            (error_detail, note_id),
        )
    return get_note(note_id)


def delete_note(note_id: str) -> bool:
    """Soft-delete: flips is_active to 0, never removes the row."""
    with get_conn() as conn:
        cursor = conn.execute(
            "UPDATE doc_registry SET is_active = 0, updated_at = datetime('now') "
            "WHERE id = ? AND is_active = 1",
            (note_id,),
        )
    return cursor.rowcount > 0


# --- Step 3: quick self-test -------------------------------------------------
# Run this file directly: python doc_registry.py
# Uses a throwaway DB file so it never touches real data.

if __name__ == "__main__":
    import os

    DB_PATH = Path(__file__).parent / "_doc_registry_selftest.db"
    if DB_PATH.exists():
        os.remove(DB_PATH)

    init_db()

    print("=== create note (pending) ===")
    note = create_note("patient_a", "session_2026_01_10.txt")
    print(note)
    assert note["status"] == "pending"

    print("\n=== reject empty patient_id ===")
    try:
        create_note("   ", "x.txt")
        print("FAILED: should have raised ValueError")
    except ValueError as e:
        print(f"OK, raised: {e}")

    print("\n=== simulate successful ingestion: mark_indexed ===")
    indexed = mark_indexed(note["id"])
    print(indexed)
    assert indexed["status"] == "indexed"
    assert indexed["error_detail"] is None

    print("\n=== simulate a failed ingestion on a second note ===")
    note2 = create_note("patient_a", "session_2026_01_17.csv")
    failed = mark_failed(note2["id"], "Chroma write timed out after 3 retries")
    print(failed)
    assert failed["status"] == "failed"
    assert failed["error_detail"]

    print("\n=== list all notes for patient_a ===")
    for n in list_notes_for_patient("patient_a"):
        print(n["filename"], "->", n["status"])

    print("\n=== list only indexed notes for patient_a ===")
    indexed_only = list_notes_for_patient("patient_a", status="indexed")
    assert len(indexed_only) == 1
    assert indexed_only[0]["id"] == note["id"]
    print(indexed_only)

    print("\n=== list only failed notes for patient_a ===")
    failed_only = list_notes_for_patient("patient_a", status="failed")
    assert len(failed_only) == 1
    assert failed_only[0]["id"] == note2["id"]
    print(failed_only)

    print("\n=== cross-patient isolation: patient_b sees nothing ===")
    assert list_notes_for_patient("patient_b") == []
    print("OK, empty as expected.")

    print("\n=== soft delete ===")
    deleted = delete_note(note["id"])
    print("deleted:", deleted)
    print("get after delete ->", get_note(note["id"]))  # should be None
    remaining = list_notes_for_patient("patient_a")
    assert len(remaining) == 1  # only note2 (failed) remains visible
    assert remaining[0]["id"] == note2["id"]

    os.remove(DB_PATH)
    print("\nSelf-test passed.")