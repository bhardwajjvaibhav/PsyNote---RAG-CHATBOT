"""
db/patients.py

Patient entity CRUD, backed by SQLite.

Why this exists (not in the original doc's folder structure):
doc_registry.py tags journal chunks with patient_id, but nothing in the
original architecture owns the patient record itself (name, id, active
status). This module is that missing piece -- the thing patient_id
actually refers to.

Soft-delete only: journal notes reference patient_id, so hard-deleting
a patient row would orphan their notes. Same pattern as doc_registry's
soft-delete tracking.
"""

import sqlite3
import uuid
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "psynote.db"


# --- Step 1: schema ---------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,               -- uuid4 hex string
    name TEXT NOT NULL,
    age INTEGER,
    place TEXT,
    gender TEXT,
    marital_status TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    is_active INTEGER NOT NULL DEFAULT 1  -- soft-delete flag
);
"""

# Columns added after the original v1 schema (registration profile fields).
# init_db() runs a tiny ALTER TABLE migration for pre-existing databases.
_POST_V1_COLUMNS = [
    ("age", "INTEGER"),
    ("place", "TEXT"),
    ("gender", "TEXT"),
    ("marital_status", "TEXT"),
]


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
        conn.execute(SCHEMA)
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(patients)")}
        for column, col_type in _POST_V1_COLUMNS:
            if column not in existing:
                conn.execute(f"ALTER TABLE patients ADD COLUMN {column} {col_type}")


# --- Step 2: CRUD functions --------------------------------------------------
# Plain functions, no HTTP here. api/routes.py calls these.
# Keeping this separation means these are testable without spinning up a
# server, and reusable later from the ingestion pipeline too.

def create_patient(name: str) -> dict:
    if not name or not name.strip():
        raise ValueError("Name cannot be empty")
    patient_id = uuid.uuid4().hex
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO patients (id, name) VALUES (?, ?)",
            (patient_id, name.strip()),
        )
    return get_patient(patient_id)


def register_patient(
    name: str,
    age: int | None = None,
    place: str | None = None,
    gender: str | None = None,
    marital_status: str | None = None,
) -> dict:
    """
    Create a patient with an optional registration profile (age, place,
    gender, marital_status). Used by /api/register. Same validation as
    create_patient, plus a numeric-age sanity check.
    """
    if not name or not name.strip():
        raise ValueError("Name cannot be empty")
    if age is not None:
        if isinstance(age, bool) or not isinstance(age, int) or age < 0 or age > 130:
            raise ValueError("Age must be a whole number between 0 and 130")

    patient_id = uuid.uuid4().hex
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO patients (id, name, age, place, gender, marital_status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                patient_id,
                name.strip(),
                age,
                (place or "").strip() or None,
                (gender or "").strip() or None,
                (marital_status or "").strip() or None,
            ),
        )
    return get_patient(patient_id)


def get_patient_by_name(name: str) -> dict | None:
    """
    Case-insensitive lookup of the first active patient matching `name`.
    Stub login until real auth (Phase 9) lands: no passwords, no tokens,
    just enough to bind the frontend session to a patient record.
    """
    if not name or not name.strip():
        return None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM patients "
            "WHERE is_active = 1 AND lower(trim(name)) = lower(trim(?)) "
            "ORDER BY created_at ASC LIMIT 1",
            (name,),
        ).fetchone()
    return dict(row) if row else None


def get_patient(patient_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE id = ? AND is_active = 1",
            (patient_id,),
        ).fetchone()
    return dict(row) if row else None


def list_patients() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM patients WHERE is_active = 1 ORDER BY name ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def update_patient(patient_id: str, name: str) -> dict | None:
    if not name or not name.strip():
        raise ValueError("Name cannot be empty")
    with get_conn() as conn:
        conn.execute(
            "UPDATE patients SET name = ?, updated_at = datetime('now') "
            "WHERE id = ? AND is_active = 1",
            (name.strip(), patient_id),
        )
    return get_patient(patient_id)


def delete_patient(patient_id: str) -> bool:
    """Soft-delete: flips is_active to 0, never removes the row."""
    with get_conn() as conn:
        cursor = conn.execute(
            "UPDATE patients SET is_active = 0, updated_at = datetime('now') "
            "WHERE id = ? AND is_active = 1",
            (patient_id,),
        )
    return cursor.rowcount > 0


# --- Step 3: quick self-test -------------------------------------------------
# Run this file directly: python patients.py
# Uses a throwaway DB file so it never touches real data.

if __name__ == "__main__":
    import os

    DB_PATH = Path(__file__).parent / "_patients_selftest.db"
    if DB_PATH.exists():
        os.remove(DB_PATH)

    init_db()

    print("=== create ===")
    p1 = create_patient("Asha Verma")
    p2 = create_patient("  Ravi Shah  ")  # whitespace should be trimmed
    print(p1)
    print(p2)

    print("\n=== reject empty name ===")
    try:
        create_patient("   ")
        print("FAILED: should have raised ValueError")
    except ValueError as e:
        print(f"OK, raised: {e}")

    print("\n=== list ===")
    for p in list_patients():
        print(p)

    print("\n=== get single ===")
    print(get_patient(p1["id"]))
    print("missing id ->", get_patient("nonexistent"))

    print("\n=== update ===")
    updated = update_patient(p1["id"], "Asha Verma-Iyer")
    print(updated)
    assert updated["name"] == "Asha Verma-Iyer"

    print("\n=== soft delete ===")
    deleted = delete_patient(p2["id"])
    print("deleted:", deleted)
    print("get after delete ->", get_patient(p2["id"]))  # should be None
    print("delete again ->", delete_patient(p2["id"]))   # should be False (already inactive)

    print("\n=== list after delete (only p1 should show) ===")
    remaining = list_patients()
    print(remaining)
    assert len(remaining) == 1
    assert remaining[0]["id"] == p1["id"]

    print("\n=== register with profile fields ===")
    reg = register_patient("Meera Nair", age=34, place="Kochi", gender="female", marital_status="married")
    print(reg)
    assert reg["age"] == 34
    assert reg["place"] == "Kochi"
    assert reg["gender"] == "female"
    assert reg["marital_status"] == "married"

    print("\n=== register rejects invalid age ===")
    try:
        register_patient("Bad Age", age=-1)
        print("FAILED: should have raised ValueError")
    except ValueError as e:
        print(f"OK, raised: {e}")

    print("\n=== get_patient_by_name: case-insensitive ===")
    by_name = get_patient_by_name("  meera nair ")
    assert by_name is not None and by_name["id"] == reg["id"]
    assert get_patient_by_name("nobody") is None
    print(f"OK, found {by_name['name']}")

    os.remove(DB_PATH)
    print("\nSelf-test passed.")