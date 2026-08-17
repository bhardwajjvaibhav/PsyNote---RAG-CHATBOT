/**
 * api/notes.js
 *
 * Thin fetch wrapper for the ingestion + note-listing routes:
 * POST /api/ingest (multipart/form-data) and
 * GET /api/patients/{id}/notes. Same shape as api/patients.js -- no
 * state, no React, just HTTP calls that return parsed JSON or throw.
 */

async function handleResponse(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // body wasn't JSON -- fall back to statusText
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  if (res.status === 204) return null;
  return res.json();
}

/**
 * Upload a session note file for a patient.
 *
 * NOTE: a 2xx response here does NOT necessarily mean the note is
 * retrievable. The backend writes doc_registry BEFORE the Chroma/BM25
 * writes are confirmed (architecture doc, Section 3's write-ordering
 * fix), so this can resolve with { status: "failed", error: "..." }
 * on a normal 201 -- that's not an HTTP error, it's a pipeline failure
 * the note stays visible (and retryable) for, rather than a request to
 * retry blindly. Callers must check result.status, not just res.ok.
 */
export async function ingestNote(patientId, file) {
  const formData = new FormData();
  formData.append("patient_id", patientId);
  formData.append("file", file);

  const res = await fetch("/api/ingest", {
    method: "POST",
    body: formData,
  });
  return handleResponse(res);
}

export async function listNotes(patientId, status) {
  const url = status
    ? `/api/patients/${patientId}/notes?status=${encodeURIComponent(status)}`
    : `/api/patients/${patientId}/notes`;
  const res = await fetch(url);
  return handleResponse(res);
}
