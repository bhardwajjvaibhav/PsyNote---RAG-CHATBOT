/**
 * api/patients.js
 *
 * Thin fetch wrapper around the backend's /api/patients routes.
 * Deliberately no state, no React here -- just HTTP calls that return
 * parsed JSON or throw. Components decide what to do with the result.
 */

const BASE_URL = "/api/patients";

async function handleResponse(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // body wasn't JSON (e.g. 204 No Content) -- fall back to statusText
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  // 204 No Content has no body to parse
  if (res.status === 204) return null;
  return res.json();
}

export async function listPatients() {
  const res = await fetch(BASE_URL);
  return handleResponse(res);
}

export async function getPatient(id) {
  const res = await fetch(`${BASE_URL}/${id}`);
  return handleResponse(res);
}

export async function createPatient(name) {
  const res = await fetch(BASE_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return handleResponse(res);
}

export async function updatePatient(id, name) {
  const res = await fetch(`${BASE_URL}/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return handleResponse(res);
}

export async function deletePatient(id) {
  const res = await fetch(`${BASE_URL}/${id}`, { method: "DELETE" });
  return handleResponse(res);
}
