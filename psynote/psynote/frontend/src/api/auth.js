/**
 * api/auth.js
 *
 * Thin fetch wrapper around POST /api/register and POST /api/login.
 * Both return the patient record, which the app stores as the session.
 * Stub auth (no passwords/tokens yet) -- see routes.py's known-gap note.
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
    throw new Error(detail);
  }
  return res.json();
}

/**
 * @param {string} name
 * @param {{age?: number, place?: string, gender?: string, marital_status?: string}} profile
 * @returns {Promise<object>} patient record
 */
export async function registerUser(name, profile = {}) {
  const res = await fetch("/api/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, ...profile }),
  });
  return handleResponse(res);
}

/** @returns {Promise<object>} patient record for the matching name */
export async function loginUser(name) {
  const res = await fetch("/api/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return handleResponse(res);
}