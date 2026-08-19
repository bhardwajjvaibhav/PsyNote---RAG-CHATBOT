/**
 * api/insights.js
 *
 * Thin fetch wrapper around GET /api/insights/{id} -- the emotional
 * energy report (per-note emotion scores, overall averages, and
 * per-emotion reasons). No state, just HTTP -> JSON or throw.
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
 * @param {string} userId
 * @returns {Promise<{notes: object[], overall: object, reasons: object}>}
 */
export async function getInsights(userId) {
  const res = await fetch(`/api/insights/${userId}`);
  return handleResponse(res);
}