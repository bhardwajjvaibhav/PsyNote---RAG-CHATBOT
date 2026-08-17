/**
 * api/chat.js
 *
 * Thin fetch wrapper around POST /api/chat (architecture doc, Section 4).
 * patient_id currently travels in the request body -- see routes.py's
 * known-gap note: no auth yet, so this is not a security boundary, just
 * the shape the backend expects today. Once Phase 9 lands, this is the
 * one place that will need to change (drop patient_id from the payload).
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
  return res.json();
}

/**
 * @param {string} patientId
 * @param {string} question
 * @param {{role: "user"|"assistant", content: string}[]} chatHistory
 * @returns {Promise<{answer: string, model: string, citations: any[],
 *   safety_hits: {category: string, note_id: string|null, matched_term: string}[],
 *   grounding: {fully_grounded: boolean, grounding_score: number, flagged_sentences: string[]}}>}
 */
export async function askQuestion(patientId, question, chatHistory = []) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      patient_id: patientId,
      question,
      chat_history: chatHistory,
    }),
  });
  return handleResponse(res);
}
