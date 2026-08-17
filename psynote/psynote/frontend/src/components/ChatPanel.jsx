import { useState, useRef, useEffect } from "react";
import { askQuestion } from "../api/chat";

/**
 * ChatPanel
 *
 * Query-time flow (architecture doc, Section 4) wired to POST
 * /api/chat. Renders the answer plus everything the pipeline computed
 * alongside it -- citations, safety_hits, and the grounding check --
 * since those are the point of the pipeline, not incidental metadata
 * to hide behind a tooltip.
 *
 * patient_id currently travels in the request body (see routes.py's
 * known-gap docstring: no auth yet, patient_id is not derived from a
 * session). This component just passes through whatever patientId its
 * parent gives it.
 */
export default function ChatPanel({ patientId, patientName }) {
  const [messages, setMessages] = useState([]); // {role, content, meta?}
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("idle"); // idle | sending | error
  const [errorMessage, setErrorMessage] = useState("");
  const scrollRef = useRef(null);

  // Reset the thread when the selected patient changes -- a chat about
  // patient A must never be sent as history alongside a question about
  // patient B.
  useEffect(() => {
    setMessages([]);
    setErrorMessage("");
    setStatus("idle");
  }, [patientId]);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, status]);

  async function handleSubmit(e) {
    e.preventDefault();
    const question = input.trim();
    if (!question || status === "sending") return;

    const history = messages.map((m) => ({ role: m.role, content: m.content }));
    const nextMessages = [...messages, { role: "user", content: question }];
    setMessages(nextMessages);
    setInput("");
    setStatus("sending");
    setErrorMessage("");

    try {
      const result = await askQuestion(patientId, question, history);
      setMessages((prev) => [...prev, { role: "assistant", content: result.answer, meta: result }]);
      setStatus("idle");
    } catch (err) {
      setStatus("error");
      setErrorMessage(err.message || "The question could not be answered.");
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  }

  if (!patientId) return null;

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.label}>Chat</span>
        <span style={styles.patientTag}>with {patientName}'s notes</span>
      </div>

      <div ref={scrollRef} style={styles.scroll}>
        {messages.length === 0 && status !== "sending" && (
          <div style={styles.emptyText}>Ask a question grounded in this patient's ingested notes.</div>
        )}

        {messages.map((m, i) => (
          <div
            key={i}
            style={{ ...styles.messageRow, alignItems: m.role === "user" ? "flex-end" : "flex-start" }}
          >
            <div style={{ ...styles.bubble, ...(m.role === "user" ? styles.bubbleUser : styles.bubbleAssistant) }}>
              {m.content}
            </div>
            {m.role === "assistant" && m.meta && <AnswerMeta meta={m.meta} />}
          </div>
        ))}

        {status === "sending" && <div style={styles.emptyText}>Retrieving, reranking, generating…</div>}
      </div>

      {status === "error" && <div style={styles.statusError}>{errorMessage}</div>}

      <form onSubmit={handleSubmit} style={styles.inputRow}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about this patient's notes…"
          disabled={status === "sending"}
          style={styles.textarea}
        />
        <button
          type="submit"
          disabled={status === "sending" || !input.trim()}
          style={{
            ...styles.button,
            ...(status === "sending" || !input.trim() ? styles.buttonDisabled : {}),
          }}
        >
          Send
        </button>
      </form>
    </div>
  );
}

/**
 * AnswerMeta -- renders citations, safety_hits, and the grounding
 * summary that come back alongside result.answer. Kept as a small
 * sub-component purely for readability; it owns no state of its own.
 */
function AnswerMeta({ meta }) {
  const groundingPct = Math.round((meta.grounding?.grounding_score ?? 1) * 100);
  const hasSafetyHits = meta.safety_hits && meta.safety_hits.length > 0;
  const hasFlaggedSentences =
    !meta.grounding?.fully_grounded && meta.grounding?.flagged_sentences?.length > 0;

  return (
    <div style={styles.metaColumn}>
      {hasSafetyHits && (
        <div style={styles.safetyBanner}>
          <div style={styles.bannerTitle}>Safety flags surfaced ({meta.safety_hits.length})</div>
          {meta.safety_hits.map((h, j) => (
            <div key={j} style={styles.bannerLine}>
              {h.category} — matched "{h.matched_term}" in note {h.note_id ? h.note_id.slice(0, 8) : "—"}
            </div>
          ))}
        </div>
      )}

      {hasFlaggedSentences && (
        <div style={styles.flaggedBanner}>
          <div style={styles.bannerTitle}>Weakly grounded sentence(s)</div>
          {meta.grounding.flagged_sentences.map((s, j) => (
            <div key={j} style={styles.bannerLine}>
              {s}
            </div>
          ))}
        </div>
      )}

      <div style={styles.chipRow}>
        <span style={styles.chip}>model: {meta.model}</span>
        <span style={styles.chip}>citations: {meta.citations ? meta.citations.length : 0}</span>
        <span style={styles.chip}>grounding: {groundingPct}%</span>
      </div>
    </div>
  );
}

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    gap: "0.6rem",
    maxWidth: "560px",
    fontFamily: "system-ui, sans-serif",
    marginTop: "1.25rem",
  },
  header: {
    display: "flex",
    alignItems: "baseline",
    gap: "0.4rem",
  },
  label: {
    fontSize: "0.85rem",
    fontWeight: 600,
    color: "#333",
  },
  patientTag: {
    fontSize: "0.8rem",
    color: "#666",
  },
  scroll: {
    display: "flex",
    flexDirection: "column",
    gap: "0.6rem",
    maxHeight: "420px",
    overflowY: "auto",
    padding: "0.5rem",
    border: "1px solid #e0e0e0",
    borderRadius: "8px",
    background: "#fafafa",
  },
  emptyText: {
    fontSize: "0.8rem",
    color: "#666",
  },
  messageRow: {
    display: "flex",
    flexDirection: "column",
    gap: "0.35rem",
  },
  bubble: {
    maxWidth: "440px",
    padding: "0.55rem 0.8rem",
    borderRadius: "10px",
    fontSize: "0.9rem",
    lineHeight: 1.5,
  },
  bubbleUser: {
    background: "#2f6fed",
    color: "#fff",
  },
  bubbleAssistant: {
    background: "#fff",
    border: "1px solid #e0e0e0",
    color: "#222",
  },
  metaColumn: {
    display: "flex",
    flexDirection: "column",
    gap: "0.35rem",
  },
  chipRow: {
    display: "flex",
    gap: "0.4rem",
    flexWrap: "wrap",
  },
  chip: {
    fontSize: "0.7rem",
    color: "#666",
    background: "#eee",
    padding: "0.15rem 0.5rem",
    borderRadius: "6px",
  },
  safetyBanner: {
    border: "1px solid #b4491f",
    background: "#fbe4dd",
    color: "#7a3115",
    borderRadius: "8px",
    padding: "0.5rem 0.7rem",
    fontSize: "0.78rem",
    maxWidth: "440px",
  },
  flaggedBanner: {
    border: "1px solid #d9a441",
    background: "#fdf3d8",
    color: "#5c4712",
    borderRadius: "8px",
    padding: "0.5rem 0.7rem",
    fontSize: "0.78rem",
    maxWidth: "440px",
  },
  bannerTitle: {
    fontSize: "0.7rem",
    fontWeight: 700,
    textTransform: "uppercase",
    letterSpacing: "0.03em",
    marginBottom: "0.25rem",
  },
  bannerLine: {
    marginBottom: "0.15rem",
  },
  inputRow: {
    display: "flex",
    gap: "0.5rem",
  },
  textarea: {
    flex: 1,
    minHeight: "44px",
    resize: "none",
    padding: "0.5rem 0.7rem",
    fontSize: "0.9rem",
    border: "1px solid #ccc",
    borderRadius: "6px",
    outline: "none",
    fontFamily: "inherit",
  },
  button: {
    alignSelf: "flex-end",
    padding: "0.5rem 1.1rem",
    fontSize: "0.9rem",
    fontWeight: 600,
    color: "#fff",
    background: "#2f6fed",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
  },
  buttonDisabled: {
    background: "#a9c0f0",
    cursor: "not-allowed",
  },
};
