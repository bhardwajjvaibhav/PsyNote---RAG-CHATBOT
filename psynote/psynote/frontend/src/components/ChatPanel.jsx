import { useState, useRef, useEffect } from "react";
import { askQuestion } from "../api/chat";
import { theme } from "../theme";

/**
 * ChatPanel
 *
 * Query-time flow (architecture doc, Section 4) wired to POST
 * /api/chat. Renders the answer plus everything the pipeline computed
 * alongside it -- citations, safety_hits, and the grounding check --
 * since those are the point of the pipeline, not incidental metadata
 * to hide behind a tooltip.
 *
 * The account id currently travels in the request body (see routes.py's
 * known-gap docstring: no auth yet, the id is not derived from a
 * session). This component just passes through whatever account id its
 * parent gives it.
 */
export default function ChatPanel({ patientId, patientName }) {
  const [messages, setMessages] = useState([]); // {role, content, meta?}
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("idle"); // idle | sending | error
  const [errorMessage, setErrorMessage] = useState("");
  const scrollRef = useRef(null);

  // Reset the thread when the selected account changes -- a chat about
  // one person must never be sent as history alongside a question about
  // another.
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
          <div style={styles.emptyText}>Ask a question grounded in your notes.</div>
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
          placeholder="Ask about your notes…"
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
    width: "100%",
    maxWidth: "680px",
    fontFamily: theme.font,
  },
  header: {
    display: "flex",
    alignItems: "baseline",
    gap: "0.4rem",
  },
  label: {
    fontSize: "0.85rem",
    fontWeight: 600,
    color: theme.text,
  },
  patientTag: {
    fontSize: "0.8rem",
    color: theme.textMuted,
  },
  scroll: {
    display: "flex",
    flexDirection: "column",
    gap: "0.6rem",
    maxHeight: "60vh",
    minHeight: "320px",
    overflowY: "auto",
    padding: "1rem",
    border: `1px solid ${theme.border}`,
    borderRadius: theme.radius,
    background: theme.bgRaised,
  },
  emptyText: {
    fontSize: "0.8rem",
    color: theme.textMuted,
  },
  messageRow: {
    display: "flex",
    flexDirection: "column",
    gap: "0.35rem",
  },
  bubble: {
    maxWidth: "520px",
    padding: "0.6rem 0.85rem",
    borderRadius: "10px",
    fontSize: "0.9rem",
    lineHeight: 1.5,
    whiteSpace: "pre-wrap",
  },
  bubbleUser: {
    background: `linear-gradient(135deg, ${theme.gold}, ${theme.mustard})`,
    color: "#1a1405",
  },
  bubbleAssistant: {
    background: theme.bgPanel,
    border: `1px solid ${theme.border}`,
    color: theme.text,
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
    color: theme.textMuted,
    background: theme.bg,
    border: `1px solid ${theme.border}`,
    padding: "0.15rem 0.5rem",
    borderRadius: "6px",
  },
  safetyBanner: {
    border: `1px solid ${theme.danger}`,
    background: "#331d15",
    color: theme.danger,
    borderRadius: "8px",
    padding: "0.5rem 0.7rem",
    fontSize: "0.78rem",
    maxWidth: "520px",
  },
  flaggedBanner: {
    border: "1px solid #6d5a1d",
    background: "#332b16",
    color: theme.mustard,
    borderRadius: "8px",
    padding: "0.5rem 0.7rem",
    fontSize: "0.78rem",
    maxWidth: "520px",
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
    padding: "0.6rem 0.8rem",
    fontSize: "0.9rem",
    color: theme.text,
    background: theme.bg,
    border: `1px solid ${theme.border}`,
    borderRadius: theme.radius,
    outline: "none",
    fontFamily: theme.font,
    boxSizing: "border-box",
  },
  button: {
    alignSelf: "flex-end",
    padding: "0.5rem 1.1rem",
    fontSize: "0.9rem",
    fontWeight: 600,
    color: "#1a1405",
    background: `linear-gradient(135deg, ${theme.gold}, ${theme.mustard})`,
    border: "none",
    borderRadius: theme.radius,
    cursor: "pointer",
    fontFamily: theme.font,
  },
  buttonDisabled: {
    opacity: 0.6,
    cursor: "not-allowed",
  },
};
