import { useState, useEffect, useCallback } from "react";
import { listNotes } from "../api/notes";

/**
 * NotesList
 *
 * Read-only view of a patient's ingested notes (GET
 * /api/patients/{id}/notes), so the "pending -> indexed / failed"
 * lifecycle from doc_registry.py's status field is actually visible
 * somewhere, not just implied by SessionNotesUpload's one-shot message.
 *
 * `refreshKey` is a bump-to-refetch prop: SessionNotesUpload's
 * onUploaded callback increments it in the parent (App.jsx) so a fresh
 * upload shows up here without polling.
 */
export default function NotesList({ patientId, refreshKey }) {
  const [notes, setNotes] = useState([]);
  const [status, setStatus] = useState("loading"); // loading | ready | error
  const [errorMessage, setErrorMessage] = useState("");

  const load = useCallback(async () => {
    setStatus("loading");
    try {
      const data = await listNotes(patientId);
      setNotes(data);
      setStatus("ready");
    } catch (err) {
      setErrorMessage(err.message || "Failed to load notes");
      setStatus("error");
    }
  }, [patientId]);

  useEffect(() => {
    if (patientId) load();
  }, [patientId, refreshKey, load]);

  if (!patientId) return null;

  return (
    <div style={styles.container}>
      <div style={styles.headerRow}>
        <span style={styles.label}>Notes</span>
        <button type="button" onClick={load} style={styles.refreshButton}>
          refresh
        </button>
      </div>

      {status === "loading" && <div style={styles.statusText}>Loading notes…</div>}
      {status === "error" && (
        <div style={styles.statusError}>Couldn't load notes: {errorMessage}</div>
      )}
      {status === "ready" && notes.length === 0 && (
        <div style={styles.statusText}>No notes ingested for this patient yet.</div>
      )}

      {status === "ready" && notes.length > 0 && (
        <ul style={styles.list}>
          {notes.map((n) => (
            <li key={n.id} style={styles.row}>
              <span style={styles.filename}>{n.filename}</span>
              <span style={{ ...styles.pill, ...pillStyleFor(n.status) }}>{n.status}</span>
              {n.status === "failed" && n.error_detail && (
                <div style={styles.errorDetail}>{n.error_detail}</div>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function pillStyleFor(status) {
  if (status === "indexed") return { background: "#e6f4ea", color: "#237a44" };
  if (status === "pending") return { background: "#fdf3d8", color: "#8a6a1a" };
  if (status === "failed") return { background: "#fbe4dd", color: "#b4491f" };
  return { background: "#eee", color: "#666" };
}

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    gap: "0.5rem",
    maxWidth: "420px",
    fontFamily: "system-ui, sans-serif",
    marginTop: "1.25rem",
  },
  headerRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  label: {
    fontSize: "0.85rem",
    fontWeight: 600,
    color: "#333",
  },
  refreshButton: {
    background: "none",
    border: "none",
    color: "#2f6fed",
    fontSize: "0.8rem",
    cursor: "pointer",
    padding: 0,
  },
  statusText: {
    fontSize: "0.8rem",
    color: "#666",
  },
  statusError: {
    fontSize: "0.8rem",
    color: "#c0392b",
  },
  list: {
    listStyle: "none",
    margin: 0,
    padding: 0,
    display: "flex",
    flexDirection: "column",
    gap: "0.4rem",
  },
  row: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    padding: "0.4rem 0",
    borderBottom: "1px solid #eee",
    flexWrap: "wrap",
  },
  filename: {
    fontSize: "0.85rem",
    flex: 1,
    minWidth: 0,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  pill: {
    fontSize: "0.7rem",
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: "0.03em",
    padding: "0.15rem 0.5rem",
    borderRadius: "999px",
  },
  errorDetail: {
    flexBasis: "100%",
    fontSize: "0.75rem",
    color: "#b4491f",
  },
};
