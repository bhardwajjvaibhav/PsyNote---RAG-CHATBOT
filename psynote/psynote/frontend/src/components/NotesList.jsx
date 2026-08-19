import { useState, useEffect, useCallback } from "react";
import { listNotes } from "../api/notes";
import { theme } from "../theme";

/**
 * NotesList
 *
 * Read-only view of an account's ingested notes (GET
 * /api/patients/{id}/notes), so the "pending -> indexed / failed"
 * lifecycle from doc_registry.py's status field is actually visible
 * somewhere, not just implied by SessionNotesUpload's one-shot message.
 *
 * `refreshKey` is a bump-to-refetch prop: an upload/created note bumps
 * it in the parent (Dashboard.jsx) so a fresh note shows up here
 * without polling.
 */
export default function NotesList({ patientId, patientName, refreshKey }) {
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
        <div style={styles.statusText}>No notes ingested yet{patientName ? ` for ${patientName}` : ""}.</div>
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
  if (status === "indexed") return { background: "#1a3323", color: theme.success };
  if (status === "pending") return { background: "#332b16", color: theme.mustard };
  if (status === "failed") return { background: "#331d15", color: theme.danger };
  return { background: theme.bg, color: theme.textMuted };
}

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    gap: "0.5rem",
    fontFamily: theme.font,
  },
  headerRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  label: {
    fontSize: "0.85rem",
    fontWeight: 600,
    color: theme.text,
  },
  refreshButton: {
    background: "none",
    border: "none",
    color: theme.gold,
    fontSize: "0.8rem",
    cursor: "pointer",
    padding: 0,
    fontFamily: theme.font,
  },
  statusText: {
    fontSize: "0.8rem",
    color: theme.textMuted,
  },
  statusError: {
    fontSize: "0.8rem",
    color: theme.danger,
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
    padding: "0.45rem 0.6rem",
    border: `1px solid ${theme.border}`,
    borderRadius: "8px",
    background: theme.bg,
    flexWrap: "wrap",
  },
  filename: {
    fontSize: "0.85rem",
    color: theme.text,
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
    color: theme.danger,
  },
};
