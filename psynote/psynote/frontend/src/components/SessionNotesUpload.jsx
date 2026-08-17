import { useState, useRef } from "react";
import { ingestNote } from "../api/notes";

/**
 * SessionNotesUpload
 *
 * Shown once a patient is selected (parent guarantees selectedPatientId
 * is non-null before mounting this -- see architecture doc, Section 2.5:
 * "the upload form has nothing to tag notes with until a patient is
 * chosen first").
 *
 * Accepts .txt / .pdf / .csv. On submit, POSTs multipart/form-data to
 * /api/ingest with { patient_id, file }.
 *
 * /api/ingest is now real (Phases 2-4 shipped). The success response
 * shape is { note_id, status, chunk_count, error }, and status can be
 * "pending"/"indexed" (retrievable soon/now) OR "failed" -- a pipeline
 * failure (bad PDF, embedding error, a Chroma/BM25 write failure) is
 * NOT surfaced as an HTTP error by the backend (see routes.py's
 * docstring): the note is written to doc_registry as "failed" with a
 * reason attached instead, so it stays visible and retryable rather
 * than just a request to retry blindly. This component checks
 * result.status, not just res.ok, or a failed ingest would look
 * identical to a successful one.
 */

const ACCEPTED_EXTENSIONS = [".txt", ".pdf", ".csv"];

export default function SessionNotesUpload({ patientId, patientName, onUploaded }) {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | uploading | success | failed | error
  const [message, setMessage] = useState("");
  const fileInputRef = useRef(null);

  function isAcceptedFile(f) {
    if (!f) return false;
    const name = f.name.toLowerCase();
    return ACCEPTED_EXTENSIONS.some((ext) => name.endsWith(ext));
  }

  function handleFileChange(e) {
    const selected = e.target.files?.[0] || null;
    if (selected && !isAcceptedFile(selected)) {
      setFile(null);
      setStatus("error");
      setMessage(`Unsupported file type. Accepted: ${ACCEPTED_EXTENSIONS.join(", ")}`);
      return;
    }
    setFile(selected);
    setStatus("idle");
    setMessage("");
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!file) {
      setStatus("error");
      setMessage("Choose a file before uploading.");
      return;
    }
    if (!patientId) {
      // Defensive only -- parent should never mount this without a
      // selected patient (see Section 2.5). Still, never send an
      // ingest request with no patient_id to tag it with.
      setStatus("error");
      setMessage("No patient selected.");
      return;
    }

    setStatus("uploading");
    setMessage("");

    try {
      const result = await ingestNote(patientId, file);

      if (result.status === "failed") {
        setStatus("failed");
        setMessage(result.error || "Ingestion failed for an unknown reason.");
      } else {
        setStatus("success");
        setMessage(
          `Ingested as ${result.chunk_count} chunk${result.chunk_count === 1 ? "" : "s"} (${result.status}).`
        );
      }

      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      onUploaded?.(result);
    } catch (err) {
      // A thrown error here means the HTTP request itself failed
      // (network, unknown patient -> 404, unsupported extension -> 400)
      // -- distinct from a pipeline-level "failed" status above, which
      // arrives as a normal 201.
      setStatus("error");
      setMessage(err.message || "Upload request failed.");
    }
  }

  return (
    <form onSubmit={handleSubmit} style={styles.container}>
      <div style={styles.header}>
        <span style={styles.label}>Session notes</span>
        <span style={styles.patientTag}>for {patientName}</span>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED_EXTENSIONS.join(",")}
        onChange={handleFileChange}
        disabled={status === "uploading"}
        style={styles.fileInput}
      />

      <button
        type="submit"
        disabled={status === "uploading" || !file}
        style={{
          ...styles.button,
          ...(status === "uploading" || !file ? styles.buttonDisabled : {}),
        }}
      >
        {status === "uploading" ? "Uploading…" : "Upload"}
      </button>

      <div
        aria-live="polite"
        style={{
          ...styles.statusText,
          ...(status === "error" || status === "failed" ? styles.statusError : {}),
          ...(status === "success" ? styles.statusSuccess : {}),
        }}
      >
        {message}
      </div>
    </form>
  );
}

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    gap: "0.6rem",
    maxWidth: "360px",
    fontFamily: "system-ui, sans-serif",
    marginTop: "1rem",
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
  fileInput: {
    fontSize: "0.85rem",
  },
  button: {
    alignSelf: "flex-start",
    padding: "0.45rem 1rem",
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
  statusText: {
    fontSize: "0.8rem",
    color: "#666",
    minHeight: "1.1em",
  },
  statusError: {
    color: "#c0392b",
  },
  statusSuccess: {
    color: "#2f8f5b",
  },
};
