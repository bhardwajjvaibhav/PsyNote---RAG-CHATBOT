import { useState, useEffect, useMemo } from "react";
import { listPatients } from "../api/patients";

/**
 * PatientPicker
 *
 * On mount: GET /api/patients -> renders a searchable list of active patients.
 * On select: lifts selectedPatientId up via onSelectPatient(patient).
 *
 * This component owns nothing beyond "which patient is selected" -- it
 * does not know about ingestion, chat, or anything downstream. Session
 * notes upload is a hard dependent (see architecture doc, Section 2.5):
 * nothing can be tagged with patient_id until a patient exists here.
 * Creating a patient is likewise a separate concern -- see
 * NewPatientForm.jsx -- kept out of this component on purpose.
 */
export default function PatientPicker({ selectedPatientId, onSelectPatient }) {
  const [patients, setPatients] = useState([]);
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("loading"); // loading | ready | error
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setStatus("loading");
      try {
        const data = await listPatients();
        if (!cancelled) {
          setPatients(data);
          setStatus("ready");
        }
      } catch (err) {
        if (!cancelled) {
          setErrorMessage(err.message || "Failed to load patients");
          setStatus("error");
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return patients;
    return patients.filter((p) => p.name.toLowerCase().includes(q));
  }, [patients, query]);

  return (
    <div style={styles.container}>
      <label htmlFor="patient-search" style={styles.label}>
        Patient
      </label>
      <input
        id="patient-search"
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search patients by name"
        style={styles.input}
        disabled={status === "loading"}
        aria-describedby="patient-picker-status"
      />

      <div id="patient-picker-status" aria-live="polite" style={styles.statusText}>
        {status === "loading" && "Loading patients…"}
        {status === "error" && `Couldn't load patients: ${errorMessage}`}
        {status === "ready" && patients.length === 0 && "No patients yet."}
        {status === "ready" &&
          patients.length > 0 &&
          filtered.length === 0 &&
          "No patients match your search."}
      </div>

      {status === "ready" && filtered.length > 0 && (
        <ul style={styles.list} role="listbox" aria-label="Active patients">
          {filtered.map((p) => {
            const isSelected = p.id === selectedPatientId;
            return (
              <li key={p.id}>
                <button
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  onClick={() => onSelectPatient(p)}
                  style={{
                    ...styles.patientButton,
                    ...(isSelected ? styles.patientButtonSelected : {}),
                  }}
                >
                  {p.name}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    gap: "0.5rem",
    maxWidth: "360px",
    fontFamily: "system-ui, sans-serif",
  },
  label: {
    fontSize: "0.85rem",
    fontWeight: 600,
    color: "#333",
  },
  input: {
    padding: "0.5rem 0.75rem",
    fontSize: "0.95rem",
    border: "1px solid #ccc",
    borderRadius: "6px",
    outline: "none",
  },
  statusText: {
    fontSize: "0.8rem",
    color: "#666",
    minHeight: "1.1em",
  },
  list: {
    listStyle: "none",
    margin: 0,
    padding: 0,
    maxHeight: "260px",
    overflowY: "auto",
    border: "1px solid #e0e0e0",
    borderRadius: "6px",
  },
  patientButton: {
    width: "100%",
    textAlign: "left",
    padding: "0.5rem 0.75rem",
    background: "none",
    border: "none",
    borderBottom: "1px solid #eee",
    cursor: "pointer",
    fontSize: "0.9rem",
  },
  patientButtonSelected: {
    background: "#eef4ff",
    fontWeight: 600,
  },
};
