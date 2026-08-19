import { useState, useEffect, useMemo } from "react";
import { listPatients } from "../api/patients";

/**
 * PatientPicker
 *
 * On mount: GET /api/patients -> renders a searchable list of registered
 * users. On select: lifts the selected id up via onSelectUser(user).
 *
 * This component owns nothing beyond "which user is selected" -- it
 * does not know about ingestion, chat, or anything downstream. Session
 * notes upload is a hard dependent (see architecture doc, Section 2.5):
 * nothing can be tagged with an account until one exists here.
 * Creating a user is likewise a separate concern -- see
 * NewPatientForm.jsx -- kept out of this component on purpose.
 */
export default function PatientPicker({ selectedUserId, onSelectUser }) {
  const [users, setUsers] = useState([]);
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
          setUsers(data);
          setStatus("ready");
        }
      } catch (err) {
        if (!cancelled) {
          setErrorMessage(err.message || "Failed to load users");
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
    if (!q) return users;
    return users.filter((p) => p.name.toLowerCase().includes(q));
  }, [users, query]);

  return (
    <div style={styles.container}>
      <label htmlFor="user-search" style={styles.label}>
        User
      </label>
      <input
        id="user-search"
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search users by name"
        style={styles.input}
        disabled={status === "loading"}
        aria-describedby="user-picker-status"
      />

      <div id="user-picker-status" aria-live="polite" style={styles.statusText}>
        {status === "loading" && "Loading users…"}
        {status === "error" && `Couldn't load users: ${errorMessage}`}
        {status === "ready" && users.length === 0 && "No users yet."}
        {status === "ready" &&
          users.length > 0 &&
          filtered.length === 0 &&
          "No users match your search."}
      </div>

      {status === "ready" && filtered.length > 0 && (
        <ul style={styles.list} role="listbox" aria-label="Users">
          {filtered.map((p) => {
            const isSelected = p.id === selectedUserId;
            return (
              <li key={p.id}>
                <button
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  onClick={() => onSelectUser(p)}
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
