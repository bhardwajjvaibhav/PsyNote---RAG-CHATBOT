import { useState } from "react";
import { createPatient } from "../api/patients";

/**
 * NewPatientForm
 *
 * Separate from PatientPicker on purpose -- PatientPicker's docstring
 * says it "owns nothing beyond which user is selected". Creating a user
 * is a different concern (a POST, and a decision about what happens to
 * the new record), so it gets its own small component rather than
 * growing PatientPicker's responsibilities.
 *
 * On success, calls onCreated(user) so the parent can refresh
 * PatientPicker's list and select the new user immediately.
 */
export default function NewPatientForm({ onCreated }) {
  const [name, setName] = useState("");
  const [status, setStatus] = useState("idle"); // idle | saving | error
  const [message, setMessage] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) {
      setStatus("error");
      setMessage("Name cannot be empty.");
      return;
    }

    setStatus("saving");
    setMessage("");
    try {
      const user = await createPatient(trimmed);
      setName("");
      setStatus("idle");
      onCreated?.(user);
    } catch (err) {
      setStatus("error");
      setMessage(err.message || "Failed to create user.");
    }
  }

  return (
    <form onSubmit={handleSubmit} style={styles.container}>
      <label htmlFor="new-user-name" style={styles.label}>
        Add user
      </label>
      <div style={styles.row}>
        <input
          id="new-user-name"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="User name"
          disabled={status === "saving"}
          style={styles.input}
        />
        <button
          type="submit"
          disabled={status === "saving" || !name.trim()}
          style={{
            ...styles.button,
            ...(status === "saving" || !name.trim() ? styles.buttonDisabled : {}),
          }}
        >
          {status === "saving" ? "Adding…" : "Add"}
        </button>
      </div>
      {status === "error" && (
        <div aria-live="polite" style={styles.statusError}>
          {message}
        </div>
      )}
    </form>
  );
}

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    gap: "0.4rem",
    maxWidth: "360px",
    fontFamily: "system-ui, sans-serif",
  },
  label: {
    fontSize: "0.85rem",
    fontWeight: 600,
    color: "#333",
  },
  row: {
    display: "flex",
    gap: "0.4rem",
  },
  input: {
    flex: 1,
    padding: "0.5rem 0.75rem",
    fontSize: "0.9rem",
    border: "1px solid #ccc",
    borderRadius: "6px",
    outline: "none",
  },
  button: {
    padding: "0.5rem 0.9rem",
    fontSize: "0.85rem",
    fontWeight: 600,
    color: "#fff",
    background: "#2f6fed",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
    whiteSpace: "nowrap",
  },
  buttonDisabled: {
    background: "#a9c0f0",
    cursor: "not-allowed",
  },
  statusError: {
    fontSize: "0.8rem",
    color: "#c0392b",
  },
};
