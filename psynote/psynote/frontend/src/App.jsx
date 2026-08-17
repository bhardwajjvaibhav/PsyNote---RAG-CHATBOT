import { useState } from "react";
import PatientPicker from "./components/PatientPicker";
import NewPatientForm from "./components/NewPatientForm";
import SessionNotesUpload from "./components/SessionNotesUpload";
import NotesList from "./components/NotesList";
import ChatPanel from "./components/ChatPanel";

/**
 * App
 *
 * Holds selectedPatientId (architecture doc, Section 2.5 / Section 7's
 * folder structure) and renders the picker + upload flow, plus the
 * query-time chat flow (Section 4) once a patient is selected.
 *
 * patientListVersion is a bump-to-remount key passed to PatientPicker:
 * PatientPicker only fetches on mount, by design (see its own
 * docstring), so creating a patient via NewPatientForm has to force a
 * remount to make the new patient show up in the list -- this is that
 * mechanism, not a workaround for a bug in PatientPicker.
 */
export default function App() {
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [patientListVersion, setPatientListVersion] = useState(0);
  const [notesRefreshKey, setNotesRefreshKey] = useState(0);
  const [tab, setTab] = useState("notes"); // notes | chat

  function handlePatientCreated(patient) {
    setPatientListVersion((v) => v + 1);
    setSelectedPatient(patient);
    setTab("notes");
  }

  function handleSelectPatient(patient) {
    setSelectedPatient(patient);
  }

  function handleNoteUploaded() {
    setNotesRefreshKey((k) => k + 1);
  }

  return (
    <div style={styles.shell}>
      <aside style={styles.sidebar}>
        <h1 style={styles.brand}>PsyNote</h1>

        <NewPatientForm onCreated={handlePatientCreated} />

        <div style={styles.divider} />

        <PatientPicker
          key={patientListVersion}
          selectedPatientId={selectedPatient?.id ?? null}
          onSelectPatient={handleSelectPatient}
        />
      </aside>

      <main style={styles.main}>
        {!selectedPatient && (
          <div style={styles.noPatient}>Select or create a patient on the left to get started.</div>
        )}

        {selectedPatient && (
          <>
            <div style={styles.topbar}>
              <h2 style={styles.patientTitle}>{selectedPatient.name}</h2>
              <div style={styles.tabs}>
                <button
                  type="button"
                  onClick={() => setTab("notes")}
                  style={{ ...styles.tab, ...(tab === "notes" ? styles.tabActive : {}) }}
                >
                  Notes
                </button>
                <button
                  type="button"
                  onClick={() => setTab("chat")}
                  style={{ ...styles.tab, ...(tab === "chat" ? styles.tabActive : {}) }}
                >
                  Chat
                </button>
              </div>
            </div>

            {tab === "notes" && (
              <>
                <SessionNotesUpload
                  patientId={selectedPatient.id}
                  patientName={selectedPatient.name}
                  onUploaded={handleNoteUploaded}
                />
                <NotesList patientId={selectedPatient.id} refreshKey={notesRefreshKey} />
              </>
            )}

            {tab === "chat" && (
              <ChatPanel patientId={selectedPatient.id} patientName={selectedPatient.name} />
            )}
          </>
        )}
      </main>
    </div>
  );
}

const styles = {
  shell: {
    display: "flex",
    minHeight: "100vh",
    fontFamily: "system-ui, sans-serif",
    background: "#f7f8fa",
  },
  sidebar: {
    width: "300px",
    flexShrink: 0,
    background: "#fff",
    borderRight: "1px solid #e0e0e0",
    padding: "1.25rem",
    display: "flex",
    flexDirection: "column",
    gap: "1rem",
  },
  brand: {
    fontSize: "1.2rem",
    margin: 0,
  },
  divider: {
    height: "1px",
    background: "#eee",
  },
  main: {
    flex: 1,
    padding: "1.5rem 2rem",
    minWidth: 0,
  },
  noPatient: {
    color: "#666",
    fontSize: "0.95rem",
  },
  topbar: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: "1rem",
  },
  patientTitle: {
    fontSize: "1.3rem",
    margin: 0,
  },
  tabs: {
    display: "flex",
    gap: "0.4rem",
  },
  tab: {
    padding: "0.4rem 0.9rem",
    fontSize: "0.85rem",
    fontWeight: 600,
    background: "none",
    border: "1px solid #e0e0e0",
    borderRadius: "6px",
    cursor: "pointer",
    color: "#666",
  },
  tabActive: {
    background: "#eef4ff",
    color: "#2f6fed",
    borderColor: "#2f6fed",
  },
};
