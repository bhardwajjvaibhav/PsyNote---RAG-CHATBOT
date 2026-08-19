import { useState } from "react";
import { createTextNote } from "../api/notes";
import SessionNotesUpload from "./SessionNotesUpload";
import NotesList from "./NotesList";
import ChatPanel from "./ChatPanel";
import InsightsPanel from "./InsightsPanel";
import { theme, inputStyle, buttonStyle } from "../theme";

/**
 * Dashboard
 *
 * Post-auth screen: the header shows the signed-in user's name plus age
 * and gender. The left panel owns the notes (write a new note, upload a
 * file, see the ingested list); the center switches between the query-
 * time chat (ChatPanel) and the emotional-energy insights (InsightsPanel),
 * both grounded in those notes.
 */
export default function Dashboard({ session, onSignOut }) {
  const [refreshKey, setRefreshKey] = useState(0);
  const [tab, setTab] = useState("chat"); // chat | insights

  function handleNoteSaved() {
    setRefreshKey((k) => k + 1);
  }

  return (
    <div style={styles.shell}>
      <header style={styles.header}>
        <div style={styles.brand}>
          <span style={styles.brandGold}>Psy</span>
          <span style={styles.brandText}>Note</span>
        </div>

        <div style={styles.profile}>
          <span style={styles.profileName}>{session.name}</span>
          <span style={styles.profileChip}>{session.age != null ? `${session.age} yrs` : "—"}</span>
          <span style={styles.profileChip}>{session.gender || "—"}</span>
        </div>

        <button type="button" onClick={onSignOut} style={buttonStyle("ghost")}>
          Sign out
        </button>
      </header>

      <div style={styles.body}>
        <aside style={styles.sidebar}>
          <NewNoteForm patientId={session.id} onSaved={handleNoteSaved} />
          <SectionDivider label="or upload a session note" />
          <SessionNotesUpload
            patientId={session.id}
            patientName={session.name}
            onUploaded={handleNoteSaved}
          />
          <SectionDivider label="ingested notes" />
          <NotesList patientId={session.id} patientName={session.name} refreshKey={refreshKey} />
        </aside>

        <main style={styles.main}>
          <div style={styles.tabs}>
            <button
              type="button"
              onClick={() => setTab("chat")}
              style={{ ...styles.tab, ...(tab === "chat" ? styles.tabActive : {}) }}
            >
              Chat
            </button>
            <button
              type="button"
              onClick={() => setTab("insights")}
              style={{ ...styles.tab, ...(tab === "insights" ? styles.tabActive : {}) }}
            >
              Insights
            </button>
          </div>

          <div style={styles.tabBody}>
            {tab === "chat" ? (
              <ChatPanel patientId={session.id} patientName={session.name} />
            ) : (
              <InsightsPanel userId={session.id} userName={session.name} />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

function SectionDivider({ label }) {
  return (
    <div style={styles.divider}>
      <span style={styles.dividerLine} />
      <span style={styles.dividerLabel}>{label}</span>
      <span style={styles.dividerLine} />
    </div>
  );
}

function NewNoteForm({ patientId, onSaved }) {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [status, setStatus] = useState("idle"); // idle | saving | error
  const [message, setMessage] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    if (!title.trim() || !content.trim()) {
      setStatus("error");
      setMessage("A title and some content are required.");
      return;
    }

    setStatus("saving");
    setMessage("");
    try {
      const result = await createTextNote(patientId, title.trim(), content);
      if (result.status === "failed") {
        setStatus("error");
        setMessage(result.error || "Could not save the note.");
      } else {
        setTitle("");
        setContent("");
        setStatus("idle");
        setMessage(`Saved as ${result.chunk_count} chunk${result.chunk_count === 1 ? "" : "s"}.`);
        onSaved?.();
      }
    } catch (err) {
      setStatus("error");
      setMessage(err.message || "Could not save the note.");
    }
  }

  return (
    <form onSubmit={handleSubmit} style={styles.noteForm}>
      <div style={styles.formHeading}>Create a new note</div>
      <input
        type="text"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="Title (e.g. Session 12)"
        disabled={status === "saving"}
        style={inputStyle}
      />
      <textarea
        value={content}
        onChange={(e) => setContent(e.target.value)}
        placeholder="Write today's reflections, moods, events…"
        rows={5}
        disabled={status === "saving"}
        style={{ ...inputStyle, resize: "vertical", minHeight: "96px", lineHeight: 1.5 }}
      />
      <button
        type="submit"
        disabled={status === "saving"}
        style={{
          ...buttonStyle("primary"),
          ...(status === "saving" ? styles.submitDisabled : {}),
        }}
      >
        {status === "saving" ? "Saving…" : "Save note"}
      </button>
      <div
        aria-live="polite"
        style={status === "error" ? styles.formError : styles.formHint}
      >
        {message}
      </div>
    </form>
  );
}

const styles = {
  shell: {
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    background: theme.bg,
    color: theme.text,
    fontFamily: theme.font,
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "1rem",
    padding: "0.9rem 1.5rem",
    background: theme.bgRaised,
    borderBottom: `1px solid ${theme.border}`,
  },
  brand: {
    fontSize: "1.35rem",
    fontWeight: 800,
    letterSpacing: "0.04em",
  },
  brandGold: { color: theme.gold },
  brandText: { color: theme.text },
  profile: {
    display: "flex",
    alignItems: "center",
    gap: "0.6rem",
  },
  profileName: {
    fontSize: "1.05rem",
    fontWeight: 700,
  },
  profileChip: {
    fontSize: "0.78rem",
    color: theme.textMuted,
    background: theme.bgPanel,
    border: `1px solid ${theme.border}`,
    borderRadius: "999px",
    padding: "0.2rem 0.7rem",
  },
  body: {
    flex: 1,
    display: "flex",
    minHeight: 0,
  },
  sidebar: {
    width: "340px",
    flexShrink: 0,
    borderRight: `1px solid ${theme.border}`,
    padding: "1.25rem 1.25rem 2rem",
    overflowY: "auto",
    background: theme.bgRaised,
    boxSizing: "border-box",
    display: "flex",
    flexDirection: "column",
    gap: "1.1rem",
  },
  main: {
    flex: 1,
    minWidth: 0,
    padding: "1.5rem 2rem",
    overflowY: "auto",
  },
  tabs: {
    display: "flex",
    gap: "0.4rem",
    marginBottom: "1.1rem",
  },
  tab: {
    padding: "0.45rem 1rem",
    fontSize: "0.85rem",
    fontWeight: 600,
    background: theme.bgRaised,
    border: `1px solid ${theme.border}`,
    borderRadius: theme.radius,
    cursor: "pointer",
    color: theme.textMuted,
    fontFamily: theme.font,
  },
  tabActive: {
    background: `linear-gradient(135deg, ${theme.gold}, ${theme.mustard})`,
    color: "#1a1405",
    borderColor: "transparent",
  },
  tabBody: {
    display: "flex",
    justifyContent: "center",
  },
  divider: {
    display: "flex",
    alignItems: "center",
    gap: "0.6rem",
    margin: "0.25rem 0",
  },
  dividerLine: {
    flex: 1,
    height: "1px",
    background: theme.border,
  },
  dividerLabel: {
    fontSize: "0.68rem",
    color: theme.textFaint,
    textTransform: "uppercase",
    letterSpacing: "0.08em",
    whiteSpace: "nowrap",
  },
  noteForm: {
    display: "flex",
    flexDirection: "column",
    gap: "0.5rem",
  },
  formHeading: {
    fontSize: "0.8rem",
    fontWeight: 700,
    color: theme.mustard,
    textTransform: "uppercase",
    letterSpacing: "0.06em",
  },
  submitDisabled: {
    opacity: 0.6,
    cursor: "not-allowed",
  },
  formError: {
    fontSize: "0.78rem",
    color: theme.danger,
    minHeight: "1.1em",
  },
  formHint: {
    fontSize: "0.78rem",
    color: theme.success,
    minHeight: "1.1em",
  },
};