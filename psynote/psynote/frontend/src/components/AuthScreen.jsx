import { useState } from "react";
import { loginUser, registerUser } from "../api/auth";
import { theme, inputStyle, buttonStyle } from "../theme";

/**
 * AuthScreen
 *
 * The landing page: PsyNote brand + "think analyze act" tagline, with a
 * Sign in / Register toggle. Registration collects the profile fields the
 * dashboard header shows (name, age, place, gender, marital status).
 * On success calls onAuthenticated(user) with the user record.
 */
export default function AuthScreen({ onAuthenticated }) {
  const [mode, setMode] = useState("login"); // login | register
  const [form, setForm] = useState({
    name: "",
    age: "",
    place: "",
    gender: "",
    marital_status: "",
  });
  const [status, setStatus] = useState("idle"); // idle | submitting | error
  const [message, setMessage] = useState("");

  function updateField(field, value) {
    setForm((f) => ({ ...f, [field]: value }));
    setMessage("");
  }

  function switchMode(next) {
    setMode(next);
    setMessage("");
    setStatus("idle");
  }

  async function handleSubmit(e) {
    e.preventDefault();
    const name = form.name.trim();
    if (!name) {
      setStatus("error");
      setMessage("Name is required.");
      return;
    }
    if (mode === "register") {
      const age = form.age.trim();
      if (age && (Number.isNaN(Number(age)) || Number(age) < 0 || Number(age) > 130)) {
        setStatus("error");
        setMessage("Age must be a whole number between 0 and 130.");
        return;
      }
    }

    setStatus("submitting");
    setMessage("");

    try {
      const user =
        mode === "register"
          ? await registerUser(name, {
              age: form.age.trim() ? Number(form.age.trim()) : null,
              place: form.place.trim(),
              gender: form.gender.trim(),
              marital_status: form.marital_status.trim(),
            })
          : await loginUser(name);
      setStatus("idle");
      onAuthenticated(user);
    } catch (err) {
      setStatus("error");
      setMessage(err.message || "Something went wrong. Please try again.");
    }
  }

  return (
    <div style={styles.wrap}>
      <div style={styles.card}>
        <div style={styles.brand}>
          <span style={styles.brandGold}>Psy</span>
          <span style={styles.brandText}>Note</span>
        </div>
        <div style={styles.tagline}>think · analyze · act</div>

        <div style={styles.tabs}>
          <button
            type="button"
            onClick={() => switchMode("login")}
            style={{ ...styles.tab, ...(mode === "login" ? styles.tabActive : {}) }}
          >
            Sign in
          </button>
          <button
            type="button"
            onClick={() => switchMode("register")}
            style={{ ...styles.tab, ...(mode === "register" ? styles.tabActive : {}) }}
          >
            Register
          </button>
        </div>

        <form onSubmit={handleSubmit} style={styles.form}>
          <label style={styles.fieldLabel} htmlFor="auth-name">
            Name
          </label>
          <input
            id="auth-name"
            type="text"
            value={form.name}
            onChange={(e) => updateField("name", e.target.value)}
            placeholder="Your full name"
            disabled={status === "submitting"}
            style={inputStyle}
          />

          {mode === "register" && (
            <>
              <div style={styles.twoCol}>
                <div style={styles.col}>
                  <label style={styles.fieldLabel} htmlFor="auth-age">
                    Age
                  </label>
                  <input
                    id="auth-age"
                    type="number"
                    min="0"
                    max="130"
                    value={form.age}
                    onChange={(e) => updateField("age", e.target.value)}
                    placeholder="Age"
                    disabled={status === "submitting"}
                    style={inputStyle}
                  />
                </div>
                <div style={styles.col}>
                  <label style={styles.fieldLabel} htmlFor="auth-gender">
                    Gender
                  </label>
                  <select
                    id="auth-gender"
                    value={form.gender}
                    onChange={(e) => updateField("gender", e.target.value)}
                    disabled={status === "submitting"}
                    style={inputStyle}
                  >
                    <option value="">Select</option>
                    <option value="female">Female</option>
                    <option value="male">Male</option>
                    <option value="non-binary">Non-binary</option>
                    <option value="other">Other</option>
                    <option value="prefer not to say">Prefer not to say</option>
                  </select>
                </div>
              </div>

              <label style={styles.fieldLabel} htmlFor="auth-place">
                Place
              </label>
              <input
                id="auth-place"
                type="text"
                value={form.place}
                onChange={(e) => updateField("place", e.target.value)}
                placeholder="City / region"
                disabled={status === "submitting"}
                style={inputStyle}
              />

              <label style={styles.fieldLabel} htmlFor="auth-marital">
                Marital status
              </label>
              <select
                id="auth-marital"
                value={form.marital_status}
                onChange={(e) => updateField("marital_status", e.target.value)}
                disabled={status === "submitting"}
                style={inputStyle}
              >
                <option value="">Select</option>
                <option value="single">Single</option>
                <option value="married">Married</option>
                <option value="divorced">Divorced</option>
                <option value="widowed">Widowed</option>
                <option value="separated">Separated</option>
                <option value="prefer not to say">Prefer not to say</option>
              </select>
            </>
          )}

          <button
            type="submit"
            disabled={status === "submitting"}
            style={{
              ...buttonStyle("primary"),
              ...(status === "submitting" ? styles.submitDisabled : {}),
            }}
          >
            {status === "submitting"
              ? "Please wait…"
              : mode === "register"
                ? "Create account"
                : "Sign in"}
          </button>

          <div aria-live="polite" style={status === "error" ? styles.error : styles.hint}>
            {message}
          </div>
        </form>
      </div>
    </div>
  );
}

const styles = {
  wrap: {
    minHeight: "100vh",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: `radial-gradient(1200px 600px at 50% -10%, ${theme.bgRaised}, ${theme.bg} 65%)`,
    fontFamily: theme.font,
    padding: "1rem",
    boxSizing: "border-box",
  },
  card: {
    width: "100%",
    maxWidth: "420px",
    background: theme.bgPanel,
    border: `1px solid ${theme.border}`,
    borderRadius: "18px",
    padding: "2.25rem 2rem",
    boxShadow: "0 24px 60px rgba(0,0,0,0.55), 0 0 0 1px rgba(212,175,55,0.08)",
    boxSizing: "border-box",
  },
  brand: {
    textAlign: "center",
    fontSize: "2.6rem",
    fontWeight: 800,
    letterSpacing: "0.04em",
  },
  brandGold: {
    color: theme.gold,
  },
  brandText: {
    color: theme.text,
  },
  tagline: {
    textAlign: "center",
    color: theme.mustard,
    fontSize: "0.85rem",
    letterSpacing: "0.42em",
    textTransform: "uppercase",
    marginTop: "0.4rem",
    marginBottom: "1.75rem",
  },
  tabs: {
    display: "flex",
    background: theme.bg,
    border: `1px solid ${theme.border}`,
    borderRadius: theme.radius,
    padding: "0.25rem",
    marginBottom: "1.5rem",
  },
  tab: {
    flex: 1,
    padding: "0.55rem 0",
    fontSize: "0.9rem",
    fontWeight: 600,
    color: theme.textMuted,
    background: "transparent",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    fontFamily: theme.font,
  },
  tabActive: {
    background: `linear-gradient(135deg, ${theme.gold}, ${theme.mustard})`,
    color: "#1a1405",
  },
  form: {
    display: "flex",
    flexDirection: "column",
    gap: "0.4rem",
  },
  fieldLabel: {
    fontSize: "0.75rem",
    fontWeight: 600,
    color: theme.textMuted,
    textTransform: "uppercase",
    letterSpacing: "0.06em",
    marginTop: "0.5rem",
  },
  twoCol: {
    display: "flex",
    gap: "0.75rem",
  },
  col: {
    flex: 1,
    minWidth: 0,
  },
  submitDisabled: {
    opacity: 0.6,
    cursor: "not-allowed",
  },
  error: {
    fontSize: "0.82rem",
    color: theme.danger,
    minHeight: "1.2em",
    marginTop: "0.5rem",
  },
  hint: {
    fontSize: "0.82rem",
    color: theme.textFaint,
    minHeight: "1.2em",
    marginTop: "0.5rem",
  },
};