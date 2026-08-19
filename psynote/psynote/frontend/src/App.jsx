import { useState, useEffect } from "react";
import AuthScreen from "./components/AuthScreen";
import Dashboard from "./components/Dashboard";

const SESSION_KEY = "psynote.session";

/**
 * App
 *
 * Top-level session gate: no session -> AuthScreen (Sign in / Register);
 * session present -> Dashboard. The session is just the user record
 * returned by /api/register or /api/login, persisted to localStorage so
 * a refresh doesn't drop you back to the landing page. Sign out clears it.
 */
export default function App() {
  const [session, setSession] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(SESSION_KEY));
    } catch {
      return null;
    }
  });

  useEffect(() => {
    if (session) {
      localStorage.setItem(SESSION_KEY, JSON.stringify(session));
    } else {
      localStorage.removeItem(SESSION_KEY);
    }
  }, [session]);

  function handleAuthenticated(user) {
    setSession(user);
  }

  function handleSignOut() {
    setSession(null);
  }

  if (!session) {
    return <AuthScreen onAuthenticated={handleAuthenticated} />;
  }

  return <Dashboard session={session} onSignOut={handleSignOut} />;
}