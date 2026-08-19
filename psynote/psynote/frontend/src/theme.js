/**
 * theme.js
 *
 * Shared design tokens for PsyNote's light-black + gold/mustard palette.
 * Every component imports these so the look stays consistent across the
 * auth screen and the dashboard without each file hard-coding hex values.
 */

export const theme = {
  bg: "#1b1b1e",
  bgRaised: "#222226",
  bgPanel: "#26262b",
  border: "#3b3b43",
  borderGold: "#5c4b1e",

  gold: "#d4af37",
  goldBright: "#e8c96a",
  goldDim: "#8a742a",
  mustard: "#e3b341",

  text: "#efe8d8",
  textMuted: "#a9a291",
  textFaint: "#8a8474",

  danger: "#e08b6a",
  success: "#8fca7a",
  warn: "#e3b341",

  radius: "10px",
  font: "'Segoe UI', system-ui, -apple-system, sans-serif",
};

export const inputStyle = {
  width: "100%",
  padding: "0.6rem 0.8rem",
  fontSize: "0.95rem",
  color: theme.text,
  background: theme.bg,
  border: `1px solid ${theme.border}`,
  borderRadius: theme.radius,
  outline: "none",
  boxSizing: "border-box",
  fontFamily: theme.font,
};

export const buttonStyle = (variant = "primary") => {
  const base = {
    padding: "0.6rem 1.2rem",
    fontSize: "0.9rem",
    fontWeight: 600,
    border: "none",
    borderRadius: theme.radius,
    cursor: "pointer",
    fontFamily: theme.font,
    letterSpacing: "0.02em",
  };
  if (variant === "ghost") {
    return {
      ...base,
      background: "transparent",
      color: theme.gold,
      border: `1px solid ${theme.borderGold}`,
    };
  }
  return {
    ...base,
    background: `linear-gradient(135deg, ${theme.gold} 0%, ${theme.mustard} 100%)`,
    color: "#1a1405",
  };
};