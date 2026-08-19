import { useState, useEffect, useCallback } from "react";
import { getInsights } from "../api/insights";
import { theme } from "../theme";

const EMOTIONS = ["happy", "sad", "stressed", "anxious"];

const EMOTION_META = {
  happy: { label: "Happy", color: "#8fca7a" },
  sad: { label: "Sad", color: "#7a9bd8" },
  stressed: { label: "Stressed", color: "#e3b341" },
  anxious: { label: "Anxious", color: "#e08b6a" },
};

/**
 * InsightsPanel
 *
 * The dashboard's emotional-energy view, from GET /api/insights/{id}:
 * overall energy bars for the four emotions, a per-note trend chart
 * (SVG, no chart dependency), and the "why" -- the matched terms +
 * snippets that drove each emotion, attributed to their note.
 */
export default function InsightsPanel({ userId, userName }) {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("loading"); // loading | ready | error
  const [errorMessage, setErrorMessage] = useState("");

  const load = useCallback(async () => {
    setStatus("loading");
    try {
      const insights = await getInsights(userId);
      setData(insights);
      setStatus("ready");
    } catch (err) {
      setErrorMessage(err.message || "Could not load insights");
      setStatus("error");
    }
  }, [userId]);

  useEffect(() => {
    if (userId) load();
  }, [userId, load]);

  if (!userId) return null;

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.label}>Insights</span>
        <span style={styles.tag}>emotional energy for {userName}</span>
        <button type="button" onClick={load} style={styles.refresh}>
          refresh
        </button>
      </div>

      {status === "loading" && <div style={styles.muted}>Analyzing notes…</div>}
      {status === "error" && <div style={styles.error}>Couldn't load insights: {errorMessage}</div>}

      {status === "ready" && (!data.notes || data.notes.length === 0) && (
        <div style={styles.muted}>
          No notes yet — add a note or upload a session note and this chart will appear here.
        </div>
      )}

      {status === "ready" && data.notes.length > 0 && (
        <>
          <OverallBars overall={data.overall} />
          <TrendChart notes={data.notes} />
          <Reasons reasons={data.reasons} />
        </>
      )}
    </div>
  );
}

function OverallBars({ overall }) {
  return (
    <section style={styles.section}>
      <div style={styles.sectionTitle}>Overall emotional energy</div>
      <div style={styles.bars}>
        {EMOTIONS.map((emotion) => {
          const meta = EMOTION_META[emotion];
          const value = overall?.[emotion] ?? 0;
          return (
            <div key={emotion} style={styles.barRow}>
              <span style={styles.barLabel}>{meta.label}</span>
              <div style={styles.barTrack}>
                <div
                  style={{
                    ...styles.barFill,
                    width: `${Math.round(value * 100)}%`,
                    background: `linear-gradient(90deg, ${meta.color}55, ${meta.color})`,
                  }}
                />
              </div>
              <span style={styles.barValue}>{Math.round(value * 100)}%</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function TrendChart({ notes }) {
  const width = 680;
  const height = 240;
  const pad = { left: 38, right: 14, top: 18, bottom: 30 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const points = notes.map((n, i) => ({
    x: pad.left + (notes.length === 1 ? plotW / 2 : (i * plotW) / (notes.length - 1)),
    y: pad.top + plotH,
    label: (n.created_at || "").slice(0, 10) || `#${i + 1}`,
  }));

  const yPos = (value) => pad.top + plotH - Math.max(0, Math.min(1, value)) * plotH;

  return (
    <section style={styles.section}>
      <div style={styles.sectionTitle}>Over time (per note)</div>
      <svg viewBox={`0 0 ${width} ${height}`} style={styles.chart} role="img" aria-label="Emotional energy over time">
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}>
            <line
              x1={pad.left}
              x2={width - pad.right}
              y1={yPos(g)}
              y2={yPos(g)}
              stroke={theme.border}
              strokeWidth="1"
            />
            <text x={pad.left - 6} y={yPos(g) + 3} textAnchor="end" fontSize="9" fill={theme.textFaint}>
              {Math.round(g * 100)}
            </text>
          </g>
        ))}

        {points.map((p, i) => (
          <text key={`x${i}`} x={p.x} y={height - 8} textAnchor="middle" fontSize="9" fill={theme.textFaint}>
            {p.label}
          </text>
        ))}

        {EMOTIONS.map((emotion) => {
          const meta = EMOTION_META[emotion];
          const coords = notes
            .map((n, i) => ({
              x: points[i].x,
              y: yPos(n.scores?.[emotion] ?? 0),
            }))
            .filter((c) => Number.isFinite(c.y));
          const linePath = coords.map((c, i) => `${i === 0 ? "M" : "L"}${c.x},${c.y}`).join(" ");
          return (
            <g key={emotion}>
              {coords.length > 1 && (
                <polyline points={linePath} fill="none" stroke={meta.color} strokeWidth="2" strokeLinejoin="round" />
              )}
              {coords.map((c, i) => (
                <circle key={`${emotion}${i}`} cx={c.x} cy={c.y} r="3.5" fill={meta.color} stroke={theme.bg} strokeWidth="1.5" />
              ))}
            </g>
          );
        })}
      </svg>

      <div style={styles.legend}>
        {EMOTIONS.map((emotion) => (
          <span key={emotion} style={styles.legendItem}>
            <span style={{ ...styles.legendDot, background: EMOTION_META[emotion].color }} />
            {EMOTION_META[emotion].label}
          </span>
        ))}
      </div>
    </section>
  );
}

function Reasons({ reasons }) {
  const present = EMOTIONS.filter((emotion) => (reasons?.[emotion] || []).length > 0);
  if (present.length === 0) return null;

  return (
    <section style={styles.section}>
      <div style={styles.sectionTitle}>Why</div>
      <div style={styles.reasonsGrid}>
        {present.map((emotion) => {
          const meta = EMOTION_META[emotion];
          return (
            <div key={emotion} style={{ ...styles.reasonCard, borderTopColor: meta.color }}>
              <div style={{ ...styles.reasonHeader, color: meta.color }}>{meta.label}</div>
              <ul style={styles.reasonList}>
                {reasons[emotion].map((r, i) => (
                  <li key={i} style={styles.reasonItem}>
                    <div style={styles.reasonTerm}>
                      “{r.term}” <span style={styles.reasonCount}>×{r.count}</span>
                    </div>
                    <div style={styles.reasonSnippet}>…{r.snippet}…</div>
                    <div style={styles.reasonNote}>{r.title}</div>
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
      </div>
    </section>
  );
}

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    gap: "1.1rem",
    width: "100%",
    maxWidth: "720px",
    fontFamily: theme.font,
  },
  header: {
    display: "flex",
    alignItems: "baseline",
    gap: "0.5rem",
  },
  label: {
    fontSize: "0.9rem",
    fontWeight: 700,
    color: theme.text,
  },
  tag: {
    fontSize: "0.8rem",
    color: theme.textMuted,
  },
  refresh: {
    marginLeft: "auto",
    background: "none",
    border: "none",
    color: theme.gold,
    fontSize: "0.8rem",
    cursor: "pointer",
    fontFamily: theme.font,
  },
  muted: {
    fontSize: "0.85rem",
    color: theme.textMuted,
  },
  error: {
    fontSize: "0.85rem",
    color: theme.danger,
  },
  section: {
    background: theme.bgRaised,
    border: `1px solid ${theme.border}`,
    borderRadius: theme.radius,
    padding: "1rem 1.1rem",
  },
  sectionTitle: {
    fontSize: "0.78rem",
    fontWeight: 700,
    color: theme.textMuted,
    textTransform: "uppercase",
    letterSpacing: "0.06em",
    marginBottom: "0.75rem",
  },
  bars: {
    display: "flex",
    flexDirection: "column",
    gap: "0.6rem",
  },
  barRow: {
    display: "flex",
    alignItems: "center",
    gap: "0.6rem",
  },
  barLabel: {
    width: "76px",
    fontSize: "0.82rem",
    color: theme.text,
  },
  barTrack: {
    flex: 1,
    height: "12px",
    background: theme.bg,
    border: `1px solid ${theme.border}`,
    borderRadius: "999px",
    overflow: "hidden",
  },
  barFill: {
    height: "100%",
    borderRadius: "999px",
    transition: "width 0.4s ease",
  },
  barValue: {
    width: "38px",
    textAlign: "right",
    fontSize: "0.78rem",
    color: theme.textMuted,
  },
  chart: {
    width: "100%",
    height: "auto",
    display: "block",
  },
  legend: {
    display: "flex",
    gap: "1rem",
    flexWrap: "wrap",
    marginTop: "0.6rem",
  },
  legendItem: {
    display: "flex",
    alignItems: "center",
    gap: "0.35rem",
    fontSize: "0.78rem",
    color: theme.textMuted,
  },
  legendDot: {
    width: "10px",
    height: "10px",
    borderRadius: "50%",
    display: "inline-block",
  },
  reasonsGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
    gap: "0.75rem",
  },
  reasonCard: {
    background: theme.bg,
    border: `1px solid ${theme.border}`,
    borderTop: "3px solid",
    borderRadius: "8px",
    padding: "0.7rem 0.85rem",
  },
  reasonHeader: {
    fontSize: "0.8rem",
    fontWeight: 700,
    marginBottom: "0.4rem",
  },
  reasonList: {
    listStyle: "none",
    margin: 0,
    padding: 0,
    display: "flex",
    flexDirection: "column",
    gap: "0.5rem",
  },
  reasonItem: {
    fontSize: "0.8rem",
  },
  reasonTerm: {
    color: theme.text,
    fontWeight: 600,
  },
  reasonCount: {
    color: theme.textFaint,
    fontWeight: 400,
  },
  reasonSnippet: {
    color: theme.textMuted,
    fontSize: "0.75rem",
    marginTop: "0.15rem",
    lineHeight: 1.4,
  },
  reasonNote: {
    color: theme.textFaint,
    fontSize: "0.7rem",
    marginTop: "0.15rem",
  },
};