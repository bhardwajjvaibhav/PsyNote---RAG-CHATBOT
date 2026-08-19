"""
core/emotion.py

Deterministic, lexicon-based emotion analysis over a user's note text,
for the dashboard's "emotional energy" chart and the per-emotion
reasons list.

Deliberately NOT model-based (same choice as safety_scanner.py and
post_validate.py, architecture doc Section 5): a fixed keyword lexicon
is auditable, deterministic, offline, and repeatable -- the same note
always yields the same scores, with no API key and no latency. It can't
catch tone the way a model could, but for a per-note trend chart over
the user's own writing it's a cheap, trustworthy baseline. Swap the
scoring function for an LLM call later without touching the report
shape if desired.

Four categories the UI asks for: happy, sad, stressed, anxious.
Scores are 0.0-1.0, soft-capped by match count. "Reasons" are the exact
matched terms with surrounding snippet + note attribution, so the UI can
answer "why is stress this high?" by showing the actual sentences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

EMOTIONS = ["happy", "sad", "stressed", "anxious"]

# emotion -> list of terms (words or short phrases). Starter set for
# engineering/testing only -- like safety_scanner's rule table, it should
# be reviewed/tuned against real user notes before any real deployment.
LEXICON: dict[str, list[str]] = {
    "happy": [
        "happy", "happier", "joy", "joyful", "joyous", "great", "good day",
        "better", "improved", "improving", "excited", "excellent", "grateful",
        "hopeful", "calm", "relaxed", "peaceful", "content", "smiling",
        "laughed", "positive", "bright", "wonderful", "glad", "pleased",
        "thrilled", "proud", "relief", "relieved", "energized",
    ],
    "sad": [
        "sad", "sadder", "crying", "cried", "tears", "unhappy", "feeling down",
        "low mood", "depressed", "depression", "hopeless", "alone", "lonely",
        "empty", "grief", "grieving", "miserable", "heartbroken", "blue",
        "no energy", "lost interest", "worthless", "numb",
    ],
    "stressed": [
        "stress", "stressed", "pressure", "overwhelmed", "burnout",
        "deadline", "deadlines", "tension", "tight schedule", "workload",
        "overworked", "frazzled", "juggling", "demanding", "swamped",
        "under pressure", "can't cope", "unable to cope", "sleep deprived",
    ],
    "anxious": [
        "anxious", "anxiety", "nervous", "worried", "panic", "panicked",
        "panic attack", "fear", "afraid", "scared", "uneasy", "on edge",
        "racing thoughts", "restless", "dread", "overthinking", "jittery",
        "butterflies", "chest tightness", "sweaty",
    ],
}

_COMPILED: dict[str, list[re.Pattern]] = {
    emotion: [re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in terms]
    for emotion, terms in LEXICON.items()
}

SNIPPET_RADIUS = 60  # chars of context on each side of a match

# Matches needed to reach a 1.0 score for an emotion in one note
SOFT_CAP_MATCHES = 4.0


def _snippet(text: str, start: int, end: int) -> str:
    lo = max(0, start - SNIPPET_RADIUS)
    hi = min(len(text), end + SNIPPET_RADIUS)
    return text[lo:hi].strip()


def analyze_text(text: str) -> tuple[dict[str, float], dict[str, dict[str, dict]]]:
    """
    Score one string against every emotion.

    Returns (scores, hits) where:
      scores: {emotion: 0.0-1.0}
      hits:   {emotion: {term: {"snippet": str, "count": int}}}
    """
    if not text:
        return {e: 0.0 for e in EMOTIONS}, {e: {} for e in EMOTIONS}

    scores: dict[str, float] = {}
    hits: dict[str, dict[str, dict]] = {e: {} for e in EMOTIONS}

    for emotion, patterns in _COMPILED.items():
        count = 0
        for pattern in patterns:
            for match in pattern.finditer(text):
                count += 1
                term = match.group(0)
                entry = hits[emotion].setdefault(
                    term, {"snippet": _snippet(text, match.start(), match.end()), "count": 0}
                )
                entry["count"] += 1
        scores[emotion] = min(1.0, count / SOFT_CAP_MATCHES)

    return scores, hits


@dataclass
class NoteEmotion:
    note_id: str
    title: str
    created_at: str
    scores: dict[str, float]
    hits: dict[str, dict[str, dict]] = field(default_factory=dict)


@dataclass
class ReasonHit:
    note_id: str
    title: str
    term: str
    snippet: str
    count: int


@dataclass
class MoodReport:
    notes: list[NoteEmotion] = field(default_factory=list)
    overall: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, list[ReasonHit]] = field(default_factory=dict)


def analyze_notes(
    records: list[dict],
    max_reasons_per_emotion: int = 5,
) -> MoodReport:
    """
    Analyze a list of note records. Each record:
      {"note_id": str, "title": str, "created_at": str, "text": str}

    Returns a MoodReport with per-note scores, overall averages across
    notes, and per-emotion reason hits (most-frequent matched terms,
    attributed to their note).
    """
    report = MoodReport(overall={e: 0.0 for e in EMOTIONS}, reasons={e: [] for e in EMOTIONS})

    if not records:
        return report

    notes: list[NoteEmotion] = []
    reason_pool: dict[str, list[ReasonHit]] = {e: [] for e in EMOTIONS}

    for rec in records:
        scores, hits = analyze_text(rec.get("text", ""))
        notes.append(NoteEmotion(
            note_id=rec["note_id"],
            title=rec.get("title", ""),
            created_at=rec.get("created_at", ""),
            scores=scores,
            hits=hits,
        ))
        for emotion in EMOTIONS:
            for term, meta in hits[emotion].items():
                reason_pool[emotion].append(ReasonHit(
                    note_id=rec["note_id"],
                    title=rec.get("title", ""),
                    term=term,
                    snippet=meta["snippet"],
                    count=meta["count"],
                ))

    report.notes = notes

    for emotion in EMOTIONS:
        vals = [n.scores[emotion] for n in notes]
        report.overall[emotion] = round(sum(vals) / len(vals), 3)
        pool = sorted(reason_pool[emotion], key=lambda r: r.count, reverse=True)
        # Dedupe by term, keep the strongest note for it
        seen: set[str] = set()
        for hit in pool:
            if hit.term.lower() in seen:
                continue
            seen.add(hit.term.lower())
            report.reasons[emotion].append(hit)
            if len(report.reasons[emotion]) >= max_reasons_per_emotion:
                break

    return report


# --- Quick self-test ----------------------------------------------------------
# Run this file directly: python emotion.py

if __name__ == "__main__":
    print("=== analyze_text: scores and hits ===")
    text = (
        "Today was stressful, a tight deadline at work. "
        "I felt anxious and nervous all day, racing thoughts at night. "
        "But the evening was calm and I felt grateful and relaxed. "
        "Still low mood at times and a little lonely."
    )
    scores, hits = analyze_text(text)
    print(scores)
    assert scores["stressed"] >= 0.25
    assert scores["anxious"] >= 0.5
    assert scores["happy"] >= 0.5
    assert scores["sad"] >= 0.25
    assert "racing thoughts" in hits["anxious"]
    assert hits["anxious"]["racing thoughts"]["count"] == 1

    print("\n=== empty text -> all zeros ===")
    zero_scores, _ = analyze_text("")
    assert zero_scores == {e: 0.0 for e in EMOTIONS}

    print("\n=== score cap: many matches still 1.0 ===")
    caps, _ = analyze_text(("anxious nervous worried scared afraid panic dread " * 3))
    assert caps["anxious"] == 1.0

    print("\n=== word boundary: 'harmless' doesn't hit 'sad' style false positives ===")
    boundary_scores, _ = analyze_text("This is an anxious-sounding generic sentence.")
    assert boundary_scores["happy"] == 0.0

    print("\n=== analyze_notes: report shape, overall, reasons attribution ===")
    report = analyze_notes([
        {"note_id": "n1", "title": "session_1.txt", "created_at": "2026-08-01", "text": "Great session, felt hopeful and calm. Slept well."},
        {"note_id": "n2", "title": "session_2.txt", "created_at": "2026-08-08", "text": "Stressful week, deadlines and pressure. Feeling anxious, nervous."},
        {"note_id": "n3", "title": "session_3.txt", "created_at": "2026-08-15", "text": "Low mood, lonely, cried today. Stress at work again."},
    ])
    assert len(report.notes) == 3
    assert 0 < report.overall["happy"] < 1
    assert all(r.note_id for rs in report.reasons.values() for r in rs)
    stressed_reasons = report.reasons["stressed"]
    assert any(r.term.lower() in {"deadlines", "pressure", "stress"} for r in stressed_reasons)
    print("overall:", report.overall)
    print("top stress reasons:", [(r.term, r.title) for r in report.reasons["stressed"]])

    print("\n=== empty records -> empty report ===")
    empty_report = analyze_notes([])
    assert empty_report.notes == []
    assert empty_report.overall == {e: 0.0 for e in EMOTIONS}

    print("\nSelf-test passed.")
