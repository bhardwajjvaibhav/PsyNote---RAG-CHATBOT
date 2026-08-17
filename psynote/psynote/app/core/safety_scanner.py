"""
safety_scanner.py

Rule-based (regex/keyword) safety scanning, scoped to a single patient_id.

Deliberately NOT model-based (see architecture doc, Section 5):
regex rules are auditable, deterministic, and don't silently miss or
hallucinate on safety-critical text the way an LLM call could.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


# --- Step 1: data shape ---------------------------------------------------
# This is what every downstream consumer (query flow, prompt builder,
# audit_log) will receive. Decide this shape before writing scan logic.

@dataclass
class SafetyHit:
    patient_id: str
    category: str              # e.g. "self_harm_risk"
    matched_term: str          # the literal text that triggered the rule
    snippet: str                # surrounding context, for clinician review
    note_id: str | None = None  # which note this came from (set later, Phase 2+)


# --- Step 2: rule table -----------------------------------------------------
# category -> list of regex patterns (case-insensitive, word-boundary matched).
#
# This is a STARTER set for engineering/testing purposes only. In a real
# deployment this list should be authored/reviewed by a clinician, not
# just an engineer — false negatives here have real consequences.
#
# \b...\b = whole-word match, so "harmless" doesn't trigger on "harm".

SAFETY_RULES: dict[str, list[str]] = {
    "self_harm_risk": [
        r"\bsuicid\w*\b",
        r"\bself[- ]harm\w*\b",
        r"\bkill(ing)? myself\b",
        r"\bwant(ed|ing|s)? to die\b",
        r"\bend(ing)? (it|my life)\b",
    ],
    "harm_to_others": [
        r"\bkill(ing)? (him|her|them|you)\b",
        r"\bhurt (him|her|them|someone)\b",
        r"\bthreat(en(ed|ing)?)?\b",
    ],
    "substance_crisis": [
        r"\boverdos\w*\b",
        r"\brelaps\w*\b",
    ],
    "abuse_disclosure": [
        r"\babus\w*\b",
        r"\bassault\w*\b",
    ],
}

# Pre-compile once at import time (cheap at startup, avoids recompiling
# the same regex on every scan call).
_COMPILED_RULES: dict[str, list[re.Pattern]] = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in SAFETY_RULES.items()
}


# --- Step 3: core scan function --------------------------------------------
# Pure function: given raw text, return hits. No patient_id, no DB, no I/O.
# This is the unit we can test in complete isolation before wiring in
# anything else.

SNIPPET_RADIUS = 40  # chars of context on each side of the match


def scan_text(text: str) -> list[dict]:
    """
    Scan a single string against all safety rules.

    Returns a list of raw hit dicts: {category, matched_term, snippet}.
    (Not yet tagged with patient_id — that happens in Step 4.)
    """
    hits: list[dict] = []
    for category, patterns in _COMPILED_RULES.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                start = max(0, match.start() - SNIPPET_RADIUS)
                end = min(len(text), match.end() + SNIPPET_RADIUS)
                hits.append({
                    "category": category,
                    "matched_term": match.group(0),
                    "snippet": text[start:end].strip(),
                })
    return hits


# --- Step 4: patient-scoped wrapper -----------------------------------------
# This is what the query flow (Section 3 of the doc) will eventually call.
# It takes ONLY this patient's notes -- the caller (query module, once it
# exists) is responsible for fetching notes already filtered to patient_id.
#
# This function does not query the DB itself; it just guarantees every
# hit it returns is correctly attributed to the patient whose notes came in.

def scan_patient_records(
    patient_id: str,
    notes: list[tuple[str, str]],  # list of (note_id, note_text)
) -> list[SafetyHit]:
    """
    Scan all of a single patient's notes and return tagged SafetyHits.

    `notes` should already be filtered to this patient_id by the caller --
    this function does not do any cross-patient filtering itself, it
    only tags what it's given.
    """
    results: list[SafetyHit] = []
    for note_id, text in notes:
        for raw_hit in scan_text(text):
            results.append(SafetyHit(
                patient_id=patient_id,
                category=raw_hit["category"],
                matched_term=raw_hit["matched_term"],
                snippet=raw_hit["snippet"],
                note_id=note_id,
            ))
    return results


# --- Step 5: quick self-test ------------------------------------------------
# Run this file directly: python safety_scanner.py

if __name__ == "__main__":
    patient_a_notes = [
        ("note_1", "Patient reports feeling low energy this week, sleeping poorly."),
        ("note_2", "Patient disclosed thoughts of wanting to die during session."),
    ]
    patient_b_notes = [
        ("note_1", "Patient discussed work stress, no risk indicators noted."),
    ]

    print("=== Patient A scan ===")
    for hit in scan_patient_records("patient_a", patient_a_notes):
        print(hit)

    print("\n=== Patient B scan ===")
    hits_b = scan_patient_records("patient_b", patient_b_notes)
    print(hits_b if hits_b else "No hits (expected -- clean notes).")

    # Sanity check: patient A's hits must never carry patient B's id, and
    # vice versa. This is the isolation guarantee the doc calls out.
    assert all(h.patient_id == "patient_a" for h in scan_patient_records("patient_a", patient_a_notes))
    print("\nSanity check passed.")