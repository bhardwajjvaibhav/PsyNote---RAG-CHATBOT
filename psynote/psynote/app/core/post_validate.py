"""
core/post_validate.py

Last step of the query-time flow before the response goes out (Section
4): a cheap lexical overlap check on the generated answer against the
chunks it was actually generated from, and per-sentence citations back
to those chunks.

Deliberately NOT a second LLM call (consistent with safety_scanner.py's
"deliberately not model-based" choice, Section 5, and the doc's
explicit build-order note in Section 8 Phase 8: "cheap lexical overlap
check", not an LLM-as-judge check): an LLM grading its own sibling
call's output is slow, costly, and can hallucinate its OWN judgment
about whether the first call hallucinated. A deterministic token-
overlap score can't catch semantic drift (an answer that's fluent but
subtly wrong), but it reliably catches the failure mode that matters
most here -- an answer sentence that has essentially no lexical
grounding in ANY retrieved chunk, which is the strongest cheap signal
that the LLM is asserting something the notes never said.

This module does not decide what to DO with a low-confidence answer
(block it, flag it in the UI, log it) -- same separation of concerns as
llm_client.py's LLMAllProvidersExhaustedError: this module reports the
per-sentence grounding signal, and the caller (rag_pipeline.py) decides
what the user sees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Below this overlap ratio, a sentence is flagged as weakly grounded.
# A starter threshold for engineering purposes -- like safety_scanner's
# rule table, this should be tuned/validated against the golden eval
# set (Section 5 / Phase 10) once that exists, not hand-picked here.
DEFAULT_GROUNDING_THRESHOLD = 0.20

# Sentences with fewer content tokens than this aren't checked -- "Yes."
# or "I'm not sure." will always score near-zero overlap and flagging
# them just adds noise, not signal.
MIN_TOKENS_TO_CHECK = 4

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "and", "or", "but",
    "this", "that", "these", "those", "it", "its", "as", "by", "from",
    "has", "have", "had", "will", "would", "could", "should", "can",
    "not", "no", "do", "does", "did", "so", "if", "than", "then",
}


def _tokenize(text: str) -> set[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


@dataclass
class SentenceValidation:
    sentence: str
    overlap_ratio: float          # fraction of the sentence's content tokens found in ANY chunk
    grounded: bool                # overlap_ratio >= threshold, or skipped as too short to judge
    best_chunk_id: str | None     # chunk with the highest overlap for this sentence, if any
    skipped: bool = False         # True for sentences too short to meaningfully score


@dataclass
class ValidationResult:
    sentences: list[SentenceValidation] = field(default_factory=list)
    flagged_count: int = 0        # sentences below threshold and NOT skipped
    checked_count: int = 0        # sentences actually scored (excludes skipped)

    @property
    def is_fully_grounded(self) -> bool:
        return self.flagged_count == 0

    @property
    def grounding_score(self) -> float:
        """Fraction of checked (non-skipped) sentences that were grounded. 1.0 if nothing to check."""
        if self.checked_count == 0:
            return 1.0
        return (self.checked_count - self.flagged_count) / self.checked_count


def validate_answer(
    answer_text: str,
    retrieved_chunks: list[dict],
    threshold: float = DEFAULT_GROUNDING_THRESHOLD,
    min_tokens_to_check: int = MIN_TOKENS_TO_CHECK,
) -> ValidationResult:
    """
    Score each sentence of `answer_text` for lexical grounding against
    `retrieved_chunks` (reranker.rerank()'s output shape: dicts with at
    least chunk_id and text).

    Grounding for one sentence = the highest, over all chunks, of
    (content tokens shared with that chunk) / (content tokens in the
    sentence). "Content tokens" excludes a small stopword list so
    grounding isn't inflated by function words every chunk shares.
    """
    result = ValidationResult()
    if not answer_text or not answer_text.strip():
        return result

    chunk_tokens = [(c.get("chunk_id"), _tokenize(c.get("text", ""))) for c in retrieved_chunks]

    for sentence in _split_sentences(answer_text):
        sent_tokens = _tokenize(sentence)

        if len(sent_tokens) < min_tokens_to_check:
            result.sentences.append(SentenceValidation(
                sentence=sentence, overlap_ratio=1.0, grounded=True,
                best_chunk_id=None, skipped=True,
            ))
            continue

        best_ratio, best_chunk_id = 0.0, None
        for chunk_id, tokens in chunk_tokens:
            if not tokens:
                continue
            overlap = len(sent_tokens & tokens) / len(sent_tokens)
            if overlap > best_ratio:
                best_ratio, best_chunk_id = overlap, chunk_id

        grounded = best_ratio >= threshold
        result.sentences.append(SentenceValidation(
            sentence=sentence, overlap_ratio=best_ratio, grounded=grounded,
            best_chunk_id=best_chunk_id, skipped=False,
        ))
        result.checked_count += 1
        if not grounded:
            result.flagged_count += 1

    return result


# --- Quick self-test ----------------------------------------------------------
# Run this file directly: python post_validate.py

if __name__ == "__main__":
    chunks = [
        {"chunk_id": "c1", "text": "Patient reports significantly improved sleep this week, averaging seven hours nightly."},
        {"chunk_id": "c2", "text": "Patient discussed increasing sertraline dosage to 100mg with their prescriber."},
    ]

    print("=== fully grounded answer: every sentence traces to a chunk ===")
    grounded_answer = (
        "The patient reports significantly improved sleep this week. "
        "They also discussed increasing their sertraline dosage to 100mg."
    )
    result = validate_answer(grounded_answer, chunks)
    for s in result.sentences:
        print(f"{s.overlap_ratio:.2f}", s.grounded, "|", s.sentence)
    assert result.is_fully_grounded
    assert result.flagged_count == 0
    assert result.checked_count == 2

    print("\n=== hallucinated sentence: flagged, low overlap ===")
    mixed_answer = (
        "The patient reports significantly improved sleep this week. "
        "The patient was diagnosed with bipolar disorder in early childhood."
    )
    result_mixed = validate_answer(mixed_answer, chunks)
    for s in result_mixed.sentences:
        print(f"{s.overlap_ratio:.2f}", s.grounded, "|", s.sentence)
    assert not result_mixed.is_fully_grounded
    assert result_mixed.flagged_count == 1
    assert result_mixed.sentences[0].grounded is True
    assert result_mixed.sentences[1].grounded is False

    print("\n=== short filler sentences are skipped, not flagged ===")
    filler_answer = "Yes. The patient reports significantly improved sleep this week."
    result_filler = validate_answer(filler_answer, chunks)
    for s in result_filler.sentences:
        print(s.skipped, f"{s.overlap_ratio:.2f}", s.grounded, "|", s.sentence)
    assert result_filler.sentences[0].skipped is True
    assert result_filler.checked_count == 1  # only the real sentence counted
    assert result_filler.is_fully_grounded

    print("\n=== best_chunk_id points at the actual supporting chunk ===")
    assert result.sentences[0].best_chunk_id == "c1"
    assert result.sentences[1].best_chunk_id == "c2"
    print("OK.")

    print("\n=== grounding_score reflects partial grounding ===")
    print(f"grounding_score = {result_mixed.grounding_score:.2f}")
    assert result_mixed.grounding_score == 0.5

    print("\n=== empty answer / no chunks handled gracefully ===")
    assert validate_answer("", chunks).sentences == []
    empty_chunks_result = validate_answer(grounded_answer, [])
    assert all(not s.grounded for s in empty_chunks_result.sentences if not s.skipped)
    print("OK.")

    print("\nSelf-test passed.")