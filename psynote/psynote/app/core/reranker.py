"""
core/reranker.py

Cross-encoder reranking: takes the fused/diversified top-20 candidates
from fusion.py and cuts down to the top-5 chunks that actually go into
the prompt (architecture doc, Section 4: "Cross-encoder rerank —
ms-marco-MiniLM-L-6-v2 → top-5 chunks").

Why a cross-encoder after RRF+MMR, not instead of it: Chroma/BM25/RRF
all score a query against a chunk independently of each other -- fast,
but each one only sees half the picture. A cross-encoder reads the
(query, chunk) pair TOGETHER through one model, which is far more
accurate at "does this chunk actually answer this question" -- but
too slow to run over a whole patient's corpus, which is why it only
runs on fusion's already-narrowed top-20, not the raw candidate pool.

Design note -- injectable scorer: the model itself (CrossEncoder from
sentence-transformers) is loaded lazily and only on first real use, and
callers can inject a `scorer` callable instead (query, texts) -> scores.
This isn't a testing convenience bolted on after the fact -- it's the
same dependency-injection shape used throughout this codebase
(get_conn() in db/patients.py, MCP servers in fusion.py's design) so the
reranking LOGIC (sorting, safety-flag handling, top-k truncation) can be
verified deterministically without a model download, while production
usage just doesn't pass `scorer` and gets the real cross-encoder.

Consistent with fusion.py's design: safety-flagged chunks are never
subject to relevance-based truncation. They passed through fusion
specifically because they're clinically critical regardless of
relevance to the current question (see fusion.py docstring) -- it would
undermine that guarantee if the reranker were free to cut them here.
"""

from __future__ import annotations

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None  # lazy-loaded singleton, only used when no scorer is injected


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(MODEL_NAME)
    return _model


def _default_scorer(query: str, texts: list[str]) -> list[float]:
    model = _load_model()
    pairs = [[query, text] for text in texts]
    return [float(s) for s in model.predict(pairs)]


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    scorer=None,
) -> list[dict]:
    """
    Rerank fusion's candidate list against the actual query text.

    `candidates` is expected in fusion.py's output shape: dicts with at
    least chunk_id, text, patient_id, safety_flag. Safety-flagged
    entries are always kept (see module docstring) and always sorted to
    the front; the remaining budget (top_k - number of safety entries)
    is filled by the highest cross-encoder scores among the rest.

    `scorer(query, texts) -> list[float]` is optional dependency
    injection point, used by tests to avoid a real model load. Higher
    score = more relevant, matching CrossEncoder.predict's convention.

    Returns candidates with a `rerank_score` field added (None for
    safety-flagged pseudo/forced entries that were never scored).
    """
    if not candidates:
        return []

    score_fn = scorer or _default_scorer

    safety_entries = [c for c in candidates if c.get("safety_flag")]
    rankable = [c for c in candidates if not c.get("safety_flag")]

    scored: list[dict] = []
    if rankable:
        scores = score_fn(query, [c["text"] for c in rankable])
        if len(scores) != len(rankable):
            raise ValueError(
                f"scorer returned {len(scores)} scores for {len(rankable)} candidates"
            )
        for candidate, score in zip(rankable, scores):
            entry = dict(candidate)
            entry["rerank_score"] = score
            scored.append(entry)
        scored.sort(key=lambda c: c["rerank_score"], reverse=True)

    for entry in safety_entries:
        entry.setdefault("rerank_score", None)

    budget = max(top_k - len(safety_entries), 0)
    result = safety_entries + scored[:budget]

    # If safety entries alone already exceed top_k, we still return all
    # of them -- see module docstring: safety-critical content is never
    # truncated here. This is a deliberate deviation from a hard top_k
    # cap, not a bug; the prompt builder's token budget (Section 4) is
    # the next place volume gets managed, not this function.
    return result


# --- Quick self-test ----------------------------------------------------------
# Run this file directly: python reranker.py
# Uses an injected fake scorer so this never needs network access to
# download the real cross-encoder model.

if __name__ == "__main__":

    def fake_scorer(query: str, texts: list[str]) -> list[float]:
        """
        Deterministic stand-in for the real cross-encoder: score = how
        many query tokens appear in the text. Good enough to test
        sorting/truncation logic without a model.
        """
        query_tokens = set(query.lower().split())
        scores = []
        for text in texts:
            text_tokens = set(text.lower().split())
            scores.append(float(len(query_tokens & text_tokens)))
        return scores

    print("=== basic rerank: most relevant text scores highest ===")
    candidates = [
        {"chunk_id": "c1", "text": "patient started a new hobby painting weekends", "patient_id": "patient_a", "safety_flag": False},
        {"chunk_id": "c2", "text": "patient reports medication side effects nausea", "patient_id": "patient_a", "safety_flag": False},
        {"chunk_id": "c3", "text": "patient discussed medication dosage changes", "patient_id": "patient_a", "safety_flag": False},
    ]
    result = rerank("medication side effects", candidates, top_k=5, scorer=fake_scorer)
    for r in result:
        print(r["rerank_score"], r["chunk_id"], "-", r["text"])
    assert result[0]["chunk_id"] == "c2"  # most token overlap with query
    assert result[0]["rerank_score"] >= result[1]["rerank_score"] >= result[2]["rerank_score"]

    print("\n=== top_k truncation ===")
    top2 = rerank("medication side effects", candidates, top_k=2, scorer=fake_scorer)
    assert len(top2) == 2
    print([r["chunk_id"] for r in top2])

    print("\n=== safety-flagged entries always survive, never scored, always lead ===")
    candidates_with_safety = candidates + [
        {"chunk_id": "safety::n9::self_harm_risk::wanting to die", "text": "disclosed thoughts of wanting to die", "patient_id": "patient_a", "safety_flag": True},
    ]
    result_safety = rerank("medication side effects", candidates_with_safety, top_k=2, scorer=fake_scorer)
    assert len(result_safety) == 2  # 1 safety + budget of 1 rankable
    assert result_safety[0]["safety_flag"] is True
    assert result_safety[0]["rerank_score"] is None
    print([(r["chunk_id"], r["safety_flag"], r["rerank_score"]) for r in result_safety])

    print("\n=== safety entries alone exceeding top_k are never truncated ===")
    many_safety = [
        {"chunk_id": f"safety::{i}", "text": "risk snippet", "patient_id": "patient_a", "safety_flag": True}
        for i in range(3)
    ]
    result_many_safety = rerank("irrelevant query", many_safety + candidates, top_k=2, scorer=fake_scorer)
    assert len(result_many_safety) == 3  # all 3 safety entries kept despite top_k=2
    assert all(r["safety_flag"] for r in result_many_safety)
    print(f"OK, {len(result_many_safety)} safety entries kept despite top_k=2.")

    print("\n=== empty candidates ===")
    assert rerank("anything", [], top_k=5, scorer=fake_scorer) == []
    print("OK, empty in -> empty out.")

    print("\n=== scorer/candidate length mismatch raises ===")
    def broken_scorer(query, texts):
        return [1.0]  # wrong length on purpose
    try:
        rerank("q", candidates, top_k=5, scorer=broken_scorer)
        print("FAILED: should have raised ValueError")
    except ValueError as e:
        print(f"OK, raised: {e}")

    print("\nSelf-test passed.")