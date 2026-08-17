"""
core/fusion.py

Query-time flow, Section 4: merges the three parallel retrieval
signals -- Chroma (dense), BM25 (lexical), and safety_scanner (rule-based)
-- into one candidate list, then diversifies it with MMR before handing
off to reranker.py's cross-encoder.

Why RRF (Reciprocal Rank Fusion) to combine Chroma + BM25, not a raw
score blend: Chroma returns cosine/L2 distances and BM25 returns
unbounded term-frequency scores -- the two are not on comparable
scales, so averaging or weighting them directly would let whichever
score happens to have the larger numeric range dominate. RRF sidesteps
this entirely by using each list's RANK, not its raw score:
score(chunk) = sum over lists containing chunk of 1 / (k + rank).
Scale-free by construction, and a well-established IR baseline for
exactly this "combine two differently-scaled ranked lists" problem.

Why safety hits are folded in here, not scored at all: a safety_flag
hit exists because a patient's own words matched a clinically critical
pattern (self-harm risk, abuse disclosure, etc. -- safety_scanner.py),
not because it's relevant to the CURRENT question. Giving it an RRF
score would let an off-topic question rank it low enough to be cut.
Instead, every safety hit is converted straight into a candidate with
safety_flag=True and unioned into the output unconditionally,
independent of query relevance -- reranker.py inherits and preserves
this guarantee (see its module docstring), and this is the one place
in the pipeline where that guarantee actually originates.

Why MMR after RRF, not instead of it: RRF alone can return five near-
duplicate chunks from the same note if that note dominates both the
dense and lexical rankings -- true relevance, but poor coverage. MMR
(Maximal Marginal Relevance) re-orders the RRF-ranked list by
repeatedly picking the next candidate that balances relevance
(RRF score) against similarity to what's already been picked, so the
top-20 handed to the reranker spans more of the patient's record
instead of repeating one note five times.

Design note -- similarity_fn is injectable, and defaults to a cheap
text-overlap proxy: true MMR wants embedding cosine similarity between
candidates, but vector_store.query() doesn't currently return the
chunk embeddings themselves (only distances to the query), and
re-embedding candidates here would be a second, redundant model call
on the query-time hot path. A Jaccard token-overlap similarity is a
deliberately cheap stand-in that still catches the common case this
step exists for (near-duplicate/overlapping chunks), and is fully
swappable: pass `similarity_fn=cosine_over(embeddings)` once
vector_store.query() is extended to return embeddings, with zero
change to the MMR loop itself. Same DI shape as reranker.py's `scorer`
and llm_client.py's `transport`.
"""

from __future__ import annotations

import re

RRF_K = 60  # standard RRF constant (Cormack et al.) -- de-emphasizes rank 1 vs rank 2
DEFAULT_TOP_N = 20  # fusion's output feeds reranker.py's top-20 input (Section 4)
DEFAULT_MMR_LAMBDA = 0.7  # weight on relevance vs. diversity; higher = more relevance-weighted

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _default_similarity_fn(text_a: str, text_b: str) -> float:
    """Jaccard token overlap -- see module docstring for why this is the default."""
    tokens_a, tokens_b = _tokenize(text_a), _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _safety_hit_to_candidate(hit) -> dict:
    """
    Normalize a safety_scanner.SafetyHit (dataclass) or an equivalent
    dict into fusion's candidate shape. chunk_id is synthesized since
    safety hits are matched against raw note text, not an existing
    chunk_id -- format mirrors reranker.py's self-test fixtures
    ("safety::{note_id}::{category}::{matched_term}").
    """
    if not isinstance(hit, dict):
        hit = {
            "patient_id": hit.patient_id,
            "category": hit.category,
            "matched_term": hit.matched_term,
            "snippet": hit.snippet,
            "note_id": hit.note_id,
        }
    chunk_id = f"safety::{hit.get('note_id')}::{hit['category']}::{hit['matched_term']}"
    return {
        "chunk_id": chunk_id,
        "text": hit["snippet"],
        "note_id": hit.get("note_id"),
        "patient_id": hit["patient_id"],
        "safety_flag": True,
        "safety_category": hit["category"],
    }


def _rrf_merge(chroma_hits: list[dict], bm25_hits: list[dict], k: int = RRF_K) -> dict[str, dict]:
    """
    Returns {chunk_id: candidate_dict} with an added "rrf_score" field.
    chroma_hits are assumed sorted ascending by distance (closer = better,
    matching vector_store.query()'s output order). bm25_hits are assumed
    sorted descending by score (matching bm25_index.search()'s output order).
    """
    merged: dict[str, dict] = {}

    for rank, hit in enumerate(chroma_hits):
        entry = dict(hit)
        entry["safety_flag"] = False
        entry["rrf_score"] = 1.0 / (k + rank + 1)
        merged[hit["chunk_id"]] = entry

    for rank, hit in enumerate(bm25_hits):
        cid = hit["chunk_id"]
        contribution = 1.0 / (k + rank + 1)
        if cid in merged:
            merged[cid]["rrf_score"] += contribution
        else:
            entry = dict(hit)
            entry["safety_flag"] = False
            entry["rrf_score"] = contribution
            merged[cid] = entry

    return merged


def _mmr_select(
    candidates: list[dict],
    top_n: int,
    similarity_fn,
    mmr_lambda: float,
) -> list[dict]:
    """
    Greedy MMR: repeatedly pick the remaining candidate maximizing
        lambda * relevance(candidate) - (1 - lambda) * max_sim(candidate, already_picked)
    `candidates` must already be a list (any order) with an "rrf_score"
    field for relevance; scores are min-max normalized first so lambda's
    weighting is meaningful regardless of RRF's absolute score range.
    """
    if not candidates:
        return []

    scores = [c["rrf_score"] for c in candidates]
    lo, hi = min(scores), max(scores)
    spread = (hi - lo) or 1.0  # avoid divide-by-zero when all scores are equal

    remaining = list(candidates)
    selected: list[dict] = []

    while remaining and len(selected) < top_n:
        best_idx, best_value = None, None
        for i, cand in enumerate(remaining):
            relevance = (cand["rrf_score"] - lo) / spread
            if selected:
                max_sim = max(similarity_fn(cand["text"], s["text"]) for s in selected)
            else:
                max_sim = 0.0
            value = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim
            if best_value is None or value > best_value:
                best_idx, best_value = i, value
        selected.append(remaining.pop(best_idx))

    return selected


def fuse(
    chroma_hits: list[dict],
    bm25_hits: list[dict],
    safety_hits: list | None = None,
    top_n: int = DEFAULT_TOP_N,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    similarity_fn=None,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """
    Merge Chroma + BM25 via RRF, diversify the merged list via MMR, and
    unconditionally union in every safety hit (never scored, never
    subject to top_n truncation -- see module docstring).

    Returns a list of candidate dicts (chunk_id, text, note_id,
    patient_id, safety_flag, plus rrf_score for non-safety entries) in
    fusion.py's canonical output shape, which reranker.rerank() expects
    as its `candidates` input.
    """
    safety_hits = safety_hits or []
    similarity = similarity_fn or _default_similarity_fn

    safety_candidates = [_safety_hit_to_candidate(h) for h in safety_hits]
    # De-dupe safety hits by chunk_id (the same matched_term can recur
    # across identical rule matches in edge cases) while preserving order.
    seen_ids: set[str] = set()
    deduped_safety: list[dict] = []
    for c in safety_candidates:
        if c["chunk_id"] not in seen_ids:
            seen_ids.add(c["chunk_id"])
            deduped_safety.append(c)

    merged = _rrf_merge(chroma_hits, bm25_hits, k=rrf_k)
    rankable = list(merged.values())

    diversified = _mmr_select(rankable, top_n=top_n, similarity_fn=similarity, mmr_lambda=mmr_lambda)

    return deduped_safety + diversified


# --- Quick self-test ----------------------------------------------------------
# Run this file directly: python fusion.py

if __name__ == "__main__":

    print("=== RRF: a chunk ranked well in BOTH lists outranks one ranked well in only one ===")
    chroma_hits = [
        {"chunk_id": "c1", "text": "patient sleep improved this week", "note_id": "n1", "patient_id": "p1", "distance": 0.10},
        {"chunk_id": "c2", "text": "patient discussed medication dosage", "note_id": "n1", "patient_id": "p1", "distance": 0.20},
        {"chunk_id": "c3", "text": "patient enjoyed a weekend hike", "note_id": "n2", "patient_id": "p1", "distance": 0.30},
    ]
    bm25_hits = [
        {"chunk_id": "c2", "text": "patient discussed medication dosage", "note_id": "n1", "patient_id": "p1", "score": 5.0},
        {"chunk_id": "c1", "text": "patient sleep improved this week", "note_id": "n1", "patient_id": "p1", "score": 1.0},
    ]
    merged = _rrf_merge(chroma_hits, bm25_hits)
    ranked = sorted(merged.values(), key=lambda c: c["rrf_score"], reverse=True)
    for r in ranked:
        print(f"{r['rrf_score']:.5f}", r["chunk_id"])
    assert abs(ranked[0]["rrf_score"] - ranked[1]["rrf_score"]) < 1e-9  # c1/c2 tie: symmetric ranks
    assert {ranked[0]["chunk_id"], ranked[1]["chunk_id"]} == {"c1", "c2"}
    assert ranked[2]["chunk_id"] == "c3"  # only appears in one list -> lowest combined score
    print("OK, c1/c2 (in both lists) both outrank c3 (in only one list).")

    print("\n=== fuse: safety hits always included, never scored ===")
    from dataclasses import dataclass

    @dataclass
    class FakeSafetyHit:
        patient_id: str
        category: str
        matched_term: str
        snippet: str
        note_id: str | None = None

    safety_hits = [
        FakeSafetyHit("p1", "self_harm_risk", "wanting to die", "disclosed thoughts of wanting to die", "n3"),
    ]
    result = fuse(chroma_hits, bm25_hits, safety_hits=safety_hits, top_n=2)
    print([(r["chunk_id"], r["safety_flag"]) for r in result])
    assert result[0]["safety_flag"] is True
    assert len(result) == 1 + 2  # 1 safety + top_n=2 diversified

    print("\n=== MMR: near-duplicate chunks get spread out, not both picked first ===")
    dup_hits_chroma = [
        {"chunk_id": "d1", "text": "patient reports feeling anxious about work deadlines", "note_id": "n1", "patient_id": "p1", "distance": 0.05},
        {"chunk_id": "d2", "text": "patient reports feeling anxious about work deadlines again", "note_id": "n1", "patient_id": "p1", "distance": 0.06},
        {"chunk_id": "d3", "text": "patient started a new painting hobby on weekends", "note_id": "n2", "patient_id": "p1", "distance": 0.15},
    ]
    diversified = fuse(dup_hits_chroma, [], top_n=2, mmr_lambda=0.5)
    print([d["chunk_id"] for d in diversified])
    ids = [d["chunk_id"] for d in diversified]
    assert "d3" in ids, "MMR should surface the diverse chunk d3 over picking both near-duplicates"

    print("\n=== empty inputs -> empty output ===")
    assert fuse([], [], []) == []
    print("OK.")

    print("\n=== top_n truncates the non-safety portion only ===")
    many = [
        {"chunk_id": f"m{i}", "text": f"unique content number {i}", "note_id": "n1", "patient_id": "p1", "distance": 0.1 * i}
        for i in range(30)
    ]
    result_many = fuse(many, [], top_n=20)
    assert len(result_many) == 20
    print(f"OK, {len(result_many)} candidates returned from 30 inputs with top_n=20.")

    print("\nSelf-test passed.")