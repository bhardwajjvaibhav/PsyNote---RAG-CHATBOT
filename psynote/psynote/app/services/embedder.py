"""
services/embedder.py

Step 3 of the ingestion flow, and also the first step of the query-time
flow (architecture doc, Sections 3 and 4): all-MiniLM-L6-v2, 384-dim
vectors (Section 6's model choice table).

Design note -- same DI shape as reranker.py: the real SentenceTransformer
model is loaded lazily, once, on first real use. Callers (ingestion
pipeline, rag_pipeline.py's query embedding step) can inject an
`embed_fn` instead, so tests and self-checks never need to download a
~90MB model or touch the network.

Design note -- one function for both ingestion and query embedding:
all-MiniLM-L6-v2 is a symmetric embedding model (not a separate
query-encoder / passage-encoder pair), so embed_texts() is correct to
use for both a note's chunks at ingest time and a user's question at
query time. If the model ever changes to an asymmetric one, that's the
seam where this module would need query vs. passage variants -- not
today.
"""

from __future__ import annotations

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model = None  # lazy-loaded singleton, only used when no embed_fn is injected


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _default_embed_fn(texts: list[str]) -> list[list[float]]:
    model = _load_model()
    vectors = model.encode(texts, convert_to_numpy=True)
    return [[float(x) for x in vec] for vec in vectors]


def embed_texts(texts: list[str], embed_fn=None) -> list[list[float]]:
    """
    Embed a batch of texts (a note's chunk texts, at ingest time).

    Returns one 384-dim vector per input text, same order. Raises
    ValueError if the injected embed_fn returns a mismatched count --
    same defensive shape as reranker.rerank()'s scorer length check, so
    a silently-misaligned embed_fn fails loudly here rather than
    corrupting chunk_id <-> embedding pairing downstream in
    vector_store.upsert_chunks.
    """
    if not texts:
        return []

    fn = embed_fn or _default_embed_fn
    vectors = fn(texts)
    if len(vectors) != len(texts):
        raise ValueError(f"embed_fn returned {len(vectors)} vectors for {len(texts)} texts")
    return vectors


def embed_query(query: str, embed_fn=None) -> list[float]:
    """
    Embed a single query string (query-time flow, Section 4). Thin
    wrapper over embed_texts so query and ingestion embedding always go
    through the exact same code path -- no drift between the two.
    """
    return embed_texts([query], embed_fn=embed_fn)[0]


# --- Quick self-test ----------------------------------------------------------
# Run this file directly: python embedder.py
# Uses an injected fake embed_fn so this never needs to download the
# real model.

if __name__ == "__main__":

    def fake_embed_fn(texts: list[str]) -> list[list[float]]:
        """Deterministic 384-dim stand-in: hash each text into a fixed-size vector."""
        vectors = []
        for text in texts:
            h = abs(hash(text))
            vectors.append([((h >> (i % 32)) % 1000) / 1000.0 for i in range(EMBEDDING_DIM)])
        return vectors

    print("=== embed_texts: batch, order preserved, right dimensionality ===")
    texts = ["Patient reports improved sleep.", "Patient discussed work stress."]
    vectors = embed_texts(texts, embed_fn=fake_embed_fn)
    for t, v in zip(texts, vectors):
        print(t, "->", len(v), "dims, first 3:", v[:3])
    assert len(vectors) == 2
    assert all(len(v) == EMBEDDING_DIM for v in vectors)

    print("\n=== embed_query: single string -> single vector ===")
    qvec = embed_query("How has the patient's mood been?", embed_fn=fake_embed_fn)
    assert len(qvec) == EMBEDDING_DIM
    print(f"OK, {len(qvec)}-dim query vector.")

    print("\n=== deterministic: same text -> same vector ===")
    v1 = embed_texts(["consistent text"], embed_fn=fake_embed_fn)[0]
    v2 = embed_texts(["consistent text"], embed_fn=fake_embed_fn)[0]
    assert v1 == v2
    print("OK, embedding is deterministic for identical input.")

    print("\n=== empty input -> empty output, no crash ===")
    assert embed_texts([], embed_fn=fake_embed_fn) == []
    print("OK.")

    print("\n=== mismatched embed_fn length raises ===")
    def broken_embed_fn(texts):
        return [[0.0] * EMBEDDING_DIM]  # wrong length on purpose
    try:
        embed_texts(["a", "b"], embed_fn=broken_embed_fn)
        print("FAILED: should have raised ValueError")
    except ValueError as e:
        print(f"OK, raised: {e}")

    print("\nSelf-test passed.")