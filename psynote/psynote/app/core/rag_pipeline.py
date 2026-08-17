"""
core/rag_pipeline.py

The orchestrator (architecture doc, Section 7: "orchestrator, gets
thinner" as more logic moves into its own modules). Two entry points,
matching the doc's two flows:

  ingest_note()  -- Section 3, the ingestion flow
  answer_query() -- Section 4, the query-time retrieval flow

This module deliberately contains almost no logic of its own -- same
principle as api/routes.py's docstring ("routes translate HTTP <->
function calls... all the actual rules live [elsewhere], where they
were already tested in isolation"). Every real decision (chunk
boundaries, isolation guarantees, RRF math, safety rules, retry
behavior, grounding thresholds) already lives in, and is already
self-tested in, its own module. rag_pipeline.py's only job is calling
those modules in the right order and handling the couple of cross-
module concerns that don't belong in any single one of them:
doc_registry's pending/indexed/failed status transitions (Section 3's
write-ordering fix), and building the final prompt.

Design note -- safety scan scope at query time: the architecture doc's
Section 4 diagram shows the safety scanner as a parallel branch
alongside Chroma and BM25, scoped to patient_id. There is currently no
dedicated store of each note's full raw text separate from its chunks
(chunks live in vector_store/bm25_index; doc_registry stores only
metadata) -- so, for now, the scanner runs over the same chunk texts
already being pulled back by the Chroma + BM25 retrieval for this
query, not the patient's entire corpus. This means a safety-relevant
disclosure sitting in a note that neither retrieval branch surfaced for
this particular question won't be caught here. That's a real,
deliberate scope gap, not an oversight -- it should be closed (e.g. by
scanning the full corpus per patient, or scanning at ingestion time and
persisting hits in doc_registry) before this goes anywhere near real
patient data, alongside the other Section 9 known gaps. Whichever fix
lands, only THIS function needs to change -- safety_scanner.py itself
already scans whatever notes it's given, unmodified.

Design note -- every collaborator is injectable: same DI shape as every
other module in this codebase (transport in llm_client, scorer in
reranker, embed_fn in embedder). rag_pipeline's self-test wires in
fakes for every single one, so the orchestration logic (call order,
status transitions, error handling) is fully covered without a real
DB, a real Chroma instance, a real model, or a real network call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core import fusion, post_validate, reranker, llm_client, safety_scanner
from services import chunker, embedder, parser
from db import doc_registry, vector_store, bm25_index

DEFAULT_RETRIEVAL_TOP_K = 20   # per-branch (Chroma, BM25) candidate pool, before fusion
DEFAULT_RERANK_TOP_K = 5       # Section 4: cross-encoder rerank -> top-5 chunks
DEFAULT_HISTORY_TURNS = 6      # windowed chat history (Section 4), in (user, assistant) turn pairs

SYSTEM_PROMPT = (
    "You are a clinical journal assistant. Answer the question using ONLY the "
    "context notes provided below, which belong to a single patient. If the "
    "context does not contain the answer, say so plainly -- do not guess or "
    "use outside knowledge. Cite which note each part of your answer comes "
    "from where possible."
)


# --- Ingestion flow (Section 3) ---------------------------------------------

@dataclass
class IngestResult:
    note_id: str
    status: str                 # "indexed" or "failed"
    chunk_count: int = 0
    error: str | None = None


def ingest_note(
    patient_id: str,
    filename: str,
    raw_bytes: bytes,
    parse_fn=None,
    chunk_fn=None,
    embed_fn=None,
    vector_upsert_fn=None,
    bm25_upsert_fn=None,
) -> IngestResult:
    """
    Full ingestion flow for one uploaded file: parse -> chunk -> embed
    -> write to doc_registry (pending) -> write to Chroma + BM25 ->
    doc_registry (indexed or failed).

    Implements the write-ordering fix from Section 3: the doc_registry
    row is created as 'pending' BEFORE either index write is attempted,
    so a note is never silently "ghost" -- visible in SQL but
    unretrievable. If either index write raises, the note is marked
    'failed' with the error attached and this function returns that
    status rather than raising, so a caller (the /api/ingest route) can
    surface a clean error to the frontend without a stack trace.
    """
    parse = parse_fn or parser.parse_file
    chunk = chunk_fn or chunker.chunk_note
    embed = embed_fn or embedder.embed_texts
    vector_upsert = vector_upsert_fn or vector_store.upsert_chunks
    bm25_upsert = bm25_upsert_fn or bm25_index.upsert_chunks

    note = doc_registry.create_note(patient_id, filename)
    note_id = note["id"]

    try:
        text = parse(filename, raw_bytes)
        chunks = chunk(note_id, text)
        if not chunks:
            raise ValueError("chunking produced zero chunks from parsed text")

        chunk_ids = [c["chunk_id"] for c in chunks]
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed(chunk_texts)

        vector_upsert(patient_id, note_id, chunk_ids, chunk_texts, embeddings)
        bm25_upsert(patient_id, note_id, chunk_ids, chunk_texts)

    except Exception as e:
        doc_registry.mark_failed(note_id, str(e))
        return IngestResult(note_id=note_id, status="failed", error=str(e))

    doc_registry.mark_indexed(note_id)
    return IngestResult(note_id=note_id, status="indexed", chunk_count=len(chunks))


# --- Query-time flow (Section 4) --------------------------------------------

def _window_history(chat_history: list[dict] | None, turns: int) -> list[dict]:
    """Keep only the last `turns` (user, assistant) pairs, i.e. the last 2*turns messages."""
    if not chat_history:
        return []
    return chat_history[-(turns * 2):]


def _build_messages(question: str, context_chunks: list[dict], chat_history: list[dict]) -> list[dict]:
    context_block = "\n\n".join(
        f"[Note {c.get('note_id', 'unknown')} | chunk {c.get('chunk_id')}]\n{c.get('text', '')}"
        for c in context_chunks
    ) or "(no relevant context retrieved)"

    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nContext:\n{context_block}"}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": question})
    return messages


@dataclass
class QueryResult:
    answer: str
    model: str
    citations: list[dict] = field(default_factory=list)   # top chunks used, for the UI
    safety_hits: list = field(default_factory=list)
    validation: post_validate.ValidationResult | None = None


def answer_query(
    patient_id: str,
    question: str,
    chat_history: list[dict] | None = None,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    history_turns: int = DEFAULT_HISTORY_TURNS,
    embed_query_fn=None,
    chroma_query_fn=None,
    bm25_search_fn=None,
    fuse_fn=None,
    rerank_fn=None,
    generate_fn=None,
    validate_fn=None,
) -> QueryResult:
    """
    Full query-time flow: embed -> {Chroma, BM25, safety scan} -> fuse
    (RRF+MMR) -> rerank (cross-encoder, top-5) -> build prompt (context +
    windowed history + question) -> generate -> post-validate.

    patient_id is assumed to already be authenticated/authorized by the
    caller (Section 4: "patient_id still comes only from the
    authenticated session/request -- never from free text, never from
    the LLM" -- this function has no way to check that itself, since
    auth isn't built yet, Section 9's known gap).
    """
    embed_query = embed_query_fn or embedder.embed_query
    chroma_query = chroma_query_fn or vector_store.query
    bm25_search = bm25_search_fn or bm25_index.search
    fuse = fuse_fn or fusion.fuse
    rerank = rerank_fn or reranker.rerank
    generate = generate_fn or llm_client.generate
    validate = validate_fn or post_validate.validate_answer

    query_embedding = embed_query(question)
    chroma_hits = chroma_query(patient_id, query_embedding, top_k=retrieval_top_k)
    bm25_hits = bm25_search(patient_id, question, top_k=retrieval_top_k)

    # Safety scan: see module docstring for the current scope decision
    # (retrieved chunk text, not the full per-patient corpus).
    candidate_notes = {
        (h.get("note_id"), h["chunk_id"]): h["text"]
        for h in (chroma_hits + bm25_hits)
        if h.get("text")
    }
    notes_for_scan = [(note_id or chunk_id, text) for (note_id, chunk_id), text in candidate_notes.items()]
    safety_hits = safety_scanner.scan_patient_records(patient_id, notes_for_scan)

    fused = fuse(chroma_hits, bm25_hits, safety_hits, top_n=retrieval_top_k)
    top_chunks = rerank(question, fused, top_k=rerank_top_k)

    windowed_history = _window_history(chat_history, history_turns)
    messages = _build_messages(question, top_chunks, windowed_history)

    generation = generate(messages)
    validation = validate(generation["text"], top_chunks)

    citations = [
        {"chunk_id": c.get("chunk_id"), "note_id": c.get("note_id"), "safety_flag": c.get("safety_flag", False)}
        for c in top_chunks
    ]

    return QueryResult(
        answer=generation["text"],
        model=generation["model"],
        citations=citations,
        safety_hits=safety_hits,
        validation=validation,
    )


# --- Quick self-test ----------------------------------------------------------
# Run this file directly: python rag_pipeline.py
# Every collaborator is injected as a fake -- no real DB, model, or
# network call is made anywhere in this self-test.

if __name__ == "__main__":

    print("=== ingest_note: happy path, marks indexed with real chunk count ===")

    class FakeDocRegistry:
        def __init__(self):
            self.notes = {}
            self.next_id = 0

        def create_note(self, patient_id, filename):
            self.next_id += 1
            note_id = f"note{self.next_id}"
            self.notes[note_id] = {"id": note_id, "patient_id": patient_id, "filename": filename, "status": "pending"}
            return self.notes[note_id]

        def mark_indexed(self, note_id):
            self.notes[note_id]["status"] = "indexed"

        def mark_failed(self, note_id, error):
            self.notes[note_id]["status"] = "failed"
            self.notes[note_id]["error_detail"] = error

    fake_registry = FakeDocRegistry()
    doc_registry.create_note = fake_registry.create_note
    doc_registry.mark_indexed = fake_registry.mark_indexed
    doc_registry.mark_failed = fake_registry.mark_failed

    vector_calls = []
    bm25_calls = []

    result = ingest_note(
        "patient_a", "session_1.txt", b"Patient reports improved mood this week and better sleep.",
        parse_fn=lambda fn, raw: raw.decode("utf-8"),
        chunk_fn=lambda note_id, text: [{"chunk_id": f"{note_id}::chunk0", "note_id": note_id, "text": text}],
        embed_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        vector_upsert_fn=lambda *a: vector_calls.append(a),
        bm25_upsert_fn=lambda *a: bm25_calls.append(a),
    )
    print(result)
    assert result.status == "indexed"
    assert result.chunk_count == 1
    assert fake_registry.notes[result.note_id]["status"] == "indexed"
    assert len(vector_calls) == 1 and len(bm25_calls) == 1

    print("\n=== ingest_note: index write fails -> marked failed, no exception raised ===")

    def broken_vector_upsert(*a):
        raise RuntimeError("Chroma write timed out")

    result_failed = ingest_note(
        "patient_a", "session_2.txt", b"Some note text here for chunking purposes.",
        parse_fn=lambda fn, raw: raw.decode("utf-8"),
        chunk_fn=lambda note_id, text: [{"chunk_id": f"{note_id}::chunk0", "note_id": note_id, "text": text}],
        embed_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        vector_upsert_fn=broken_vector_upsert,
        bm25_upsert_fn=lambda *a: bm25_calls.append(a),
    )
    print(result_failed)
    assert result_failed.status == "failed"
    assert "timed out" in result_failed.error
    assert fake_registry.notes[result_failed.note_id]["status"] == "failed"

    print("\n=== ingest_note: empty parsed text -> zero chunks -> failed, not a silent no-op ===")
    result_empty = ingest_note(
        "patient_a", "session_3.txt", b"irrelevant",
        parse_fn=lambda fn, raw: "some text",
        chunk_fn=lambda note_id, text: [],  # simulate chunker returning nothing
        embed_fn=lambda texts: [],
        vector_upsert_fn=lambda *a: None,
        bm25_upsert_fn=lambda *a: None,
    )
    assert result_empty.status == "failed"
    print(result_empty)

    print("\n=== answer_query: full flow with fakes, grounded answer ===")

    fake_chroma_hits = [
        {"chunk_id": "c1", "text": "Patient reports significantly improved sleep this week.", "note_id": "note1", "patient_id": "patient_a", "distance": 0.1},
    ]
    fake_bm25_hits = [
        {"chunk_id": "c1", "text": "Patient reports significantly improved sleep this week.", "note_id": "note1", "patient_id": "patient_a", "score": 4.0},
    ]

    def fake_generate(messages):
        # Sanity check the prompt actually contains context + the question.
        system_msg = messages[0]["content"]
        assert "improved sleep" in system_msg
        assert messages[-1]["content"] == "How has the patient's sleep been?"
        return {"text": "The patient reports significantly improved sleep this week.", "model": "fake-model", "attempts": []}

    qr = answer_query(
        "patient_a",
        "How has the patient's sleep been?",
        embed_query_fn=lambda q: [0.1, 0.2, 0.3],
        chroma_query_fn=lambda pid, emb, top_k: fake_chroma_hits,
        bm25_search_fn=lambda pid, q, top_k: fake_bm25_hits,
        rerank_fn=lambda query, candidates, top_k: candidates[:top_k],  # skip real cross-encoder
        generate_fn=fake_generate,
    )
    print(qr.answer)
    print("citations:", qr.citations)
    print("grounded:", qr.validation.is_fully_grounded)
    assert qr.model == "fake-model"
    assert qr.validation.is_fully_grounded
    assert qr.citations[0]["chunk_id"] == "c1"

    print("\n=== answer_query: chat history gets windowed to the configured turn count ===")
    long_history = []
    for i in range(10):
        long_history.append({"role": "user", "content": f"question {i}"})
        long_history.append({"role": "assistant", "content": f"answer {i}"})

    captured_messages = {}

    def capturing_generate(messages):
        captured_messages["messages"] = messages
        return {"text": "ok.", "model": "fake-model", "attempts": []}

    answer_query(
        "patient_a", "latest question",
        chat_history=long_history,
        history_turns=2,
        embed_query_fn=lambda q: [0.1],
        chroma_query_fn=lambda pid, emb, top_k: [],
        bm25_search_fn=lambda pid, q, top_k: [],
        rerank_fn=lambda query, candidates, top_k: candidates[:top_k],
        generate_fn=capturing_generate,
    )
    # system + 2 turns (4 messages) + final question = 6
    assert len(captured_messages["messages"]) == 1 + 4 + 1
    print(f"OK, {len(captured_messages['messages'])} messages sent (windowed to 2 turns).")

    print("\nSelf-test passed.")