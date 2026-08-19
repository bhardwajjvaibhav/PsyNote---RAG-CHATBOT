"""
api/routes.py

Thin HTTP layer over db/patients.py, plus (new) the /api/ingest and
/api/chat endpoints that turn the ingestion and query-time flows
(architecture doc, Sections 3 and 4) into actual HTTP surface, replacing
the frontend's Section 2.5 stub.

Deliberately no logic here -- routes translate HTTP <-> function calls
and turn ValueError / None into the right status codes. All the actual
rules (empty name rejection, soft-delete, chunking, retrieval, safety
scanning, generation, grounding checks) live in db/patients.py and
core/rag_pipeline.py, where they were already tested in isolation --
this file's only job is the HTTP <-> function-call translation, same as
the existing patient endpoints below.

Known gap (carried forward from architecture doc, Section 9):
these endpoints have no authentication in front of them. That is
acceptable only for local development while the frontend is being
built -- must close before Phase 9 (security/) is skipped or delayed.
Section 4 is explicit that patient_id must come only from the
authenticated session, never from free text -- until auth exists, these
routes take patient_id as a path/body parameter instead, which is the
same known gap, not a new one.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pydantic import BaseModel

from db import patients, doc_registry
from core import rag_pipeline
from services.parser import SUPPORTED_EXTENSIONS

app = FastAPI(title="PsyNote API")

# Allows the React dev server (different port) to call this API during
# local development. Tighten this before anything resembling production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

patients.init_db()
doc_registry.init_db()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


class PatientCreate(BaseModel):
    name: str


class PatientUpdate(BaseModel):
    name: str


class ChatMessage(BaseModel):
    role: str    # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    patient_id: str
    question: str
    chat_history: list[ChatMessage] = []


class RegisterRequest(BaseModel):
    name: str
    age: int | None = None
    place: str | None = None
    gender: str | None = None
    marital_status: str | None = None


class LoginRequest(BaseModel):
    name: str


class TextNoteRequest(BaseModel):
    patient_id: str
    title: str
    content: str


@app.get("/api/patients")
def api_list_patients():
    return patients.list_patients()


@app.post("/api/patients", status_code=201)
def api_create_patient(body: PatientCreate):
    try:
        return patients.create_patient(body.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/patients/{patient_id}")
def api_get_patient(patient_id: str):
    patient = patients.get_patient(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="User not found")
    return patient


@app.put("/api/patients/{patient_id}")
def api_update_patient(patient_id: str, body: PatientUpdate):
    try:
        updated = patients.update_patient(patient_id, body.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if updated is None:
        raise HTTPException(status_code=404, detail="User not found")
    return updated


@app.delete("/api/patients/{patient_id}", status_code=204)
def api_delete_patient(patient_id: str):
    deleted = patients.delete_patient(patient_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")


@app.get("/api/health")
def health():
    return {"status": "ok"}


# --- Registration / sign-in (frontend auth flow) -----------------------------
# Stub auth carried forward from Phase 9: no passwords or tokens yet, just
# enough to bind the frontend session to a patient record. The moment real
# auth lands, these routes are the ones that change.

@app.post("/api/register", status_code=201)
def api_register(body: RegisterRequest):
    try:
        return patients.register_patient(
            body.name,
            age=body.age,
            place=body.place,
            gender=body.gender,
            marital_status=body.marital_status,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/login")
def api_login(body: LoginRequest):
    patient = patients.get_patient_by_name(body.name)
    if patient is None:
        raise HTTPException(status_code=404, detail="No registered user with that name")
    return patient


# --- Ingestion (Section 3) ---------------------------------------------------

@app.post("/api/ingest", status_code=201)
async def api_ingest(patient_id: str = Form(...), file: UploadFile = File(...)):
    """
    Replaces the Section 2.5 stub. Validates the patient exists and the
    file extension is supported (fast, cheap checks worth doing before
    the full parse/chunk/embed/write pipeline runs), then delegates
    everything else to rag_pipeline.ingest_note.

    A pipeline-level failure (bad PDF, embedding error, Chroma/BM25
    write failure) does NOT become an HTTP error here -- ingest_note
    already turns that into a note marked 'failed' in doc_registry with
    the reason attached (Section 3's write-ordering fix), which is more
    useful to the frontend than a generic 500: the note is visible and
    retryable, not just a failed request to retry blindly.
    """
    patient = patients.get_patient(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="User not found")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type {ext!r} -- expected one of {sorted(SUPPORTED_EXTENSIONS)}",
        )

    raw_bytes = await file.read()
    result = rag_pipeline.ingest_note(patient_id, file.filename, raw_bytes)

    return {
        "note_id": result.note_id,
        "status": result.status,
        "chunk_count": result.chunk_count,
        "error": result.error,
    }


@app.post("/api/notes/text", status_code=201)
def api_create_text_note(body: TextNoteRequest):
    """
    Ingest a note typed directly into the app (no file upload). Same
    pipeline as /api/ingest minus the parse step -- see
    rag_pipeline.ingest_text. Pipeline failures surface as status="failed"
    on a normal 201, same contract as /api/ingest.
    """
    patient = patients.get_patient(body.patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        result = rag_pipeline.ingest_text(body.patient_id, body.title, body.content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "note_id": result.note_id,
        "status": result.status,
        "chunk_count": result.chunk_count,
        "error": result.error,
    }


@app.get("/api/patients/{patient_id}/notes")
def api_list_notes(patient_id: str, status: str | None = None):
    patient = patients.get_patient(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="User not found")
    return doc_registry.list_notes_for_patient(patient_id, status=status)


@app.get("/api/insights/{patient_id}")
def api_insights(patient_id: str):
    """
    Emotional-energy report for the dashboard: per-note emotion scores
    (happy / sad / stressed / anxious), overall averages, and per-emotion
    reasons (matched terms + snippets attributed to their notes). See
    core/emotion.py -- deterministic, offline, no model call.
    """
    patient = patients.get_patient(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="User not found")

    report = rag_pipeline.analyze_mood(patient_id)

    return {
        "notes": [
            {
                "note_id": n.note_id,
                "title": n.title,
                "created_at": n.created_at,
                "scores": n.scores,
                "dominant": max(n.scores, key=n.scores.get) if any(n.scores.values()) else "neutral",
            }
            for n in report.notes
        ],
        "overall": report.overall,
        "reasons": {
            emotion: [
                {"note_id": r.note_id, "title": r.title, "term": r.term, "snippet": r.snippet, "count": r.count}
                for r in hits
            ]
            for emotion, hits in report.reasons.items()
        },
    }


# --- Query / chat (Section 4) -------------------------------------------------

@app.post("/api/chat")
def api_chat(body: ChatRequest):
    """
    patient_id currently comes from the request body -- see the known-
    gap note at the top of this file. Once auth (Phase 9) exists,
    patient_id should be derived from the authenticated session instead
    and this field removed from the request schema, per Section 4.
    """
    patient = patients.get_patient(body.patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="User not found")

    if not body.question or not body.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    history = [{"role": m.role, "content": m.content} for m in body.chat_history]

    try:
        result = rag_pipeline.answer_query(body.patient_id, body.question, chat_history=history)
    except rag_pipeline.llm_client.LLMAllProvidersExhaustedError as e:
        # Every fallback model failed -- see llm_client.py's docstring:
        # this is the exception this module decides what the user sees for.
        raise HTTPException(status_code=503, detail=f"Generation temporarily unavailable: {e}")

    return {
        "answer": result.answer,
        "model": result.model,
        "citations": result.citations,
        "safety_hits": [
            {
                "category": h.category,
                "note_id": h.note_id,
                "matched_term": h.matched_term,
            }
            for h in result.safety_hits
        ],
        "grounding": {
            "fully_grounded": result.validation.is_fully_grounded,
            "grounding_score": result.validation.grounding_score,
            "flagged_sentences": [
                s.sentence for s in result.validation.sentences if not s.grounded and not s.skipped
            ],
        },
    }