
# app.py (single-file FastAPI app)
# RAG-based Course Recommendation Engine using ONLY chromadb for the Vector DB
# - Matches your working requirements.txt pins (LangChain 1.1.0 / langchain-google-genai 3.2.0 / chromadb 1.3.5 / google-genai 1.52.0)
# - Accepts ONLY a single free-text `query`
# - On startup: auto-downloads CSV and builds Chroma index (PersistentClient)
# typing import Dict, Any, List# - Pipeline: LLM extracts profile -> Chroma retrieval (with distances) -> LLM re-ranks -> JSON output

from __future__ import annotations

import os
from pathlib import Path
from contextlib import asynccontextmanager
from functools import lru_cache


import requests
import pandas as pd
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# LangChain LLM & Embeddings (works with langchain-google-genai==3.2.0 and google-genai>=1.52.0)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Optional: load .env
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# --------------------------------------------------------------------------------------
# Configuration / Paths
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
if load_dotenv:
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)

# Paths
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
for p in (DATA_DIR, REPORTS_DIR, STORAGE_DIR, CHROMA_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Dataset
DATASET_URL = (
    "https://raw.githubusercontent.com/Bluedata-Consulting/"
    "GAAPB01-training-code-base/refs/heads/main/Assignments/assignment2dataset.csv"
)
DATASET_PATH = DATA_DIR / "assignment2dataset.csv"

# Embeddings / LLM (env-driven)
USE_GOOGLE_EMBEDDINGS = os.getenv("USE_GOOGLE_EMBEDDINGS", "false").lower() == "true"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")  # or "text-embedding-004"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))

# Chroma collection per provider (prevents mixed-dimension errors when switching)
CHROMA_COLLECTION = f"courses_collection_{'gemini' if USE_GOOGLE_EMBEDDINGS else 'hf'}"

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _assert_api_key():
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY missing in .env")

def _download_dataset_if_missing() -> Path:
    if DATASET_PATH.exists():
        return DATASET_PATH
    resp = requests.get(DATASET_URL, timeout=60)
    resp.raise_for_status()
    DATASET_PATH.write_bytes(resp.content)
    return DATASET_PATH

def _load_catalog() -> pd.DataFrame:
    _download_dataset_if_missing()
    df = pd.read_csv(DATASET_PATH)
    expected = {"course_id", "title", "description"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    return df.fillna("")

def _make_doc_text(title: str, description: str) -> str:
    return f"Title: {title}\nDescription: {description}"

def _user_query_text(completed_courses: List[str], interests: str) -> str:
    completed = "; ".join(completed_courses) if completed_courses else "None"
    interests = (interests or "").strip()
    return (
        f"Completed courses: {completed}\n"
        f"Interests: {interests}\n"
        "Goal: Recommend next best courses."
    )

def _clip(s: str, n: int = 220) -> str:
    return (s or "")[:n].replace("\n", " ").strip()

# --------------------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_embeddings():
    """
    Returns a LangChain-compatible embeddings object.
    Default: HuggingFace sentence-transformers (offline-friendly).
    Optional: Google Gemini embeddings when USE_GOOGLE_EMBEDDINGS=true.
    """
    if USE_GOOGLE_EMBEDDINGS:
        _assert_api_key()
        return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    else:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --------------------------------------------------------------------------------------
# LLM (Gemini) – structured outputs
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_llm():
    """
    Returns a LangChain-compatible Gemini chat model.
    """
    _assert_api_key()
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )

# JSON schema for final recommendations
REC_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "course_id": {"type": "string"},
            "title": {"type": "string"},
            "score": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["course_id", "title", "score", "reason"],
    },
}

# JSON schema for extracting profile from free-text query
PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "completed_courses": {"type": "array", "items": {"type": "string"}},
        "interests": {"type": "string"},
    },
    "required": ["completed_courses", "interests"],
}

# --------------------------------------------------------------------------------------
# Chroma index (PersistentClient) — no langchain_chroma wrapper
# --------------------------------------------------------------------------------------
def _get_client() -> chromadb.PersistentClient:
    """Persistent client; data auto-persisted at CHROMA_DIR."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))

def _index_exists() -> bool:
    """Returns True if an existing collection appears non-empty."""
    client = _get_client()
    try:
        col = client.get_collection(name=CHROMA_COLLECTION)
        try:
            return col.count() > 0
        except Exception:
            return True
    except Exception:
        return False

def _build_index_idempotent():
    """
    Build the index if empty/missing; idempotent.
    Precomputes embeddings and upserts with vectors for deterministic behavior.
    Implemented via chromadb.PersistentClient for precise control.
    """
    if _index_exists():
        print("[index] Existing collection detected; skipping rebuild.")
        return

    client = _get_client()

    # Clean slate
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(name=CHROMA_COLLECTION)
    df = _load_catalog()
    emb = _get_embeddings()

    ids, docs, metas, texts = [], [], [], []
    for _, row in df.iterrows():
        cid = str(row["course_id"])
        title = str(row["title"])
        desc = str(row["description"])
        doc_text = _make_doc_text(title, desc)
        ids.append(cid)
        docs.append(doc_text)
        metas.append({"course_id": cid, "title": title, "description": desc})
        texts.append(doc_text)

    # Precompute vectors once (deterministic & faster at query time)
    vectors = emb.embed_documents(texts)
    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=vectors,
    )
    print("[index] Built collection with PersistentClient at", CHROMA_DIR)

def _load_collection():
    client = _get_client()
    try:
        return client.get_collection(name=CHROMA_COLLECTION)
    except Exception:
        return client.get_or_create_collection(name=CHROMA_COLLECTION)

# --------------------------------------------------------------------------------------
# Profile extraction from single query
# --------------------------------------------------------------------------------------
def _extract_profile_from_query(query: str) -> Dict[str, Any]:
    """
    Use LLM to extract completed course names & interests from a single free-text query.
    Fallback: if extraction fails, treat entire query as 'interests', completed_courses = [].
    """
    structured_profile_llm = _get_llm().with_structured_output(
        schema=PROFILE_SCHEMA, method="json_mode", include_raw=False
    )
    try:
        res = structured_profile_llm.invoke(
            f"""Extract the user's completed courses and interests from the message.
User message: {query}
Return JSON with keys: completed_courses (array of course names), interests (short string)."""
        )
        if isinstance(res, dict):
            return {
                "completed_courses": res.get("completed_courses", []) or [],
                "interests": res.get("interests", "") or "",
            }
    except Exception:
        pass
    return {"completed_courses": [], "interests": query.strip()}

# --------------------------------------------------------------------------------------
# Retrieval (Chroma) — returns docs + distances mapped to similarity
# --------------------------------------------------------------------------------------
def _to_similarity(distance: float) -> float:
    """
    Chroma's low-level query API returns distances (lower is better).
    Convert to an intuitive similarity score in [0, 1].
    """
    sim = max(0.0, min(1.0, 1.0 - float(distance)))
    return round(sim, 4)

def _retrieve_from_chroma(profile_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve candidates directly from Chroma using the embedded query vector.
    """
    collection = _load_collection()
    emb = _get_embeddings()
    qvec = emb.embed_query(profile_text)

    res = collection.query(
        query_embeddings=[qvec],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )
    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    out: List[Dict[str, Any]] = []
    for meta, dist in zip(metadatas, distances):
        out.append({
            "course_id": meta.get("course_id"),
            "title": meta.get("title"),
            "description": meta.get("description"),
            "relevance": _to_similarity(dist),
        })
    return out

def _format_kb_context(retrieved: List[Dict[str, Any]]) -> str:
    lines = []
    for r in retrieved:
        desc_snip = _clip(r.get("description", ""))
        lines.append(f"[{r['relevance']}] {r['course_id']} {r['title']}: {desc_snip}...")
    return "\n".join(lines)

# --------------------------------------------------------------------------------------
# RAG: Use Chroma-based Knowledge during LLM call
# --------------------------------------------------------------------------------------
def run_rag_recommendation_from_query(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Full RAG pipeline:
    - Extract profile (completed courses + interests) from single free-text query
    - Build 'profile_text' and retrieve top-k course candidates from Chroma
    - Re-rank & score with LLM (JSON mode) to return structured list
    """
    profile = _extract_profile_from_query(query)
    profile_text = _user_query_text(profile["completed_courses"], profile["interests"])

    # Retrieve from Chroma (Vector DB) with distances converted to similarity
    retrieved = _retrieve_from_chroma(profile_text, top_k)
    kb_context = _format_kb_context(retrieved)

    structured_llm = _get_llm().with_structured_output(
        schema=REC_SCHEMA, method="json_mode", include_raw=False
    )

    prompt_text = f"""You are a course recommender.
USER PROFILE:
{profile_text}

KNOWLEDGE BASE CONTEXT (courses retrieved from a vector database):
{kb_context}

TASK:
Return exactly {top_k} courses as structured JSON (no markdown). For each:
- course_id: string
- title: string
- score: float in [0.0, 1.0] (rounded to 2 decimals)
- reason: one sentence

Scoring rules:
- Baseline: use the numeric [relevance] shown for each candidate (already in [0.0, 1.0]).
- Interest alignment: if title/description strongly match the user's interests, add up to +0.35.
- Next-step progression: if the course clearly extends the user's completed courses (specialization or production best-practices),
  add up to +0.35.
- Redundancy penalty: if the course looks already completed or repeats the user's foundation, subtract up to -0.60.

Normalisation:
- Clamp final score to [0.0, 1.0].
- Round to 2 decimals.
- If two items tie, prefer stronger interest alignment, then stronger next-step progression, then higher baseline [relevance].

Return ONLY the JSON array with the required keys, no extra text."""
    llm_parsed = structured_llm.invoke(prompt_text)
    return {"profile": profile, "raw": retrieved, "llm_json": llm_parsed}

# --------------------------------------------------------------------------------------
# FastAPI
# --------------------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., description="Single free-text user query")
    top_k: int = Field(default=TOP_K_DEFAULT)
    # Use LLM format for output to align with 'Use Knowledge base, user prompt, and call LLM'
    use_llm_format: bool = Field(default=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build once at startup: auto-download dataset and persist Chroma index
    _build_index_idempotent()
    yield
    # Shutdown: file-based persistence, nothing to do

app = FastAPI(
    title="Course Recommendation Engine (Chroma-only, pins per working requirements.txt)",
    version="3.4.0",
    lifespan=lifespan,
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(req: QueryRequest) -> Dict[str, Any]:
    if req.use_llm_format and not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="GOOGLE_API_KEY missing; set use_llm_format=false or add GOOGLE_API_KEY in .env."
        )
    result = run_rag_recommendation_from_query(query=req.query, top_k=req.top_k)
    # Return LLM JSON by default, also include extracted profile & raw retrieval for transparency
    return {"result": result}

