"""
RAG Document Assistant -- Web API

Run with:
    uvicorn app:app --reload

Then open http://localhost:8000 for the UI, or use the API directly:
    POST /documents   (multipart file upload)
    POST /ask         ({"question": "..."})
    GET  /status
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag_engine

app = FastAPI(title="RAG Document Assistant")

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str
    top_k: int = 4


@app.get("/")
def serve_ui():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    raise HTTPException(404, "UI not found -- static/index.html is missing.")


@app.get("/status")
def status():
    store = rag_engine.get_store()
    return {"chunks_stored": store.document_count()}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = rag_engine.ingest_document(tmp_path, source_name=file.filename)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not result["success"]:
        raise HTTPException(422, result["message"])
    return result


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    return rag_engine.answer_question(req.question, top_k=req.top_k)