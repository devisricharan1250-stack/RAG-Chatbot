"""
The web server. Wires the RAG pipeline to HTTP endpoints the frontend calls.

Endpoints:
  GET  /api/health           -> quick liveness + index stats
  POST /api/upload           -> upload & index a document (multipart file)
  POST /api/chat             -> ask a question, get an answer + sources
  POST /api/reset            -> clear the whole index
  GET  /                     -> serves the frontend (if present)
"""
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
from ingest import DocumentStore
from rag import RAGPipeline

app = FastAPI(title="RAG Chat")

# Allow the frontend to call us during development from any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build these once at startup (loading the embedding model takes a moment).
store = DocumentStore()
pipeline = RAGPipeline(store)

os.makedirs(config.UPLOAD_DIR, exist_ok=True)

ALLOWED = {".pdf", ".docx", ".txt", ".md"}


class ChatRequest(BaseModel):
    question: str


@app.get("/api/health")
def health():
    return {"status": "ok", **store.stats()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported type '{ext}'. Allowed: pdf, docx, txt, md.")

    dest = os.path.join(config.UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())

    try:
        n = store.add_document(dest, file.filename)
    except Exception as e:
        raise HTTPException(500, f"Failed to index document: {e}")

    if n == 0:
        raise HTTPException(400, "No readable text found in that file.")
    return {"filename": file.filename, "chunks_added": n, **store.stats()}


@app.post("/api/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty.")
    try:
        return pipeline.answer(question)
    except Exception as e:
        raise HTTPException(500, f"Error generating answer: {e}")


@app.post("/api/reset")
def reset():
    store.reset()
    return {"status": "cleared", **store.stats()}


# ----- Serve the frontend so you can run everything with one command -----
FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")


@app.get("/")
def index():
    if os.path.exists(FRONTEND):
        return FileResponse(FRONTEND)
    return {"message": "Frontend not found. Open frontend/index.html manually."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
