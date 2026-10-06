from contextlib import asynccontextmanager
from pathlib import Path
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .documents import SUPPORTED, extract_text, chunk_text
from .schemas import AskRequest, AskResponse, Source
from .vector_store import VectorStore
from .llm import generate_answer

store: VectorStore | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    store = VectorStore(settings.data_dir, settings.embedding_model)
    yield

app = FastAPI(title="AI Knowledge Assistant", version="1.0.0", description="Document Q&A using retrieval-augmented generation.", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok", "indexed_documents": len(store.documents()) if store else 0}

@app.post("/documents", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    if store is None:
        raise HTTPException(503, "Index is initializing")
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in SUPPORTED:
        raise HTTPException(400, "Supported formats: TXT, Markdown, PDF, DOCX")
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
    safe_name = re.sub(r"[^\w.() -]", "_", filename).strip()[:180]
    path = Path(settings.data_dir) / safe_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    try:
        text = extract_text(path)
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise ValueError("No extractable text found. Scanned PDFs need OCR before upload.")
        count = store.add_document(safe_name, chunks)
    except Exception as exc:
        path.unlink(missing_ok=True)
        if isinstance(exc, ValueError):
            raise HTTPException(400, str(exc)) from exc
        raise HTTPException(500, "Could not process document") from exc
    return {"filename": safe_name, "chunks_indexed": count}

@app.get("/documents")
def list_documents():
    return store.documents() if store else []

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    if store is None or not store.delete_document(filename):
        raise HTTPException(404, "Document not found")
    (Path(settings.data_dir) / Path(filename).name).unlink(missing_ok=True)
    return {"deleted": filename}

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if store is None:
        raise HTTPException(503, "Index is initializing")
    matches = store.search(request.question, request.top_k or settings.top_k, settings.similarity_threshold)
    if not matches:
        return AskResponse(
            answer="I couldn't find the answer in the uploaded documents.",
            sources=[],
            grounded=False
        )
    answer = await generate_answer(request.question, matches)
    grounded = answer is not None
    if answer is None:
        answer = (
            "The local LLM is unavailable. Here are the retrieved excerpts:\n\n"
            + "\n\n".join(
                f"[{i}] {m['text']}" for i, m in enumerate(matches, 1)
            )
        )
    sources = [Source(document=m["document"], chunk=m["chunk"], score=round(m["score"], 4), excerpt=m["text"][:500]) for m in matches]
    return AskResponse(answer=answer, sources=sources, grounded=grounded)
