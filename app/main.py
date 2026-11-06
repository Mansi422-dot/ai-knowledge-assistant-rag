import logging
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .documents import SUPPORTED, chunk_text, extract_text
from .llm import generate_answer
from .schemas import AskRequest, AskResponse, Source
from .vector_store import VectorStore


logger = logging.getLogger(__name__)

store: VectorStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store

    store = VectorStore(
        settings.data_dir,
        settings.embedding_model,
    )

    yield

    store = None


app = FastAPI(
    title="AI Knowledge Assistant",
    version="1.0.0",
    description="Document Q&A using retrieval-augmented generation.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


def get_store() -> VectorStore:
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="The document index is initializing.",
        )

    return store


def normalize_filename(original: str | None) -> str:
    if not original:
        raise HTTPException(400, "A filename is required.")

    # Handle both POSIX and Windows path separators.
    original = original.replace("\\", "/")
    filename = original.rsplit("/", 1)[-1].strip()

    if not filename or filename in {".", ".."}:
        raise HTTPException(400, "Invalid filename.")

    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED:
        raise HTTPException(
            400,
            "Supported formats: TXT, Markdown, PDF, DOCX.",
        )

    safe_name = re.sub(
        r"[^\w.() -]",
        "_",
        filename,
    ).strip(" .")

    if not safe_name:
        raise HTTPException(400, "Invalid filename.")

    if safe_name.startswith("."):
        raise HTTPException(400, "Hidden filenames are not allowed.")

    if len(safe_name) > 180:
        stem = Path(safe_name).stem[:150]
        safe_name = f"{stem}{suffix}"

    if Path(safe_name).suffix.lower() != suffix:
        raise HTTPException(400, "Invalid filename extension.")

    return safe_name


@app.get("/health")
def health():
    current_store = get_store()

    return {
        "status": "ok",
        "indexed_documents": len(current_store.documents()),
    }


@app.post("/documents", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    current_store = get_store()

    filename = normalize_filename(file.filename)
    destination = current_store.root / filename

    maximum_bytes = settings.max_upload_mb * 1024 * 1024

    try:
        content = await file.read(maximum_bytes + 1)

        if len(content) > maximum_bytes:
            raise HTTPException(
                413,
                f"File exceeds {settings.max_upload_mb} MB limit.",
            )

        if not content:
            raise HTTPException(400, "The uploaded file is empty.")

        # Serialize filename checks, file operations and index updates.
        with current_store.lock:
            indexed_names = {
                item["filename"]
                for item in current_store.documents()
            }

            if destination.exists() or filename in indexed_names:
                raise HTTPException(
                    409,
                    "A document with this filename already exists.",
                )

            staged_path = None
            published = False

            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=current_store.root,
                    prefix=".upload-",
                    suffix=Path(filename).suffix.lower(),
                    delete=False,
                ) as temporary:
                    staged_path = Path(temporary.name)
                    temporary.write(content)
                    temporary.flush()

                text = extract_text(staged_path)

                chunks = chunk_text(
                    text,
                    settings.chunk_size,
                    settings.chunk_overlap,
                )

                if not chunks:
                    raise HTTPException(
                        400,
                        "No extractable text found. Scanned PDFs require OCR.",
                    )

                # Publish the file without overwriting an existing file.
                # The shared lock prevents concurrent API uploads from
                # claiming the same destination.
                if destination.exists():
                    raise HTTPException(
                        409,
                        "A document with this filename already exists.",
                    )

                staged_path.replace(destination)
                staged_path = None
                published = True

                count = current_store.add_document(
                    filename,
                    chunks,
                )

                return {
                    "filename": filename,
                    "chunks_indexed": count,
                }

            except HTTPException:
                if published:
                    destination.unlink(missing_ok=True)
                raise

            except ValueError as exc:
                if published:
                    destination.unlink(missing_ok=True)

                raise HTTPException(
                    400,
                    str(exc),
                ) from exc

            except Exception as exc:
                if published:
                    destination.unlink(missing_ok=True)

                logger.exception("Document processing failed.")

                raise HTTPException(
                    500,
                    "Could not process document.",
                ) from exc

            finally:
                if staged_path is not None:
                    staged_path.unlink(missing_ok=True)

    finally:
        await file.close()


@app.get("/documents")
def list_documents():
    current_store = get_store()

    with current_store.lock:
        return current_store.documents()


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    current_store = get_store()

    # Do not permit path components in the route parameter.
    if (
        not filename
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or filename in {".", ".."}
    ):
        raise HTTPException(400, "Invalid filename.")

    with current_store.lock:
        indexed_names = {
            item["filename"]
            for item in current_store.documents()
        }

        if filename not in indexed_names:
            raise HTTPException(404, "Document not found.")

        document_path = current_store.root / filename

        if not document_path.is_file():
            raise HTTPException(
                409,
                "The document file is missing. Index recovery is required.",
            )

        quarantine_path = None

        try:
            # Temporarily move the file so a failed index update can
            # restore it without overwriting another file.
            with tempfile.NamedTemporaryFile(
                dir=current_store.root,
                prefix=".delete-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                quarantine_path = Path(temporary.name)

            quarantine_path.unlink()
            document_path.replace(quarantine_path)

            if not current_store.delete_document(filename):
                quarantine_path.replace(document_path)
                raise HTTPException(409, "Could not update the document index.")

            quarantine_path.unlink(missing_ok=True)

            return {"deleted": filename}

        except HTTPException:
            raise

        except Exception as exc:
            if quarantine_path and quarantine_path.exists():
                quarantine_path.replace(document_path)

            logger.exception("Document deletion failed.")

            raise HTTPException(
                500,
                "Could not delete document safely.",
            ) from exc


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    current_store = get_store()

    matches = current_store.search(
        request.question,
        request.top_k or settings.top_k,
        settings.similarity_threshold,
    )

    if not matches:
        return AskResponse(
            answer="I couldn't find the answer in the uploaded documents.",
            sources=[],
            grounded=False,
        )

    answer = await generate_answer(
        request.question,
        matches,
    )

    context_used = answer is not None

    if answer is None:
        answer = (
            "The local LLM is unavailable. "
            "Here are the retrieved excerpts:\n\n"
            + "\n\n".join(
                f"[{i}] {match['text']}"
                for i, match in enumerate(matches, 1)
            )
        )

    sources = [
        Source(
            document=match["document"],
            chunk=match["chunk"],
            score=round(match["score"], 4),
            excerpt=match["text"][:500],
        )
        for match in matches
    ]

    return AskResponse(
        answer=answer,
        sources=sources,
        grounded=context_used,
    )
