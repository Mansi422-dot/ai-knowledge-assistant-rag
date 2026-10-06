from pathlib import Path
from pypdf import PdfReader
from docx import Document as DocxDocument

SUPPORTED = {".txt", ".md", ".pdf", ".docx"}

def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if suffix == ".docx":
        doc = DocxDocument(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"Unsupported file type: {suffix}")

def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + size // 2:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
