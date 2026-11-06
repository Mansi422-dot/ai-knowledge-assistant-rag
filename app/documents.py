from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader


SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        reader = PdfReader(str(path))

        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    if suffix == ".docx":
        doc = DocxDocument(str(path))

        sections = []

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                sections.append(paragraph.text)

        for table in doc.tables:
            for row in table.rows:
                cells = [
                    cell.text.strip()
                    for cell in row.cells
                    if cell.text.strip()
                ]

                if cells:
                    sections.append(" | ".join(cells))

        return "\n".join(sections)

    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(
    text: str,
    size: int = 900,
    overlap: int = 150,
) -> list[str]:

    if size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")

    if overlap >= size:
        raise ValueError("Overlap must be smaller than chunk size.")

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

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
