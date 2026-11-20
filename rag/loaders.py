"""Read .txt, .pdf and .docx files into plain text. (Reused from Project 1.)"""

from docx import Document
from pypdf import PdfReader


def load_txt(path):
    return path.read_text(encoding="utf-8")


def load_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)


def load_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_text(path):
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return load_txt(path)
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".docx":
        return load_docx(path)
    raise ValueError(f"Unsupported file type: {path.name}")
