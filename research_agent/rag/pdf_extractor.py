"""PDF text extraction for academic papers."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from research_agent.observability.monitoring import log_tool_call, trace_span


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract full text from a PDF file."""
    with trace_span("pdf_extraction", path=str(pdf_path)):
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        full_text = "\n\n".join(pages)
        log_tool_call("pdf_extractor", path=str(pdf_path), pages=len(pages), chars=len(full_text))
        return full_text


def extract_all_pdfs(papers_dir: Path) -> dict[str, str]:
    """Extract text from all PDFs in a directory."""
    documents: dict[str, str] = {}
    if not papers_dir.exists():
        return documents
    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        documents[pdf_path.name] = extract_text_from_pdf(pdf_path)
    return documents
