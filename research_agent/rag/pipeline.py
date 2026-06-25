"""End-to-end RAG ingestion and retrieval pipeline."""

from __future__ import annotations

from pathlib import Path

from research_agent.config import PAPERS_DIR, RAG_TOP_K, VECTOR_STORE_PATH
from research_agent.observability.monitoring import trace_span
from research_agent.rag.chunker import semantic_chunk
from research_agent.rag.pdf_extractor import extract_all_pdfs, extract_text_from_pdf
from research_agent.rag.vector_store import FaissVectorStore

_store: FaissVectorStore | None = None


def get_vector_store() -> FaissVectorStore:
    global _store
    if _store is None:
        _store = FaissVectorStore(VECTOR_STORE_PATH)
    return _store


def ingest_papers(papers_dir: Path | None = None) -> dict:
    """Ingest all PDFs from papers directory into FAISS."""
    papers_dir = papers_dir or PAPERS_DIR
    store = get_vector_store()
    documents = extract_all_pdfs(papers_dir)

    total_chunks = 0
    for source, text in documents.items():
        chunks = semantic_chunk(text, source=source)
        total_chunks += store.add_chunks(chunks)

    return {
        "documents_processed": len(documents),
        "chunks_indexed": total_chunks,
        "total_in_store": store.document_count,
    }


def ingest_single_pdf(pdf_path: Path) -> dict:
    """Ingest a single PDF file."""
    store = get_vector_store()
    text = extract_text_from_pdf(pdf_path)
    chunks = semantic_chunk(text, source=pdf_path.name)
    added = store.add_chunks(chunks)
    return {"source": pdf_path.name, "chunks_added": added, "total_in_store": store.document_count}


def retrieve_context(query: str, top_k: int | None = None) -> list[dict]:
    """Retrieve relevant chunks for a query."""
    with trace_span("rag_retrieve", query=query[:80]):
        return get_vector_store().search(query, top_k=top_k or RAG_TOP_K)
