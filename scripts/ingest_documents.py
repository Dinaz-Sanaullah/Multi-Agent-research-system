#!/usr/bin/env python3
"""CLI script to ingest PDF papers into the FAISS vector store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from research_agent.config import PAPERS_DIR
from research_agent.rag.pipeline import ingest_papers, ingest_single_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDF papers into RAG vector store")
    parser.add_argument("--dir", type=Path, default=PAPERS_DIR, help="Directory with PDF files")
    parser.add_argument("--file", type=Path, help="Single PDF file to ingest")
    args = parser.parse_args()

    if args.file:
        result = ingest_single_pdf(args.file)
        print(f"Ingested {result['source']}: {result['chunks_added']} chunks")
    else:
        result = ingest_papers(args.dir)
        print(
            f"Processed {result['documents_processed']} documents, "
            f"indexed {result['chunks_indexed']} chunks "
            f"(total: {result['total_in_store']})"
        )


if __name__ == "__main__":
    main()
