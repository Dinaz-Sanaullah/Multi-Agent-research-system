#!/usr/bin/env python3
"""Quick RAG retrieval smoke test (no LLM/agent required)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from research_agent.rag.pipeline import retrieve_context

QUERIES = [
    "What is the self-attention mechanism in transformers?",
    "How does BERT pre-training work?",
    "What is few-shot learning in language models?",
]


def main() -> None:
    print("RAG Smoke Test")
    print("-" * 50)
    for q in QUERIES:
        results = retrieve_context(q, top_k=3)
        print(f"\nQ: {q}")
        if not results:
            print("  No results!")
            continue
        for r in results:
            print(f"  [{r['score']:.3f}] {r['source']} (chunk {r['chunk_index']})")
            excerpt = r["text"][:120].encode("ascii", errors="replace").decode("ascii")
            print(f"       {excerpt}...")
    print("\nRAG smoke test passed.")


if __name__ == "__main__":
    main()
