"""FAISS vector store with Vertex AI text-embedding-005."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from research_agent.config import (
    EMBEDDING_MODEL,
    GOOGLE_CLOUD_LOCATION,
    GOOGLE_CLOUD_PROJECT,
    VECTOR_STORE_PATH,
)
from research_agent.observability.monitoring import log_tool_call, trace_span
from research_agent.rag.chunker import DocumentChunk


class FaissVectorStore:
    """FAISS-backed vector store using Vertex AI embeddings."""

    def __init__(self, store_path: Path | None = None) -> None:
        self.store_path = store_path or VECTOR_STORE_PATH
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_path / "faiss.index"
        self.chunks_path = self.store_path / "chunks.pkl"
        self.meta_path = self.store_path / "meta.json"

        self.chunks: list[DocumentChunk] = []
        self.index: faiss.IndexFlatIP | None = None
        self._embedding_model: TextEmbeddingModel | None = None
        self._dimension: int | None = None

        vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)
        self._load()

    @property
    def embedding_model(self) -> TextEmbeddingModel:
        if self._embedding_model is None:
            self._embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
        return self._embedding_model

    def _embed(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        all_vectors: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = [TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT") for t in batch]
            embeddings = self.embedding_model.get_embeddings(inputs)
            vectors = np.array([e.values for e in embeddings], dtype=np.float32)
            faiss.normalize_L2(vectors)
            all_vectors.append(vectors)
        return np.vstack(all_vectors) if all_vectors else np.array([], dtype=np.float32)

    def _embed_query(self, query: str) -> np.ndarray:
        inputs = [TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")]
        embedding = self.embedding_model.get_embeddings(inputs)[0]
        vector = np.array([embedding.values], dtype=np.float32)
        faiss.normalize_L2(vector)
        return vector

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        with trace_span("vector_store_add", count=len(chunks)):
            texts = [c.text for c in chunks]
            vectors = self._embed(texts)
            if self.index is None:
                self._dimension = vectors.shape[1]
                self.index = faiss.IndexFlatIP(self._dimension)
            self.index.add(vectors)
            self.chunks.extend(chunks)
            self._save()
            log_tool_call("faiss_add", chunks_added=len(chunks), total=len(self.chunks))
            return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.index is None or not self.chunks:
            return []
        with trace_span("vector_store_search", query=query[:80], top_k=top_k):
            query_vector = self._embed_query(query)
            scores, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))
            results = []
            for score, idx in zip(scores[0], indices[0], strict=False):
                if idx < 0:
                    continue
                chunk = self.chunks[idx]
                results.append(
                    {
                        "text": chunk.text,
                        "source": chunk.source,
                        "score": float(score),
                        "chunk_index": chunk.chunk_index,
                    }
                )
            log_tool_call("faiss_search", results=len(results))
            return results

    def _save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"count": len(self.chunks), "dimension": self._dimension}, f)

    def _load(self) -> None:
        if self.chunks_path.exists():
            with open(self.chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self._dimension = self.index.d

    @property
    def document_count(self) -> int:
        return len(self.chunks)
