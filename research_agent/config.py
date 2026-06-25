"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "multi-agent-research-system")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "True").lower() == "true"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-005")
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", PROJECT_ROOT / "vector_store"))
PAPERS_DIR = Path(os.getenv("PAPERS_DIR", PROJECT_ROOT / "data" / "papers"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_CLOUD_TRACE = os.getenv("ENABLE_CLOUD_TRACE", "false").lower() == "true"
APP_NAME = "multi_agent_research_system"
# Set true to run RAG + web sequentially (fewer concurrent Gemini calls, avoids 429).
SEQUENTIAL_RESEARCH = os.getenv("SEQUENTIAL_RESEARCH", "true").lower() == "true"
MAX_REVIEW_ITERATIONS = int(os.getenv("MAX_REVIEW_ITERATIONS", "2"))
