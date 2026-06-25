"""Streamlit UI for the Multi-Agent Research Intelligence System."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

from research_agent.agent import root_agent
from research_agent.config import APP_NAME, PAPERS_DIR, VECTOR_STORE_PATH
from research_agent.observability.monitoring import metrics
from research_agent.rag.pipeline import ingest_papers, ingest_single_pdf

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔬",
    layout="wide",
)

st.title("Multi-Agent Research Intelligence System")
st.caption("Google ADK · RAG · Tavily Web Search · Vertex AI")


USER_ID = "streamlit_user"


def init_session():
    if "runner" not in st.session_state:
        st.session_state.runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    if "session_id" not in st.session_state:
        st.session_state.session_id = "streamlit_session"
        st.session_state.session_ready = False
    if "messages" not in st.session_state:
        st.session_state.messages = []


async def ensure_session():
    if not st.session_state.get("session_ready"):
        await st.session_state.runner.session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=st.session_state.session_id,
            state={"user_query": ""},
        )
        st.session_state.session_ready = True


async def run_agent_query(query: str) -> str:
    await ensure_session()
    runner = st.session_state.runner
    session_id = st.session_state.session_id

    session = await runner.session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    if session:
        session.state["user_query"] = query

    content = Content(role="user", parts=[Part(text=query)])
    response_text = ""
    start = time.perf_counter()

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response_text = part.text

    elapsed = (time.perf_counter() - start) * 1000
    metrics.record_query("hybrid", elapsed)
    return response_text or "No response generated."


def run_query_sync(query: str) -> str:
    return asyncio.run(run_agent_query(query))


init_session()

# Sidebar
with st.sidebar:
    st.header("Knowledge Base")
    st.write(f"Papers dir: `{PAPERS_DIR}`")
    st.write(f"Vector store: `{VECTOR_STORE_PATH}`")

    uploaded = st.file_uploader("Upload PDF paper", type=["pdf"])
    if uploaded and st.button("Ingest PDF"):
        save_path = PAPERS_DIR / uploaded.name
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(uploaded.read())
        with st.spinner("Extracting, chunking, embedding..."):
            result = ingest_single_pdf(save_path)
        st.success(f"Indexed {result['chunks_added']} chunks from {result['source']}")

    if st.button("Re-ingest all papers"):
        with st.spinner("Processing all PDFs..."):
            result = ingest_papers()
        st.success(
            f"Processed {result['documents_processed']} docs, "
            f"{result['chunks_indexed']} chunks indexed"
        )

    st.divider()
    st.header("Observability")
    m = metrics.to_dict()
    st.metric("Total Queries", m["total_queries"])
    st.metric("Avg Latency (ms)", m["avg_latency_ms"])
    col1, col2 = st.columns(2)
    col1.metric("RAG", m["rag_queries"])
    col2.metric("Web", m["web_queries"])

    st.divider()
    st.header("Architecture")
    st.markdown("""
    **Agents:** Orchestrator → Planner → Researcher(s) → Reviewer

    **Patterns:**
    - Sequential pipeline
    - Parallel RAG + Web
    - Hierarchical delegation
    - Feedback loop (QA)
    """)

# Main chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a research question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Orchestrator → Research → Review..."):
            try:
                response = run_query_sync(prompt)
            except Exception as exc:
                response = f"Error: {exc}\n\nEnsure GOOGLE_CLOUD_PROJECT and credentials are configured."
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer tabs for debugging
with st.expander("Session Debug"):
    st.json(metrics.to_dict())
