"""Custom ADK tools for RAG, web search, and agent composition."""

from __future__ import annotations

import json
import time
from typing import Any

from google.adk.tools import ToolContext

from research_agent.config import TAVILY_API_KEY
from research_agent.observability.monitoring import log_tool_call, metrics, trace_span
from research_agent.rag.pipeline import retrieve_context


def classify_query_source(query: str) -> dict[str, str]:
    """
    Intelligently classify whether a query should use RAG, web search, or both.

    Academic/historical/citation queries -> RAG
    Latest news/recent developments -> Web
    Comprehensive research -> Hybrid
    """
    query_lower = query.lower()
    web_signals = [
        "latest", "recent", "2024", "2025", "2026", "today", "current",
        "news", "breaking", "new study", "just published", "this year",
    ]
    rag_signals = [
        "paper", "journal", "cite", "citation", "methodology", "abstract",
        "according to", "literature", "published in", "author", "theorem",
        "framework", "prior work", "systematic review",
    ]

    web_score = sum(1 for s in web_signals if s in query_lower)
    rag_score = sum(1 for s in rag_signals if s in query_lower)

    if web_score > 0 and rag_score > 0:
        source = "hybrid"
    elif web_score > rag_score:
        source = "web"
    elif rag_score > 0:
        source = "rag"
    else:
        # Default for research assistant: prefer RAG for academic KB
        source = "rag" if web_score == 0 else "hybrid"

    return {
        "source": source,
        "reasoning": f"web_signals={web_score}, rag_signals={rag_score}",
        "query": query,
    }


def search_knowledge_base(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    Search the academic knowledge base (RAG) for relevant paper excerpts.
    Use for questions about published research, methodologies, and citations.
    """
    start = time.perf_counter()
    with trace_span("tool_rag_search", query=query[:80]):
        results = retrieve_context(query, top_k=top_k)
        latency = (time.perf_counter() - start) * 1000
        metrics.record_query("rag", latency)
        log_tool_call("search_knowledge_base", query=query[:80], results=len(results))

        if not results:
            return {
                "status": "no_results",
                "message": "No relevant documents found. Upload PDFs to data/papers/ and run ingestion.",
                "results": [],
            }

        formatted = [
            {
                "source": r["source"],
                "relevance": round(r["score"], 3),
                "excerpt": r["text"][:1500],
            }
            for r in results
        ]
        return {"status": "success", "results": formatted, "count": len(formatted)}


def search_web(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Search the web for real-time information using Tavily API.
    Use for latest citations, recent publications, and current developments.
    """
    start = time.perf_counter()
    with trace_span("tool_web_search", query=query[:80]):
        if not TAVILY_API_KEY:
            return {
                "status": "error",
                "message": "TAVILY_API_KEY not configured. Set it in .env",
                "results": [],
            }

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=TAVILY_API_KEY)
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True,
            )
        except Exception as exc:  # noqa: BLE001
            log_tool_call("search_web_error", error=str(exc))
            return {"status": "error", "message": str(exc), "results": []}

        latency = (time.perf_counter() - start) * 1000
        metrics.record_query("web", latency)
        log_tool_call("search_web", query=query[:80], results=len(response.get("results", [])))

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:1000],
                "score": r.get("score", 0),
            }
            for r in response.get("results", [])
        ]
        return {
            "status": "success",
            "answer": response.get("answer", ""),
            "results": results,
            "count": len(results),
        }


def append_to_state(key: str, value: str, tool_context: ToolContext) -> dict[str, str]:
    """Append text to a session state key for cross-agent communication."""
    current = tool_context.state.get(key, "")
    if current:
        tool_context.state[key] = f"{current}\n\n{value}"
    else:
        tool_context.state[key] = value
    log_tool_call("append_to_state", key=key)
    return {"status": "success", "key": key}


def save_research_plan(plan: str, tool_context: ToolContext) -> dict[str, str]:
    """Save the orchestrator's research plan to session state."""
    tool_context.state["research_plan"] = plan
    tool_context.state["query_classification"] = classify_query_source(plan)
    log_tool_call("save_research_plan")
    return {"status": "success", "plan_saved": True}


def get_system_metrics() -> dict[str, Any]:
    """Return current observability metrics for monitoring dashboards."""
    return metrics.to_dict()


def format_citations(rag_results: str, web_results: str) -> dict[str, str]:
    """Custom tool demonstrating agent composition — merge citation sources."""
    citations = []
    try:
        rag_data = json.loads(rag_results) if rag_results.startswith("{") else {}
        if isinstance(rag_data, dict) and "results" in rag_data:
            for r in rag_data["results"]:
                citations.append(f"[Paper: {r.get('source', 'unknown')}]")
    except json.JSONDecodeError:
        pass

    try:
        web_data = json.loads(web_results) if web_results.startswith("{") else {}
        if isinstance(web_data, dict) and "results" in web_data:
            for r in web_data["results"]:
                citations.append(f"[Web: {r.get('title', 'unknown')} - {r.get('url', '')}]")
    except json.JSONDecodeError:
        pass

    return {"citations": "\n".join(citations) if citations else "No citations extracted"}
