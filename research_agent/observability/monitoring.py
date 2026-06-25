"""Observability layer: structured logging, metrics, and optional Cloud Trace."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

import structlog

from research_agent.config import ENABLE_CLOUD_TRACE, LOG_LEVEL

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, LOG_LEVEL, logging.INFO)),
)

logger = structlog.get_logger("research_agent")

_tracer = None
if ENABLE_CLOUD_TRACE:
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("multi-agent-research-system")
    except Exception as exc:  # noqa: BLE001
        logger.warning("cloud_trace_init_failed", error=str(exc))


@dataclass
class AgentMetrics:
    """In-memory metrics for agent runs (exportable to Cloud Monitoring)."""

    total_queries: int = 0
    rag_queries: int = 0
    web_queries: int = 0
    hybrid_queries: int = 0
    review_iterations: int = 0
    avg_latency_ms: float = 0.0
    _latencies: list[float] = field(default_factory=list, repr=False)

    def record_query(self, source: str, latency_ms: float) -> None:
        self.total_queries += 1
        if source == "rag":
            self.rag_queries += 1
        elif source == "web":
            self.web_queries += 1
        elif source == "hybrid":
            self.hybrid_queries += 1
        self._latencies.append(latency_ms)
        self.avg_latency_ms = sum(self._latencies) / len(self._latencies)

    def record_review_iteration(self) -> None:
        self.review_iterations += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "rag_queries": self.rag_queries,
            "web_queries": self.web_queries,
            "hybrid_queries": self.hybrid_queries,
            "review_iterations": self.review_iterations,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


metrics = AgentMetrics()


@contextmanager
def trace_span(name: str, **attributes: Any) -> Generator[None, None, None]:
    """Context manager for timed operations with optional distributed tracing."""
    start = time.perf_counter()
    span_ctx = _tracer.start_as_current_span(name) if _tracer else None
    try:
        if span_ctx:
            with span_ctx as span:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
                yield
        else:
            yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("span_completed", span=name, latency_ms=round(elapsed_ms, 2), **attributes)


def log_agent_event(agent_name: str, event: str, **kwargs: Any) -> None:
    logger.info("agent_event", agent=agent_name, event=event, **kwargs)


def log_tool_call(tool_name: str, **kwargs: Any) -> None:
    logger.info("tool_call", tool=tool_name, **kwargs)
