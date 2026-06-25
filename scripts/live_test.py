#!/usr/bin/env python3
"""Run a live end-to-end test query against the multi-agent research system."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

from research_agent.agent import root_agent
from research_agent.config import APP_NAME
from research_agent.observability.monitoring import metrics
from research_agent.tools.custom_tools import classify_query_source

USER_ID = "test_user"
SESSION_ID = "live_test_session"

TEST_QUERIES = [
    "What is the transformer architecture and self-attention mechanism described in the attention paper?",
]


async def run_query(runner: InMemoryRunner, query: str) -> str:
    session = await runner.session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    if session:
        session.state["user_query"] = query

    classification = classify_query_source(query)
    print(f"  Query routing: {classification['source']} ({classification['reasoning']})")

    content = Content(role="user", parts=[Part(text=query)])
    response_text = ""
    start = time.perf_counter()

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
    ):
        author = getattr(event, "author", "unknown")
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response_text = part.text
                    print(f"  [{author}] chunk received ({len(part.text)} chars)")

    elapsed = (time.perf_counter() - start) * 1000
    print(f"  Completed in {elapsed:.0f}ms")
    return response_text


async def main() -> None:
    print("=" * 60)
    print("Multi-Agent Research System — Live Test")
    print("=" * 60)

    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state={"user_query": ""},
    )

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n--- Test {i}/{len(TEST_QUERIES)} ---")
        print(f"Q: {query}")
        try:
            response = await run_query(runner, query)
            preview = response[:500] + "..." if len(response) > 500 else response
            print(f"\nA (preview):\n{preview}\n")
        except Exception as exc:
            print(f"ERROR: {exc}")
            import traceback
            traceback.print_exc()

    print("\n--- Metrics ---")
    print(metrics.to_dict())
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
