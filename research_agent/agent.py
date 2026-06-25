"""
Multi-agent research system using Google ADK.

Communication Patterns Implemented:
1. Sequential Flow     - planner -> parallel research -> synthesis -> review
2. Parallel Execution  - RAG researcher + web researcher run concurrently
3. Hierarchical Delegation - root orchestrator delegates to research pipeline
4. Feedback Loop       - reviewer/refiner loop until quality approved
"""

from __future__ import annotations

from google.adk.agents import Agent, LoopAgent, ParallelAgent, SequentialAgent

from research_agent.agents.definitions import (
    MODEL,
    planner_agent,
    rag_researcher,
    refinement_agent,
    reviewer_agent,
    synthesis_agent,
    web_researcher,
)
from research_agent.config import MAX_REVIEW_ITERATIONS, SEQUENTIAL_RESEARCH
from research_agent.model_config import GENERATE_CONTENT_CONFIG
from research_agent.tools.custom_tools import get_system_metrics

# Pattern 2: Parallel or Sequential research (sequential reduces 429 quota bursts)
research_team = (
    SequentialAgent(
        name="sequential_research_team",
        description="Runs RAG then web research sequentially to reduce API quota pressure.",
        sub_agents=[rag_researcher, web_researcher],
    )
    if SEQUENTIAL_RESEARCH
    else ParallelAgent(
        name="parallel_research_team",
        description="Runs RAG and web research in parallel (fan-out/gather pattern).",
        sub_agents=[rag_researcher, web_researcher],
    )
)

# Pattern 4: Feedback Loop — reviewer approves or triggers refinement
quality_review_loop = LoopAgent(
    name="quality_review_loop",
    description="Iterative QA loop: synthesize -> review -> refine until approved.",
    sub_agents=[synthesis_agent, reviewer_agent, refinement_agent],
    max_iterations=MAX_REVIEW_ITERATIONS,
)

# Pattern 1: Sequential Flow — plan -> research -> synthesize/review
research_pipeline = SequentialAgent(
    name="research_pipeline",
    description="Sequential research workflow: plan, parallel research, quality review.",
    sub_agents=[
        planner_agent,
        research_team,
        quality_review_loop,
    ],
)

# Pattern 3: Hierarchical Delegation — orchestrator routes to pipeline
orchestrator_agent = Agent(
    name="orchestrator",
    model=MODEL,
    generate_content_config=GENERATE_CONTENT_CONFIG,
    description=(
        "Research assistant orchestrator. Coordinates multi-agent research pipeline "
        "for academic queries using papers (RAG) and web search (latest citations)."
    ),
    instruction="""
You are the Orchestrator for a Multi-Agent Research Intelligence System.

Your role:
1. Understand the user's research question
2. Store it in session state (the pipeline reads {user_query?})
3. Delegate to the research_pipeline sub-agent for execution
4. Present the final {research_synthesis?} to the user with clear formatting

When the user asks a research question:
- Acknowledge the query briefly
- Transfer to research_pipeline to execute the full workflow
- After pipeline completes, summarize key findings and cite sources

Available metrics: use get_system_metrics if asked about system performance.

Session context:
- User query: {user_query?}
- Research plan: {research_plan?}
- Final synthesis: {research_synthesis?}
- RAG findings: {rag_findings?}
- Web findings: {web_findings?}
""",
    sub_agents=[research_pipeline],
    tools=[get_system_metrics],
)

# ADK Cloud Run requires root_agent
root_agent = orchestrator_agent
