"""Specialized agent definitions for the research multi-agent system."""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.tools import exit_loop

from research_agent.config import GEMINI_MODEL
from research_agent.model_config import GENERATE_CONTENT_CONFIG
from research_agent.tools.custom_tools import (
    append_to_state,
    format_citations,
    get_system_metrics,
    save_research_plan,
    search_knowledge_base,
    search_web,
)

MODEL = GEMINI_MODEL
_AGENT_KWARGS = {"model": MODEL, "generate_content_config": GENERATE_CONTENT_CONFIG}

# Shared research tools — both agents get both tools to prevent cross-agent tool-call errors.
_RESEARCH_TOOLS = [search_knowledge_base, search_web]

# --- Researcher Agents (used in Parallel Execution) ---

rag_researcher = Agent(
    name="rag_researcher",
    **_AGENT_KWARGS,
    description="Searches academic papers and journals in the knowledge base via RAG.",
    instruction="""
You are an academic research specialist. Search the knowledge base for information
relevant to the user's research query.

RESEARCH PLAN:
{research_plan?}

QUERY:
{user_query?}

IMPORTANT: Call ONLY the `search_knowledge_base` tool. Do NOT call `search_web`.
Use search_knowledge_base to find relevant paper excerpts.
Summarize findings with proper academic tone. Include source document names.
Save your findings concisely — they will be merged with web research results.
""",
    output_key="rag_findings",
    tools=_RESEARCH_TOOLS,
)

web_researcher = Agent(
    name="web_researcher",
    **_AGENT_KWARGS,
    description="Searches the web for latest citations, news, and recent publications.",
    instruction="""
You are a web research specialist focused on real-time academic information.

RESEARCH PLAN:
{research_plan?}

QUERY:
{user_query?}

IMPORTANT: Call ONLY the `search_web` tool. Do NOT call `search_knowledge_base`.
Use search_web to find the latest citations, recent papers, and current developments.
Prioritize authoritative sources (.edu, arxiv, pubmed, major journals).
Save concise findings with URLs for citation.
""",
    output_key="web_findings",
    tools=_RESEARCH_TOOLS,
)

# --- Reviewer / QA Agent (Feedback Loop) ---

reviewer_agent = Agent(
    name="reviewer",
    **_AGENT_KWARGS,
    description="Reviews research quality, checks citations, and approves or requests revision.",
    instruction="""
You are a rigorous academic QA reviewer. Evaluate the research synthesis below.

ORIGINAL QUERY: {user_query?}

RAG FINDINGS:
{rag_findings?}

WEB FINDINGS:
{web_findings?}

RESEARCH SYNTHESIS:
{research_synthesis?}

QUALITY CRITERIA:
1. Directly answers the user's query
2. Cites sources from both knowledge base and web where applicable
3. Distinguishes established knowledge (papers) from recent developments (web)
4. No hallucinated citations
5. Clear, academic writing quality

If the synthesis meets all criteria (score >= 7/10), call exit_loop to approve.
If improvements needed, use append_to_state to add feedback to 'review_feedback',
then explain what must be improved. Be specific and actionable.

Previous feedback (if any):
{review_feedback?}
""",
    tools=[append_to_state, exit_loop, get_system_metrics],
)

# --- Synthesis Agent (Sequential step after parallel research) ---

synthesis_agent = Agent(
    name="synthesizer",
    **_AGENT_KWARGS,
    description="Synthesizes RAG and web findings into a coherent research response.",
    instruction="""
Synthesize the research findings into a comprehensive, well-cited response.

USER QUERY: {user_query?}
RESEARCH PLAN: {research_plan?}

RAG FINDINGS (academic papers):
{rag_findings?}

WEB FINDINGS (latest information):
{web_findings?}

REVIEW FEEDBACK (if revision requested):
{review_feedback?}

Instructions:
- Integrate both knowledge base and web sources intelligently
- Clearly label which claims come from papers vs. recent web sources
- Include a References section with paper names and URLs
- If review feedback exists, address all points raised
- Write in clear academic prose suitable for a research assistant
""",
    output_key="research_synthesis",
    tools=[format_citations],
)

# --- Refinement Agent (inside feedback loop) ---

refinement_agent = Agent(
    name="refiner",
    **_AGENT_KWARGS,
    description="Refines the synthesis based on reviewer feedback.",
    instruction="""
Revise the research synthesis based on reviewer feedback.

ORIGINAL SYNTHESIS:
{research_synthesis?}

REVIEWER FEEDBACK:
{review_feedback?}

RAG FINDINGS:
{rag_findings?}

WEB FINDINGS:
{web_findings?}

Address every point in the feedback. Produce an improved synthesis.
Clear the review_feedback concern by writing a better response.
""",
    output_key="research_synthesis",
    tools=[format_citations],
)

# --- Orchestrator Planning Agent (Hierarchical Delegation entry) ---

planner_agent = Agent(
    name="planner",
    **_AGENT_KWARGS,
    description="Plans research strategy and classifies query type (RAG vs web vs hybrid).",
    instruction="""
You are the research orchestrator's planning module.

Analyze the user's query and create a research plan:
1. Determine if the query needs knowledge base search (papers/journals),
   web search (latest citations/news), or both (hybrid)
2. Break down sub-questions to investigate
3. Save the plan using save_research_plan tool

Guidelines:
- Academic methodology, established theories, paper content -> RAG
- Latest publications, current year, breaking research -> Web
- Comprehensive literature review -> Hybrid (both)

User query is in session state as user_query.
""",
    output_key="research_plan",
    tools=[save_research_plan],
)
