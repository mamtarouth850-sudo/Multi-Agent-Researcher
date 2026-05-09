"""
graph.py – LangGraph DAG for the Multi-Agent Deep Researcher.

                    ┌─────────────────────────────────────┐
                    │          User Query                  │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │        Lead Strategist               │  Decomposes query
                    │   (Router / Query Decomposer)        │  into sub-questions
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │      Contextual Retriever            │◄──────────────────┐
                    │  (Tavily / DDG / ArXiv / PDF)        │                   │
                    └──────────────┬──────────────────────┘                   │
                                   │                                           │
                    ┌──────────────▼──────────────────────┐                   │
                    │       Critical Analyst               │                   │
                    │  (Bias · Contradictions · Gaps)      │                   │
                    └──────────┬──────────────┬────────────┘                   │
                               │ enough data  │ insufficient data              │
                               │              └───────────────────────────────►│
                    ┌──────────▼──────────────────────────┐                   │
                    │      Insight Generator               │                   │
                    │   (Chain-of-Thought Synthesis)       │                   │
                    └──────────────┬──────────────────────┘                   │
                                   │                                           │
                    ┌──────────────▼──────────────────────┐                   │
                    │        Report Builder                │                   │
                    │  (Markdown · Citations · Scores)     │                   │
                    └──────────────┬──────────────────────┘                   │
                                   │                                           │
                    ┌──────────────▼──────────────────────┐                   │
                    │           QA Agent                   │ qa failed &       │
                    │    (Coverage · Logic · Caveats)      │ loops left ───────┘
                    └──────────────┬──────────────────────┘
                                   │ qa passed
                    ┌──────────────▼──────────────────────┐
                    │            END                       │
                    └─────────────────────────────────────┘
"""
from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from .agents import (
    critical_analyst_node,
    insight_generator_node,
    lead_strategist_node,
    qa_agent_node,
    report_builder_node,
    retriever_node,
)
from .state import ResearchState

# ─────────────────────────────────────────────────────────────────────────────
# Edge condition functions
# ─────────────────────────────────────────────────────────────────────────────

def route_after_analyst(
    state: ResearchState,
) -> Literal["retriever", "insight_generator"]:
    """Send back to Retriever if more data is needed, else proceed."""
    if state.get("needs_more_data", False):
        return "retriever"
    return "insight_generator"


def route_after_qa(
    state: ResearchState,
) -> Literal["retriever", END]:          # type: ignore[valid-type]
    """Loop back to Retriever for a targeted re-search, or finish."""
    if state.get("completed", False):
        return END
    # QA failed → current_agent was set to "retriever" by the QA node
    return "retriever"


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_researcher_graph() -> StateGraph:
    """
    Compile and return the LangGraph StateGraph.

    Nodes
    -----
    lead_strategist   → Contextual Retriever
    retriever         → Critical Analyst
    critical_analyst  → Insight Generator  (conditional: may loop to Retriever)
    insight_generator → Report Builder
    report_builder    → QA Agent
    qa_agent          → END                (conditional: may loop to Retriever)
    """
    graph = StateGraph(ResearchState)

    # ── Register nodes ───────────────────────────────────────────────────
    graph.add_node("lead_strategist",   lead_strategist_node)
    graph.add_node("retriever",         retriever_node)
    graph.add_node("critical_analyst",  critical_analyst_node)
    graph.add_node("insight_generator", insight_generator_node)
    graph.add_node("report_builder",    report_builder_node)
    graph.add_node("qa_agent",          qa_agent_node)

    # ── Entry point ──────────────────────────────────────────────────────
    graph.set_entry_point("lead_strategist")

    # ── Static edges ─────────────────────────────────────────────────────
    graph.add_edge("lead_strategist",   "retriever")
    graph.add_edge("retriever",         "critical_analyst")
    graph.add_edge("insight_generator", "report_builder")
    graph.add_edge("report_builder",    "qa_agent")

    # ── Conditional edges ────────────────────────────────────────────────
    graph.add_conditional_edges(
        "critical_analyst",
        route_after_analyst,
        {
            "retriever":         "retriever",
            "insight_generator": "insight_generator",
        },
    )

    graph.add_conditional_edges(
        "qa_agent",
        route_after_qa,
        {
            "retriever": "retriever",
            END:         END,
        },
    )

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level compiled graph (import and use directly)
# ─────────────────────────────────────────────────────────────────────────────
researcher_graph = build_researcher_graph()
