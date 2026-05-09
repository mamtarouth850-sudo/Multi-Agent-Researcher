"""
agents/retriever.py

The Contextual Retriever Agent searches the web (Tavily / DuckDuckGo),
academic databases (ArXiv), and optionally local PDFs for each sub-question
produced by the Lead Strategist.

It is designed to be called multiple times (once per retrieval round) when
the Critical Analyst or QA Agent sends work back.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import config
from ..tools.llm_factory import get_llm
from ..state import ResearchState, make_source
from ..tools import RETRIEVAL_TOOLS

# ─────────────────────────────────────────────────────────────────────────────
# System instructions
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the Contextual Retriever Agent on an elite research team.

Your mission: gather HIGH-QUALITY, GRANULAR evidence for each sub-question
you receive.  Do NOT produce broad summaries — extract specific data points,
statistics, named entities, dates, and verbatim findings.

## TOOLS AVAILABLE
- tavily_search(query, max_results)   → real-time web results
- duckduckgo_search(query, max_results) → fallback web results
- arxiv_search(query, max_results)    → academic papers
- parse_pdf(file_path, max_pages)     → local documents (if path provided)

## STRATEGY
1. For each sub-question, formulate 2-3 search queries that approach the
   question from different angles (e.g., "fintech liquidity SE Asia 2026" AND
   "ASEAN interest rate fintech impact" AND "Southeast Asia credit markets 2026").
2. Prefer Tavily for recency; ArXiv for technical / scientific topics.
3. Aim for at least {max_results} distinct sources per sub-question.
4. If a data gap is flagged by the Critical Analyst, search specifically for
   the missing information first.

## OUTPUT RULE
Call tools to retrieve sources and return your results.  Do NOT synthesise or
interpret — that is the Critical Analyst's job.

{gap_instruction}
"""

GAP_INSTRUCTION_TEMPLATE = """
## PRIORITY DATA GAPS (from Critical Analyst — address these first)
{gaps}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Node function
# ─────────────────────────────────────────────────────────────────────────────

def retriever_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Contextual Retriever.

    Reads:  state["sub_questions"], state["data_gaps"],
            state["retrieval_round"], state["raw_sources"]
    Writes: state["raw_sources"] (appended), state["retrieval_round"],
            state["current_agent"]
    """
    sub_questions: list[str] = state.get("sub_questions", [])
    data_gaps: list[str] = state.get("data_gaps", [])
    round_num: int = state.get("retrieval_round", 0)

    # Guard: don't loop forever
    if round_num >= config.max_retrieval_rounds:
        return {
            "current_agent": "critical_analyst",
            "retrieval_round": round_num,
        }

    # Build system prompt
    gap_instruction = ""
    if data_gaps:
        gap_instruction = GAP_INSTRUCTION_TEMPLATE.format(
            gaps="\n".join(f"  - {g}" for g in data_gaps)
        )

    llm = get_llm().bind_tools(RETRIEVAL_TOOLS)

    all_new_sources: list[dict] = []

    for question in sub_questions:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(
                max_results=config.max_search_results,
                gap_instruction=gap_instruction,
            )),
            HumanMessage(content=(
                f"RESEARCH ROUND: {round_num + 1}\n\n"
                f"SUB-QUESTION:\n{question}\n\n"
                "Search for evidence and return raw source material."
            )),
        ]

        try:
            response = llm.invoke(messages)

            # Execute any tool calls the LLM requested
            for tool_call in (response.tool_calls or []):
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                tool_fn = next(
                    (t for t in RETRIEVAL_TOOLS if t.name == tool_name), None
                )
                if tool_fn is None:
                    continue

                results = tool_fn.invoke(tool_args)
                if isinstance(results, list):
                    all_new_sources.extend(results)

        except Exception as exc:
            all_new_sources.append(make_source(
                url="",
                title=f"Retrieval error for: {question[:60]}",
                snippet=str(exc),
                source_type="error",
            ))

    return {
        "raw_sources": all_new_sources,       # appended via reducer
        "retrieval_round": round_num + 1,
        "current_agent": "critical_analyst",
    }
