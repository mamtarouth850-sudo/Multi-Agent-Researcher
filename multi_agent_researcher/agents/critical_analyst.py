"""
agents/critical_analyst.py

The Critical Analyst plays "Devil's Advocate."  It receives all raw sources,
maps them to sub-questions, flags biases, identifies contradictions between
sources, marks outdated information, and decides whether the Retriever needs
to do another round.

Output: a list of `finding` dicts (one per sub-question) + a
`needs_more_data` flag with an optional `data_gaps` list.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import config
from ..tools.llm_factory import get_llm
from ..state import ResearchState, make_finding

# ─────────────────────────────────────────────────────────────────────────────
# System instructions
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the Critical Analyst on an elite research team.  You are a sceptic
by nature — your job is to stress-test evidence before it reaches the report.

## YOUR RESPONSIBILITIES

1. **Map Sources → Sub-Questions**
   For each sub-question, identify which sources (by title/url) are relevant.
   Assign a relevance score (0.0–1.0) mentally, and discard sources below 0.4.

2. **Synthesise an Answer**
   Write a factual, evidence-based answer to each sub-question using only
   the supplied source material.  Do NOT hallucinate.

3. **Flag Biases**
   - Ideological, commercial, or geographic bias in sources.
   - Over-representation of a single perspective.

4. **Flag Contradictions**
   - Where two or more sources make conflicting claims, name both and state
     the discrepancy explicitly.

5. **Mark Outdated Information**
   - If a source is clearly outdated relative to the question's time-horizon,
     flag it as stale.

6. **Assign a Confidence Score** (0.0–1.0) per sub-question:
   - 0.9+: Multiple independent, recent, high-quality sources agree.
   - 0.7–0.89: Good evidence but minor gaps or one contradiction.
   - 0.5–0.69: Significant gaps, heavy reliance on a single source.
   - Below 0.5: Evidence is weak, contradictory, or missing.

7. **Decide if More Data is Needed**
   Set `needs_more_data: true` if ANY sub-question has confidence < {threshold}
   AND another retrieval round is possible.  List the specific `data_gaps`.

## OUTPUT FORMAT (strict JSON, no markdown fences)

{{
  "findings": [
    {{
      "question": "...",
      "answer": "...",
      "sources": [{{"url": "...", "title": "..."}}],
      "confidence": 0.0,
      "biases_flagged": ["..."],
      "contradictions": ["..."]
    }}
  ],
  "needs_more_data": false,
  "data_gaps": []
}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Node function
# ─────────────────────────────────────────────────────────────────────────────

def critical_analyst_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Critical Analyst.

    Reads:  state["sub_questions"], state["raw_sources"],
            state["retrieval_round"]
    Writes: state["findings"], state["needs_more_data"],
            state["data_gaps"], state["current_agent"]
    """
    sub_questions: list[str] = state.get("sub_questions", [])
    raw_sources: list[dict] = state.get("raw_sources", [])
    retrieval_round: int = state.get("retrieval_round", 0)

    # Truncate source snippets to keep context manageable
    source_summary = json.dumps([
        {
            "title": s.get("title", ""),
            "url": s.get("url", ""),
            "snippet": s.get("snippet", "")[:400],
            "source_type": s.get("source_type", "web"),
            "retrieved_at": s.get("retrieved_at", ""),
        }
        for s in raw_sources
    ], indent=2)

    llm = get_llm()

    more_rounds_possible = retrieval_round < config.max_retrieval_rounds

    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(
            threshold=config.confidence_threshold,
        )),
        HumanMessage(content=(
            f"SUB-QUESTIONS:\n{json.dumps(sub_questions, indent=2)}\n\n"
            f"RAW SOURCES ({len(raw_sources)} total):\n{source_summary}\n\n"
            f"Retrieval round completed: {retrieval_round}\n"
            f"Additional retrieval rounds possible: {more_rounds_possible}\n\n"
            "Analyse the evidence and return your structured findings."
        )),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # ── Parse JSON ───────────────────────────────────────────────────────
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed: dict = json.loads(clean)
    except json.JSONDecodeError:
        # Graceful degradation
        parsed = {
            "findings": [
                make_finding(
                    question=q,
                    answer="Analysis failed – JSON parse error.",
                    sources=[],
                    confidence=0.0,
                )
                for q in sub_questions
            ],
            "needs_more_data": False,
            "data_gaps": [],
        }

    findings = parsed.get("findings", [])
    needs_more_data = parsed.get("needs_more_data", False)
    data_gaps = parsed.get("data_gaps", [])

    # Respect hard limits
    if not more_rounds_possible:
        needs_more_data = False

    next_agent = "retriever" if needs_more_data else "insight_generator"

    return {
        "findings": findings,
        "needs_more_data": needs_more_data,
        "data_gaps": data_gaps,
        "current_agent": next_agent,
    }
