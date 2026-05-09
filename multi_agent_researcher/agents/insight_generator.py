"""
agents/insight_generator.py

The Insight Generator uses Chain-of-Thought (CoT) reasoning to:
  1. Connect findings across sub-questions.
  2. Surface non-obvious emerging trends.
  3. Formulate logical hypotheses grounded in the evidence.

It operates at a higher temperature than other agents to encourage
creative synthesis while staying within the bounds of the sourced data.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import config
from ..tools.llm_factory import get_llm
from ..state import ResearchState

# ─────────────────────────────────────────────────────────────────────────────
# System instructions
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the Insight Generator on an elite research team.

You specialise in connecting disparate findings to reveal patterns that no
single sub-question uncovers on its own.  Think like a strategic consultant
synthesising a client briefing after a week of research.

## YOUR METHOD: Chain-of-Thought Synthesis

**Step 1 – Cross-Reference**
Read all findings.  Identify where two or more findings share a common theme,
mechanism, or entity — even if they came from different sub-questions.

**Step 2 – Identify Emerging Trends**
What patterns are appearing across the evidence?  What direction is the
data pointing?  An "emerging trend" is something that is shifting or building
momentum, not just a current fact.

**Step 3 – Formulate Hypotheses**
Based only on the evidence (not prior knowledge), what logical hypotheses can
be stated?  A hypothesis is a falsifiable proposition — "If X, then Y, because Z."
Ground every hypothesis in at least one specific finding.

**Step 4 – Synthesise Key Insights**
Write 3-6 high-value insights.  An insight is a non-obvious implication or
a reframing that adds strategic value.  Avoid restating facts as insights.

## CRITICAL RULE
Every insight, trend, and hypothesis MUST be traceable to the findings.
Do not introduce external knowledge not present in the source material.

## OUTPUT FORMAT (strict JSON, no markdown fences)

{{
  "insights": [
    "Insight 1: ...",
    "Insight 2: ..."
  ],
  "emerging_trends": [
    "Trend 1: ...",
    "Trend 2: ..."
  ],
  "hypotheses": [
    "Hypothesis 1: If ... then ... because [finding reference] ...",
    "Hypothesis 2: ..."
  ],
  "cot_reasoning": "Your step-by-step reasoning process (2-4 paragraphs)."
}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Node function
# ─────────────────────────────────────────────────────────────────────────────

def insight_generator_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Insight Generator.

    Reads:  state["findings"], state["original_query"],
            state["research_plan"]
    Writes: state["insights"], state["hypotheses"],
            state["emerging_trends"], state["current_agent"]
    """
    findings: list[dict] = state.get("findings", [])
    original_query: str = state.get("original_query", "")
    research_plan: str = state.get("research_plan", "")

    findings_text = json.dumps([
        {
            "question": f.get("question", ""),
            "answer": f.get("answer", ""),
            "confidence": f.get("confidence", 0),
            "biases_flagged": f.get("biases_flagged", []),
            "contradictions": f.get("contradictions", []),
        }
        for f in findings
    ], indent=2)

    llm = get_llm(creative=True)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"ORIGINAL USER QUERY:\n{original_query}\n\n"
            f"RESEARCH PLAN:\n{research_plan}\n\n"
            f"VERIFIED FINDINGS:\n{findings_text}\n\n"
            "Apply Chain-of-Thought synthesis and return your insights, "
            "trends, and hypotheses."
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
            "insights": ["Insight generation failed – JSON parse error."],
            "emerging_trends": [],
            "hypotheses": [],
            "cot_reasoning": raw[:500],
        }

    return {
        "insights": parsed.get("insights", []),
        "emerging_trends": parsed.get("emerging_trends", []),
        "hypotheses": parsed.get("hypotheses", []),
        "current_agent": "report_builder",
    }
