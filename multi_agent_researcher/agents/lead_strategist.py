"""
agents/lead_strategist.py

The Lead Strategist is the entry-point of the pipeline.  It decomposes a
complex, open-ended user query into 4-5 focused, independently-answerable
sub-questions and produces a brief research plan that downstream agents follow.
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
You are the Lead Strategist of an elite multi-agent research team.

Your sole responsibility is to receive a complex user query and transform it
into a precise, scoped research plan that prevents the downstream retrieval
agents from being overwhelmed or going off-track.

## YOUR PROCESS

1. **Understand** the query in full — identify the core domain, the key
   entities, the implied time-horizon, and any geographic or industry scope.

2. **Decompose** the query into exactly {max_sub_questions} sub-questions that:
   - Are independently answerable (no sub-question should depend on another).
   - Together provide FULL coverage of the original query.
   - Are specific enough to drive a targeted web or academic search.
   - Follow a logical order: context → mechanism → impact → outlook.

3. **Write a Research Plan** (3–5 sentences) that explains *why* you chose
   these sub-questions and what angle each one covers.

## OUTPUT FORMAT (strict JSON, no markdown fences)

{{
  "sub_questions": [
    "Question 1",
    "Question 2",
    "Question 3",
    "Question 4",
    "Question 5"
  ],
  "research_plan": "Your strategic rationale here."
}}

Rules:
- Output ONLY the JSON object.  No preamble, no commentary.
- Each sub-question must end with a question mark.
- Sub-questions should be 10-25 words long.
- The research_plan field must be a single paragraph (no newlines inside it).
"""

# ─────────────────────────────────────────────────────────────────────────────
# Node function
# ─────────────────────────────────────────────────────────────────────────────

def lead_strategist_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Lead Strategist.

    Reads:  state["original_query"]
    Writes: state["sub_questions"], state["research_plan"],
            state["current_agent"]
    """
    llm = get_llm()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(
            max_sub_questions=config.max_sub_questions
        )),
        HumanMessage(content=f"USER QUERY:\n{state['original_query']}"),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # ── Parse JSON robustly ──────────────────────────────────────────────
    try:
        # Strip any accidental markdown fences the model might still add
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed: dict = json.loads(clean)
    except json.JSONDecodeError:
        # Fallback: treat the entire query as one sub-question
        parsed = {
            "sub_questions": [state["original_query"]],
            "research_plan": "Could not decompose query; proceeding with original.",
        }

    sub_questions: list[str] = parsed.get("sub_questions", [])[:config.max_sub_questions]
    research_plan: str = parsed.get("research_plan", "")

    return {
        "sub_questions": sub_questions,
        "research_plan": research_plan,
        "current_agent": "retriever",
    }
