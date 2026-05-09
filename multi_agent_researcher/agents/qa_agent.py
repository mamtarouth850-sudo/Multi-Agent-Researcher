"""
agents/qa_agent.py

The QA Agent is the final gatekeeper before the report is delivered.

It audits the draft report against the original user query to verify:
  1. Full coverage  — every part of the query is addressed.
  2. Logical validity — no circular reasoning or unsupported claims.
  3. Confidence floor — no critical claim sits below the threshold.
  4. Contradiction resolution — flagged contradictions are acknowledged.

If the QA Agent is unsatisfied, it triggers a loop-back to the Retriever
with specific feedback.  It may do this at most `config.max_qa_loops` times.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import config
from ..tools.llm_factory import get_llm
from ..state import ResearchState, make_qa_feedback

# ─────────────────────────────────────────────────────────────────────────────
# System instructions
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the Quality Assurance (QA) Agent for an elite research team.
You are the last line of defence before the report reaches the end user.

## YOUR AUDIT CHECKLIST

For each item below, assess PASS or FAIL:

1. **Coverage** — Does the report address every meaningful aspect of the
   original query?  Identify any unanswered parts.

2. **Logical Validity** — Are all conclusions supported by cited evidence?
   Flag circular reasoning (conclusion restated as evidence) or unsupported
   leaps.

3. **Confidence Floor** — Does any critical claim carry a confidence score
   below {threshold}?  If so, it needs more evidence or must be clearly
   caveated.

4. **Contradiction Handling** — Are flagged contradictions acknowledged and
   handled (either resolved or noted as unresolved)?

5. **Citation Integrity** — Are all inline citations traceable to a reference
   in the References section?

## DECISION

- If ALL checks pass → set `qa_passed: true`, `issues: []`.
- If ANY check fails AND another loop is possible →
  set `qa_passed: false` with specific `issues` and `suggested_searches`
  that will help the Retriever fix the gap.
- If ANY check fails BUT no more loops are allowed →
  set `qa_passed: true` (accept with caveats) but populate `issues`
  so the report can be annotated.

## OUTPUT FORMAT (strict JSON, no markdown fences)

{{
  "qa_passed": true,
  "issues": [
    {{
      "issue": "description of the problem",
      "affected_question": "which sub-question or query part is affected",
      "severity": "high|medium|low"
    }}
  ],
  "suggested_searches": ["search query 1", "search query 2"],
  "qa_summary": "One paragraph QA verdict."
}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Node function
# ─────────────────────────────────────────────────────────────────────────────

def qa_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: QA Agent.

    Reads:  state["original_query"], state["final_report"],
            state["findings"], state["qa_loop_count"]
    Writes: state["qa_passed"], state["qa_feedback"],
            state["qa_loop_count"], state["data_gaps"],
            state["sub_questions"] (may narrow for re-search),
            state["current_agent"], state["completed"]
    """
    original_query: str = state.get("original_query", "")
    final_report: str = state.get("final_report", "")
    findings: list[dict] = state.get("findings", [])
    qa_loop_count: int = state.get("qa_loop_count", 0)
    sub_questions: list[str] = state.get("sub_questions", [])

    loops_remaining = config.max_qa_loops - qa_loop_count
    can_loop = loops_remaining > 0

    llm = get_llm()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(
            threshold=config.confidence_threshold,
        )),
        HumanMessage(content=(
            f"ORIGINAL USER QUERY:\n{original_query}\n\n"
            f"DRAFT REPORT (first 6000 chars):\n{final_report[:6000]}\n\n"
            f"QA LOOPS REMAINING: {loops_remaining}\n"
            f"CAN TRIGGER RE-SEARCH: {can_loop}\n\n"
            "Audit the report and return your structured QA assessment."
        )),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # ── Parse JSON ───────────────────────────────────────────────────────
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed: dict = json.loads(clean)
    except json.JSONDecodeError:
        parsed = {
            "qa_passed": True,
            "issues": [],
            "suggested_searches": [],
            "qa_summary": "QA JSON parse error — accepting report as-is.",
        }

    qa_passed: bool = parsed.get("qa_passed", True)
    raw_issues: list[dict] = parsed.get("issues", [])
    suggested_searches: list[str] = parsed.get("suggested_searches", [])

    qa_feedback = [
        make_qa_feedback(
            issue=iss.get("issue", ""),
            affected_question=iss.get("affected_question", ""),
            severity=iss.get("severity", "medium"),
        )
        for iss in raw_issues
    ]

    # Determine routing
    if not qa_passed and can_loop:
        # Send back to retriever with targeted sub-questions derived from issues
        new_sub_questions = suggested_searches or [
            iss.get("affected_question", "")
            for iss in raw_issues
            if iss.get("affected_question")
        ] or sub_questions

        return {
            "qa_passed": False,
            "qa_feedback": qa_feedback,
            "qa_loop_count": qa_loop_count + 1,
            "data_gaps": [iss.get("issue", "") for iss in raw_issues],
            "sub_questions": new_sub_questions[:config.max_sub_questions],
            "current_agent": "retriever",
            "completed": False,
        }

    # QA passed (or no loops left)
    return {
        "qa_passed": True,
        "qa_feedback": qa_feedback,
        "qa_loop_count": qa_loop_count + 1,
        "current_agent": "done",
        "completed": True,
    }
