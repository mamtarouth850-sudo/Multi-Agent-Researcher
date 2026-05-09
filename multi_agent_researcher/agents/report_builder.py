"""
agents/report_builder.py

The Report Builder formats all research outputs into a professional
Markdown report with:
  - An executive summary
  - Findings per sub-question with confidence scores
  - Insights, emerging trends, and hypotheses
  - A cited references section (deduplicated)
  - A limitations / caveats section
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import config
from ..tools.llm_factory import get_llm
from ..state import ResearchState

# ─────────────────────────────────────────────────────────────────────────────
# System instructions
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the Report Builder on an elite research team.

You receive structured research outputs and produce a polished, professional
Markdown report suitable for senior stakeholders.

## REPORT STRUCTURE (follow exactly)

```
# [Descriptive Title Derived from the User Query]

## Executive Summary
2-3 paragraph synthesis of the most important findings and their implications.
Do NOT repeat the sub-questions verbatim — synthesise.

## Research Findings

### [Sub-Question 1 — paraphrased as a short heading]
**Confidence Score: X.X / 1.0** ⬛⬛⬛⬛⬜ (fill ⬛ proportionally)

[Evidence-based answer, 2-4 paragraphs.  Cite inline as [1], [2], etc.]

> ⚠️ **Analyst Notes:** [biases flagged + contradictions, if any]

[Repeat for each sub-question]

## Insights & Strategic Implications

### Key Insights
- [Each insight as a bullet with brief elaboration]

### Emerging Trends
- [Each trend]

### Hypotheses
- [Each hypothesis in "If → Then → Because" format]

## Limitations & Caveats
[List data gaps, low-confidence areas, and suggested follow-up research]

## References
[1] [Title](url) — *source_type*
[2] ...
```

## FORMATTING RULES
- Use British academic tone — precise, neutral, evidence-forward.
- Confidence score visual: use 5 blocks, fill proportionally (0.0=0 filled,
  1.0=5 filled; round to nearest block).
- Inline citations MUST match the reference list numbering.
- Deduplicate references — same URL should appear only once.
- The references section must be sorted by first appearance in the text.
- Do NOT add content beyond what the findings and insights supply.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_bar(score: float) -> str:
    filled = round(score * 5)
    return "⬛" * filled + "⬜" * (5 - filled)


def _deduplicate_sources(findings: list[dict]) -> list[dict]:
    seen_urls: set[str] = set()
    ordered: list[dict] = []
    for finding in findings:
        for src in finding.get("sources", []):
            url = src.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                ordered.append(src)
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# Node function
# ─────────────────────────────────────────────────────────────────────────────

def report_builder_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Report Builder.

    Reads:  state["original_query"], state["findings"], state["insights"],
            state["hypotheses"], state["emerging_trends"],
            state["data_gaps"], state["research_plan"]
    Writes: state["final_report"], state["current_agent"]
    """
    original_query: str = state.get("original_query", "")
    findings: list[dict] = state.get("findings", [])
    insights: list[str] = state.get("insights", [])
    hypotheses: list[str] = state.get("hypotheses", [])
    emerging_trends: list[str] = state.get("emerging_trends", [])
    data_gaps: list[str] = state.get("data_gaps", [])

    all_sources = _deduplicate_sources(findings)

    # Pre-build a reference list for the LLM to use
    reference_list = "\n".join(
        f"[{i+1}] {s.get('title', 'Untitled')} ({s.get('url', '')}) — {s.get('source_type', 'web')}"
        for i, s in enumerate(all_sources)
    )

    # Annotate findings with confidence bars for the LLM
    annotated_findings = []
    for f in findings:
        conf = f.get("confidence", 0.0)
        annotated_findings.append({
            **f,
            "confidence_bar": _confidence_bar(conf),
        })

    payload = json.dumps({
        "original_query": original_query,
        "findings": annotated_findings,
        "insights": insights,
        "hypotheses": hypotheses,
        "emerging_trends": emerging_trends,
        "data_gaps": data_gaps,
        "reference_list": reference_list,
    }, indent=2)

    llm = get_llm()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            "Using the structured research data below, produce the complete "
            "Markdown report.  Follow the exact structure specified in your "
            "instructions.\n\n"
            f"RESEARCH DATA:\n{payload}"
        )),
    ]

    response = llm.invoke(messages)
    final_report: str = response.content.strip()

    return {
        "final_report": final_report,
        "current_agent": "qa_agent",
    }
