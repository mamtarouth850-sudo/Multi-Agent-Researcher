"""
state.py – Shared State Schema for the Multi-Agent Deep Researcher.

Every agent reads from and writes to this TypedDict.  LangGraph uses the
`Annotated` operator to decide how values are merged when multiple nodes
write to the same key.
"""
from __future__ import annotations

import time
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
import operator


# ─────────────────────────────────────────────────────────────────────────────
# Helper reducers
# ─────────────────────────────────────────────────────────────────────────────

def _keep_last(existing: Any, update: Any) -> Any:
    """For scalar fields – just overwrite."""
    return update


def _append_list(existing: list, update: list) -> list:
    """Accumulate items across agent calls."""
    if existing is None:
        existing = []
    return existing + (update or [])


# ─────────────────────────────────────────────────────────────────────────────
# Individual data models (plain dicts so LangGraph can serialise them)
# ─────────────────────────────────────────────────────────────────────────────

def make_source(url: str, title: str, snippet: str,
                source_type: str = "web") -> dict:
    return {
        "url": url,
        "title": title,
        "snippet": snippet,
        "source_type": source_type,       # web | arxiv | pubmed | pdf
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def make_finding(question: str, answer: str,
                 sources: list[dict],
                 confidence: float = 0.0,
                 biases_flagged: list[str] | None = None,
                 contradictions: list[str] | None = None) -> dict:
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "biases_flagged": biases_flagged or [],
        "contradictions": contradictions or [],
    }


def make_qa_feedback(issue: str, affected_question: str,
                     severity: str = "medium") -> dict:
    """Returned by the QA agent when it wants a re-search."""
    return {
        "issue": issue,
        "affected_question": affected_question,
        "severity": severity,           # low | medium | high
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main State TypedDict
# ─────────────────────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────
    original_query: Annotated[str, _keep_last]

    # ── Lead Strategist output ───────────────────────────────────────────
    sub_questions: Annotated[list[str], _keep_last]
    research_plan: Annotated[str, _keep_last]         # brief strategic notes

    # ── Retriever output ─────────────────────────────────────────────────
    raw_sources: Annotated[list[dict], _append_list]   # make_source() dicts
    retrieval_round: Annotated[int, _keep_last]

    # ── Critical Analyst output ──────────────────────────────────────────
    findings: Annotated[list[dict], _keep_last]        # make_finding() dicts
    needs_more_data: Annotated[bool, _keep_last]
    data_gaps: Annotated[list[str], _keep_last]

    # ── Insight Generator output ─────────────────────────────────────────
    insights: Annotated[list[str], _keep_last]
    hypotheses: Annotated[list[str], _keep_last]
    emerging_trends: Annotated[list[str], _keep_last]

    # ── Report Builder output ────────────────────────────────────────────
    final_report: Annotated[str, _keep_last]           # Markdown string

    # ── QA Agent output ──────────────────────────────────────────────────
    qa_passed: Annotated[bool, _keep_last]
    qa_feedback: Annotated[list[dict], _append_list]   # make_qa_feedback() dicts
    qa_loop_count: Annotated[int, _keep_last]

    # ── Orchestration metadata ───────────────────────────────────────────
    current_agent: Annotated[str, _keep_last]
    error_log: Annotated[list[str], _append_list]
    completed: Annotated[bool, _keep_last]


def initial_state(query: str) -> ResearchState:
    """Factory – returns a blank state seeded with the user's query."""
    return ResearchState(
        original_query=query,
        sub_questions=[],
        research_plan="",
        raw_sources=[],
        retrieval_round=0,
        findings=[],
        needs_more_data=False,
        data_gaps=[],
        insights=[],
        hypotheses=[],
        emerging_trends=[],
        final_report="",
        qa_passed=False,
        qa_feedback=[],
        qa_loop_count=0,
        current_agent="lead_strategist",
        error_log=[],
        completed=False,
    )
