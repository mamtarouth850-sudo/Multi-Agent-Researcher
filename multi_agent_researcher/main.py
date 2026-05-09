"""
main.py – Entry point for the Multi-Agent Deep Researcher.

Usage
-----
From Python:
    from multi_agent_researcher.main import run_research
    report = run_research("How will the 2026 interest rate shift affect
                           fintech liquidity in SE Asia?")
    print(report)

From CLI:
    python -m multi_agent_researcher.main \\
        --query "How will the 2026 interest rate shift affect fintech liquidity in SE Asia?" \\
        --output-dir reports/

Environment variables required:
    OPENAI_API_KEY  – OpenAI API key (GPT-4o)
    TAVILY_API_KEY  – Tavily Search API key (optional; falls back to DDG)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .config import config
from .graph import researcher_graph
from .state import ResearchState, initial_state

# ─────────────────────────────────────────────────────────────────────────────
# Progress display helpers
# ─────────────────────────────────────────────────────────────────────────────

AGENT_LABELS: dict[str, str] = {
    "lead_strategist":   "🧭  Lead Strategist    – decomposing query",
    "retriever":         "🔍  Retriever          – gathering evidence",
    "critical_analyst":  "🔬  Critical Analyst   – auditing sources",
    "insight_generator": "💡  Insight Generator  – synthesising patterns",
    "report_builder":    "📝  Report Builder     – drafting report",
    "qa_agent":          "✅  QA Agent           – validating report",
    "done":              "🎉  Pipeline complete",
}


def _print_step(agent: str, detail: str = "") -> None:
    label = AGENT_LABELS.get(agent, f"⚙️  {agent}")
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    print(f"[{timestamp}] {label}", flush=True)
    if detail:
        print(f"             {detail}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core runner
# ─────────────────────────────────────────────────────────────────────────────

def run_research(
    query: str,
    stream: bool = True,
    save_report: bool = True,
    output_dir: str | None = None,
) -> str:
    """
    Run the full multi-agent research pipeline on `query`.

    Parameters
    ----------
    query       : The user's research question.
    stream      : If True, print live agent-progress updates.
    save_report : If True, save the Markdown report to disk.
    output_dir  : Directory for saved reports (default: config.output_dir).

    Returns
    -------
    str : The final Markdown report.
    """
    if not config.google_api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set.  "
            "Export it before running: export GOOGLE_API_KEY=AIza..."
        )

    state = initial_state(query)
    final_state: ResearchState | None = None
    start = time.time()

    if stream:
        print("\n" + "═" * 60)
        print("  Multi-Agent Deep Researcher")
        print("═" * 60)
        print(f"  Query: {query[:80]}{'...' if len(query) > 80 else ''}")
        print("═" * 60 + "\n")

    for event in researcher_graph.stream(state, {"recursion_limit": 30}):
        # event is a dict: {node_name: partial_state}
        for node_name, partial in event.items():
            if stream:
                detail = ""
                if node_name == "lead_strategist":
                    qs = partial.get("sub_questions", [])
                    if qs:
                        detail = f"→ {len(qs)} sub-questions generated"
                elif node_name == "retriever":
                    sources = partial.get("raw_sources", [])
                    rnd = partial.get("retrieval_round", "?")
                    detail = f"→ {len(sources)} sources (round {rnd})"
                elif node_name == "critical_analyst":
                    needs = partial.get("needs_more_data", False)
                    detail = "→ requesting more data" if needs else "→ sufficient evidence"
                elif node_name == "qa_agent":
                    passed = partial.get("qa_passed", False)
                    loops = partial.get("qa_loop_count", "?")
                    detail = f"→ {'PASSED' if passed else 'FAILED'} (loop {loops})"

                _print_step(node_name, detail)

            final_state = {**state, **partial}   # accumulate state
            state = final_state                  # type: ignore[assignment]

    elapsed = time.time() - start

    if final_state is None:
        return "❌ Pipeline produced no output."

    report: str = final_state.get("final_report", "")

    if stream:
        print(f"\n{'═' * 60}")
        print(f"  Completed in {elapsed:.1f}s")
        qa_loops = final_state.get("qa_loop_count", 0)
        retrieval_rounds = final_state.get("retrieval_round", 0)
        print(f"  Retrieval rounds: {retrieval_rounds}  |  QA loops: {qa_loops}")
        print("═" * 60 + "\n")

    # ── Save report ──────────────────────────────────────────────────────
    if save_report and report:
        out_dir = Path(output_dir or config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = "".join(c if c.isalnum() else "_" for c in query[:40]).strip("_")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = out_dir / f"report_{slug}_{timestamp}.md"
        report_path.write_text(report, encoding="utf-8")

        if config.save_intermediate_state:
            state_path = out_dir / f"state_{slug}_{timestamp}.json"
            serialisable = {
                k: v for k, v in final_state.items()
                if k != "final_report"            # report already saved as .md
            }
            state_path.write_text(
                json.dumps(serialisable, indent=2, default=str),
                encoding="utf-8",
            )

        if stream:
            print(f"  📄 Report saved to: {report_path}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        prog="multi-agent-researcher",
        description="Run the Multi-Agent Deep Researcher on a query.",
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="The research query to investigate.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=config.output_dir,
        help=f"Directory for saved reports (default: {config.output_dir})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the report to disk.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress live progress output.",
    )
    parser.add_argument(
        "--model",
        default=config.llm_model,
        help=f"OpenAI model to use (default: {config.llm_model})",
    )

    args = parser.parse_args()

    # Override config from CLI flags
    config.llm_model = args.model
    config.output_dir = args.output_dir

    report = run_research(
        query=args.query,
        stream=not args.quiet,
        save_report=not args.no_save,
        output_dir=args.output_dir,
    )

    print("\n" + "─" * 60)
    print(report)


if __name__ == "__main__":
    _cli()
