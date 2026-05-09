"""
multi_agent_researcher
======================
A LangGraph-powered Multi-Agent Deep Researcher featuring:

  • Lead Strategist     – query decomposition & research planning
  • Contextual Retriever – Tavily / DuckDuckGo / ArXiv / PDF ingestion
  • Critical Analyst    – bias detection, contradiction flagging, confidence scoring
  • Insight Generator   – Chain-of-Thought synthesis & hypothesis formation
  • Report Builder      – professional Markdown with citations & confidence bars
  • QA Agent            – coverage audit & conditional loop-back

Quick start
-----------
    from multi_agent_researcher import run_research
    report = run_research("How will 2026 rate shifts affect SE Asia fintech?")
    print(report)
"""
from .main import run_research
from .config import config, ResearcherConfig
from .state import initial_state, ResearchState
from .graph import researcher_graph, build_researcher_graph

__all__ = [
    "run_research",
    "config",
    "ResearcherConfig",
    "initial_state",
    "ResearchState",
    "researcher_graph",
    "build_researcher_graph",
]

__version__ = "1.0.0"
