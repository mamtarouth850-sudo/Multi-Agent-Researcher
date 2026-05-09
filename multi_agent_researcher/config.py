"""
Configuration for Multi-Agent Deep Researcher
"""
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ResearcherConfig:
    # ── LLM Provider ───────────────────────────────────────────────────
    # provider: "google" | "openai" | "anthropic"
    provider: str = "google"

    # Single unified API key field (set by the UI based on provider)
    api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))

    # Model name (depends on provider)
    llm_model: str = "gemini-2.0-flash"
    llm_temperature: float = 0.1
    llm_temperature_creative: float = 0.4   # used by Insight Generator

    # ── Legacy individual keys (kept for .env compatibility) ───────────
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # ── Search / Retrieval ──────────────────────────────────────────────
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    max_search_results: int = 8
    max_retrieval_rounds: int = 3           # guard against infinite loops

    # ── Researcher behaviour ────────────────────────────────────────────
    max_sub_questions: int = 5
    max_qa_loops: int = 2                   # how many times QA can send work back
    confidence_threshold: float = 0.70      # below this → QA triggers re-search

    # ── Output ──────────────────────────────────────────────────────────
    output_dir: str = "reports"
    save_intermediate_state: bool = True

    # ── Optional local docs ─────────────────────────────────────────────
    pdf_directory: Optional[str] = None


# Singleton used across modules
config = ResearcherConfig()
