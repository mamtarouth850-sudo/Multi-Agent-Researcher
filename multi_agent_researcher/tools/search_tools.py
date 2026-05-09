"""
tools/search_tools.py

Wraps Tavily, DuckDuckGo, ArXiv, PubMed, and a PDF parser into
LangChain-compatible tool objects that agents can call via the LLM.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

# ── LangChain tool primitives ────────────────────────────────────────────────
from langchain_core.tools import tool

# ── Optional heavy deps (imported lazily so tests can mock them) ─────────────
try:
    from tavily import TavilyClient
    _TAVILY_AVAILABLE = True
except ImportError:
    _TAVILY_AVAILABLE = False

try:
    from duckduckgo_search import DDGS
    _DDG_AVAILABLE = True
except ImportError:
    _DDG_AVAILABLE = False

try:
    import arxiv
    _ARXIV_AVAILABLE = True
except ImportError:
    _ARXIV_AVAILABLE = False

try:
    import pypdf
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False

from ..config import config
from ..state import make_source


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tavily Web Search
# ─────────────────────────────────────────────────────────────────────────────

@tool
def tavily_search(query: str, max_results: int = 8) -> list[dict]:
    """
    Search the web using the Tavily API for high-quality, curated results.

    Args:
        query:       The search query string.
        max_results: Maximum number of results to return (default 8).

    Returns:
        List of source dicts containing url, title, snippet, source_type.
    """
    if not _TAVILY_AVAILABLE:
        raise ImportError("Install with: pip install tavily-python")

    client = TavilyClient(api_key=config.tavily_api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
    )
    results = []
    for r in response.get("results", []):
        results.append(make_source(
            url=r.get("url", ""),
            title=r.get("title", ""),
            snippet=r.get("content", ""),
            source_type="web",
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. DuckDuckGo Fallback Search
# ─────────────────────────────────────────────────────────────────────────────

@tool
def duckduckgo_search(query: str, max_results: int = 8) -> list[dict]:
    """
    Search the web using DuckDuckGo (no API key required, good fallback).

    Args:
        query:       The search query string.
        max_results: Maximum number of results to return (default 8).

    Returns:
        List of source dicts.
    """
    if not _DDG_AVAILABLE:
        raise ImportError("Install with: pip install duckduckgo-search")

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(make_source(
                url=r.get("href", ""),
                title=r.get("title", ""),
                snippet=r.get("body", ""),
                source_type="web",
            ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. ArXiv Academic Search
# ─────────────────────────────────────────────────────────────────────────────

@tool
def arxiv_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search ArXiv for academic papers relevant to the query.

    Args:
        query:       Research topic or keywords.
        max_results: Maximum papers to retrieve (default 5).

    Returns:
        List of source dicts with paper abstracts as snippets.
    """
    if not _ARXIV_AVAILABLE:
        raise ImportError("Install with: pip install arxiv")

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = []
    for paper in client.results(search):
        results.append(make_source(
            url=paper.entry_id,
            title=paper.title,
            snippet=paper.summary[:600],
            source_type="arxiv",
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Python REPL (for numerical calculations)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def python_repl(code: str) -> str:
    """
    Execute Python code for numerical calculations or data processing.

    IMPORTANT: Only use for pure calculations. No file I/O, no imports
    of packages not already available (math, statistics, json are fine).

    Args:
        code: Python code to execute (must end with a print() statement).

    Returns:
        Captured stdout output as a string.
    """
    import io
    import contextlib
    import math
    import statistics

    allowed_globals = {
        "__builtins__": {
            k: __builtins__[k] if isinstance(__builtins__, dict)
            else getattr(__builtins__, k)
            for k in [
                "print", "len", "range", "sum", "min", "max",
                "abs", "round", "sorted", "enumerate", "zip",
                "list", "dict", "str", "int", "float", "bool",
            ]
        },
        "math": math,
        "statistics": statistics,
        "json": json,
    }

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, allowed_globals)  # noqa: S102
    except Exception as exc:
        return f"ERROR: {exc}"

    return buf.getvalue().strip() or "(no output)"


# ─────────────────────────────────────────────────────────────────────────────
# 5. PDF Parser
# ─────────────────────────────────────────────────────────────────────────────

@tool
def parse_pdf(file_path: str, max_pages: int = 20) -> list[dict]:
    """
    Extract text from a local PDF file and return it as source dicts
    (one dict per page).

    Args:
        file_path: Absolute or relative path to the PDF file.
        max_pages: Maximum pages to extract (default 20).

    Returns:
        List of source dicts with page content as snippets.
    """
    if not _PYPDF_AVAILABLE:
        raise ImportError("Install with: pip install pypdf")

    if not os.path.exists(file_path):
        return [make_source("", f"File not found: {file_path}", "", "pdf")]

    reader = pypdf.PdfReader(file_path)
    results = []
    for i, page in enumerate(reader.pages[:max_pages]):
        text = page.extract_text() or ""
        results.append(make_source(
            url=f"file://{os.path.abspath(file_path)}#page={i + 1}",
            title=f"{os.path.basename(file_path)} – Page {i + 1}",
            snippet=text[:800],
            source_type="pdf",
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────────────────────────────────────

RETRIEVAL_TOOLS = [tavily_search, duckduckgo_search, arxiv_search, parse_pdf]
CALCULATION_TOOLS = [python_repl]
ALL_TOOLS = RETRIEVAL_TOOLS + CALCULATION_TOOLS
