# Multi-Agent Researcher 🔬 (by Mamta)

A production-ready **LangGraph** implementation of a multi-agent research pipeline. It decomposes complex queries, retrieves evidence from the live web and uploaded documents, critically analyses sources, synthesises insights via Chain-of-Thought reasoning, and delivers a professional cited Markdown report.

The project features a **world-class Streamlit UI** and full support for **Google Gemini, OpenAI, and Anthropic Claude**.

---

## 🌟 Key Features

- **Multi-Provider Support:** Switch seamlessly between Google Gemini, OpenAI (ChatGPT), and Anthropic (Claude) using the sleek sidebar settings.
- **Document Upload:** Attach your own `PDF`, `DOCX`, `PPTX`, `TXT`, or `MD` files. The agents will extract the text and cite your documents alongside web search results.
- **Live Web Search:** Uses **Tavily** (or DuckDuckGo fallback) to pull real-time internet data.
- **Agent Collaboration:** Uses a LangGraph DAG with conditional routing and feedback loops (QA loop triggers targeted re-search when coverage is insufficient).
- **Beautiful UI:** A premium, dark-mode dashboard with glassmorphism effects, live execution tracking, and glowing progress widgets.

---

## Architecture

```
User Query + Uploaded Documents
    │
    ▼
┌─────────────────────────────────┐
│  🧭 Lead Strategist              │  Decomposes query → 4–5 sub-questions
│     (Router / Query Decomposer)  │  + strategic research plan
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  🔍 Contextual Retriever  ◄─────┼──────────────────────────┐
│  Tavily · DDG · ArXiv · Docs    │                          │
└──────────────┬──────────────────┘                          │
               │                                             │
               ▼                                             │ needs_more_data
┌─────────────────────────────────┐                          │
│  🔬 Critical Analyst            │──── insufficient ────────┘
│  Bias · Contradictions · Gaps   │
└──────────────┬──────────────────┘
               │ sufficient evidence
               ▼
┌─────────────────────────────────┐
│  💡 Insight Generator            │  Chain-of-Thought synthesis
│  Trends · Hypotheses · Insights  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  📝 Report Builder               │  Markdown · Citations · Confidence Bars
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  ✅ QA Agent                    │──── qa failed, loops left ──► Retriever
│  Coverage · Logic · Caveats     │
└──────────────┬──────────────────┘
               │ qa passed
               ▼
             END
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install streamlit python-dotenv langchain-google-genai langchain-openai langchain-anthropic pypdf python-docx python-pptx tavily-python
```

### 2. Run the Dashboard

The entire application is driven by a beautiful web interface.

```bash
streamlit run app.py --server.port 8080
```

### 3. Usage

1. Open **http://localhost:8080** in your browser.
2. Open the **Settings Panel** on the left.
3. Select your preferred **AI Provider** (e.g., Google Gemini) and Model.
4. Paste your API key (links are provided in the UI to get one for free).
5. (Optional) Enter a **Tavily API key** for better web search.
6. Attach any documents you want the agents to read.
7. Enter your research query and hit **🚀 Run Analysis**.

---

## Agent Roles

| Agent | Role |
|---|---|
| **Lead Strategist** | Breaks complex queries into 4–5 focused sub-questions. |
| **Contextual Retriever** | Fetches evidence via Tavily, DuckDuckGo, ArXiv, and parses uploaded documents. |
| **Critical Analyst** | Audits sources for bias, contradictions, and gaps; scores confidence. |
| **Insight Generator** | CoT synthesis — connects dots, surfaces trends, forms hypotheses. |
| **Report Builder** | Formats Markdown report with executive summary, citations, confidence bars. |
| **QA Agent** | Checks coverage, logic, and citation integrity; triggers loop-back. |

---

## License

MIT — use freely, attribution to Mamta appreciated.
