"""
app.py – Streamlit Web UI for Multi-Agent Deep Researcher
Run with:  streamlit run app.py --server.port 8080
"""
import time
import sys
import os

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nexus Deep Research",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Advanced CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #1a1b26 0%, #0d0f17 100%);
    }
    [data-testid="stSidebar"] {
        background: rgba(22, 25, 43, 0.6) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Typography Colors */
    h1, h2, h3 { color: #f8fafc !important; font-weight: 600 !important; tracking: tight; }
    p, li, span { color: #cbd5e1; }

    /* Custom Logo / Header */
    .app-logo-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0px;
        padding-bottom: 20px;
    }
    .app-logo-header svg {
        width: 42px;
        height: 42px;
        fill: url(#grad1);
    }
    .app-logo-header h1 {
        margin: 0;
        font-size: 2.2rem;
        background: -webkit-linear-gradient(45deg, #4ade80, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Agent Status Widgets */
    .agent-widget {
        background: rgba(30, 34, 53, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 20px;
        margin: 8px 0;
        color: #f1f5f9;
        font-size: 0.95rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    /* Glowing active animation */
    @keyframes pulse-border {
        0%   { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.4); border-color: #38bdf8; }
        70%  { box-shadow: 0 0 0 8px rgba(56, 189, 248, 0); border-color: #0284c7; }
        100% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); border-color: #38bdf8; }
    }
    .agent-widget.active { 
        background: rgba(14, 34, 53, 0.8);
        animation: pulse-border 2s infinite;
    }
    .agent-widget.done { 
        border-left: 4px solid #10b981; 
        background: rgba(16, 185, 129, 0.05); 
    }
    .agent-widget.error { 
        border-left: 4px solid #ef4444; 
        background: rgba(239, 68, 68, 0.05); 
    }

    /* Metrics Row */
    .metric-panel {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 16px 12px;
        text-align: center;
        color: #94a3b8;
        border: 1px solid rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .metric-panel h3 { 
        color: #e2e8f0 !important; 
        font-size: 1.8rem !important; 
        margin: 0 0 6px 0 !important; 
        font-weight: 700 !important;
    }

    /* Inputs & Buttons styling */
    .stTextArea textarea {
        background: rgba(0,0,0,0.2) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 16px !important;
        font-size: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
    }
    
    .stButton>button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 6px 24px !important;
        transition: transform 0.1s !important;
    }
    .stButton>button:active {
        transform: scale(0.98);
    }

    /* Hide standard st file uploader border in popover */
    [data-testid="stPopoverBody"] [data-testid="stFileUploadDropzone"] {
        border: 1px dashed rgba(255,255,255,0.2);
        background: transparent;
    }

    /* Sidebar Settings Box */
    .sidebar-setting-box {
        background: rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    
    .provider-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Provider → Model map ───────────────────────────────────────────────────
PROVIDERS = {
    "🌐 Google Gemini": {
        "key": "google",
        "placeholder": "AIzaSy...",
        "env_var": "GOOGLE_API_KEY",
        "get_key_url": "https://aistudio.google.com/apikey",
        "models": {
            "gemini-2.0-flash  ✦ recommended": "gemini-2.0-flash",
            "gemini-2.0-flash-lite  · fastest": "gemini-2.0-flash-lite",
            "gemini-2.5-flash  · smartest": "gemini-2.5-flash",
        },
        "color": "#4285F4",
    },
    "🤖 OpenAI (ChatGPT)": {
        "key": "openai",
        "placeholder": "sk-...",
        "env_var": "OPENAI_API_KEY",
        "get_key_url": "https://platform.openai.com/api-keys",
        "models": {
            "gpt-4o          · most capable": "gpt-4o",
            "gpt-4o-mini     · fast & affordable": "gpt-4o-mini",
        },
        "color": "#10a37f",
    },
    "🧠 Anthropic (Claude)": {
        "key": "anthropic",
        "placeholder": "sk-ant-...",
        "env_var": "ANTHROPIC_API_KEY",
        "get_key_url": "https://console.anthropic.com/settings/keys",
        "models": {
            "claude-sonnet-4-5  · recommended": "claude-sonnet-4-5-20251001",
            "claude-3-5-haiku   · fast & cheap": "claude-3-5-haiku-20241022",
        },
        "color": "#c77dff",
    },
}

AGENT_INFO = {
    "lead_strategist":   ("🧭", "Lead Strategist",   "Decomposing query into sub-questions"),
    "retriever":         ("🔍", "Retriever",          "Searching web, arXiv, and documents"),
    "critical_analyst":  ("🔬", "Critical Analyst",   "Auditing sources for bias & contradictions"),
    "insight_generator": ("💡", "Insight Generator",  "Synthesising insights via Chain-of-Thought"),
    "report_builder":    ("📝", "Report Builder",     "Drafting the final Markdown report"),
    "qa_agent":          ("✅", "QA Agent",           "Validating coverage, logic & citations"),
}

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔑 Enter your Key here")
    st.markdown("---")

    with st.container(border=True):
        st.markdown("<h4 style='color: #f8fafc; margin-bottom: 5px;'>🏢 AI Provider</h4>", unsafe_allow_html=True)
        provider_label = st.selectbox("Provider", list(PROVIDERS.keys()), label_visibility="collapsed")
        prov = PROVIDERS[provider_label]

        st.markdown("<h4 style='color: #f8fafc; margin-top: 15px; margin-bottom: 5px;'>🤖 Model Selection</h4>", unsafe_allow_html=True)
        model_label = st.selectbox("Model", list(prov["models"].keys()), label_visibility="collapsed")
        chosen_model = prov["models"][model_label]
        
        st.markdown("<h4 style='color: #f8fafc; margin-top: 15px; margin-bottom: 5px;'>🔑 Authentication</h4>", unsafe_allow_html=True)
        api_key = st.text_input(
            "API Key",
            value=os.getenv(prov["env_var"], ""),
            type="password",
            placeholder=prov["placeholder"],
            label_visibility="collapsed",
        )
        
        colA, colB = st.columns([1, 1])
        with colA:
            st.markdown(f"<a href='{prov['get_key_url']}' target='_blank' style='font-size:0.85rem;color:#60a5fa;text-decoration:none;'>🔗 Get API Key</a>", unsafe_allow_html=True)
        with colB:
            if api_key:
                st.markdown("<div style='text-align:right;font-size:0.85rem;color:#4ade80;'>✅ Key Set</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='text-align:right;font-size:0.85rem;color:#f87171;'>❌ Required</div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<h4 style='color: #f8fafc; margin-bottom: 5px;'>🔎 Web Search API</h4>", unsafe_allow_html=True)
        tavily_key = st.text_input(
            "Tavily API Key (optional)",
            value=os.getenv("TAVILY_API_KEY", ""),
            type="password",
            placeholder="tvly-...",
            label_visibility="collapsed",
        )
        colC, colD = st.columns([1, 1])
        with colC:
            st.markdown("<a href='https://tavily.com' target='_blank' style='font-size:0.85rem;color:#60a5fa;text-decoration:none;'>🔗 Get Tavily Key</a>", unsafe_allow_html=True)
        with colD:
            if tavily_key:
                st.markdown("<div style='text-align:right;font-size:0.85rem;color:#4ade80;'>✅ Key Set</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='text-align:right;font-size:0.85rem;color:#94a3b8;'>DuckDuckGo fallback</div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<h4 style='color: #f8fafc; margin-bottom: 5px;'>🎛️ Pipeline Config</h4>", unsafe_allow_html=True)
        max_sub_q = st.slider("Sub-questions", 2, 6, 4)
        max_qa    = st.slider("QA loops", 0, 3, 1)
        max_ret   = st.slider("Retrieval rounds", 1, 4, 2)

# ── Main area ──────────────────────────────────────────────────────────────

# Polished Logo Header
st.markdown("""
<div class="app-logo-header" style="flex-direction: column; align-items: flex-start; gap: 4px;">
    <div style="display: flex; align-items: center; gap: 12px;">
        <svg viewBox="0 0 24 24" style="width: 42px; height: 42px; fill: url(#grad1);">
            <defs>
                <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#4ade80;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#3b82f6;stop-opacity:1" />
                </linearGradient>
            </defs>
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="url(#grad1)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        </svg>
        <h1 style="margin: 0; font-size: 2.5rem; background: -webkit-linear-gradient(45deg, #4ade80, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Multi-Agent Researcher</h1>
    </div>
    <div style="font-size: 0.9rem; color: #94a3b8; font-weight: 500; margin-left: 54px; margin-top: -6px;">by Mamta</div>
</div>
<p style="font-size: 1.1rem; color: #94a3b8; margin-bottom: 30px; margin-top: 10px;">
    An elite, multi-agent AI framework for comprehensive, multi-source research investigations.
</p>
""", unsafe_allow_html=True)

# Query Area
query = st.text_area(
    "Research Query",
    value="",
    height=100,
    placeholder="Enter your research Query or upload documents (PDF, DOCX, PPTX, TXT, MD)...",
    label_visibility="collapsed"
)

# Underneath query: Upload popover and Run button
col_up, col_fill, col_run = st.columns([1, 2, 1])
uploaded_files = []

with col_up:
    # Popover for clean attachment UI
    with st.popover("📎 Attach Documents", use_container_width=True):
        st.markdown("**Upload files to feed directly into the research pipeline.**")
        uploaded_files = st.file_uploader(
            "Supported formats",
            type=["pdf", "docx", "doc", "pptx", "ppt", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) attached.")

with col_run:
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

st.markdown("---")

# ── Run pipeline ───────────────────────────────────────────────────────────
if run_btn:
    if not api_key:
        st.error("❌ Please provide an API key in the settings panel.")
        st.stop()
    if not query.strip() and not uploaded_files:
        st.warning("⚠️ Please enter a query or upload a document.")
        st.stop()

    # Inject keys
    os.environ[prov["env_var"]] = api_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    # Apply config
    sys.path.insert(0, os.path.dirname(__file__))
    from multi_agent_researcher.config import config
    from multi_agent_researcher.graph import researcher_graph
    from multi_agent_researcher.state import initial_state

    config.provider             = prov["key"]
    config.api_key              = api_key
    config.tavily_api_key       = tavily_key
    config.llm_model            = chosen_model
    config.max_sub_questions    = max_sub_q
    config.max_qa_loops         = max_qa
    config.max_retrieval_rounds = max_ret

    # Process uploads
    uploaded_sources: list[dict] = []
    if uploaded_files:
        from multi_agent_researcher.tools.doc_parser import extract_text_from_file
        with st.spinner("Extracting text from attachments..."):
            for uf in uploaded_files:
                file_bytes = uf.getvalue()
                parsed = extract_text_from_file(file_bytes, uf.name)
                uploaded_sources.extend(parsed)

    # Metrics row setup
    m1, m2, m3, m4 = st.columns(4)
    sources_met  = m1.empty()
    agent_met    = m2.empty()
    rounds_met   = m3.empty()
    elapsed_met  = m4.empty()

    def update_metrics(sources=0, agent="—", rounds=0, elapsed=0.0):
        sources_met.markdown(f'<div class="metric-panel"><h3>{sources}</h3><small>Sources Extracted</small></div>', unsafe_allow_html=True)
        agent_met.markdown(f'<div class="metric-panel"><h3 style="font-size:1.1rem !important; padding-top:8px;">{agent}</h3><small>Active Process</small></div>', unsafe_allow_html=True)
        rounds_met.markdown(f'<div class="metric-panel"><h3>{rounds}</h3><small>Retrieval Rounds</small></div>', unsafe_allow_html=True)
        elapsed_met.markdown(f'<div class="metric-panel"><h3>{elapsed:.0f}s</h3><small>Time Elapsed</small></div>', unsafe_allow_html=True)

    update_metrics(len(uploaded_sources), "Booting...")

    st.markdown("### <br>Live Execution Graph", unsafe_allow_html=True)
    agent_placeholder = st.empty()
    agent_log = []

    def render_agents():
        html = ""
        for icon, name, detail, status in agent_log:
            css = {"active": "active", "done": "done", "error": "error"}.get(status, "")
            badge = {"active": "⏳", "done": "✅", "error": "❌"}.get(status, "")
            html += (
                f'<div class="agent-widget {css}">'
                f'<div style="font-weight:600; font-size:1.05rem; margin-bottom:2px;">{icon} {name} <span style="float:right">{badge}</span></div>'
                f'<div style="color:#94a3b8;">{detail}</div>'
                f'</div>'
            )
        agent_placeholder.markdown(html, unsafe_allow_html=True)

    state = initial_state(query or f"Analyse the uploaded documents: {', '.join(f.name for f in uploaded_files)}")
    if uploaded_sources:
        state["raw_sources"] = uploaded_sources

    final_state = None
    start = time.time()
    total_sources = len(uploaded_sources)
    retrieval_round = 0

    try:
        for event in researcher_graph.stream(state, {"recursion_limit": 40}):
            for node_name, partial in event.items():
                icon, name, base_detail = AGENT_INFO.get(node_name, ("⚙️", node_name, "Processing task..."))
                detail = base_detail

                if node_name == "lead_strategist":
                    qs = partial.get("sub_questions", [])
                    if qs: detail = f"Generated {len(qs)} distinct research paths"
                elif node_name == "retriever":
                    sources = partial.get("raw_sources", [])
                    rnd = partial.get("retrieval_round", 0)
                    total_sources = len(sources) # it's the full accumulated list
                    retrieval_round = rnd
                    detail = f"Round {rnd} completed. Context expanded to {total_sources} fragments."
                elif node_name == "critical_analyst":
                    needs = partial.get("needs_more_data", False)
                    detail = "⚠️ Identifying data gaps — requesting more data" if needs else "✔️ Evidence meets confidence threshold"
                elif node_name == "qa_agent":
                    passed = partial.get("qa_passed", False)
                    loops  = partial.get("qa_loop_count", "?")
                    detail = f"{'✔️ Report validated' if passed else '⚠️ Flaws detected, initiating revision'} (loop {loops})"

                if agent_log:
                    p = agent_log[-1]
                    agent_log[-1] = (p[0], p[1], p[2], "done")

                agent_log.append((icon, name, detail, "active"))
                render_agents()

                elapsed = time.time() - start
                update_metrics(total_sources, name, retrieval_round, elapsed)

                final_state = {**state, **partial}
                state = final_state

        if agent_log:
            p = agent_log[-1]
            agent_log[-1] = (p[0], p[1], p[2], "done")
            render_agents()

        elapsed = time.time() - start
        update_metrics(total_sources, "Complete 🎉", retrieval_round, elapsed)

        if final_state and final_state.get("final_report"):
            st.markdown("---")
            st.success(f"✅ Mission accomplished in {elapsed:.1f}s.")
            st.markdown("## 📄 Comprehensive Report")
            st.markdown(final_state["final_report"])
            st.download_button(
                "⬇️ Export as Markdown",
                data=final_state["final_report"],
                file_name="nexus_report.md",
                mime="text/markdown",
                use_container_width=True
            )
        else:
            st.error("Pipeline terminated without producing a final report.")

    except Exception as e:
        err_str = str(e)
        if agent_log:
            p = agent_log[-1]
            agent_log[-1] = (p[0], p[1], f"Critical Error: {err_str[:150]}", "error")
            render_agents()

        if any(x in err_str for x in ["RESOURCE_EXHAUSTED", "429", "quota"]):
            st.error("⚠️ **API Quota Exceeded**")
            st.info("The selected AI provider rejected the request due to rate limits. Please select a different model in the Settings Panel.")
        else:
            st.error(f"❌ Execution Failure:\n\n```\n{err_str[:500]}\n```")
