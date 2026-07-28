"""
SARSP-LangEd: Semi-Automated Research Synthesis Protocol for Language Education
Main Streamlit Application Entry Point
"""
import streamlit as st

import spacy
import logging

logger = logging.getLogger(__name__)

REQUIRED_SPACY_MODELS = [
    "en_core_web_sm", "es_core_news_sm", "fr_core_news_sm",
    "de_core_news_sm", "it_core_news_sm", "xx_sent_ud_sm"
]

_missing_models = []
for _model in REQUIRED_SPACY_MODELS:
    try:
        spacy.load(_model)
    except OSError:
        _missing_models.append(_model)

if _missing_models:
    logger.warning(
        f"Missing spaCy models: {_missing_models}. "
        f"Run 'bash post_install.sh' or 'python -m spacy download <model>' to install. "
        f"Tokenization will fall back to NLTK/whitespace; POS heuristics will be disabled for affected languages."
    )

# =============================================================================
# PAGE CONFIGURATION & GLOBAL STYLES
# =============================================================================
st.set_page_config(
    page_title="SARSP-LangEd Pipeline",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SIDEBAR: NAVIGATION
# =============================================================================
with st.sidebar:
    st.title("📚 SARSP-LangEd")
    st.caption("Semi-Automated Research Synthesis Protocol")
    
    st.divider()
    
    st.subheader("🗂️ Pipeline Stages")
    page = st.radio(
        "Select Stage",
        [
            "🏠 Home",
            "1️⃣ Preprocessing",
            "2️⃣ Screening",
            "3️⃣ Extraction",
            "4️⃣ Validation",
            "5️⃣ Normalization",
            "6️⃣ Prompt Analysis",
            "7️⃣ Topic Discovery",
            "8️⃣ Results & Export"
        ],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # --- Replication Mode ---
    repl_toggle = st.toggle(
        "🔬 Replication Mode",
        value=st.session_state.get("replication_mode", False),
        help="Load original experiment data at each stage to verify published results."
    )
    if repl_toggle != st.session_state.get("replication_mode", False):
        st.session_state["replication_mode"] = repl_toggle
        # Clear all cached stage results on toggle
        keys_to_clear = [k for k in st.session_state.keys() 
                         if any(k.startswith(p) for p in 
                                ["preprocessing_", "screening_", "extraction_",
                                 "val_", "norm_", "pa_", "td_", "norm_figures"])]
        for key in keys_to_clear:
            del st.session_state[key]
        st.rerun()
    
    if st.session_state.get("replication_mode"):
        from core.utils import get_available_replication_files
        avail = get_available_replication_files()
        n_files = len([k for k in avail if k != "s7_topics"])
        n_topics = avail.get("s7_topics", {}).get("n_sections", 0)
        st.success(f"🔬 **Replication Mode Active**\n{n_files} artifact(s) + {n_topics} topic section(s) loaded.")
    
    st.divider()

# =============================================================================
# PAGE ROUTING
# =============================================================================

if page == "🏠 Home":
    st.title("📚 SARSP-LangEd Pipeline")
    st.markdown("""
    ### Semi-Automated Research Synthesis Protocol for Language Education
    
    This tool implements the complete semi-automated systematic review pipeline 
    described in the *Elements* book. Each stage is modular and can be run independently.
    
    #### Pipeline Overview
    | Stage | Description | Requires API? | Replicable? |
    |-------|-------------|---------------|-------------|
    | 1️⃣ Preprocessing | Deduplicate, filter journals, unify columns | ❌ No | ✅ Full (manual re-run) |
    | 2️⃣ Screening | Dual-rater LLM title/abstract screening | ✅ Yes | ✅ Full |
    | 3️⃣ Extraction | Schema-constrained RAG extraction from PDFs | ✅ Yes | ✅ Full (JSON) |
    | 4️⃣ Validation | Double-blind human-LLM agreement assessment | ❌ No | ✅ Full |
    | 5️⃣ Normalization | Rule-based categorical harmonization | ❌ No | ✅ Full |
    | 6️⃣ Prompt Analysis | Structural profiling of verbatim prompts | ❌ No | ✅ Full |
    | 7️⃣ Topic Discovery | Embedding clustering + LLM labeling | ✅ Yes | ✅ Verify |
    | 8️⃣ Results & Export | Interactive reports + reproducibility package | ❌ No | ✅ Auto |
    
    ---
    
    ### 🔬 Replicating the Published Study
    
    This app bundles **all original experiment artifacts** (except copyrighted PDFs).
    
    1. Toggle **🔬 Replication Mode** in the sidebar
    2. Navigate to any stage — original data loads automatically
    3. Compare outputs against published values
    4. Or upload your own files to run the pipeline independently
    
    > 📁 All replication files are in `replication_data/`  
    > 📖 See `replication_data/README_REPLICATION.md` for detailed instructions  
    > 📊 **Stage 8 (Results Dashboard)** automatically reads replication data when Replication Mode is active — no manual file uploads needed
    
    ---
    
    ### 💻 Local vs. Hosted Deployment
    
    > ⚠️ **Local LLM support** (Ollama, LM Studio) requires running this app on your own computer. 
    > The hosted version supports **cloud APIs only** (Mistral, OpenAI, Cerebras, Gemini).
    
    | Deployment | Local Models | Cloud APIs | Best For |
    |-----------|-------------|------------|----------|
    | **Your computer** | ✅ Yes | ✅ Yes | Zero-cost screening/extraction with Ollama/LM Studio |
    | **Hosted (Streamlit Cloud)** | ❌ No | ✅ Yes | Quick access, sharing with collaborators |
    
    #### Getting Started (Own Data)
    1. Navigate to any stage and upload your own files
    2. Configure API endpoints and keys within each LLM-dependent stage
    3. Download the full reproducibility package from Stage 8 when complete
    """)

elif page == "1️⃣ Preprocessing":
    from core.preprocessing import render_preprocessing_page
    render_preprocessing_page()

elif page == "2️⃣ Screening":
    from core.screening import render_screening_page
    render_screening_page()

elif page == "3️⃣ Extraction":
    from core.extraction import render_extraction_page
    render_extraction_page()

elif page == "4️⃣ Validation":
    from core.validation import render_validation_page
    render_validation_page()

elif page == "5️⃣ Normalization":
    from core.normalization import render_normalization_page
    render_normalization_page()

elif page == "6️⃣ Prompt Analysis":
    from core.prompt_analysis import render_prompt_analysis_page
    render_prompt_analysis_page()

elif page == "7️⃣ Topic Discovery":
    from core.topic_discovery import render_topic_discovery_page
    render_topic_discovery_page()

elif page == "8️⃣ Results & Export":
    from core.results_dashboard import render_results_page
    render_results_page()