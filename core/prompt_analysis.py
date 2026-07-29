"""
SARSP-LangEd - Stage 6: Prompt Analysis
Structural analysis, language detection, translation, and heuristic-based
prompt feature extraction with user-editable regex patterns.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import re
import os
from collections import Counter
from pathlib import Path
import time

try:
    import pycountry
    HAS_PYCOUNTRY = True
except ImportError:
    HAS_PYCOUNTRY = False


def code_to_name(code: str) -> str:
    """Map ISO-639-1/3 code → language name. Keeps original code when not found."""
    if not HAS_PYCOUNTRY:
        return code
    base = code.split("-")[0].lower()
    if base == "und":
        return "Undetermined"
    try:
        lang = pycountry.languages.get(alpha_2=base)
        if not lang:
            lang = pycountry.languages.get(alpha_3=base)
        return lang.name if lang else code
    except Exception:
        return code

# Optional heavy imports — guarded so the page loads even if not installed
try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

try:
    import nltk
    # Only download if not already present (avoids redundant checks on rerun)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from datetime import datetime

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Consistent temp directory structure
from core.utils import get_session_temp_dir
TEMP_DIR = get_session_temp_dir()
STAGE_DIR = os.path.join(TEMP_DIR, "stage6_prompt_analysis")
AUTO_SAVE_DIR = os.path.join(STAGE_DIR, "auto_save")
FIGURES_DIR = os.path.join(STAGE_DIR, "figures")

# Ensure directories exist at module load
for d in [STAGE_DIR, AUTO_SAVE_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)


# =============================================================================
# DEFAULT PATTERNS & HEURISTICS (user-editable)
# =============================================================================

DEFAULT_ROLE_PATTERNS = [
    r"you are",
    r"\bact\b",
    r"\bplay (the )?role of\b",
    r"\bas an?\b",
]

DEFAULT_CONSTRAINT_PATTERNS = [
    r"\b(in|within) \d+ (words|sentences|minutes|characters)\b",
    r"\b(max|min)\b",
    r"\blimit(ed)? to\b",
    r"no more than",
    r"\bbetween \d+ and \d+\b",
    r"\bup to \d+\b",
    r"\b\d+ words\b",
    r"\b\d+ sentences\b",
]

DEFAULT_EXAMPLE_MARKERS = [
    r"for example",
    r"e\.g\.",
    r"for instance",
    r"such as",
    r"like:",
    r"example(s)?:",
]

DEFAULT_COT_MARKERS = [
    r"chain of thought",
    r"think step",
    r"step-by-step",
    r"explain your reasoning",
    r"reasoning",
    r"step by step",
    r"follow these steps",
    r"think carefully",
    r"let's think",
    r"follow this",
]

LANG_SPACY_MAP = {
    'en': 'en_core_web_sm',
    'es': 'es_core_news_sm',
    'fr': 'fr_core_news_sm',
    'de': 'de_core_news_sm',
    'it': 'it_core_news_sm',
    'zh-cn': 'xx_sent_ud_sm',
    'zh': 'xx_sent_ud_sm',
    'ja': 'xx_sent_ud_sm',
    'ko': 'xx_sent_ud_sm',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

_spacy_cache = {}

def _download_spacy_model(model_name):
    """Attempt to download a spaCy model on-demand. Returns True on success."""
    try:
        import spacy.cli
        spacy.cli.download(model_name)
        return True
    except Exception:
        return False


def get_spacy_for_lang(lang_code):
    """Load spaCy model for language code with caching, on-demand download, and fallback."""
    if not HAS_SPACY:
        return None
    model_name = LANG_SPACY_MAP.get(lang_code, 'xx_sent_ud_sm')
    if model_name in _spacy_cache:
        return _spacy_cache[model_name]
    # Attempt 1: load directly
    try:
        nlp = spacy.load(model_name)
        _spacy_cache[model_name] = nlp
        return nlp
    except OSError:
        pass
    # Attempt 2: download on-demand then load
    if _download_spacy_model(model_name):
        try:
            nlp = spacy.load(model_name)
            _spacy_cache[model_name] = nlp
            return nlp
        except OSError:
            pass
    # Attempt 3: fall back to multilingual model
    if 'xx_sent_ud_sm' in _spacy_cache:
        return _spacy_cache['xx_sent_ud_sm']
    try:
        nlp = spacy.load('xx_sent_ud_sm')
        _spacy_cache['xx_sent_ud_sm'] = nlp
        return nlp
    except OSError:
        pass
    # Attempt 4: download multilingual fallback
    if _download_spacy_model('xx_sent_ud_sm'):
        try:
            nlp = spacy.load('xx_sent_ud_sm')
            _spacy_cache['xx_sent_ud_sm'] = nlp
            return nlp
        except OSError:
            pass
    return None


def detect_language_safe(text):
    """Detect language with langdetect; returns 'und' on failure."""
    if not HAS_LANGDETECT:
        return 'und'
    try:
        return detect(text)
    except LangDetectException:
        return 'und'


def translate_to_english(text, lang):
    """Translate non-English text to English using deep-translator."""
    if lang == 'en':
        return text
    if not HAS_TRANSLATOR:
        return text
    try:
        return GoogleTranslator(source=lang, target='en').translate(text)
    except Exception:
        try:
            return GoogleTranslator(source='auto', target='en').translate(text)
        except Exception:
            return text


def analyze_prompt_basic(text, lang):
    """Return (token_count, sentence_count) using spaCy or NLTK fallback."""
    nlp = get_spacy_for_lang(lang)
    if nlp is not None:
        doc = nlp(text)
        tokens = [t.text for t in doc if not t.is_space]
        sents = list(doc.sents)
        return len(tokens), len(sents)
    if HAS_NLTK:
        tokens = nltk.word_tokenize(text)
        sents = nltk.sent_tokenize(text)
        return len(tokens), len(sents)
    # Last resort: whitespace split
    return len(text.split()), text.count('.') + text.count('!') + text.count('?')


def detect_any(patterns, text):
    """Check if any regex pattern matches the lowercased text."""
    t = text.lower()
    for p in patterns:
        try:
            if re.search(p, t):
                return True
        except re.error:
            continue
    return False


# In both detect_imperative_heuristic and detect_interrogative_heuristic:
def detect_imperative_heuristic(text, lang='en'):
    """Detect imperative mood. Always uses English model since input is prompt_en."""
    nlp = get_spacy_for_lang('en')  # ← Force English regardless of detected lang
    if nlp is None:
        return False
    doc = nlp(text)
    sents = list(doc.sents)
    if not sents:
        return False
    sent = sents[0]
    tokens = [t for t in sent if not t.is_space]
    if not tokens:
        return False
    # ---- POSITIVE CUE: sentence starts with a base-form verb ----
    if tokens[0].tag_ == "VB" and tokens[0].pos_ == "VERB":
        return True
    # ---- RULE OUT interrogatives ----
    # Starts with WH-word → NOT imperative
    if tokens[0].tag_ in {"WDT", "WP", "WP$", "WRB"}:
        return False
    # Starts with auxiliary/modal → NOT imperative
    if tokens[0].pos_ == "AUX" or tokens[0].tag_ in {"MD"}:
        return False
    # Ends with "?" → very likely interrogative
    if sent[-1].text == "?":
        return False
    # ---- ROOT analysis ----
    root = next((t for t in sent if t.dep_ == "ROOT"), None)
    if root is None:
        return False
    # Root must be a verb in base form
    if root.tag_ != "VB":
        return False
    # No explicit subject
    has_subject = any(t.dep_ in {"nsubj", "csubj", "expl"} for t in sent)
    if has_subject:
        return False
    return True


def detect_interrogative_heuristic(text, lang='en'):
    """Detect interrogative mood. Matches Jupyter notebook implementation exactly."""
    nlp = get_spacy_for_lang('en')  # ← Force English regardless of detected lang
    if nlp is None:
        return False
    doc = nlp(text)
    sents = list(doc.sents)
    if not sents:
        return False
    first = sents[0]
    tokens = [t for t in first if not t.is_space]
    if not tokens:
        return False
    if len(first) > 0 and tokens[-1].text == "?":
        return True
    if tokens[0].tag_ in {"WDT", "WP", "WP$", "WRB"}:
        # Follows with auxiliary/modal → interrogative
        if len(tokens) > 1 and (tokens[1].pos_ == "AUX" or tokens[1].tag_ in {"MD"}):
            return True
    return False


def extract_prompts_from_df(df):
    """Expand prompts_used column into a long DataFrame of individual prompts."""
    id_col = 'source_filename' if 'source_filename' in df.columns else (
        'study_id' if 'study_id' in df.columns else None
    )
    if id_col is None:
        df = df.reset_index().rename(columns={'index': 'row_index'})
        id_col = 'row_index'

    rows = []
    for _, row in df.iterrows():
        sid = row[id_col]
        prompts_list = row.get('prompts_used', [])
        if prompts_list is None:
            continue
        if isinstance(prompts_list, str):
            try:
                prompts_list = json.loads(prompts_list)
            except Exception:
                prompts_list = [prompts_list]
        if not isinstance(prompts_list, list):
            prompts_list = [prompts_list]
        for i, p in enumerate(prompts_list):
            if p is None:
                continue
            p_text = str(p).strip()
            if p_text.upper() in {"N/A", "NA", ""}:
                continue
            entry = {id_col: sid, "prompt_index": i, "prompt": p_text}
            for c in ['title', 'authors', 'year']:
                if c in row.index:
                    entry[c] = row[c]
            rows.append(entry)

    return pd.DataFrame(rows), id_col


def create_bar_chart(series, title, top_n=20, figsize=(14, 8)):
    """Horizontal bar chart with count + percentage labels."""
    counts = series.value_counts().head(top_n)
    if counts.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14)
        ax.set_title(title, fontsize=14, fontweight="bold")
        return fig
    total = counts.sum()
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        counts.index[::-1], counts.values[::-1],
        color=plt.cm.viridis(np.linspace(0, 1, len(counts))), alpha=0.85
    )
    for bar in bars:
        width = bar.get_width()
        pct = (width / total) * 100 if total > 0 else 0
        ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{int(width)} ({pct:.1f}%)', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel("Count", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    return fig


def create_distribution_plots(prompts_df, metric_cols, titles, xlims):
    """Create side-by-side histogram distributions for multiple metrics."""
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(7 * len(metric_cols), 6))
    if len(metric_cols) == 1:
        axes = [axes]
    colors = ['C0', 'C1', 'C2', 'C3']
    for i, (metric, title) in enumerate(zip(metric_cols, titles)):
        sns.histplot(prompts_df[metric], kde=True, ax=axes[i],
                     bins=40, stat='count', color=colors[i % len(colors)])
        axes[i].set_title(title, fontsize=13)
        axes[i].set_xlim(xlims[metric])
        axes[i].xaxis.set_major_locator(plt.MaxNLocator(5))
        axes[i].grid(axis='y', alpha=0.25)
    plt.tight_layout()
    return fig


def create_comparative_histograms(structured, plain, metrics, titles):
    """Create comparative histograms: structured vs non-structured prompts."""
    fig, axes = plt.subplots(len(metrics), 2, figsize=(18, 6 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        global_xmax = max(
            np.percentile(structured[metric], 99) if len(structured) > 0 else 1,
            np.percentile(plain[metric], 99) if len(plain) > 0 else 1
        )
        struct_counts, _ = np.histogram(structured[metric], bins=40, range=(0, global_xmax)) if len(structured) > 0 else ([0], [0])
        plain_counts, _ = np.histogram(plain[metric], bins=40, range=(0, global_xmax)) if len(plain) > 0 else ([0], [0])
        global_ymax = max(
            struct_counts.max() if hasattr(struct_counts, 'max') else 0,
            plain_counts.max() if hasattr(plain_counts, 'max') else 0
        )
        sns.histplot(structured[metric], kde=True, bins=40, color='C0', ax=axes[i][0])
        axes[i][0].set_title(f"{title} – Structured prompts")
        axes[i][0].set_xlim(0, global_xmax)
        axes[i][0].set_ylim(0, global_ymax * 1.05 if global_ymax > 0 else 1)
        axes[i][0].grid(axis='y', alpha=0.2)
        sns.histplot(plain[metric], kde=True, bins=40, color='C1', ax=axes[i][1])
        axes[i][1].set_title(f"{title} – Non-structured prompts")
        axes[i][1].set_xlim(0, global_xmax)
        axes[i][1].set_ylim(0, global_ymax * 1.05 if global_ymax > 0 else 1)
        axes[i][1].grid(axis='y', alpha=0.2)
    plt.tight_layout()
    return fig


# =============================================================================
# STREAMLIT RENDER FUNCTION
# =============================================================================

def render_prompt_analysis_page():
    st.title("6️⃣ Prompt Analysis")
    st.markdown("""
    Analyze extracted prompts for features: language distribution, 
    structural metrics, functional composition, and syntactic heuristics. All regex patterns and heuristic 
    rules are **fully editable** below to adapt to your research context.
    """)
    # === PRE-TAB INITIALIZATION: Populate reproducibility metadata for replication mode ===
    # This MUST run before st.tabs() so it executes regardless of active tab
    if st.session_state.get("replication_mode") and "pa_analyzed_df" in st.session_state:
        repl_df = st.session_state["pa_analyzed_df"]
        id_col_candidates = ["source_filename", "row_index", "study_id"]
        id_col = next((c for c in id_col_candidates if c in repl_df.columns), None)
        n_source = repl_df[id_col].nunique() if id_col else "N/A"
        st.session_state["pa_source_records"] = n_source
        st.session_state["pa_prompts_df"] = repl_df
        st.session_state["pa_prompt_col_used"] = "prompts_used (replication)"
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data & Extraction",
        "⚙️ Editable Patterns & Rules",
        "📊 Analysis Results",
        "🔬 Reproducibility"
    ])

    # =========================================================================
    # TAB 1: DATA LOADING & PROMPT EXTRACTION
    # =========================================================================
    with tab1:
                # === REPLICATION MODE INJECTION ===
        if st.session_state.get("replication_mode"):
            from core.utils import get_replication_path

            st.info("🔬 **Replication Mode:** Original prompt analysis results loaded. Toggle off in sidebar to run your own analysis.")

            repl_path = get_replication_path("s6_prompts")
            if repl_path and os.path.exists(repl_path):
                try:
                    repl_df = pd.read_csv(repl_path, dtype=str).fillna("")
                    # Cast numeric metric columns that were stored as strings in CSV
                    numeric_cols = ['char_len', 'token_count', 'sentence_count']
                    for col in numeric_cols:
                        if col in repl_df.columns:
                            repl_df[col] = pd.to_numeric(repl_df[col], errors='coerce')
                    # Cast boolean-like columns back to proper booleans
                    bool_cols = ['has_role', 'has_constraints', 'has_example', 'has_cot',
                                 'has_any_structure', 'has_any_functional',
                                 'imperative_like', 'interrogative_like']
                    for col in bool_cols:
                        if col in repl_df.columns:
                            repl_df[col] = repl_df[col].astype(str).str.strip().str.lower().map(
                                {'true': True, 'false': False, '1': True, '0': False}
                            ).fillna(False).astype(bool)
                    # Compatibility: ensure 'has_any_functional' exists regardless of CSV vintage
                    if 'has_any_functional' not in repl_df.columns and 'has_any_structure' in repl_df.columns:
                        repl_df['has_any_functional'] = repl_df['has_any_structure']
                    elif 'has_any_functional' not in repl_df.columns:
                        # Derive from component columns if available
                        func_cols = [c for c in ['has_role', 'has_constraints', 'has_example', 'has_cot'] if c in repl_df.columns]
                        if func_cols:
                            repl_df['has_any_functional'] = repl_df[func_cols].any(axis=1)
                        else:
                            repl_df['has_any_functional'] = False

                    st.session_state["pa_analyzed_df"] = repl_df
                    # Clear cached figures so they regenerate from replication data
                    st.session_state.pop("pa_figures", None)
                    st.success(f"✅ Loaded {len(repl_df)} analyzed prompts from replication data.")
                except Exception as e:
                    st.error(f"❌ Failed to load replication prompt analysis file: {e}")
            else:
                st.warning("⚠️ Replication prompt analysis file not found. Check `replication_data/stage6_analyzed_prompts.csv`.")

            st.divider()
            # Skip normal upload/extraction — fall through to results display in Tab 3
            if "pa_analyzed_df" not in st.session_state:
                return
        else:
            # === NORMAL MODE ===
            # --- AUTO-RECOVER FROM DISK IF SESSION STATE LOST ---
            if "pa_analyzed_df" not in st.session_state:
                auto_save_csv = os.path.join(AUTO_SAVE_DIR, "analyzed_prompts.csv")
                if os.path.exists(auto_save_csv):
                    try:
                        recovered_df = pd.read_csv(auto_save_csv, dtype=str).fillna("")
                        # Cast numeric metric columns that were stored as strings in CSV
                        numeric_cols = ['char_len', 'token_count', 'sentence_count']
                        for col in numeric_cols:
                            if col in recovered_df.columns:
                                recovered_df[col] = pd.to_numeric(recovered_df[col], errors='coerce')
                        bool_cols = ['has_role', 'has_constraints', 'has_example', 'has_cot',
                                    'has_any_structure', 'has_any_functional',
                                    'imperative_like', 'interrogative_like']
                        for col in bool_cols:
                            if col in recovered_df.columns:
                                recovered_df[col] = recovered_df[col].astype(str).str.strip().str.lower().map(
                                    {'true': True, 'false': False, '1': True, '0': False}
                                ).fillna(False).astype(bool)
                        # Compatibility: ensure 'has_any_functional' exists regardless of CSV vintage
                        if 'has_any_functional' not in recovered_df.columns and 'has_any_structure' in recovered_df.columns:
                            recovered_df['has_any_functional'] = recovered_df['has_any_structure']
                        elif 'has_any_functional' not in recovered_df.columns:
                            func_cols = [c for c in ['has_role', 'has_constraints', 'has_example', 'has_cot'] if c in recovered_df.columns]
                            if func_cols:
                                recovered_df['has_any_functional'] = recovered_df[func_cols].any(axis=1)
                            else:
                                recovered_df['has_any_functional'] = False

                        st.session_state["pa_analyzed_df"] = recovered_df
                        st.info(f"♻️ Recovered analyzed prompts from `{auto_save_csv}` ({len(recovered_df)} records)")
                    except Exception as e:
                        st.warning(f"⚠️ Could not recover auto-save: {e}")

            st.subheader("Upload Extracted Data")
            json_upload = st.file_uploader(
                "Upload structured_output.json (from Stage 3)",
                type=["json"],
                key="pa_json_upload"
            )

            if json_upload:
                try:
                    json_upload.seek(0)
                    data = json.loads(json_upload.read().decode("utf-8"))
                    json_upload.seek(0)
                    df = pd.DataFrame(data)
                    st.success(f"✅ Loaded {len(df)} records")

                    # --- Auto-detect prompt field ---
                    prompt_candidates = [c for c in df.columns if any(
                        kw in c.lower() for kw in ['prompt', 'instruction', 'query', 'input_text', 'user_message']
                    )]
                    default_prompt_col = 'prompts_used' if 'prompts_used' in df.columns else (
                        prompt_candidates[0] if prompt_candidates else None
                    )

                    if default_prompt_col:
                        st.info(f"🔍 Auto-detected prompt field: **`{default_prompt_col}`**")

                    prompt_col = st.selectbox(
                        "Select prompt field",
                        options=list(df.columns),
                        index=list(df.columns).index(default_prompt_col) if default_prompt_col and default_prompt_col in df.columns else 0,
                        help="Choose the column containing prompts. Auto-detected based on common naming patterns.",
                        key="pa_prompt_col_selector"
                    )

                    if prompt_col not in df.columns:
                        st.error(f"❌ Selected column '{prompt_col}' not found in dataset.")
                        return

                    # Temporarily rename selected column to 'prompts_used' for downstream compatibility
                    df_for_extraction = df.copy()
                    if prompt_col != 'prompts_used':
                        df_for_extraction = df_for_extraction.rename(columns={prompt_col: 'prompts_used'})

                    prompts_df, id_col = extract_prompts_from_df(df_for_extraction)

                    if prompts_df.empty:
                        st.warning(f"⚠️ No valid prompts found in column `{prompt_col}`. Check that it contains non-empty, non-N/A values.")
                    else:
                        st.success(f"✅ Extracted {len(prompts_df)} individual prompts from column `{prompt_col}`")

                    st.session_state["pa_prompts_df"] = prompts_df
                    st.session_state["pa_id_col"] = id_col
                    st.session_state["pa_source_records"] = len(df)
                    st.session_state["pa_prompt_col_used"] = prompt_col

                    if not prompts_df.empty:
                        with st.expander("👀 Preview extracted prompts (first 10)"):
                            preview_cols = [id_col, 'prompt_index', 'prompt']
                            preview_cols = [c for c in preview_cols if c in prompts_df.columns]
                            st.dataframe(prompts_df[preview_cols].head(10), use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Failed to load JSON: {e}")
            else:
                st.warning("⚠️ Upload a JSON file to proceed.")

    # =========================================================================
    # TAB 2: EDITABLE PATTERNS & HEURISTIC RULES
    # =========================================================================
    with tab2:
        if st.session_state.get("replication_mode"):
            st.info("🔬 **Replication Mode:** Pattern editing is disabled. The patterns used in the original analysis are archived in the Reproducibility tab.")
        else:
            st.subheader("Editable Regex Patterns & Heuristic Rules")
            st.caption("Modify these patterns to match your research context. Changes apply immediately when you run analysis in Tab 3.")

            # Initialize session state defaults
            if "pa_role_patterns" not in st.session_state:
                st.session_state["pa_role_patterns"] = DEFAULT_ROLE_PATTERNS.copy()
            if "pa_constraint_patterns" not in st.session_state:
                st.session_state["pa_constraint_patterns"] = DEFAULT_CONSTRAINT_PATTERNS.copy()
            if "pa_example_markers" not in st.session_state:
                st.session_state["pa_example_markers"] = DEFAULT_EXAMPLE_MARKERS.copy()
            if "pa_cot_markers" not in st.session_state:
                st.session_state["pa_cot_markers"] = DEFAULT_COT_MARKERS.copy()

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("#### 🎭 Role Assignment Patterns")
                role_text = st.text_area(
                    "Role patterns (one regex per line)",
                    value="\n".join(st.session_state["pa_role_patterns"]),
                    height=150,
                    key="pa_role_editor",
                    help="Regex patterns that detect role assignment in prompts."
                )
                st.session_state["pa_role_patterns"] = [
                    line.strip() for line in role_text.strip().split("\n") if line.strip()
                ]

                st.markdown("#### 📏 Constraint Patterns")
                constraint_text = st.text_area(
                    "Constraint patterns (one regex per line)",
                    value="\n".join(st.session_state["pa_constraint_patterns"]),
                    height=200,
                    key="pa_constraint_editor",
                    help="Regex patterns that detect length/time/quantity constraints."
                )
                st.session_state["pa_constraint_patterns"] = [
                    line.strip() for line in constraint_text.strip().split("\n") if line.strip()
                ]

            with col_b:
                st.markdown("#### 💡 Example Markers")
                example_text = st.text_area(
                    "Example markers (one regex per line)",
                    value="\n".join(st.session_state["pa_example_markers"]),
                    height=150,
                    key="pa_example_editor",
                    help="Regex patterns that detect few-shot/example provision."
                )
                st.session_state["pa_example_markers"] = [
                    line.strip() for line in example_text.strip().split("\n") if line.strip()
                ]

                st.markdown("#### 🧠 Chain-of-Thought Markers")
                cot_text = st.text_area(
                    "CoT markers (one regex per line)",
                    value="\n".join(st.session_state["pa_cot_markers"]),
                    height=200,
                    key="pa_cot_editor",
                    help="Regex patterns that detect chain-of-thought/reasoning instructions."
                )
                st.session_state["pa_cot_markers"] = [
                    line.strip() for line in cot_text.strip().split("\n") if line.strip()
                ]

            st.divider()
            st.info("💡 **Imperative/Interrogative heuristics** use spaCy POS tagging and cannot be edited via regex. They require a spaCy model to be installed. Structural cues above are fully customizable.")

            if st.button("🔄 Reset to Defaults", key="pa_reset_patterns_btn"):
                st.session_state["pa_role_patterns"] = DEFAULT_ROLE_PATTERNS.copy()
                st.session_state["pa_constraint_patterns"] = DEFAULT_CONSTRAINT_PATTERNS.copy()
                st.session_state["pa_example_markers"] = DEFAULT_EXAMPLE_MARKERS.copy()
                st.session_state["pa_cot_markers"] = DEFAULT_COT_MARKERS.copy()
                st.rerun()
            st.divider()
            st.markdown("#### ➕ Custom Markers")
            st.caption("Define additional pattern-based markers beyond the defaults. Each marker creates a new boolean column in the analysis output.")

            # Initialize custom markers in session state
            if "pa_custom_markers" not in st.session_state:
                st.session_state["pa_custom_markers"] = []

            # Display existing custom markers
            markers_to_remove = []
            for i, marker in enumerate(st.session_state["pa_custom_markers"]):
                col_name, col_patterns = st.columns([1, 3])
                with col_name:
                    new_name = st.text_input(
                        f"Marker name",
                        value=marker["name"],
                        key=f"pa_custom_marker_name_{i}",
                        placeholder="e.g., Tone"
                    )
                    if st.button("🗑️ Remove", key=f"pa_remove_marker_{i}"):
                        markers_to_remove.append(i)
                with col_patterns:
                    new_patterns_text = st.text_area(
                        f"Patterns (one regex per line)",
                        value="\n".join(marker["patterns"]),
                        height=100,
                        key=f"pa_custom_marker_patterns_{i}"
                    )
                    st.session_state["pa_custom_markers"][i] = {
                        "name": new_name.strip(),
                        "patterns": [line.strip() for line in new_patterns_text.strip().split("\n") if line.strip()]
                    }

            # Remove deleted markers
            if markers_to_remove:
                st.session_state["pa_custom_markers"] = [
                    m for i, m in enumerate(st.session_state["pa_custom_markers"])
                    if i not in markers_to_remove
                ]
                st.rerun()

            # Add new marker button
            if st.button("➕ Add Custom Marker", key="pa_add_custom_marker_btn"):
                st.session_state["pa_custom_markers"].append({
                    "name": "",
                    "patterns": []
                })
                st.rerun()           

    # =========================================================================
    # TAB 3: ANALYSIS RESULTS
    # =========================================================================
    with tab3:
        if st.session_state.get("replication_mode"):
            # Replication mode: skip run logic, go straight to results display
            if "pa_analyzed_df" not in st.session_state:
                st.warning("⚠️ Replication data not loaded. Check Tab 1 or toggle Replication Mode off and on.")
                return
        else:
            # Normal mode: require raw prompts and show run button
            if "pa_prompts_df" not in st.session_state:
                st.info("ℹ️ Load data in Tab 1 first.")
                return
            prompts_df = st.session_state["pa_prompts_df"].copy()
            if st.button("🚀 Run Full Prompt Analysis", type="primary", key="pa_run_btn"):
                total_steps = 7
                progress_bar = st.progress(0, text="Starting analysis...")

                # --- Step 0/7: Ensure required spaCy models are available ---
                progress_bar.progress(0.5 / total_steps, text="🧠 Checking spaCy models...")
                if HAS_SPACY:
                    # Always ensure English model is available (needed for POS heuristics)
                    _ = get_spacy_for_lang('en')
                    # Detect which languages will be needed and pre-download models
                    if HAS_LANGDETECT:
                        sample_langs = prompts_df['prompt'].head(50).apply(detect_language_safe).unique()
                        needed_models = set()
                        for lang in sample_langs:
                            model = LANG_SPACY_MAP.get(lang, 'xx_sent_ud_sm')
                            needed_models.add(model)
                        for model in needed_models:
                            if model not in _spacy_cache:
                                try:
                                    spacy.load(model)
                                    _spacy_cache[model] = spacy.load(model)
                                except OSError:
                                    _download_spacy_model(model)
                                    try:
                                        _spacy_cache[model] = spacy.load(model)
                                    except OSError:
                                        pass

                # --- Step 1/7: Character length ---
                progress_bar.progress(1 / total_steps, text="📏 Computing character lengths...")
                prompts_df['char_len'] = prompts_df['prompt'].apply(len)

                # --- Step 2/7: Language detection ---
                progress_bar.progress(2 / total_steps, text="🌍 Detecting languages...")
                if HAS_LANGDETECT:
                    prompts_df['lang'] = prompts_df['prompt'].apply(detect_language_safe)
                else:
                    prompts_df['lang'] = 'und'
                    st.warning("⚠️ langdetect not installed. Language detection skipped.")

                # --- Step 3/7: Token & sentence counts ---
                progress_bar.progress(3 / total_steps, text="🔢 Counting tokens and sentences...")
                basic_results = prompts_df.apply(
                    lambda r: pd.Series(analyze_prompt_basic(r['prompt'], r['lang'])),
                    axis=1
                )
                prompts_df[['token_count', 'sentence_count']] = basic_results

                # --- Step 4/7: Translation ---
                progress_bar.progress(4 / total_steps, text="🔄 Translating non-English prompts...")
                if HAS_TRANSLATOR:
                    prompts_df['prompt_en'] = prompts_df.apply(
                        lambda r: translate_to_english(r['prompt'], r['lang']), axis=1
                    )
                else:
                    prompts_df['prompt_en'] = prompts_df['prompt']
                    st.warning("⚠️ deep-translator not installed. Using original text for structural analysis.")

                # --- Step 5/7: Regex-based functional features ---
                progress_bar.progress(5 / total_steps, text="🔍 Detecting functional features (role, constraints, examples, CoT)...")
                role_pats = st.session_state.get("pa_role_patterns", DEFAULT_ROLE_PATTERNS)
                constraint_pats = st.session_state.get("pa_constraint_patterns", DEFAULT_CONSTRAINT_PATTERNS)
                example_pats = st.session_state.get("pa_example_markers", DEFAULT_EXAMPLE_MARKERS)
                cot_pats = st.session_state.get("pa_cot_markers", DEFAULT_COT_MARKERS)

                prompts_df['has_role'] = prompts_df['prompt_en'].apply(lambda t: detect_any(role_pats, t))
                prompts_df['has_constraints'] = prompts_df['prompt_en'].apply(lambda t: detect_any(constraint_pats, t))
                prompts_df['has_example'] = prompts_df['prompt_en'].apply(lambda t: detect_any(example_pats, t))
                prompts_df['has_cot'] = prompts_df['prompt_en'].apply(lambda t: detect_any(cot_pats, t))
                prompts_df['has_any_functional'] = prompts_df[
                    ['has_role', 'has_constraints', 'has_example', 'has_cot']
                ].any(axis=1)

                # --- Custom markers (user-defined) ---
                custom_markers = st.session_state.get("pa_custom_markers", [])
                custom_col_names = []
                for cm in custom_markers:
                    cm_name = cm.get("name", "").strip()
                    cm_pats = cm.get("patterns", [])
                    if cm_name and cm_pats:
                        safe_col = f"has_custom_{re.sub(r'[^a-z0-9]', '_', cm_name.lower())}"
                        prompts_df[safe_col] = prompts_df['prompt_en'].apply(
                            lambda t: detect_any(cm_pats, t)
                        )
                        custom_col_names.append((safe_col, cm_name))

                # Include custom markers in "has_any_functional" if any exist
                if custom_col_names:
                    all_func_cols = ['has_role', 'has_constraints', 'has_example', 'has_cot'] + [c[0] for c in custom_col_names]
                    prompts_df['has_any_functional'] = prompts_df[all_func_cols].any(axis=1)

                # --- Step 6/7: Syntactic heuristics (imperative / interrogative) ---
                progress_bar.progress(6 / total_steps, text="🧠 Running syntactic heuristics (imperative / interrogative)...")
                prompts_df['imperative_like'] = prompts_df.apply(
                    lambda r: detect_imperative_heuristic(r['prompt_en'], r['lang']), axis=1
                )
                prompts_df['interrogative_like'] = prompts_df.apply(
                    lambda r: detect_interrogative_heuristic(r['prompt_en'], r['lang']), axis=1
                )

                progress_bar.progress(1.0, text="✅ Analysis complete!")
                time.sleep(0.5)  # Brief pause so user sees 100%

                st.session_state["pa_analyzed_df"] = prompts_df
                st.session_state.pop("pa_figures", None)

                # ---- AUTO-SAVE TO TEMP (safety net) ----
                try:
                    auto_save_csv = os.path.join(AUTO_SAVE_DIR, "analyzed_prompts.csv")
                    prompts_df.to_csv(auto_save_csv, index=False)
                    auto_save_json = os.path.join(AUTO_SAVE_DIR, "analysis_metadata.json")
                    with open(auto_save_json, "w", encoding="utf-8") as f:
                        json.dump({
                            "total_prompts": len(prompts_df),
                            "prompt_col_used": st.session_state.get("pa_prompt_col_used", "N/A"),
                            "timestamp": datetime.now().isoformat(),
                        }, f, indent=2, ensure_ascii=False)
                    st.caption(f"💾 Auto-saved to `{AUTO_SAVE_DIR}`")
                except Exception as save_err:
                    st.warning(f"⚠️ Auto-save failed: {save_err}")

                st.success(f"✅ Analysis complete for {len(prompts_df)} prompts")
                st.rerun()
        # --- Display results ---
        if "pa_analyzed_df" not in st.session_state:
            st.info("ℹ️ Click 'Run Full Prompt Analysis' to generate results.")
            return

        adf = st.session_state["pa_analyzed_df"]

        # Generate figures if not cached
        if "pa_figures" not in st.session_state:
            with st.spinner("Generating visualizations..."):
                figures = {}

                # Language distribution (codes mapped to names)
                if 'lang' in adf.columns:
                    lang_named = adf['lang'].apply(code_to_name)
                    figures["Language Distribution"] = create_bar_chart(
                        lang_named, "Detected Languages (Top 20)", top_n=20
                    )

                # Basic distributions
                xlims = {
                    'char_len': (0, np.percentile(adf['char_len'], 99)),
                    'token_count': (0, np.percentile(adf['token_count'], 99)),
                    'sentence_count': (0, np.percentile(adf['sentence_count'], 99)),
                }
                figures["Prompt Length Distributions"] = create_distribution_plots(
                    adf,
                    ['char_len', 'token_count', 'sentence_count'],
                    ['Character Length', 'Token Count', 'Sentence Count'],
                    xlims
                )

                # Comparative: functional vs non-functional
                func_col = 'has_any_functional' if 'has_any_functional' in adf.columns else (
                    'has_any_structure' if 'has_any_structure' in adf.columns else None
                )
                if func_col is None:
                    # Derive from component columns
                    func_cols = [c for c in ['has_role', 'has_constraints', 'has_example', 'has_cot'] if c in adf.columns]
                    if func_cols:
                        adf['has_any_functional'] = adf[func_cols].any(axis=1)
                        func_col = 'has_any_functional'

                if func_col:
                    structured = adf[adf[func_col]]
                    plain = adf[~adf[func_col]]
                else:
                    structured = pd.DataFrame()
                    plain = pd.DataFrame()
                if len(structured) > 0 and len(plain) > 0:
                    figures["Functional vs Non-Functional"] = create_comparative_histograms(
                        structured, plain,
                        ['char_len', 'token_count', 'sentence_count'],
                        ['Character Length', 'Token Count', 'Sentence Count']
                    )
                    # Compute descriptive statistics matching notebook: .describe().round(2)
                    comp_stats_rows = []
                    for metric in ['char_len', 'token_count', 'sentence_count']:
                        s_desc = structured[metric].describe().round(2) if len(structured) > 0 else pd.Series(dtype=float)
                        p_desc = plain[metric].describe().round(2) if len(plain) > 0 else pd.Series(dtype=float)
                        for stat_name in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                            comp_stats_rows.append({
                                "Metric": metric.replace('_', ' ').title(),
                                "Statistic": stat_name,
                                "Functional": s_desc.get(stat_name, np.nan),
                                "Non-Functional": p_desc.get(stat_name, np.nan),
                            })
                    st.session_state["pa_comp_stats"] = pd.DataFrame(comp_stats_rows)

                st.session_state["pa_figures"] = figures

        figures = st.session_state.get("pa_figures", {})

        # === SYNTACTIC MOOD (self-contained distribution, independent from functional features) ===
        st.subheader("🗣️ Syntactic Mood Distribution")
        mood_cols = st.columns(3)

        imp_count = int(adf['imperative_like'].sum()) if 'imperative_like' in adf.columns else 0
        int_count = int(adf['interrogative_like'].sum()) if 'interrogative_like' in adf.columns else 0
        dec_count = len(adf) - imp_count - int_count
        mood_total = imp_count + int_count + dec_count  # Self-contained denominator

        if mood_total > 0:
            mood_cols[0].metric("Imperative", f"{imp_count} ({imp_count/mood_total*100:.1f}%)")
            mood_cols[1].metric("Interrogative", f"{int_count} ({int_count/mood_total*100:.1f}%)")
            mood_cols[2].metric("Declarative / Other", f"{dec_count} ({dec_count/mood_total*100:.1f}%)")
        else:
            mood_cols[0].metric("Imperative", "0 (0.0%)")
            mood_cols[1].metric("Interrogative", "0 (0.0%)")
            mood_cols[2].metric("Declarative / Other", "0 (0.0%)")

        st.caption("Syntactic mood is determined by POS-tagging heuristics on the first sentence. Declarative/Other includes all prompts not classified as imperative or interrogative.")
        st.divider()

        # === FUNCTIONAL FEATURES (regex-based, separate from syntactic mood) ===
        st.subheader("🏗️ Functional Features (Regex-Based)")
        n_total = len(adf)  # Total prompts in dataset — denominator for functional features
        struct_cols = st.columns(4)
        struct_features = [
            ('has_role', 'Role Assignment'),
            ('has_constraints', 'Constraints'),
            ('has_example', 'Examples'),
            ('has_cot', 'Chain-of-Thought'),
        ]
        for col_idx, (feat, label) in enumerate(struct_features):
            if feat in adf.columns:
                count = int(adf[feat].sum())
                pct = (count / n_total) * 100 if n_total > 0 else 0
                struct_cols[col_idx].metric(label, f"{count} ({pct:.1f}%)")

        # Display custom marker metrics if any exist
        custom_markers = st.session_state.get("pa_custom_markers", [])
        custom_col_names = [(f"has_custom_{re.sub(r'[^a-z0-9]', '_', cm['name'].lower())}", cm['name'])
                           for cm in custom_markers if cm.get("name", "").strip() and cm.get("patterns")]
        if custom_col_names:
            st.markdown("#### ➕ Custom Markers")
            custom_cols_display = st.columns(min(len(custom_col_names), 4))
            for col_idx, (feat, label) in enumerate(custom_col_names):
                if feat in adf.columns:
                    count = int(adf[feat].sum())
                    pct = (count / n_total) * 100 if n_total > 0 else 0
                    custom_cols_display[col_idx % len(custom_cols_display)].metric(label, f"{count} ({pct:.1f}%)")
            st.caption("Custom markers are user-defined regex patterns configured in Tab 2. Included in the 'has_any_functional' flag for comparative histograms.")

        st.caption("Functional features are detected via user-editable regex patterns (see Tab 2).")
        st.divider()

        # Descriptive stats table
        st.markdown("#### Descriptive Statistics")
        desc_cols = ['char_len', 'token_count', 'sentence_count']
        desc_cols = [c for c in desc_cols if c in adf.columns]
        if desc_cols:
            st.dataframe(adf[desc_cols].describe().round(2), use_container_width=True)

        st.divider()

        # Figures
        if figures:
            st.subheader(f"📊 Visualizations ({len(figures)} charts)")
            chart_names = list(figures.keys())
            selected_charts = st.multiselect(
                "Select charts to display",
                options=chart_names,
                default=chart_names,
                key="pa_viz_selector"
            )
            for chart_name in selected_charts:
                if chart_name in figures:
                    st.markdown(f"#### {chart_name}")
                    st.pyplot(figures[chart_name])
                    # Show descriptive stats table for functional vs non-functional comparison
                    if chart_name == "Functional vs Non-Functional" and "pa_comp_stats" in st.session_state:
                        st.markdown("**Descriptive Statistics: Functional vs Non-Functional Prompts**")
                        st.dataframe(
                            st.session_state["pa_comp_stats"],
                            use_container_width=True,
                            hide_index=True
                        )
                    st.divider()

            # Download figures as ZIP + auto-save individual PNGs to disk
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, fig in figures.items():
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
                    img_buf.seek(0)
                    safe_name = f"{name.lower().replace(' ', '_').replace('/', '_')}.png"
                    zf.writestr(safe_name, img_buf.read())
                    # Also auto-save individual PNG to disk
                    try:
                        fig.savefig(os.path.join(FIGURES_DIR, safe_name), dpi=150, bbox_inches="tight")
                    except Exception:
                        pass
            zip_buffer.seek(0)
            st.download_button(
                label="💾 Download All Figures (ZIP)",
                data=zip_buffer,
                file_name="prompt_analysis_figures.zip",
                mime="application/zip",
                type="secondary",
                key="pa_download_figs_zip"
            )
            st.caption(f"💾 Individual figures also saved to `{FIGURES_DIR}`")

        # Download analyzed DataFrame
        st.divider()
        csv_buf = io.BytesIO()
        adf.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        st.download_button(
            label="📄 Download Analyzed Prompts (CSV)",
            data=csv_buf,
            file_name="analyzed_prompts.csv",
            mime="text/csv",
            key="pa_download_csv"
        )

    # =========================================================================
    # TAB 4: REPRODUCIBILITY & ACCESSIBILITY
    # =========================================================================
    with tab4:
        st.subheader("🔬 Reproducibility & Accessibility Information")
        # Collect spaCy model versions for audit trail
        spacy_models_used = {}
        if HAS_SPACY:
            for lang_code, model_name in LANG_SPACY_MAP.items():
                if model_name in _spacy_cache and _spacy_cache[model_name] is not None:
                    try:
                        spacy_models_used[model_name] = _spacy_cache[model_name].meta.get("version", "unknown")
                    except Exception:
                        spacy_models_used[model_name] = "loaded"

        custom_markers_snapshot = [
            {"name": cm.get("name", ""), "patterns": cm.get("patterns", [])}
            for cm in st.session_state.get("pa_custom_markers", [])
            if cm.get("name", "").strip() and cm.get("patterns")
        ]

        repro = {
            "stage": "prompt_analysis",
            "timestamp": datetime.now().isoformat(),
            "audit_trail": {
                "source_records": st.session_state.get("pa_source_records", 0),
                "total_prompts_extracted": len(st.session_state.get("pa_prompts_df", [])),
                "total_prompts_analyzed": len(st.session_state.get("pa_analyzed_df", [])),
                "prompt_field_used": st.session_state.get("pa_prompt_col_used", "N/A"),
                "auto_save_directory": AUTO_SAVE_DIR,
                "figures_directory": FIGURES_DIR,
                "output_files": {
                    "analyzed_csv": os.path.join(AUTO_SAVE_DIR, "analyzed_prompts.csv"),
                    "metadata_json": os.path.join(AUTO_SAVE_DIR, "analysis_metadata.json"),
                },
            },
            "spacy_models_loaded": spacy_models_used,
            "editable_patterns": {
                "role_assignment": st.session_state.get("pa_role_patterns", DEFAULT_ROLE_PATTERNS),
                "constraints": st.session_state.get("pa_constraint_patterns", DEFAULT_CONSTRAINT_PATTERNS),
                "example_markers": st.session_state.get("pa_example_markers", DEFAULT_EXAMPLE_MARKERS),
                "cot_markers": st.session_state.get("pa_cot_markers", DEFAULT_COT_MARKERS),
                "custom_markers": custom_markers_snapshot,
            },
            "heuristic_rules": {
                "imperative": "spaCy POS: first token VB+VERB, no subject, no WH/AUX start, no trailing '?'",
                "interrogative": "spaCy POS: trailing '?' or WH-word + AUX/modal",
            },
        }

        st.json(repro)
        st.code(json.dumps(repro, indent=2), language="json")

        st.divider()

        # Reproducibility Checklist
        st.markdown("#### ✅ Reproducibility Checklist")
        st.markdown(f"""
        - [x] All regex patterns used for structural detection archived above
        - [x] Imperative/interrogative heuristic rules documented above
        - [x] Analyzed DataFrame auto-saved to `{AUTO_SAVE_DIR}`
        - [x] All figures auto-saved to `{FIGURES_DIR}`
        - [ ] Document any manual corrections or fallback mechanisms triggered during analysis
        """)

        st.divider()

        # Accessibility Alternatives
        st.markdown("#### ♿ Accessibility Alternatives")
        st.markdown("""
        | Scenario | Alternative |
        |---|---|
        | **Offline language detection** | Use `fasttext-wheel` for fast, fully offline language identification |
        | **Collaborative teams** | Host a shared local translation API or use pre-translated prompt corpora to avoid repeated API calls |
        | **Without spaCy installed** | Token/sentence counts fall back to NLTK or whitespace splitting; imperative/interrogative heuristics gracefully disabled |
        | **Without langdetect installed** | Language detection returns `'und'`; translation skipped; structural analysis proceeds on original text |
        """)
