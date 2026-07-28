
"""
SARSP-LangEd - Stage 7: Data-Driven Topic Discovery & Theme Construction
Semantic clustering with adaptive KMeans + constrained LLM-assisted labeling.
Universal API provider support.
"""

import streamlit as st
import os
import io
import json
import time
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from openai import OpenAI
    HAS_OPENAI_CLIENT = True
except ImportError:
    HAS_OPENAI_CLIENT = False

import zipfile
from datetime import datetime

# Consistent temp directory structure
from core.utils import get_session_temp_dir
TEMP_DIR = get_session_temp_dir()
STAGE_DIR = os.path.join(TEMP_DIR, "stage7_topic_discovery")
TOPIC_OUTPUT_DIR = os.path.join(STAGE_DIR, "results")
AUTO_SAVE_DIR = os.path.join(STAGE_DIR, "auto_save")

# Ensure directories exist at module load
for d in [STAGE_DIR, TOPIC_OUTPUT_DIR, AUTO_SAVE_DIR]:
    os.makedirs(d, exist_ok=True)
# =============================================================================
# API PROVIDERS (matches Stage 2 screening)
# =============================================================================
_ALL_API_PROVIDERS = {
    "Cerebras (Cloud)": "https://api.cerebras.ai/v1",
    "Google Gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "Mistral AI": "https://api.mistral.ai/v1",
    "OpenAI": "https://api.openai.com/v1",
    "LM Studio (Local)": "http://localhost:1234/v1",
    "Ollama (Local)": "http://localhost:11434/v1",
    "Custom / Other": "custom",
}

def _get_available_providers():
    """Filter providers based on deployment context. Local/custom only available locally."""
    is_cloud = bool(os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("STREAMLIT_CLOUD"))
    if is_cloud:
        return {k: v for k, v in _ALL_API_PROVIDERS.items()
                if "Local" not in k and "Custom" not in k}
    return _ALL_API_PROVIDERS

EMBEDDING_MODELS = [
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "all-distilroberta-v1",
]

# =============================================================================
# EXACT PROMPTS 
# =============================================================================

# Cluster labeling prompt template — uses .format() at call time
DEFAULT_CLUSTER_LABEL_PROMPT = """### CONTEXT ###
You are analysing research findings in GenAI and language education.
These {n_examples} statements form a theme cluster.

{examples}

### TASKS ###
1. Give a concise label (≤6 words).
2. Provide 2–3‑sentence description.
3. Provide 5–10 key words.

### OUTPUT ###
Output JSON ONLY as:
{{"cluster_id": {cluster_id}, "label": "", "description": "", "key_terms": [] }}"""

# Theme merging prompt template — uses .format() at call time
DEFAULT_THEME_MERGE_PROMPT = """### TASK ###
Below are summaries of clusters for {section_name}.
If any describe the same or overlapping topic, merge them into unified themes.

Each theme must have:
- theme_label (≤6 words)
- cluster_ids [list of ints]
- summary (1–2 sentences)

### OUTPUT ###
Output JSON list:
[{{"theme_label": "...", "cluster_ids": [...], "summary": "..." }}, ...]
{block}"""

# Default JSON schemas
DEFAULT_LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "cluster_id": {"type": "integer"},
        "label": {"type": "string"},
        "description": {"type": "string"},
        "key_terms": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["cluster_id", "label", "description", "key_terms"]
}

DEFAULT_MERGE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "theme_label": {"type": "string", "description": "Short (≤6 words) label for the unified theme"},
            "cluster_ids": {"type": "array", "items": {"type": "integer"}, "description": "List of cluster IDs grouped under this theme"},
            "summary": {"type": "string", "description": "1–2 sentence summary explaining the shared topic"}
        },
        "required": ["theme_label", "cluster_ids", "summary"]
    }
}

SCHEMA_GENERATION_PROMPT = """You are an expert JSON Schema architect for structured data extraction from academic papers.

### YOUR TASK ###
Below is a TASK description listing fields to extract from research papers. Generate a STRICT JSON Schema (draft-07) that enforces exactly those fields.

### RULES ###
1. Root must be {{"type": "object"}} with "required" and "properties".
2. Every field in the TASK description MUST appear in "properties".
3. Include enums exactly as listed. Add "N/A" to all optional string fields.
4. Arrays use {{"type": "array", "items": {{...}}}}. Nested objects use {{"type": "object", "properties": {{...}}, "required": [...]}}.
5. Use "description" for complex fields.
6. Do NOT add or omit fields.
7. Return ONLY raw JSON. No markdown fences. No explanation. First char = {{, last char = }}.

### TASK DESCRIPTION ###
{prompt}
"""

# Standard pipeline sections
SECTIONS = {
    "learning_perceptions": lambda d: _flatten_list_field(d, "learning_perceptions"),
    "outcomes_benefits": lambda d: _flatten_dict_field(d, "outcomes", "benefits_affordances"),
    "outcomes_drawbacks": lambda d: _flatten_dict_field(d, "outcomes", "drawbacks"),
    "outcomes_limitations": lambda d: _flatten_dict_field(d, "outcomes", "study_limitations"),
    "stakeholder_students": lambda d: _flatten_dict_field(d, "stakeholder_impact", "students"),
    "stakeholder_teachers": lambda d: _flatten_dict_field(d, "stakeholder_impact", "teachers"),
    "stakeholder_institutions": lambda d: _flatten_dict_field(d, "stakeholder_impact", "institutions"),
    "stakeholder_other": lambda d: _flatten_dict_field(d, "stakeholder_impact", "other"),
    "policy_guidance": lambda d: _flatten_list_field(d, "policy_guidance"),
    "emergent_themes": lambda d: _flatten_list_field(d, "emergent_themes"),
    "emergent_issues": lambda d: _flatten_list_field(d, "emergent_issues"),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_api_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key or "sk-no-key")


def _flatten_list_field(record, field_name):
    val = record.get(field_name, [])
    if isinstance(val, list):
        return [s for s in val]
    return []


def _flatten_dict_field(record, dict_field, sub_key):
    val = record.get(dict_field, {})
    if isinstance(val, dict):
        sub = val.get(sub_key, [])
        if isinstance(sub, list):
            return [s for s in sub]
    return []


def extract_texts_from_data(data, section_key):
    """Extract texts for standard or arbitrary fields. Preserves original JSON order."""
    fn = SECTIONS.get(section_key)
    if fn:
        all_texts = []
        for record in data:
            all_texts.extend(fn(record))
    else:
        all_texts = []
        for record in data:
            val = record
            for part in section_key.split("."):
                if isinstance(val, dict):
                    val = val.get(part, [])
                else:
                    val = []
                    break
            if isinstance(val, list):
                all_texts.extend([s for s in val])
            elif isinstance(val, str):
                all_texts.append(val)
    return all_texts

def compute_embeddings(texts, model_name="all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def auto_kmeans(embeddings, max_k=30, min_k=5, random_state=42):
    """Adaptive k using silhouette optimization. Fixed step=5."""
    best_score, best_k = -1, min_k
    for k in range(min_k, max_k + 1, 5):
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k, best_score

def llm_json_call(client, model, prompt, schema, system_msg=None, max_tokens=None, temperature=0, max_retries=3):
    """
    Call LLM with JSON schema enforcement, intelligent backoff, 
    and Retry-After header parsing for Mistral 429 errors.
    max_tokens=None means the model decides its own output length (matches notebook).
    """
    import time
    
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})

    # Build API kwargs — only include max_tokens if explicitly set
    api_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema}
        }
    }
    if max_tokens is not None:
        api_kwargs["max_tokens"] = max_tokens

    content = ""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**api_kwargs)
            content = response.choices[0].message.content.strip()
            
            # Strip markdown fences if present
            fence_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?\s*```', content)
            if fence_match:
                content = fence_match.group(1).strip()
            return json.loads(content)

        except Exception as e:
            err_str = str(e).lower()
            
            # Handle 429 Rate Limit with Retry-After header
            if "429" in err_str or "rate_limit" in err_str or "rate_limited" in err_str:
                # Try to extract Retry-After from the exception
                retry_after = None
                if hasattr(e, 'headers') and e.headers:
                    retry_after = e.headers.get('retry-after')
                
                if retry_after:
                    try:
                        wait_time = int(retry_after) + 1  # Add 1s buffer
                    except ValueError:
                        wait_time = 60 * (2 ** attempt)  # Fallback exponential
                else:
                    # Exponential backoff: 30s, 60s, 120s, 240s, 480s
                    wait_time = 30 * (2 ** attempt)
                
                if attempt < max_retries - 1:
                    st.warning(f"⏳ Rate limited (429). Waiting {wait_time}s before retry {attempt+2}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    st.error(f"❌ Rate limit persisted after {max_retries} retries. Consider switching providers.")
                    return None
            
            # Handle other transient errors
            elif any(kw in err_str for kw in ["timeout", "503", "500", "connection"]):
                if attempt < max_retries - 1:
                    wait_time = 10 * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
            
            # Fallback JSON extraction for non-rate-limit errors
            try:
                s_idx = content.find("{") if "{" in content else content.find("[")
                e_idx = content.rfind("}") if "}" in content else content.rfind("]")
                if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
                    return json.loads(content[s_idx:e_idx + 1])
            except Exception:
                pass
            
            if attempt == max_retries - 1:
                return None
            time.sleep(5)
    
    return None


def generate_schema_for_prompt(client, model, prompt_text):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": SCHEMA_GENERATION_PROMPT.format(prompt=prompt_text)}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        fence_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?\s*```', content)
        if fence_match:
            content = fence_match.group(1).strip()
        json_match = re.search(r'[\[{][\s\S]*[\]}]', content)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception:
        pass
    return None


def save_section_results(section_name, df_texts, cluster_summaries, merged_themes):
    outdir = os.path.join(TOPIC_OUTPUT_DIR, section_name)
    os.makedirs(outdir, exist_ok=True)
    df_texts.to_csv(os.path.join(outdir, "texts_with_themes.csv"), index=False)
    cluster_summaries.to_csv(os.path.join(outdir, "cluster_summaries.csv"), index=False)
    pd.DataFrame(merged_themes).to_csv(os.path.join(outdir, "themes_merged.csv"), index=False)


# =============================================================================
# STREAMLIT RENDER FUNCTION
# =============================================================================

def render_topic_discovery_page():
    st.title("7️⃣ Data-Driven Topic Discovery & Theme Construction")
    st.markdown("""
    Process open-ended qualitative segments using **unsupervised semantic clustering** 
    (SentenceTransformers + KMeans) and **constrained LLM-assisted labeling**. 
    Supports multiple API providers, editable prompts, and auto-generated JSON schemas.
    """)

    if not HAS_EMBEDDINGS:
        st.error("❌ `sentence-transformers` not installed. Run: `pip install sentence-transformers`")
        return
    if not HAS_SKLEARN:
        st.error("❌ `scikit-learn` not installed. Run: `pip install scikit-learn`")
        return
    if not HAS_OPENAI_CLIENT:
        st.error("❌ `openai` not installed. Run: `pip install openai`")
        return

    # === PRE-TAB INITIALIZATION: Populate reproducibility metadata for replication mode ===
    # This MUST run before st.tabs() so it executes regardless of which tab is active
    if st.session_state.get("replication_mode") and "td_results" in st.session_state:
        # Set defaults that would normally be set by Tab 1 widgets
        st.session_state.setdefault("td_embedding_model", "all-mpnet-base-v2 (replication)")
        st.session_state.setdefault("td_provider", "N/A (replication)")
        st.session_state.setdefault("td_base_url", "N/A")
        st.session_state.setdefault("td_model_name", "N/A (replication)")
        st.session_state.setdefault("td_min_k", "N/A")
        st.session_state.setdefault("td_max_k", "N/A")
        st.session_state.setdefault("td_examples", "N/A")
        st.session_state.setdefault("td_seed", "N/A")
        st.session_state.setdefault("td_temperature", "N/A")
        st.session_state.setdefault("td_prompt_mode", "Default (replication)")
        st.session_state.setdefault("td_label_prompt", DEFAULT_CLUSTER_LABEL_PROMPT)
        st.session_state.setdefault("td_merge_prompt", DEFAULT_THEME_MERGE_PROMPT)
        st.session_state.setdefault("td_label_schema", DEFAULT_LABEL_SCHEMA)
        st.session_state.setdefault("td_merge_schema", DEFAULT_MERGE_SCHEMA)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data & API Configuration",
        "✏️ Prompts & Schemas",
        "🚀 Run Discovery",
        "🔬 Reproducibility"
    ])

    # =========================================================================
    # TAB 1: DATA & API CONFIGURATION
    # =========================================================================
    with tab1:
        # === REPLICATION MODE INJECTION ===
        if st.session_state.get("replication_mode"):
            from core.utils import get_replication_path, REPLICATION_DIR

            st.info("🔬 **Replication Mode:** Original topic discovery results loaded. Toggle off in sidebar to run your own discovery.")

            topic_dir = os.path.join(REPLICATION_DIR, "stage7_topic_discovery")
            if os.path.exists(topic_dir):
                recovered_sections = {}
                for sec_dir in sorted(Path(topic_dir).iterdir()):
                    if not sec_dir.is_dir():
                        continue
                    texts_csv = sec_dir / "texts_with_themes.csv"
                    summaries_csv = sec_dir / "cluster_summaries.csv"
                    themes_csv = sec_dir / "themes_merged.csv"
                    if texts_csv.exists() and summaries_csv.exists():
                        try:
                            themes_data = []
                            if themes_csv.exists():
                                try:
                                    themes_df = pd.read_csv(themes_csv, dtype=str).fillna("")
                                    themes_data = themes_df.to_dict(orient="records")
                                except Exception:
                                    pass
                            sec_df_texts = pd.read_csv(texts_csv, dtype=str).fillna("")
                            sec_summaries = pd.read_csv(summaries_csv, dtype=str).fillna("")
                            recovered_sections[sec_dir.name] = {
                                "n_texts": len(sec_df_texts),
                                "optimal_k": len(sec_summaries),
                                "silhouette": 0.0,
                                "n_themes": len(themes_data) if themes_data else 0,
                                "df_texts": sec_df_texts,
                                "cluster_summaries": sec_summaries,
                                "merged_themes": themes_data,
                            }
                        except Exception as e:
                            st.warning(f"⚠️ Failed to load section `{sec_dir.name}`: {e}")

                if recovered_sections:
                    st.session_state["td_results"] = recovered_sections
                    total_texts = sum(v["n_texts"] for v in recovered_sections.values())
                    total_themes = sum(v["n_themes"] for v in recovered_sections.values())
                    st.success(
                        f"✅ Loaded {len(recovered_sections)} topic sections from replication data "
                        f"({total_texts} text segments, {total_themes} unified themes)."
                    )
                else:
                    st.warning("⚠️ No valid topic sections found in replication data.")
            else:
                st.warning(f"⚠️ Topic discovery replication folder not found: `{topic_dir}`")

            st.divider()
            # Skip normal config — fall through to results display in Tab 3
            if "td_results" not in st.session_state:
                return
        else:
            # === NORMAL MODE ===
            # --- AUTO-RECOVER FROM DISK IF SESSION STATE LOST ---
            if "td_results" not in st.session_state and os.path.exists(AUTO_SAVE_DIR):
                recovered_sections = {}
                for meta_file in sorted(Path(AUTO_SAVE_DIR).glob("*_metadata.json")):
                    try:
                        with open(meta_file, "r", encoding="utf-8") as mf:
                            meta = json.load(mf)
                        sec_key = meta.get("section", meta_file.stem.replace("_metadata", ""))
                        sec_dir = os.path.join(TOPIC_OUTPUT_DIR, sec_key)
                        texts_csv = os.path.join(sec_dir, "texts_with_themes.csv")
                        summaries_csv = os.path.join(sec_dir, "cluster_summaries.csv")
                        themes_csv = os.path.join(sec_dir, "themes_merged.csv")
                        if os.path.exists(texts_csv) and os.path.exists(summaries_csv):
                            recovered_sections[sec_key] = {
                                "n_texts": meta.get("n_texts", 0),
                                "optimal_k": meta.get("optimal_k", 0),
                                "silhouette": meta.get("silhouette", 0.0),
                                "n_themes": meta.get("n_themes", 0),
                                "df_texts": pd.read_csv(texts_csv, dtype=str).fillna(""),
                                "cluster_summaries": pd.read_csv(summaries_csv, dtype=str).fillna(""),
                                "merged_themes": json.loads(open(themes_csv, "r", encoding="utf-8").read()) if os.path.exists(themes_csv) else [],
                            }
                    except Exception:
                        continue
                if recovered_sections:
                    st.session_state["td_results"] = recovered_sections
                    st.info(f"♻️ Recovered {len(recovered_sections)} section results from `{AUTO_SAVE_DIR}`")

            st.subheader("Upload Extracted Data")
            json_upload = st.file_uploader(
                "Upload structured_output.json (from Stage 3)",
                type=["json"],
                key="td_json_upload"
            )

            if json_upload:
                try:
                    json_upload.seek(0)
                    data = json.loads(json_upload.read().decode("utf-8"))
                    json_upload.seek(0)
                    st.session_state["td_data"] = data
                    st.success(f"✅ Loaded {len(data)} records")

                    section_counts = {}
                    for sec_key in SECTIONS:
                        texts = extract_texts_from_data(data, sec_key)
                        section_counts[sec_key] = len(texts)
                    sec_df = pd.DataFrame([
                        {"Section": k.replace("_", " ").title(), "Text Segments": v}
                        for k, v in section_counts.items()
                    ]).sort_values("Text Segments", ascending=False)
                    st.dataframe(sec_df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"❌ Failed to load JSON: {e}")
            else:
                st.warning("⚠️ Upload a the structured JSON file to proceed.")

            st.divider()

            # --- API Provider Selection ---
            st.subheader("🔌 API Configuration")
            provider_col1, provider_col2 = st.columns([1, 1])

            available_providers = _get_available_providers()

            with provider_col1:
                selected_provider = st.selectbox(
                    "API Service Provider",
                    options=list(available_providers.keys()),
                    index=0,
                    help="Select your LLM provider. Base URL will be auto-filled."
                        + (" Local/custom providers require running the app on your own machine."
                            if "Local" not in str(available_providers.keys()) else ""),
                    key="td_provider"
                )
            default_url = available_providers.get(selected_provider, "")
            if "td_last_provider" not in st.session_state or st.session_state["td_last_provider"] != selected_provider:
                st.session_state["td_base_url"] = default_url if default_url != "custom" else ""
                st.session_state["td_last_provider"] = selected_provider

            with provider_col2:
                if default_url == "custom":
                    base_url = st.text_input(
                        "Custom API Base URL",
                        value=st.session_state.get("td_base_url", ""),
                        placeholder="https://your-api-endpoint.com/v1",
                        key="td_custom_url"
                    )
                else:
                    base_url = st.text_input(
                        "API Base URL (auto-filled)",
                        value=st.session_state.get("td_base_url", default_url),
                        key="td_base_url"
                    )

            api_key = st.text_input("API Key", type="password", value="", key="td_api_key")

            st.caption("Click below to discover available models from your selected endpoint.")
            if st.button("🔍 Discover Models", key="td_discover_btn"):
                if not base_url:
                    st.error("❌ Please provide an API Base URL first.")
                else:
                    try:
                        with st.spinner("Fetching model list..."):
                            client = get_api_client(base_url, api_key)
                            models_response = client.models.list()
                            available_models = sorted([m.id for m in models_response])
                            st.session_state["td_available_models"] = available_models
                            st.success(f"✅ Found {len(available_models)} models")
                    except Exception as e:
                        st.error(f"❌ Failed to fetch models: {str(e)}")
                        st.session_state["td_available_models"] = []

            available_models = st.session_state.get("td_available_models", [])
            if available_models:
                model_name = st.selectbox("Select Model", options=available_models, index=0, key="td_model_name")
            else:
                model_name = st.text_input("Model Name (manual)", value="mistral-large-latest", key="td_model_name_manual")
                st.info("👆 Click 'Discover Models' to load available models.")

            st.divider()

            # --- Clustering Parameters ---
            st.subheader("⚙️ Clustering Parameters")
            col_a, col_b = st.columns(2)
            with col_a:
                embedding_model = st.selectbox("Embedding Model", options=EMBEDDING_MODELS, index=0, key="td_embedding_model")
                min_clusters = st.slider("Min Clusters (k)", 5, 20, 10, key="td_min_k")
                max_clusters = st.slider("Max Clusters (k)", 15, 50, 30, key="td_max_k")
            with col_b:
                examples_per_cluster = st.slider("Examples per Cluster", 5, 30, 20, key="td_examples")
                random_state = st.number_input("Random Seed", value=42, step=1, key="td_seed")
                temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.0, 0.1, key="td_temperature")

            # --- Dynamic Section Selection ---
            if "td_data" in st.session_state:
                st.divider()
                st.subheader("Select Text Fields to Analyse")

                standard_sections = list(SECTIONS.keys())
                standard_available = [k for k in standard_sections if extract_texts_from_data(st.session_state["td_data"], k)]

                # Discover ALL eligible fields across ALL records (not just first)
                all_fields = set()
                for record in st.session_state["td_data"]:
                    for key, val in record.items():
                        if isinstance(val, list) and all(isinstance(item, str) for item in val if item is not None):
                            all_fields.add(key)
                        elif isinstance(val, str):
                            all_fields.add(key)
                        elif isinstance(val, dict):
                            for sub_key, sub_val in val.items():
                                if isinstance(sub_val, list) and all(isinstance(item, str) for item in sub_val if item is not None):
                                    all_fields.add(f"{key}.{sub_key}")
                                elif isinstance(sub_val, str):
                                    all_fields.add(f"{key}.{sub_key}")

                extra_fields = sorted([f for f in all_fields if f not in standard_sections])
                all_available = standard_available + extra_fields
                default_selection = standard_available

                selected_sections = st.multiselect(
                    "Text Fields to Cluster",
                    options=all_available,
                    default=default_selection,
                    help="Standard pipeline sections are pre-selected. All list-of-strings fields detected in your JSON are listed below. Use dot notation for nested fields (e.g., outcomes.benefits_affordances). Custom fields are processed using the same extraction logic as standard sections.",
                    key="td_selected_sections"
                )

    # =========================================================================
    # TAB 2: EDITABLE PROMPTS & SCHEMAS
    # =========================================================================
    with tab2:
        if st.session_state.get("replication_mode"):
            st.info("🔬 **Replication Mode:** Prompt and schema editing is disabled. The prompts and schemas used in the original analysis are archived in the Reproducibility tab.")
        else:
            st.subheader("Editable Prompts & Auto-Generated Schemas")
            st.caption("Edit prompts to customize cluster labeling and theme merging. The default prompts are loaded by default.")

            prompt_mode = st.radio("Prompt Mode", ["🔒 Default", "✏️ Custom"], horizontal=True, key="td_prompt_mode")

            if prompt_mode == "🔒 Default":
                st.session_state["td_label_prompt"] = DEFAULT_CLUSTER_LABEL_PROMPT
                st.session_state["td_merge_prompt"] = DEFAULT_THEME_MERGE_PROMPT
                st.session_state["td_label_schema"] = DEFAULT_LABEL_SCHEMA
                st.session_state["td_merge_schema"] = DEFAULT_MERGE_SCHEMA
                st.info("Using default prompts and schemas.")

                with st.expander("👁️ Preview Cluster Label Prompt"):
                    st.code(DEFAULT_CLUSTER_LABEL_PROMPT, language="text")
                with st.expander("👁️ Preview Theme Merge Prompt"):
                    st.code(DEFAULT_THEME_MERGE_PROMPT, language="text")
            else:
                col_p1, col_p2 = st.columns(2)

                with col_p1:
                    st.markdown("#### Cluster Labeling Prompt")
                    label_prompt = st.text_area(
                        "Label Prompt",
                        value=st.session_state.get("td_label_prompt", DEFAULT_CLUSTER_LABEL_PROMPT),
                        height=300,
                        key="td_label_prompt_editor"
                    )
                    st.session_state["td_label_prompt"] = label_prompt

                    if st.button("🧠 Generate Label Schema", key="td_gen_label_schema"):
                        if not base_url:
                            st.error("❌ API Base URL required.")
                        else:
                            with st.spinner("Generating schema..."):
                                client = get_api_client(base_url, api_key)
                                schema = generate_schema_for_prompt(client, model_name, label_prompt)
                                if schema:
                                    st.session_state["td_label_schema"] = schema
                                    st.success("✅ Label schema generated!")
                                else:
                                    st.error("❌ Schema generation failed.")

                    if "td_label_schema" in st.session_state:
                        with st.expander("👁️ Preview Label Schema"):
                            st.json(st.session_state["td_label_schema"])

                with col_p2:
                    st.markdown("#### Theme Merging Prompt")
                    merge_prompt = st.text_area(
                        "Merge Prompt",
                        value=st.session_state.get("td_merge_prompt", DEFAULT_THEME_MERGE_PROMPT),
                        height=300,
                        key="td_merge_prompt_editor"
                    )
                    st.session_state["td_merge_prompt"] = merge_prompt

                    if st.button("🧠 Generate Merge Schema", key="td_gen_merge_schema"):
                        if not base_url:
                            st.error("❌ API Base URL required.")
                        else:
                            with st.spinner("Generating schema..."):
                                client = get_api_client(base_url, api_key)
                                schema = generate_schema_for_prompt(client, model_name, merge_prompt)
                                if schema:
                                    st.session_state["td_merge_schema"] = schema
                                    st.success("✅ Merge schema generated!")
                                else:
                                    st.error("❌ Schema generation failed.")

                    if "td_merge_schema" in st.session_state:
                        with st.expander("👁️ Preview Merge Schema"):
                            st.json(st.session_state["td_merge_schema"])

    # =========================================================================
    # TAB 3: RUN DISCOVERY
    # =========================================================================
    with tab3:
        if st.session_state.get("replication_mode"):
            # Replication mode: skip run logic entirely, go straight to results display
            if "td_results" not in st.session_state:
                st.warning("⚠️ Replication data not loaded. Check Tab 1 or toggle Replication Mode off and on.")
                return
        else:
            # Normal mode: require data, sections, and API config
            if "td_data" not in st.session_state:
                st.info("ℹ️ Load data in Tab 1 first.")
                return
            if not selected_sections:
                st.warning("⚠️ Select at least one section in Tab 1.")
                return
            if not base_url:
                st.warning("⚠️ Configure an API endpoint in Tab 1.")
                return

        # Only show Run button and processing logic in normal mode
        if not st.session_state.get("replication_mode"):
            label_schema = st.session_state.get("td_label_schema", DEFAULT_LABEL_SCHEMA)
            merge_schema = st.session_state.get("td_merge_schema", DEFAULT_MERGE_SCHEMA)
            label_prompt_template = st.session_state.get("td_label_prompt", DEFAULT_CLUSTER_LABEL_PROMPT)
            merge_prompt_template = st.session_state.get("td_merge_prompt", DEFAULT_THEME_MERGE_PROMPT)

            if st.button("🚀 Run Topic Discovery", type="primary", key="td_run_btn"):
                client = get_api_client(base_url, api_key)
                all_results = {}
                total_sections = len(selected_sections)
                section_progress = st.progress(0, text="Starting...")

                for sec_idx, section_key in enumerate(selected_sections):
                    # NO SORTING — preserve original JSON order for deterministic embeddings
                    texts = extract_texts_from_data(st.session_state["td_data"], section_key)
                    section_progress.progress(
                        sec_idx / total_sections,
                        text=f"Processing: {section_key.replace('_', ' ').title()} ({len(texts)} segments)"
                    )

                    if len(texts) < 10:
                        st.warning(f"⚠️ Skipping {section_key}: only {len(texts)} texts (minimum 10 required).")
                        continue

                    with st.spinner(f"📊 Computing embeddings for {section_key}..."):
                        embeddings = compute_embeddings(texts, embedding_model)

                    with st.spinner(f"🔢 Finding optimal k for {section_key}..."):
                        optimal_k, sil_score = auto_kmeans(embeddings, max_clusters, min_clusters, random_state)
                        st.info(f"**{section_key}**: Optimal k={optimal_k}, Silhouette={sil_score:.3f}")

                    kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=random_state)
                    cluster_ids = kmeans.fit_predict(embeddings)
                    df_texts = pd.DataFrame({"text": texts, "cluster_id": cluster_ids})

                    np.random.seed(random_state)
                    samples = {}
                    for cid in sorted(df_texts.cluster_id.unique()):
                        cluster_texts = df_texts[df_texts.cluster_id == cid]["text"].values
                        n_sample = min(examples_per_cluster, len(cluster_texts))
                        samples[cid] = list(np.random.choice(cluster_texts, n_sample, replace=False))

                    # ---- LLM: Label each cluster individually ----
                    cluster_summaries = []
                    label_progress = st.progress(0, text=f"Labeling clusters for {section_key}...")
                    
                    for i, (cid, ex) in enumerate(samples.items()):
                        label_progress.progress(
                            (i + 1) / len(samples), 
                            text=f"Labeling cluster {cid}/{optimal_k}..."
                        )
                        listed = "\n".join(f"{j+1}. {t}" for j, t in enumerate(ex))
                        
                        # EXACT prompt formatting
                        prompt = label_prompt_template.format(
                            n_examples=len(ex),
                            examples=listed,
                            cluster_id=cid
                        )
                        
                        result = llm_json_call(
                            client, model_name, prompt, label_schema,
                            system_msg="You are an expert research analyst specialized in the use of large language models in language education.",
                            temperature=temperature
                        )
                        
                        if result:
                            result["cluster_id"] = cid
                            cluster_summaries.append(result)
                        else:
                            cluster_summaries.append({
                                "cluster_id": cid,
                                "label": f"Cluster {cid}",
                                "description": "LLM labeling failed for this cluster.",
                                "key_terms": []
                            })
                    
                    label_progress.empty()

                    cluster_df = pd.DataFrame(cluster_summaries)
                    label_progress.empty()

                    # ---- Reorganisation across clusters  ----
                    with st.spinner(f"🔗 Merging themes for {section_key}..."):
                        # EXACT block formatting
                        block = "\n".join(
                            f"{r['cluster_id']}. {r['label']}: {r['description']}"
                            for r in cluster_summaries
                        )
                        merge_prompt = merge_prompt_template.format(
                            section_name=section_key,
                            block=block
                        )
                        merged = llm_json_call(
                            client, model_name, merge_prompt, merge_schema,
                            system_msg="You are an expert research analyst specialized in the use of large language models in language education.",
                            temperature=temperature
                        )

                        if merged and isinstance(merged, list):
                            cluster_map = {}
                            for theme in merged:
                                for cid in theme.get("cluster_ids", []):
                                    cluster_map[cid] = theme.get("theme_label", "Unassigned")
                            df_texts["theme"] = df_texts["cluster_id"].map(cluster_map).fillna("Unassigned")
                        else:
                            st.warning(f"⚠️ Theme merging failed for {section_key}. Using cluster labels as themes.")
                            df_texts["theme"] = df_texts["cluster_id"].apply(lambda x: f"Cluster {x}")
                            merged = []

                    save_section_results(section_key, df_texts, cluster_df, merged)

                    # ---- AUTO-SAVE SECTION METADATA (safety net) ----
                    try:
                        meta_path = os.path.join(AUTO_SAVE_DIR, f"{section_key}_metadata.json")
                        with open(meta_path, "w", encoding="utf-8") as mf:
                            json.dump({
                                "section": section_key,
                                "n_texts": len(texts),
                                "optimal_k": optimal_k,
                                "silhouette": round(sil_score, 3),
                                "n_themes": len(merged) if merged else optimal_k,
                                "timestamp": datetime.now().isoformat(),
                            }, mf, indent=2, ensure_ascii=False)
                    except Exception:
                        pass

                    all_results[section_key] = {
                        "n_texts": len(texts),
                        "optimal_k": optimal_k,
                        "silhouette": round(sil_score, 3),
                        "n_themes": len(merged) if merged else optimal_k,
                        "df_texts": df_texts,
                        "cluster_summaries": cluster_df,
                        "merged_themes": merged,
                    }

                section_progress.progress(1.0, text="✅ All sections processed!")
                st.session_state["td_results"] = all_results
                st.success(f"✅ Topic discovery complete for {len(all_results)} sections!")
                st.rerun()

        # Display results
        if "td_results" in st.session_state:
            results = st.session_state["td_results"]
            st.divider()
            st.subheader(f"📊 Results ({len(results)} sections)")

            for sec_key, sec_data in results.items():
                with st.expander(f"🔹 {sec_key.replace('_', ' ').title()} — {sec_data['n_texts']} texts, {sec_data['optimal_k']} clusters, {sec_data['n_themes']} themes"):
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("Text Segments", sec_data["n_texts"])
                    metric_cols[1].metric("Optimal k", sec_data["optimal_k"])
                    metric_cols[2].metric("Silhouette", f"{sec_data['silhouette']:.3f}")
                    metric_cols[3].metric("Unified Themes", sec_data["n_themes"])

                    st.markdown("**Cluster Summaries:**")
                    display_cols = ["cluster_id", "label", "description", "key_terms"]
                    display_cols = [c for c in display_cols if c in sec_data["cluster_summaries"].columns]
                    st.dataframe(sec_data["cluster_summaries"][display_cols], use_container_width=True, hide_index=True)

                    if sec_data["merged_themes"]:
                        st.markdown("**Merged Themes:**")
                        themes_df = pd.DataFrame(sec_data["merged_themes"])
                        # Ensure consistent column order for display
                        theme_display_cols = ["theme_label", "cluster_ids", "summary"]
                        theme_display_cols = [c for c in theme_display_cols if c in themes_df.columns]
                        if theme_display_cols:
                            st.dataframe(themes_df[theme_display_cols], use_container_width=True, hide_index=True)
                        else:
                            st.dataframe(themes_df, use_container_width=True, hide_index=True)
            st.divider()
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                # Include all section CSVs
                for sec_key in results:
                    sec_dir = os.path.join(TOPIC_OUTPUT_DIR, sec_key)
                    for fname in ["texts_with_themes.csv", "cluster_summaries.csv", "themes_merged.csv"]:
                        fpath = os.path.join(sec_dir, fname)
                        if os.path.exists(fpath):
                            zf.write(fpath, os.path.join(sec_key, fname))
                # Include auto-save metadata
                for meta_file in Path(AUTO_SAVE_DIR).glob("*_metadata.json"):
                    zf.write(str(meta_file), os.path.join("auto_save", meta_file.name))
            zip_buffer.seek(0)
            st.download_button(
                label="💾 Download All Results (ZIP)",
                data=zip_buffer,
                file_name="topic_discovery_results.zip",
                mime="application/zip",
                type="primary",
                key="td_download_zip"
            )
            st.caption(f"💾 Results also saved to `{TOPIC_OUTPUT_DIR}` | Metadata in `{AUTO_SAVE_DIR}`")

    # =========================================================================
    # TAB 4: REPRODUCIBILITY
    # =========================================================================
    with tab4:
        st.subheader("🔬 Reproducibility & Accessibility Information")

        repro = {
            "stage": "topic_discovery",
            "timestamp": datetime.now().isoformat(),
            "audit_trail": {
                "embedding_model": st.session_state.get("td_embedding_model", "N/A"),
                "api_service_provider": st.session_state.get("td_provider", "N/A"),
                "api_base_url": st.session_state.get("td_base_url", "N/A"),
                "llm_model": st.session_state.get("td_model_name", st.session_state.get("td_model_name_manual", "N/A")),
                "output_directory": TOPIC_OUTPUT_DIR,
                "auto_save_directory": AUTO_SAVE_DIR,
            },
            "clustering_params": {
                "min_k": st.session_state.get("td_min_k", 10),
                "max_k": st.session_state.get("td_max_k", 30),
                "k_search_step": 5,
                "examples_per_cluster": st.session_state.get("td_examples", 20),
                "random_seed": st.session_state.get("td_seed", 42),
                "temperature": st.session_state.get("td_temperature", 0.0),
                "kmeans_n_init": 10,
                "embedding_normalization": True,
            },
            "prompt_mode": st.session_state.get("td_prompt_mode", "Default (replication)"),
            "cluster_label_prompt": st.session_state.get("td_label_prompt", DEFAULT_CLUSTER_LABEL_PROMPT),
            "theme_merge_prompt": st.session_state.get("td_merge_prompt", DEFAULT_THEME_MERGE_PROMPT),
            "cluster_label_schema": st.session_state.get("td_label_schema", DEFAULT_LABEL_SCHEMA),
            "theme_merge_schema": st.session_state.get("td_merge_schema", DEFAULT_MERGE_SCHEMA),
            "sections_analyzed": list(st.session_state.get("td_results", {}).keys()),
            "section_results_summary": {
                k: {
                    "n_texts": v["n_texts"],
                    "optimal_k": v["optimal_k"],
                    "silhouette": v["silhouette"],
                    "n_themes": v["n_themes"],
                }
                for k, v in st.session_state.get("td_results", {}).items()
            },
            "libraries": {
                "sentence_transformers": HAS_EMBEDDINGS,
                "scikit_learn": HAS_SKLEARN,
                "openai_client": HAS_OPENAI_CLIENT,
            },
        }

        st.json(repro)
        st.code(json.dumps(repro, indent=2), language="json")

        st.divider()

        # Reproducibility Checklist
        st.markdown("#### ✅ Reproducibility Checklist")
        st.markdown(f"""
        - [x] Exact prompts and JSON schemas archived above
        - [x] Embedding model, clustering params, and random seed recorded
        - [x] LLM provider, model, and temperature recorded
        - [x] Per-section results auto-saved to `{TOPIC_OUTPUT_DIR}`
        - [x] Section metadata auto-saved to `{AUTO_SAVE_DIR}`
        - [x] Silhouette search step fixed at 5
        - [ ] Document any manual interventions or prompt adjustments made during discovery
        """)

        st.divider()

        # Accessibility Alternatives
        st.markdown("#### ♿ Accessibility & Open-Source Notes")
        is_cloud = bool(os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("STREAMLIT_CLOUD"))
        if is_cloud:
            st.markdown("""
            | Scenario | Alternative |
            |---|---|
            | **No API budget / rate limits** | Install and run this app locally to use free local models (Ollama, LM Studio). The hosted version supports cloud APIs only. |
            | **Faster embeddings on large corpora** | Switch to `all-MiniLM-L6-v2` (lighter, ~5x faster than mpnet) |
            | **Schema portability** | JSON schemas are model-agnostic; export via this tab and reuse with any provider |
            | **Non-programmers** | Downloaded CSVs can be reviewed in Excel; theme labels and cluster assignments are human-readable |
            """)
        else:
            st.markdown("""
            | Scenario | Alternative |
            |---|---|
            | **No API budget / rate limits** | Use local inference with Ollama/LMStudio + Qwen3, Gemma3, Mistral 7B, etc. |
            | **Faster embeddings on large corpora** | Switch to `all-MiniLM-L6-v2` (lighter, ~5x faster than mpnet) |
            | **Schema portability** | JSON schemas are model-agnostic; export via this tab and reuse with any provider |
            | **Non-programmers** | Downloaded CSVs can be reviewed in Excel; theme labels and cluster assignments are human-readable |
            """)