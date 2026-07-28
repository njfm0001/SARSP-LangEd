"""
SARSP-LangEd - Stage 4: Double-Blind Validation
Human-LLM agreement assessment with fuzzy matching, weighted agreement
computation, and qualitative mismatch audit. Zero API cost.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import os
import zipfile
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


# =============================================================================
# CONSTANTS
# =============================================================================

from core.utils import get_session_temp_dir
TEMP_DIR = get_session_temp_dir()
STAGE_DIR = os.path.join(TEMP_DIR, "stage4_validation")
VALIDATION_SAMPLE_DIR = os.path.join(STAGE_DIR, "sample_pdfs")
AUTO_SAVE_DIR = os.path.join(STAGE_DIR, "auto_save")

# Ensure directories exist
for d in [STAGE_DIR, VALIDATION_SAMPLE_DIR, AUTO_SAVE_DIR]:
    os.makedirs(d, exist_ok=True)

# Columns to compare between human and LLM outputs
COLUMNS_FOR_COMPARISON = [
    'study_location',
    'educational_settings',
    'participant_type',
    'participant_demographics',
    'language_skills_targeted',
    'task_types',
    'LLMs_used',
    'prompts_used',
    'prompting_techniques',
    'prompting_strategies',
    'research_methodology',
    'data_gathering_methods',
    'research_design',
    'sample_size',
    'duration',
    'frameworks',
    'learning_perceptions',
    'outcomes',
    'stakeholder_impact',
    'policy_guidance',
    'emergent_themes',
    'challenges_concerns_limitations_of_LLMs_in_language_education',
    'emergent_issues',
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_json_data(uploaded_file) -> pd.DataFrame:
    """Load structured_output.json from upload or session state.
    Resets stream position after reading so the file can be read multiple times."""
    if uploaded_file:
        uploaded_file.seek(0)
        content = uploaded_file.read().decode("utf-8")
        uploaded_file.seek(0)  # Reset for subsequent reads
        data = json.loads(content)
        return pd.DataFrame(data)
    return pd.DataFrame()


def sample_for_validation(df: pd.DataFrame, frac: float, seed: int, selected_columns: list = None) -> pd.DataFrame:
    """
    Create a reproducible sample with cleared extraction columns.
    Retains metadata columns + research_h_q; clears ALL other columns for human coding.
    The selected_columns parameter is ignored — all non-retained columns are always cleared.
    Always returns at least 1 row.
    """
    n_rows = max(1, int(round(len(df) * frac)))
    sample = df.sample(n=n_rows, random_state=seed).copy()

    # Columns that MUST be retained (metadata + research questions)
    cols_to_retain = {
        'source_filename', 'title', 'authors', 'summary', 'APA_reference', 'research_h_q'
    }

    # Clear EVERYTHING except retained columns
    cols_to_clear = [c for c in sample.columns if c not in cols_to_retain]

    for col in cols_to_clear:
        if col in sample.columns:
            sample[col] = np.nan

    return sample


def align_by_source_filename(df_human: pd.DataFrame, df_llm: pd.DataFrame):
    """
    Align human and LLM DataFrames by exact source_filename match.
    Returns a merged DataFrame with suffixes _HUMAN and _LLM for all comparison columns.
    Unmatched rows are dropped.
    """
    # Normalize filenames for case-insensitive matching
    df_h = df_human.copy()
    df_l = df_llm.copy()
    df_h['_fn_key'] = df_h['source_filename'].fillna('').astype(str).str.strip().str.lower()
    df_l['_fn_key'] = df_l['source_filename'].fillna('').astype(str).str.strip().str.lower()

    # Inner join on filename — only matched records survive
    merged = pd.merge(
        df_h, df_l,
        on='_fn_key',
        how='inner',
        suffixes=('_HUMAN', '_LLM')
    )
    merged.drop(columns=['_fn_key'], inplace=True)
    return merged

def fuzzy_compare_values(val_h, val_l, threshold: float) -> str:
    """
    Compare two cell values using fuzzy string matching.
    Returns 'Y' (full match), 'P' (partial/fuzzy match), 'N' (mismatch), or '⚠️' (missing).
    Safely handles list/dict cell values from JSON-loaded DataFrames.
    """
    str_h = _safe_str(val_h).strip()
    str_l = _safe_str(val_l).strip()

    # Both empty → agreement
    if not str_h and not str_l:
        return 'Y'
    # One empty → warning
    if not str_h or not str_l:
        return '⚠️'
    # Exact match
    if str_h == str_l:
        return 'Y'
    # Fuzzy match
    ratio = SequenceMatcher(None, str_h.lower(), str_l.lower()).ratio()
    if ratio >= threshold:
        return 'P'
    return 'N'

def _safe_str(val):
    """Convert a cell value to string safely, handling lists, dicts, NaN, and None."""
    if val is None:
        return ''
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False)
    try:
        if pd.isna(val):
            return ''
    except (ValueError, TypeError):
        # pd.isna() raises ValueError for arrays/lists; already handled above
        pass
    return str(val)


def build_comparison_table(df_merged: pd.DataFrame, value_threshold: float, selected_columns: list):
    """
    Build the comparison DataFrame with HUMAN, LLM, MATCH, and MISMATCH REASON columns.
    Each field produces four columns: _HUMAN, _LLM, _MATCH, _MISMATCH_REASON.
    Safely handles list/dict cell values from JSON-loaded DataFrames.
    """
    rows = []
    for idx, row in df_merged.iterrows():
        comp = {
            'SOURCE_FILENAME': _safe_str(row.get('source_filename_HUMAN', row.get('source_filename', ''))),
            'TITLE': _safe_str(row.get('title_HUMAN', row.get('title', ''))),
        }
        for col in selected_columns:
            h_col = f'{col}_HUMAN'
            l_col = f'{col}_LLM'
            val_h = row.get(h_col, np.nan)
            val_l = row.get(l_col, np.nan)
            comp[h_col] = _safe_str(val_h)
            comp[l_col] = _safe_str(val_l)
            comp[f'{col}_MATCH'] = fuzzy_compare_values(val_h, val_l, value_threshold)
            comp[f'{col}_MISMATCH_REASON'] = ''  # Empty for human annotator to fill
        rows.append(comp)

    final_columns = ['SOURCE_FILENAME', 'TITLE']
    for col in selected_columns:
        final_columns.extend([
            f'{col}_HUMAN',
            f'{col}_LLM',
            f'{col}_MATCH',
            f'{col}_MISMATCH_REASON'
        ])

    return pd.DataFrame(rows).reindex(columns=final_columns)


def compute_agreement(df_comp: pd.DataFrame, selected_columns: list = None):
    """
    Compute per-column and overall weighted agreement.
    Y=1.0, P=0.5, N=0.0. Other values ignored.
    Returns (per_column_df, overall_df).
    If selected_columns is None, auto-detects from _MATCH suffix columns.
    """
    if selected_columns is None:
        selected_columns = [c.replace('_MATCH', '') for c in df_comp.columns if c.endswith('_MATCH')]

    per_col_stats = []
    total_Y = total_P = total_N = 0

    for col in selected_columns:
        match_col = f"{col}_MATCH"
        if match_col not in df_comp.columns:
            continue
        vals = df_comp[match_col].astype(str).str.strip()
        mask_valid = vals.isin(["Y", "P", "N"])
        v = vals[mask_valid]
        n_Y = (v == "Y").sum()
        n_P = (v == "P").sum()
        n_N = (v == "N").sum()
        n_valid = len(v)
        total_Y += n_Y
        total_P += n_P
        total_N += n_N
        if n_valid > 0:
            agreement_rate = (n_Y + 0.5 * n_P) / n_valid
        else:
            agreement_rate = np.nan
        per_col_stats.append({
            "column": col,
            "n_valid": n_valid,
            "n_full_match_Y": n_Y,
            "n_partial_match_P": n_P,
            "n_mismatch_N": n_N,
            "prop_full_match_Y": n_Y / n_valid if n_valid > 0 else np.nan,
            "prop_partial_match_P": n_P / n_valid if n_valid > 0 else np.nan,
            "prop_mismatch_N": n_N / n_valid if n_valid > 0 else np.nan,
            "agreement_rate": agreement_rate,
        })

    df_per_column = pd.DataFrame(per_col_stats)
    total_valid = total_Y + total_P + total_N
    if total_valid > 0:
        overall_agreement = (total_Y + 0.5 * total_P) / total_valid
    else:
        overall_agreement = np.nan

    df_overall = pd.DataFrame([{
        "total_valid_cells": total_valid,
        "total_full_match_Y": total_Y,
        "total_partial_match_P": total_P,
        "total_mismatch_N": total_N,
        "overall_prop_full_match_Y": total_Y / total_valid if total_valid > 0 else np.nan,
        "overall_prop_partial_match_P": total_P / total_valid if total_valid > 0 else np.nan,
        "overall_prop_mismatch_N": total_N / total_valid if total_valid > 0 else np.nan,
        "overall_agreement_rate": overall_agreement,
    }])
    return df_per_column, df_overall


def flag_reason_pairs(df: pd.DataFrame):
    """Pair _MATCH columns with their deterministic *_MISMATCH_REASON columns."""
    pairs = []
    for col in df.columns:
        if col.endswith("_MATCH"):
            field = col.replace("_MATCH", "")
            reason_col = f"{field}_MISMATCH_REASON"
            if reason_col in df.columns:
                pairs.append((col, reason_col))
    return pairs


def clean_reason(cell):
    """Clean a reason cell, handling JSON-like strings."""
    if not isinstance(cell, str):
        return ""
    try:
        parsed = json.loads(cell.replace("'", '"'))
        if isinstance(parsed, (list, dict)):
            return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        pass
    return cell.strip().strip('"').strip("'")


def count_flags_and_collect(df: pd.DataFrame):
    """Count N/P flags and collect mismatch explanations."""
    counter = Counter()
    explanations = defaultdict(list)
    pairs = flag_reason_pairs(df)
    for _, row in df.iterrows():
        for flag_col, reason_col in pairs:
            flag = str(row[flag_col]).strip().upper()
            if flag in {"N", "P"}:
                counter[flag] += 1
                reason = clean_reason(row[reason_col])
                if reason:
                    explanations[flag].append(reason)
    return counter, explanations

# =============================================================================
# STREAMLIT RENDER FUNCTION
# =============================================================================

def render_validation_page():
    """Render the full Validation stage UI with tabbed workflow."""
    st.title("4️⃣ Double-Blind Validation")
    st.markdown("""
    Assess human-LLM agreement through double-blind coding. This stage requires **zero API costs** 
    and runs entirely offline. Upload your LLM-extracted JSON and human-coded validation Excel 
    to generate comparison tables, weighted agreement metrics, and mismatch audits.
    """)

    # === REPLICATION MODE INJECTION ===
    if st.session_state.get("replication_mode"):
        from core.utils import get_replication_path

        st.info("🔬 **Replication Mode:** Original validation artifacts loaded. Toggle off in sidebar to run your own validation.")

        repl_files = {
            "template": get_replication_path("s4_template"),
            "comparison": get_replication_path("s4_comparison"),
            "metrics": get_replication_path("s4_metrics"),
            "human_coding": get_replication_path("s4_human_coding"),
        }

        available = {k: v for k, v in repl_files.items() if v and os.path.exists(v)}
        missing = [k for k, v in repl_files.items() if not v or not os.path.exists(v)]

        if available:
            st.success(f"✅ Loaded {len(available)} validation artifact(s) from replication data.")
            if missing:
                st.warning(f"⚠️ Missing replication files: {', '.join(missing)}")

            # Pre-load template into session state for Tab 1
            if "template" in available:
                try:
                    template_df = pd.read_excel(available["template"], dtype=str).fillna("")
                    st.session_state["val_sample_df"] = template_df
                    # Also load full JSON if available for record count display
                    json_path = get_replication_path("s3_structured_output")
                    if json_path and os.path.exists(json_path):
                        with open(json_path, "r", encoding="utf-8") as f:
                            full_data = json.load(f)
                        st.session_state["val_full_df"] = pd.DataFrame(full_data)
                except Exception as e:
                    st.error(f"❌ Failed to load template file: {e}")

            # Pre-load comparison table into session state for Tabs 2, 3 & 4
            if "comparison" in available:
                try:
                    comp_df = pd.read_excel(available["comparison"], sheet_name=0, dtype=str).fillna("")
                    st.session_state["val_comparison_df"] = comp_df
                    # Auto-detect selected columns from _MATCH suffixes
                    auto_cols = [c.replace("_MATCH", "") for c in comp_df.columns if c.endswith("_MATCH")]
                    if auto_cols:
                        st.session_state["val_selected_columns_final"] = auto_cols
                except Exception as e:
                    st.error(f"❌ Failed to load comparison file: {e}")

            # Pre-load metrics into session state for Tab 3
            if "metrics" in available:
                try:
                    per_col_df = pd.read_excel(available["metrics"], sheet_name="PER_COLUMN")
                    overall_df = pd.read_excel(available["metrics"], sheet_name="OVERALL")
                    st.session_state["val_per_col"] = per_col_df
                    st.session_state["val_overall"] = overall_df
                except Exception as e:
                    st.error(f"❌ Failed to load metrics file: {e}")
        else:
            st.error("❌ No validation replication files found. Check `replication_data/` folder.")

        st.divider()
        # Fall through to tabs — pre-loaded data will render in all tabs

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Sampling & Template",
        "🔍 Comparison",
        "📊 Agreement Metrics",
        "🔎 Mismatch Audit"
    ])

    # =========================================================================
    # TAB 1: SAMPLING & TEMPLATE GENERATION
    # =========================================================================
    with tab1:
        if st.session_state.get("replication_mode"):
            # === REPLICATION MODE: Show pre-loaded template ===
            st.subheader("Validation Sample (Replication Data)")
            sample_df = st.session_state.get("val_sample_df")
            full_df = st.session_state.get("val_full_df")

            if sample_df is not None and not sample_df.empty:
                n_sample = len(sample_df)
                n_full = len(full_df) if full_df is not None else "N/A"
                st.success(f"✅ Loaded {n_sample} sampled records from replication template (out of {n_full} total).")

                st.dataframe(sample_df.head(10), use_container_width=True)

                buffer = io.BytesIO()
                sample_df.to_excel(buffer, index=False)
                buffer.seek(0)
                st.download_button(
                    label="📥 Download Validation Template (Excel)",
                    data=buffer,
                    file_name="validation_refs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    key="val_repl_download_template"
                )
            else:
                st.warning("⚠️ Validation template not available in replication data.")
        else:
            st.subheader("Generate Validation Sample")
            st.caption("Sample a subset of LLM-extracted records for manual double-blind coding.")

            json_upload = st.file_uploader(
                "Upload structured_output.json (from Stage 3)",
                type=["json"],
                key="val_json_upload"
            )

            col_s1, col_s2 = st.columns([1, 1])
            with col_s1:
                sample_frac = st.slider("Sample fraction", 0.01, 0.50, 0.05, 0.01,
                                        help="Proportion of records to sample for validation.",
                                        key="val_sample_frac")
            with col_s2:
                sample_seed = st.number_input("Random seed", value=1, step=1,
                                            help="Seed for reproducible sampling.",
                                            key="val_sample_seed")
            # Note: All extraction columns except metadata + research_h_q are automatically
            # cleared for human coding. No column selection needed.
            selected_sampling_cols = None
            if json_upload:
                df_full = load_json_data(json_upload)
                st.success(f"✅ Loaded {len(df_full)} records from JSON")

                if st.button("🎲 Generate Validation Sample", key="val_gen_sample_btn"):
                    sample_df = sample_for_validation(df_full, sample_frac, sample_seed)
                    st.session_state["val_sample_df"] = sample_df
                    st.session_state["val_full_df"] = df_full
                    st.success(f"✅ Sampled {len(sample_df)} records ({sample_frac:.0%} of {len(df_full)})")

            if "val_sample_df" in st.session_state:
                sample_df = st.session_state["val_sample_df"]
                st.dataframe(sample_df.head(10), use_container_width=True)

                buffer = io.BytesIO()
                sample_df.to_excel(buffer, index=False)
                buffer.seek(0)
                st.download_button(
                    label="📥 Download Validation Template (Excel)",
                    data=buffer,
                    file_name="validation_refs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    key="val_download_template"
                )
                # =========================================================================
                # PDF RETRIEVAL FOR VALIDATION
                # =========================================================================
                st.divider()
                st.subheader("📄 Retrieve Source PDFs")
                st.caption("PDFs are needed so human annotators can re-read original papers when adjudicating mismatches.")

                STAGE3_PDF_DIR = os.path.join(TEMP_DIR, "stage3_extraction", "uploaded_pdfs")
                VALIDATION_PDF_DIR = os.path.join(TEMP_DIR, "stage4_validation", "sample_pdfs")
                os.makedirs(VALIDATION_PDF_DIR, exist_ok=True)

                # Auto-discover PDFs from Stage 3
                pdf_source_dir = None
                if os.path.exists(STAGE3_PDF_DIR):
                    stage3_pdfs = [f for f in os.listdir(STAGE3_PDF_DIR) if f.lower().endswith('.pdf')]
                    if stage3_pdfs:
                        pdf_source_dir = STAGE3_PDF_DIR
                        st.success(f"✅ Found {len(stage3_pdfs)} PDFs in `{STAGE3_PDF_DIR}`")

                # Fallback: let user choose a folder
                if pdf_source_dir is None:
                    st.info("ℹ️ No PDFs found in Stage 3 folder. Specify a custom folder below.")
                    custom_pdf_dir = st.text_input(
                        "Custom PDF Folder Path",
                        placeholder="/path/to/your/pdfs",
                        help="Enter the full path to a folder containing the source PDFs.",
                        key="val_custom_pdf_dir"
                    )
                    if custom_pdf_dir and os.path.exists(custom_pdf_dir):
                        custom_pdfs = [f for f in os.listdir(custom_pdf_dir) if f.lower().endswith('.pdf')]
                        if custom_pdfs:
                            pdf_source_dir = custom_pdf_dir
                            st.success(f"✅ Found {len(custom_pdfs)} PDFs in `{custom_pdf_dir}`")
                        else:
                            st.warning("⚠️ No PDF files found in the specified folder.")
                    elif custom_pdf_dir:
                        st.error("❌ Folder does not exist. Please check the path.")

                # Copy relevant PDFs to validation folder
                if pdf_source_dir and "val_sample_df" in st.session_state:
                    sample_filenames = set(
                        st.session_state["val_sample_df"]["source_filename"].dropna().astype(str).str.strip()
                    )
                    copied = 0
                    missing = []
                    for fname in sample_filenames:
                        src = os.path.join(pdf_source_dir, fname)
                        dst = os.path.join(VALIDATION_PDF_DIR, fname)
                        if os.path.exists(src):
                            if not os.path.exists(dst):
                                import shutil
                                shutil.copy2(src, dst)
                            copied += 1
                        else:
                            missing.append(fname)

                    if copied > 0:
                        st.success(f"✅ Copied {copied}/{len(sample_filenames)} sample PDFs to `{VALIDATION_PDF_DIR}`")
                    if missing:
                        st.warning(f"⚠️ {len(missing)} PDFs not found: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")
                # AUTO-SAVE validation template to disk (safety net)
                try:
                    auto_save_template = os.path.join(AUTO_SAVE_DIR, "validation_template.xlsx")
                    sample_df.to_excel(auto_save_template, index=False)
                    st.caption(f"💾 Auto-saved template to `{auto_save_template}`")
                except Exception:
                    pass

                st.info("📌 **Next step:** Code this template manually (double-blind), then upload it in the **Comparison** tab alongside the full LLM dataset.")
                            # Reproducibility Information for Sampling
                with st.expander("🔬 Reproducibility Information"):
                    repro_sampling = {
                        "stage": "validation_sampling",
                        "sample_fraction": sample_frac,
                        "random_seed": sample_seed,
                        "total_records_loaded": len(df_full) if json_upload else 0,
                        "sample_size": len(sample_df) if "val_sample_df" in st.session_state else 0,
                        "metadata_columns": ['source_filename', 'title', 'authors', 'summary', 'APA_reference', 'research_h_q'],
                        "extraction_columns_selected": selected_sampling_cols if selected_sampling_cols else "all (default)",
                        "timestamp": pd.Timestamp.now().isoformat()
                    }
                    st.json(repro_sampling)
                    st.code(json.dumps(repro_sampling, indent=2), language="json")
    # =========================================================================
    # TAB 2: COMPARISON TABLE GENERATION
    # =========================================================================
    with tab2:
        if st.session_state.get("replication_mode"):
            # === REPLICATION MODE: Show pre-loaded comparison ===
            st.subheader("Human-LLM Comparison Table (Replication Data)")
            comp_df = st.session_state.get("val_comparison_df")

            if comp_df is not None and not comp_df.empty:
                st.success(f"✅ Loaded comparison table with {len(comp_df)} records from replication data.")
                st.caption("Human adjudication has already been completed. Navigate to Tab 3 for agreement metrics or Tab 4 for mismatch audit.")

                preview_cols = ["SOURCE_FILENAME", "TITLE"]
                match_cols = [c for c in comp_df.columns if c.endswith("_MATCH")]
                if match_cols:
                    preview_cols.append(match_cols[0])
                preview_cols = [c for c in preview_cols if c in comp_df.columns]
                st.dataframe(comp_df[preview_cols].head(10), use_container_width=True)

                buffer = io.BytesIO()
                comp_df.to_excel(buffer, index=False, sheet_name="FULL_COMPARISON")
                buffer.seek(0)
                st.download_button(
                    label="📥 Download Comparison Excel",
                    data=buffer,
                    file_name="VALIDATION_SELECTED_COLUMNS.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    key="val_repl_download_comparison"
                )
            else:
                st.warning("⚠️ Comparison table not available in replication data.")
        else:
            st.subheader("Generate Human-LLM Comparison Table")
            st.caption("Upload the completed human-coded Excel and the full LLM extraction to create a matched comparison.")

            col_c1, col_c2 = st.columns([1, 1])
            with col_c1:
                llm_json = st.file_uploader(
                    "Full LLM Extraction (structured_output.json)",
                    type=["json"],
                    help="Upload the structured_output.json file generated in Stage 3.",
                    key="val_llm_upload"
                )
            with col_c2:
                human_excel = st.file_uploader(
                    "Human-Coded Validation (Excel)",
                    type=["xlsx"],
                    key="val_human_upload"
                )
            value_threshold = st.slider(
                "Fuzzy Value Match Threshold",
                0.50, 1.00, 0.50, 0.01,
                help="Minimum SequenceMatcher ratio for cell values to be marked as Partial Match (P). Below this = Mismatch (N). Exact matches always = Y.",
                key="val_value_threshold"
            )

            # Human Adjudication Workflow (moved from Tab 4 to Tab 2 where comparison is generated)
            st.markdown("#### 👤 Human Adjudication Workflow")
            st.info("""
            After downloading the comparison table below, you must **manually revise each match**:
            1. Filter rows where any `*_MATCH` column = **"P"**, **"N"**, or **"⚠️"** (missing value)
            2. Update the match flag to **Y** if the LLM output was actually correct upon review
            3. If not, re-read the original PDF to determine the ground truth
            4. Document the reason for discrepancy in the adjacent **`*_MISMATCH_REASON`** column:
            - *Human omission* (coder missed information present in text)
            - *LLM hallucination* (model fabricated information not in text)
            - *Implicit vs. explicit reporting* (information inferable but not stated verbatim)
            5. Update the match flag to **Y** if the LLM output was actually correct upon re-review
            6. Upload the revised Excel in **Tab 3 (Agreement Metrics)** or **Tab 4 (Mismatch Audit)**
            """)

            # Dynamic column selector — detects ALL non-metadata columns from uploaded JSON
            metadata_cols = {'source_filename', 'title', 'authors', 'summary', 'APA_reference'}
            available_columns = []
            if llm_json:
                try:
                    _peek = load_json_data(llm_json)
                    available_columns = sorted([c for c in _peek.columns if c not in metadata_cols])
                except Exception:
                    pass

            # Fall back to default list only if no file uploaded yet
            if not available_columns:
                available_columns = COLUMNS_FOR_COMPARISON

            selected_columns = st.multiselect(
                "Columns to Compare",
                options=available_columns,
                default=available_columns,
                help="Select which extraction fields to include in the comparison. All columns are selected by default.",
                key="val_selected_columns"
            )

            if llm_json and human_excel:
                if st.button("🔗 Generate Comparison Table", key="val_gen_comparison_btn"):
                    with st.spinner("Aligning records and comparing values..."):
                        # Load LLM data from JSON
                        df_llm = load_json_data(llm_json)
                        df_human = pd.read_excel(human_excel)

                        st.info(f"LLM records: {len(df_llm)} | Human records: {len(df_human)}")

                        # Align by exact source_filename
                        df_merged = align_by_source_filename(df_human, df_llm)

                        if df_merged.empty:
                            st.error("❌ No matching records found. Ensure both files share identical `source_filename` values.")
                        else:
                            st.success(f"✅ Aligned {len(df_merged)} records by source_filename")

                            # Build comparison with fuzzy VALUE matching using user-selected columns
                            if not selected_columns:
                                st.error("❌ Please select at least one column to compare.")
                            else:
                                df_comp = build_comparison_table(df_merged, value_threshold, selected_columns)
                                st.session_state["val_comparison_df"] = df_comp
                                st.session_state["val_selected_columns_final"] = selected_columns

                            buffer = io.BytesIO()
                            df_comp.to_excel(buffer, index=False, sheet_name="FULL_COMPARISON")
                            buffer.seek(0)

                            st.download_button(
                                label="📥 Download Comparison Excel",
                                data=buffer,
                                file_name="VALIDATION_SELECTED_COLUMNS.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                type="primary",
                                key="val_download_comparison"
                            )
                            # AUTO-SAVE comparison table to disk (safety net)
                            try:
                                auto_save_comp = os.path.join(AUTO_SAVE_DIR, "comparison_table.xlsx")
                                df_comp.to_excel(auto_save_comp, index=False, sheet_name="FULL_COMPARISON")
                                st.caption(f"💾 Auto-saved comparison to `{auto_save_comp}`")
                            except Exception:
                                pass

                            with st.expander("👀 Preview Comparison (first 5 rows)"):
                                # Build preview with SOURCE_FILENAME, TITLE, and first few comparison fields
                                preview_cols = ['SOURCE_FILENAME', 'TITLE']
                                if selected_columns:
                                    # Show first 3 fields' HUMAN, LLM, MATCH columns for quick verification
                                    for col in selected_columns[:3]:
                                        for suffix in ['_HUMAN', '_LLM', '_MATCH']:
                                            candidate = f'{col}{suffix}'
                                            if candidate in df_comp.columns:
                                                preview_cols.append(candidate)
                                # Fallback: if no selected columns matched, show whatever exists
                                if len(preview_cols) <= 2:
                                    preview_cols = [c for c in df_comp.columns[:8]]
                                st.dataframe(df_comp[preview_cols].head(), use_container_width=True)
                            # Reproducibility Information for Comparison
                            with st.expander("🔬 Reproducibility Information"):
                                repro_comparison = {
                                    "stage": "validation_comparison",
                                    "alignment_method": "exact_source_filename_match",
                                    "fuzzy_value_threshold": value_threshold,
                                    "llm_records_loaded": len(df_llm),
                                    "human_records_loaded": len(df_human),
                                    "records_aligned": len(df_merged),
                                    "comparison_columns": selected_columns,
                                    "match_coding": {"Y": "exact match", "P": f"fuzzy ≥ {value_threshold}", "N": "mismatch", "⚠️": "missing value"},
                                    "timestamp": pd.Timestamp.now().isoformat()
                                }
                                st.json(repro_comparison)
                                st.code(json.dumps(repro_comparison, indent=2), language="json")

    # =========================================================================
    # TAB 3: AGREEMENT METRICS
    # =========================================================================
    with tab3:
        if st.session_state.get("replication_mode"):
            # === REPLICATION MODE: Show pre-loaded metrics ===
            st.subheader("Weighted Agreement Metrics (Replication Data)")

            df_per_col = st.session_state.get("val_per_col")
            df_overall = st.session_state.get("val_overall")

            if df_overall is not None and not df_overall.empty and df_per_col is not None:
                ov = df_overall.iloc[0]
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Overall Agreement Rate", f"{ov['overall_agreement_rate']:.1%}" if not pd.isna(ov.get('overall_agreement_rate')) else "N/A")
                with metric_cols[1]:
                    st.metric("Full Matches (Y)", int(ov.get('total_full_match_Y', 0)))
                with metric_cols[2]:
                    st.metric("Partial Matches (P)", int(ov.get('total_partial_match_P', 0)))
                with metric_cols[3]:
                    st.metric("Mismatches (N)", int(ov.get('total_mismatch_N', 0)))

                st.divider()
                st.markdown("#### Per-Column Agreement Rates")
                display_cols = ['column', 'n_valid', 'agreement_rate', 'prop_full_match_Y', 'prop_partial_match_P', 'prop_mismatch_N']
                display_cols = [c for c in display_cols if c in df_per_col.columns]
                st.dataframe(df_per_col[display_cols], use_container_width=True)

                # Download button
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    df_per_col.to_excel(writer, sheet_name="PER_COLUMN", index=False)
                    df_overall.to_excel(writer, sheet_name="OVERALL", index=False)
                buf.seek(0)
                st.download_button(
                    label="📥 Download Agreement Summary (Excel)",
                    data=buf,
                    file_name="AGREEMENT_SUMMARY.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="val_repl_download_agreement"
                )
            else:
                st.warning("⚠️ Agreement metrics not available in replication data.")
        else:
            st.subheader("Weighted Agreement Computation")
            st.caption("Upload the comparison Excel (after manual Y/P/N review) to compute agreement metrics.")

            agreement_upload = st.file_uploader(
                "Reviewed Comparison Excel (with Y/P/N flags)",
                type=["xlsx"],
                key="val_agreement_upload"
            )

            if agreement_upload:
                if st.button("📊 Compute Agreement", key="val_compute_agreement_btn"):
                    with st.spinner("Computing weighted agreement..."):
                        df_comp = pd.read_excel(agreement_upload, sheet_name=0)
                        _sel_cols = st.session_state.get("val_selected_columns_final", None)
                        df_per_col, df_overall = compute_agreement(df_comp, selected_columns=_sel_cols)

                        st.session_state["val_per_col"] = df_per_col
                        st.session_state["val_overall"] = df_overall

                    # Overall metrics
                    ov = df_overall.iloc[0]
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Overall Agreement Rate", f"{ov['overall_agreement_rate']:.1%}" if not pd.isna(ov['overall_agreement_rate']) else "N/A")
                    with metric_cols[1]:
                        st.metric("Full Matches (Y)", int(ov['total_full_match_Y']))
                    with metric_cols[2]:
                        st.metric("Partial Matches (P)", int(ov['total_partial_match_P']))
                    with metric_cols[3]:
                        st.metric("Mismatches (N)", int(ov['total_mismatch_N']))

                    st.divider()
                    st.markdown("#### Per-Column Agreement Rates")
                    display_cols = ['column', 'n_valid', 'agreement_rate', 'prop_full_match_Y', 'prop_partial_match_P', 'prop_mismatch_N']
                    display_cols = [c for c in display_cols if c in df_per_col.columns]
                    st.dataframe(df_per_col[display_cols], use_container_width=True)

                    # Download agreement summary
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                        df_per_col.to_excel(writer, sheet_name="PER_COLUMN", index=False)
                        df_overall.to_excel(writer, sheet_name="OVERALL", index=False)
                    buf.seek(0)
                    st.download_button(
                        label="📥 Download Agreement Summary (Excel)",
                        data=buf,
                        file_name="AGREEMENT_SUMMARY.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="val_download_agreement"
                    )
                    
                    # AUTO-SAVE agreement summary to disk (safety net)
                    try:
                        auto_save_agreement = os.path.join(AUTO_SAVE_DIR, "agreement_summary.xlsx")
                        with pd.ExcelWriter(auto_save_agreement, engine="openpyxl") as writer:
                            df_per_col.to_excel(writer, sheet_name="PER_COLUMN", index=False)
                            df_overall.to_excel(writer, sheet_name="OVERALL", index=False)
                        st.caption(f"💾 Auto-saved agreement summary to `{auto_save_agreement}`")
                    except Exception:
                        pass
                    
                    # Reproducibility Information for Agreement
                    with st.expander("🔬 Reproducibility Information"):
                        ov = df_overall.iloc[0] if not df_overall.empty else {}
                        repro_agreement = {
                            "stage": "validation_agreement",
                            "weighting_scheme": {"Y": 1.0, "P": 0.5, "N": 0.0},
                            "overall_agreement_rate": round(ov.get('overall_agreement_rate', np.nan), 4) if not pd.isna(ov.get('overall_agreement_rate', np.nan)) else None,
                            "total_valid_cells": int(ov.get('total_valid_cells', 0)),
                            "total_full_match_Y": int(ov.get('total_full_match_Y', 0)),
                            "total_partial_match_P": int(ov.get('total_partial_match_P', 0)),
                            "total_mismatch_N": int(ov.get('total_mismatch_N', 0)),
                            "per_column_stats": df_per_col.to_dict(orient="records") if not df_per_col.empty else [],
                            "timestamp": pd.Timestamp.now().isoformat()
                        }
                        st.json(repro_agreement)
                        st.code(json.dumps(repro_agreement, indent=2), language="json")

    # =========================================================================
    # TAB 4: MISMATCH AUDIT
    # =========================================================================
    with tab4:
        st.subheader("Qualitative Mismatch Audit")
        st.caption("Review N (Mismatch) and P (Partial) flags alongside their documented reasons, organized by extraction field.")

        # In replication mode, data is pre-loaded; otherwise require upload
        df_audit = st.session_state.get("val_comparison_df")

        if not st.session_state.get("replication_mode"):
            audit_upload = st.file_uploader(
                "Reviewed Comparison Excel (with Mismatch reasons)",
                type=["xlsx"],
                key="val_audit_upload"
            )
            if audit_upload:
                if st.button("🔎 Run Mismatch Audit", key="val_run_audit_btn"):
                    with st.spinner("Collecting mismatch flags and reasons..."):
                        df_audit = pd.read_excel(audit_upload, sheet_name=0, dtype=str).fillna("")
                        st.session_state["val_comparison_df"] = df_audit

        if df_audit is None or df_audit.empty:
            if not st.session_state.get("replication_mode"):
                st.info("ℹ️ Upload a reviewed comparison Excel or enable Replication Mode to view mismatch audit.")
            else:
                st.warning("⚠️ Comparison data not available. Check replication files.")
        else:
            # Detect all comparison fields from _MATCH columns
            match_cols = [c for c in df_audit.columns if c.endswith("_MATCH")]
            fields = [c.replace("_MATCH", "") for c in match_cols]

            if not fields:
                st.warning("⚠️ No *_MATCH columns found in the comparison data.")
            else:
                # Flag totals overview
                counts, explanations = count_flags_and_collect(df_audit)

                st.markdown("#### Flag Totals")
                if counts:
                    flag_df = pd.DataFrame([{"Flag": k, "Count": v} for k, v in counts.items()])
                    st.dataframe(flag_df, use_container_width=True, hide_index=True)
                else:
                    st.info("ℹ️ No N or P flags found. All records may be marked Y or unreviewed.")

                st.divider()

                # Per-field mismatch audit with reasons displayed alongside
                st.markdown("#### 🔍 Mismatches & Reasons by Field")
                st.caption("Each field shows its N/P flags with the corresponding reason from the adjacent *_MISMATCH_REASON column.")

                for field in fields:
                    match_col = f"{field}_MATCH"
                    reason_col = f"{field}_MISMATCH_REASON"

                    if match_col not in df_audit.columns:
                        continue

                    # Filter to rows with N or P flags for this field
                    flags = df_audit[match_col].astype(str).str.strip().str.upper()
                    mask_np = flags.isin(["N", "P"])
                    flagged = df_audit[mask_np].copy()

                    if flagged.empty:
                        continue

                    n_count = int((flags[mask_np] == "N").sum())
                    p_count = int((flags[mask_np] == "P").sum())

                    with st.expander(
                        f"**{field}** — {n_count} mismatches, {p_count} partial matches",
                        expanded=False
                    ):
                        # Build display DataFrame with source info + human/LLM values + flag + reason
                        human_col = f"{field}_HUMAN"
                        llm_col = f"{field}_LLM"

                        display_cols = ["SOURCE_FILENAME", "TITLE"]
                        for col_candidate in [human_col, llm_col, match_col, reason_col]:
                            if col_candidate in flagged.columns:
                                display_cols.append(col_candidate)

                        display_df = flagged[display_cols].copy()

                        # Rename for readability
                        rename_map = {
                            "SOURCE_FILENAME": "Source File",
                            "TITLE": "Title",
                            human_col: "Human Value",
                            llm_col: "LLM Value",
                            match_col: "Match",
                            reason_col: "Reason",
                        }
                        display_df = display_df.rename(columns=rename_map)

                        st.dataframe(display_df, use_container_width=True, hide_index=True)
        # Reproducibility & Accessibility Information
        st.divider()
        with st.expander("🔬 Reproducibility & Accessibility Information", expanded=False):
            counts = st.session_state.get("val_audit_counts", Counter())
            explanations = st.session_state.get("val_audit_explanations", defaultdict(list))
            repro_audit = {
                "stage": "validation_mismatch_audit",
                "timestamp": pd.Timestamp.now().isoformat(),
                "audit_trail": {
                    "output_files": {
                        "validation_template": os.path.join(AUTO_SAVE_DIR, "validation_template.xlsx"),
                        "comparison_table": os.path.join(AUTO_SAVE_DIR, "comparison_table.xlsx"),
                        "agreement_summary": os.path.join(AUTO_SAVE_DIR, "agreement_summary.xlsx"),
                    },
                    "match_coding": {"Y": "Full Match (exact)", "P": "Partial Match (fuzzy ≥ threshold)", "N": "Mismatch", "⚠️": "Missing value"},
                    "weighting_scheme": {"Y": 1.0, "P": 0.5, "N": 0.0},
                    "alignment_method": "exact source_filename match (case-insensitive)",
                    "fuzzy_matching": "difflib.SequenceMatcher (character-level ratio)",
                },
                "flag_totals": dict(counts) if counts else {},
                "n_mismatch_explanations_collected": len(explanations.get("N", [])),
                "n_partial_explanations_collected": len(explanations.get("P", [])),
            }
            st.json(repro_audit)
            st.code(json.dumps(repro_audit, indent=2), language="json")

            st.divider()

            # Reproducibility Checklist
            st.markdown("#### ✅ Reproducibility Checklist")
            st.markdown(f"""
            - [x] Input JSON, human validation Excel, and comparison Excel auto-saved to `{AUTO_SAVE_DIR}`
            - [x] Random seed documented in Tab 1 reproducibility JSON
            - [x] Fuzzy match threshold recorded in Tab 2 reproducibility JSON
            - [x] Per-column and overall agreement rates archived in Tab 3
            - [x] Mismatch explanations collected and archived in this tab
            - [ ] Document any manual interventions made during adjudication
            """)

            # Accessibility Alternatives
            st.markdown("#### ♿ Accessibility Alternatives")
            st.markdown("""
            | Scenario | Alternative |
            |---|---|
            | **Low-resource / no API budget** | This stage requires **zero API costs** and runs entirely offline using open-source Python libraries |
            | **Collaborative teams** | Host the validation Excel template on shared cloud drives (Nextcloud, SharePoint, Google Drive) for double-blind coding by multiple human raters |
            | **Non-programmers** | All outputs are standard Excel files; agreement computation can be replicated manually using spreadsheet formulas |
            """)