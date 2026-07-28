"""
SARSP-LangEd - Stage 1: Preprocessing
Deduplication, configurable quartile filtering, column unification,
and reproducibility documentation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import unidecode
from tqdm.auto import tqdm
import io
from pathlib import Path
from datetime import datetime
import os

from core.utils import get_session_temp_dir
TEMP_DIR = get_session_temp_dir()
# =============================================================================
# HELPER FUNCTIONS (from original script, unchanged logic)
# =============================================================================

def normalize_text(s):
    """Normalise text for matching: lowercase, remove special chars, handle ampersands."""
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = s.replace("&", "and")
    s = unidecode.unidecode(s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_issn(s):
    """Extract and clean ISSN numbers from a string."""
    if pd.isna(s):
        return []
    s = re.sub(r"[^\d;,]", "", str(s))
    parts = re.split(r"[;,]", s)
    return [p.strip().replace("-", "") for p in parts if p.strip()]

def normalize_doi(doi):
    """Clean and standardise DOI strings."""
    if pd.isna(doi):
        return ""
    doi = str(doi).lower().strip()
    return doi.replace("https://doi.org/", "")

def merge_values(val1, val2):
    """Merge two values from potentially different sources, preserving unique information."""
    if pd.isna(val1) and pd.isna(val2):
        return np.nan
    if pd.isna(val1):
        return val2
    if pd.isna(val2):
        return val1
    str1 = str(val1).strip()
    str2 = str(val2).strip()
    if str1 == str2:
        return val1
    if str1 in str2:
        return val2
    if str2 in str1:
        return val1
    combined_parts = []
    parts1 = [p.strip() for p in str1.split(';')]
    parts2 = [p.strip() for p in str2.split(';')]
    all_parts = parts1 + parts2
    seen = set()
    unique_parts = []
    for part in all_parts:
        if part and part not in seen:
            seen.add(part)
            unique_parts.append(part)
    if len(unique_parts) == 1:
        return unique_parts[0]
    else:
        return '; '.join(unique_parts)

def merge_rows(row1, row2):
    """Merge two DataFrame rows based on the merge_values function."""
    merged_row = row1.copy()
    for col in row2.index:
        if col in merged_row:
            merged_value = merge_values(merged_row[col], row2[col])
            merged_row[col] = merged_value
        else:
            merged_row[col] = row2[col]
    return merged_row

def unify_columns(df):
    """Merge synonymous columns and remove empty ones."""
    if "Title" in df.columns and "Article Title" in df.columns:
        df["Title"] = df["Title"].fillna(df["Article Title"])
        df.drop(columns=["Article Title"], inplace=True)
    if "Source title" in df.columns and "Source Title" in df.columns:
        df["Source Title"] = df["Source title"].fillna(df["Source Title"])
        df.drop(columns=["Source title"], inplace=True)
    if "ISSN" in df.columns and "eISSN" in df.columns:
        df['ISSN'] = df.apply(lambda row: '; '.join([x for x in [row.get('ISSN', ''), row.get('eISSN', '')] 
                                                    if pd.notna(x) and str(x).strip() != '' and str(x).lower() != 'nan']), 
                              axis=1)
        df.drop(columns=["eISSN"], inplace=True)
    df = df.loc[:, df.notna().any()]
    return df

def resolve_column(df, expected_names, default=None):
    """
    Case-insensitive column resolver for external database files.
    Returns the actual column name found in df, or default if none match.
    
    Parameters
    ----------
    df : pd.DataFrame
    expected_names : list of str
        Possible column name variants (e.g., ["ISSN", "Issn", "issn"])
    default : str or None
        Value to return if no match is found
    
    Returns
    -------
    str or None : The actual column name in df, or default
    """
    col_map = {c.lower().strip(): c for c in df.columns}
    for name in expected_names:
        if name.lower().strip() in col_map:
            return col_map[name.lower().strip()]
    return default

# =============================================================================
# STREAMLIT RENDER FUNCTION
# =============================================================================

def render_preprocessing_page():
    """Render the full Preprocessing stage UI with configurable quartile filtering."""
    st.title("1️⃣ Preprocessing & Quality Filtering")
    st.markdown("""
    Upload your **Web of Science** and **Scopus** exports along with **SJR** and **JCR** 
    journal ranking files. The pipeline will deduplicate, filter by your chosen quartile 
    thresholds, and produce a unified corpus ready for screening.
    """)
    
    # === REPLICATION MODE INJECTION ===
    wos_file = None
    scopus_file = None
    sjr_file = None
    jcr_file = None
    all_uploaded = False

    if st.session_state.get("replication_mode"):
        from core.utils import get_replication_path

        st.info("🔬 **Replication Mode:** Original experiment files loaded automatically. Toggle off in sidebar to use your own data.")

        repl_paths = {
            "wos": get_replication_path("s1_wos"),
            "scopus": get_replication_path("s1_scopus"),
            "sjr": get_replication_path("s1_sjr"),
            "jcr": get_replication_path("s1_jcr"),
        }

        missing = [k for k, v in repl_paths.items() if v is None]
        if missing:
            st.error(f"❌ Missing replication files: {', '.join(missing)}. Check `replication_data/` folder.")
        else:
            wos_file = repl_paths["wos"]
            scopus_file = repl_paths["scopus"]
            sjr_file = repl_paths["sjr"]
            jcr_file = repl_paths["jcr"]
            all_uploaded = True

            st.success("✅ All 4 Stage 1 replication files loaded and ready.")

            # Show expected outputs for verification
            final_refs = get_replication_path("s1_final_refs")
            removed_refs = get_replication_path("s1_removed_refs")
            manual_verified = get_replication_path("s1_manual_verification")

            with st.expander("📋 Expected Outputs (for verification)"):
                if final_refs:
                    try:
                        df_preview = pd.read_csv(final_refs, sep=";", nrows=5, dtype=str)
                        st.markdown(f"**Final filtered refs** (`{os.path.basename(final_refs)}`):")
                        st.dataframe(df_preview, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not preview final refs: {e}")
                if removed_refs:
                    st.markdown(f"**Removed refs:** `{os.path.basename(removed_refs)}`")
                if manual_verified:
                    st.markdown(f"**After manual book chapter verification:** `{os.path.basename(manual_verified)}`")

        st.divider()

    # --- File Upload Section (only when NOT in replication mode) ---
    if not st.session_state.get("replication_mode"):
        st.subheader("📁 Upload Database Exports")

        col1, col2 = st.columns(2)
        with col1:
            wos_file = st.file_uploader("Web of Science Export (.xls/.xlsx)", type=["xls", "xlsx"], key="wos_upload")
            sjr_file = st.file_uploader("SJR Journal Rankings (.csv)", type=["csv"], key="sjr_upload")
        with col2:
            scopus_file = st.file_uploader("Scopus Export (.csv)", type=["csv"], key="scopus_upload")
            jcr_file = st.file_uploader("JCR Impact Factors (.xlsx)", type=["xlsx"], key="jcr_upload")

        all_uploaded = all([wos_file, scopus_file, sjr_file, jcr_file])

        if not all_uploaded:
            st.warning("⚠️ Please upload all four files to proceed.")
            return  # Only returns when NOT in replication mode and files are missing

    # If we reach here, all_uploaded is guaranteed True (either via replication or upload)
    
    # =========================================================================
    # QUARTILE FILTERING CONFIGURATION PANEL
    # =========================================================================
    st.divider()
    with st.expander("⚙️ Quartile Filtering Criteria", expanded=True):
        st.caption(
            "Configure which journal quartiles to retain. Leave a selector empty "
            "to skip filtering for that database entirely. Articles must meet "
            "**all active** criteria to be kept."
        )
        
        q_col1, q_col2 = st.columns(2)
        
        quartile_options = ["Q1", "Q2", "Q3", "Q4"]
        
        with q_col1:
            sjr_quartiles = st.multiselect(
                "SJR (Scimago) Quartiles",
                options=quartile_options,
                default=["Q1", "Q2"],
                key="sjr_q_select",
                help="Select acceptable SJR quartiles. Leave empty to disable SJR filtering."
            )
        
        with q_col2:
            jcr_quartiles = st.multiselect(
                "JCR (Web of Science) Quartiles",
                options=quartile_options,
                default=["Q1", "Q2"],
                key="jcr_q_select",
                help="Select acceptable JCR quartiles. Leave empty to disable JCR filtering."
            )
        
        # Build human-readable summary of active criteria
        sjr_active = len(sjr_quartiles) > 0
        jcr_active = len(jcr_quartiles) > 0
        
        if sjr_active and jcr_active:
            criteria_summary = f"SJR ∈ {{{', '.join(sjr_quartiles)}}} **AND** JCR ∈ {{{', '.join(jcr_quartiles)}}}"
        elif sjr_active and not jcr_active:
            criteria_summary = f"SJR ∈ {{{', '.join(sjr_quartiles)}}} only (JCR filtering disabled)"
        elif not sjr_active and jcr_active:
            criteria_summary = f"JCR ∈ {{{', '.join(jcr_quartiles)}}} only (SJR filtering disabled)"
        else:
            criteria_summary = "**No quartile filtering** (all matched journals retained)"
        
        st.info(f"📋 **Active filter:** {criteria_summary}")
    
    st.divider()
    
    # --- Run Preprocessing Button ---
    run_disabled = not all_uploaded or st.session_state.get("preprocessing_complete", False)
    if st.button("🚀 Run Preprocessing Pipeline", type="primary", disabled=run_disabled):
        
        progress_bar = st.progress(0, text="Loading database exports...")
        status_text = st.empty()
        
        try:
            # ---- STEP 1: Load Data ----
            status_text.text("📂 Step 1/9: Loading database exports...")
            progress_bar.progress(10)
            
            wos = pd.read_excel(wos_file, dtype=str).fillna("")
            scopus = pd.read_csv(scopus_file, sep=',', dtype=str).fillna("")
            sjr = pd.read_csv(sjr_file, sep=';', dtype=str).fillna("")
            jcr = pd.read_excel(jcr_file, dtype=str).fillna("")
            
            st.metric("Loaded Records", f"WoS: {len(wos)} | Scopus: {len(scopus)} | SJR: {len(sjr)} | JCR: {len(jcr)}")
            
            # ---- STEP 2: Deduplicate ----
            status_text.text("🔄 Step 2/9: Deduplicating references...")
            progress_bar.progress(20)
            
            wos["DOI_norm"] = wos["DOI"].apply(normalize_doi)
            scopus["DOI_norm"] = scopus["DOI"].apply(normalize_doi)
            wos["Title_norm"] = wos.get("Article Title", "").apply(normalize_text)
            scopus["Title_norm"] = scopus.get("Title", "").apply(normalize_text)
            
            merged = pd.concat([wos, scopus], ignore_index=True)
            
            # --- IMPROVED DEDUPLICATION: Cross-reference DOI ↔ Title ---
            # Build a mapping from normalized titles to DOIs so that records
            # without DOI can still match records that have one for the same study.
            title_to_doi = {}
            for _, row in merged.iterrows():
                t = row.get("Title_norm", "")
                d = row.get("DOI_norm", "")
                if t and d:
                    title_to_doi[t] = d
            
            def make_merge_key(row):
                doi = row.get("DOI_norm", "")
                title = row.get("Title_norm", "")
                if doi:
                    return f"doi:{doi}"
                # No DOI: check if another record with the same title HAS a DOI
                if title and title in title_to_doi:
                    return f"doi:{title_to_doi[title]}"
                # Fall back to title-only key
                if title:
                    return f"title:{title}"
                return f"row:{row.name}"  # Last resort: unique per row
            
            merged['merge_key'] = merged.apply(make_merge_key, axis=1)
            grouped = merged.groupby('merge_key', dropna=False)
            merged_rows_list = []
            total_groups = len(grouped)
            
            for idx, (key, group_df) in enumerate(grouped):
                if len(group_df) == 1:
                    merged_rows_list.append(group_df.iloc[0])
                else:
                    current_merged_row = group_df.iloc[0]
                    for i in range(1, len(group_df)):
                        next_row = group_df.iloc[i]
                        current_merged_row = merge_rows(current_merged_row, next_row)
                    merged_rows_list.append(current_merged_row)
                
                if idx % 50 == 0:
                    progress_bar.progress(20 + int((idx / total_groups) * 15))
            
            merged = pd.DataFrame(merged_rows_list)
            merged.drop(columns=['merge_key'], inplace=True)
            
            st.metric("After Deduplication", f"{len(merged)} unique records")
            
            # ---- STEP 3: Split by Document Type ----
            status_text.text("📑 Step 3/9: Separating books from articles...")
            progress_bar.progress(40)
            
            book_types = ["Book chapter", "Book", "Article; Book Chapter", "Article; Book Chapter; Book chapter"]
            non_articles = merged[
                merged.get("Document Type", "").isin(book_types) |
                merged.get("Publication Type", "").isin(book_types)
            ]
            articles_to_check = merged.drop(non_articles.index).copy()
            
            # ---- STEP 4: Normalize Journal Names ----
            status_text.text("📰 Step 4/9: Normalizing journal names & ISSNs...")
            progress_bar.progress(45)
            
            # --- Resolve column names case-insensitively ---
            sjr_title_col = resolve_column(sjr, ["Title", "title", "TITLE", "Source Title", "Journal Title"])
            sjr_issn_col = resolve_column(sjr, ["ISSN", "Issn", "issn", "E-ISSN", "eissn"])
            sjr_quartile_col = resolve_column(sjr, ["SJR Best Quartile", "sjr best quartile", "Best Quartile", "Quartile"])
            
            jcr_name_col = resolve_column(jcr, ["Journal Name", "journal name", "JOURNAL NAME", "Full Journal Title"])
            jcr_abbr_col = resolve_column(jcr, ["Abbreviated Journal", "abbreviated journal", "ABBREVIATED JOURNAL", "Abbrev Title"])
            jcr_issn_col = resolve_column(jcr, ["ISSN", "Issn", "issn"])
            jcr_eissn_col = resolve_column(jcr, ["eISSN", "eissn", "E-ISSN", "EISSN"])
            jcr_quartile_col = resolve_column(jcr, ["JIF Quartile", "jif quartile", "JIF QUARTILE", "Quartile"])
            
            # Validate critical columns exist
            missing_cols = []
            if not sjr_title_col:
                missing_cols.append(f"SJR Title (searched: Title, Source Title)")
            if not sjr_issn_col:
                missing_cols.append(f"SJR ISSN (searched: ISSN, Issn)")
            if not sjr_quartile_col:
                missing_cols.append(f"SJR Quartile (searched: SJR Best Quartile, Best Quartile)")
            if not jcr_name_col:
                missing_cols.append(f"JCR Journal Name (searched: Journal Name, Full Journal Title)")
            if not jcr_quartile_col:
                missing_cols.append(f"JCR Quartile (searched: JIF Quartile, Quartile)")
            
            if missing_cols:
                raise KeyError(
                    f"Could not find required columns in uploaded files:\n"
                    f"• {'\n• '.join(missing_cols)}\n\n"
                    f"SJR columns found: {list(sjr.columns)}\n"
                    f"JCR columns found: {list(jcr.columns)}\n\n"
                    f"Please check your file format matches expected column names."
                )
            
            # Normalize journal names using resolved columns
            articles_to_check["Journal_norm_JCR"] = articles_to_check.get("Source Title", "").apply(normalize_text)
            articles_to_check["Journal_norm_Scopus"] = articles_to_check.get("Source title", "").apply(normalize_text)
            sjr["Journal_norm"] = sjr[sjr_title_col].apply(normalize_text)
            jcr["Journal_norm"] = jcr[jcr_name_col].apply(normalize_text)
            jcr["Abbrev_norm"] = jcr[jcr_abbr_col].apply(normalize_text) if jcr_abbr_col else ""
            
            # Build ISSN lists using resolved columns
            sjr["ISSN_list"] = sjr[sjr_issn_col].apply(normalize_issn)
            
            if jcr_eissn_col:
                jcr["ISSN_list"] = jcr[jcr_issn_col].apply(normalize_issn) + jcr[jcr_eissn_col].apply(normalize_issn)
            else:
                jcr["ISSN_list"] = jcr[jcr_issn_col].apply(normalize_issn)
            
            # Store resolved quartile column names for use in Step 5
            _sjr_quartile_col = sjr_quartile_col
            _jcr_quartile_col = jcr_quartile_col
            
            # ---- STEP 5: Journal Matching ----
            status_text.text("🔍 Step 5/9: Matching journals against SJR & JCR...")
            progress_bar.progress(50)
            
            def match_journal(row):
                journal_JCR = row["Journal_norm_JCR"]
                journal_Scopus = row["Journal_norm_Scopus"]
                issn_ref = normalize_issn(row.get("ISSN", ""))
                sjr_match = sjr[(sjr["Journal_norm"] == journal_Scopus) | (sjr["Journal_norm"] == journal_JCR)]
                if sjr_match.empty:
                    sjr_match = sjr[sjr["ISSN_list"].apply(lambda x: any(i in x for i in issn_ref))]
                jcr_match = jcr[(jcr["Journal_norm"] == journal_JCR) | (jcr["Abbrev_norm"] == journal_JCR) | (jcr["Journal_norm"] == journal_Scopus)]
                if jcr_match.empty:
                    jcr_match = jcr[jcr["ISSN_list"].apply(lambda x: any(i in x for i in issn_ref))]
                sjr_q = sjr_match[_sjr_quartile_col].iloc[0] if not sjr_match.empty else None
                jcr_q = jcr_match[_jcr_quartile_col].iloc[0] if not jcr_match.empty else None
                return pd.Series({"SJR_Q": sjr_q, "JCR_Q": jcr_q, "SJR_found": not sjr_match.empty, "JCR_found": not jcr_match.empty})
            
            match_results = articles_to_check.apply(match_journal, axis=1)
            articles_to_check = pd.concat([articles_to_check, match_results], axis=1)
            progress_bar.progress(70)
            
            # ---- STEP 6: Filter by User-Selected Quartiles ----
            status_text.text(f"✅ Step 6/9: Applying quartile filter ({criteria_summary})...")
            progress_bar.progress(75)
            
            _sjr_quartiles = [q.upper() for q in sjr_quartiles] if sjr_quartiles else []
            _jcr_quartiles = [q.upper() for q in jcr_quartiles] if jcr_quartiles else []
            _sjr_active = len(_sjr_quartiles) > 0
            _jcr_active = len(_jcr_quartiles) > 0
            
            def filter_article(row):
                keep = True
                reason = ""
                
                if not (row["SJR_found"] and row["JCR_found"]):
                    keep = False
                    reason = "Journal not found in DBs"
                else:
                    sjr_ok = True
                    jcr_ok = True
                    
                    if _sjr_active:
                        sjr_val = str(row["SJR_Q"]).upper().strip()
                        if sjr_val not in _sjr_quartiles:
                            sjr_ok = False
                    
                    if _jcr_active:
                        jcr_val = str(row["JCR_Q"]).upper().strip()
                        if jcr_val not in _jcr_quartiles:
                            jcr_ok = False
                    
                    if not sjr_ok and not jcr_ok:
                        keep = False
                        reason = "Below threshold in both SJR and JCR"
                    elif not sjr_ok:
                        keep = False
                        reason = f"Below threshold in SJR (not in {', '.join(_sjr_quartiles)})"
                    elif not jcr_ok:
                        keep = False
                        reason = f"Below threshold in JCR (not in {', '.join(_jcr_quartiles)})"
                
                row["Removal_reason"] = reason
                row["Keep"] = keep
                return row
            
            articles_to_check = articles_to_check.apply(filter_article, axis=1)
            removed = articles_to_check[articles_to_check["Keep"] == False].copy()
            kept = articles_to_check[articles_to_check["Keep"] == True].copy()
            final = pd.concat([kept, non_articles], ignore_index=True)
            
            # ---- STEP 7: Column Unification ----
            status_text.text("🧹 Step 7/9: Unifying columns...")
            progress_bar.progress(80)
            
            final = unify_columns(final)
            removed = unify_columns(removed)
            
            # ---- STEP 8: Select Requested Columns ----
            status_text.text("📋 Step 8/9: Selecting final schema columns...")
            progress_bar.progress(85)
            
            requested_cols = [
                "Authors", "Title", "Abstract", "Author Keywords", "Index Keywords", "Source Title",
                "Document Type", "Publisher", "DOI", "ISSN", "ISBN",
                "Book Authors", "Group Authors", "Book Group Authors", "Book Editors",
                "Book Series Title", "Book Series Subtitle", "Volume", "Issue",
                "Special Issue", "Start Page", "End Page", "Book DOI", "Early Access Date",
                "Publication Date", "Publication Year", "Times Cited, All Databases",
                "180 Day Usage Count", "Since 2013 Usage Count", "Indexed Date",
                "Cited by", "Link", "PubMed ID", "SJR_Q", "JCR_Q", "SJR_found", "JCR_found"
            ]
            for col in requested_cols:
                if col not in final.columns:
                    final[col] = np.nan
                if col not in removed.columns:
                    removed[col] = np.nan
            final = final[requested_cols]
            removed = removed[requested_cols + ["Removal_reason"]]
            
            # ---- STEP 9: Export ----
            status_text.text("💾 Step 9/9: Preparing downloads...")
            progress_bar.progress(95)
            
            # ---- AUTO-SAVE TO TEMP (safety net) ----
            auto_save_dir = os.path.join(TEMP_DIR, "stage1_preprocessing")
            os.makedirs(auto_save_dir, exist_ok=True)
            final.to_csv(os.path.join(auto_save_dir, "final_filtered_refs.csv"), index=False, sep=';')
            removed.to_csv(os.path.join(auto_save_dir, "removed_refs.csv"), index=False, sep=';')
            st.caption(f"💾 Auto-saved to `{auto_save_dir}`")
            # Store results AND filter criteria in session state
            st.session_state["preprocessing_final"] = final
            st.session_state["preprocessing_removed"] = removed
            st.session_state["preprocessing_criteria"] = criteria_summary
            st.session_state["preprocessing_sjr_quartiles"] = sjr_quartiles
            st.session_state["preprocessing_jcr_quartiles"] = jcr_quartiles
            st.session_state["preprocessing_loaded_counts"] = {
                "wos": len(wos), "scopus": len(scopus), "sjr": len(sjr), "jcr": len(jcr)
            }
            st.session_state["preprocessing_after_dedup"] = len(merged)
            st.session_state["preprocessing_complete"] = True
            
            progress_bar.progress(100, text="✅ Preprocessing complete!")
            status_text.empty()
            
            st.success(f"✅ **Preprocessing Complete!** Final corpus: **{len(final)}** records | Removed: **{len(removed)}** records")
            st.caption(f"Filter applied: {criteria_summary}")
            
            # Display summary statistics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Final Corpus Size", len(final))
            with col_b:
                st.metric("Removed Records", len(removed))
            with col_c:
                n_books = len(final[final["Document Type"].astype(str).str.contains("Book", na=False)])
                st.metric("Books/Chapters Included", n_books)
            
            # Preview data
            with st.expander("👀 Preview Final Corpus (first 20 records)"):
                st.dataframe(final.head(20), use_container_width=True)
            
            with st.expander("🗑️ Preview Removed Records (first 20 records)"):
                st.dataframe(removed.head(20), use_container_width=True)
            
            # Download buttons
            st.divider()
            st.subheader("📥 Download Results")
            
            dl_col1, dl_col2 = st.columns(2)
            
            final_csv = final.to_csv(index=False, sep=';').encode('utf-8')
            with dl_col1:
                st.download_button(
                    label="📄 Download Final Corpus (.csv)",
                    data=final_csv,
                    file_name="final_filtered_refs.csv",
                    mime="text/csv",
                    type="primary"
                )
            
            removed_csv = removed.to_csv(index=False, sep=';').encode('utf-8')
            with dl_col2:
                st.download_button(
                    label="🗑️ Download Removal Log (.csv)",
                    data=removed_csv,
                    file_name="removed_refs.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"❌ Error during preprocessing: {str(e)}")
            st.exception(e)
            progress_bar.empty()
            status_text.empty()
    
    # =========================================================================
    # RESULTS DISPLAY (if already completed in this session)
    # =========================================================================
    elif st.session_state.get("preprocessing_complete", False):
        saved_criteria = st.session_state.get("preprocessing_criteria", "Unknown")
        st.success(f"✅ Preprocessing already completed in this session. Filter: {saved_criteria}")
        final = st.session_state.get("preprocessing_final")
        removed = st.session_state.get("preprocessing_removed")
        
        if final is not None:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Final Corpus Size", len(final))
            with col_b:
                st.metric("Removed Records", len(removed) if removed is not None else "N/A")
            with col_c:
                n_books = len(final[final["Document Type"].astype(str).str.contains("Book", na=False)])
                st.metric("Books/Chapters Included", n_books)
            
            with st.expander("👀 Preview Final Corpus"):
                st.dataframe(final.head(20), use_container_width=True)
            
            st.divider()
            st.subheader("📥 Download Results")
            dl_col1, dl_col2 = st.columns(2)
            final_csv = final.to_csv(index=False, sep=';').encode('utf-8')
            with dl_col1:
                st.download_button("📄 Download Final Corpus (.csv)", final_csv, "final_filtered_refs.csv", "text/csv", type="primary")
            if removed is not None:
                removed_csv = removed.to_csv(index=False, sep=';').encode('utf-8')
                with dl_col2:
                    st.download_button("🗑️ Download Removal Log (.csv)", removed_csv, "removed_refs.csv", "text/csv")
    
    # =========================================================================
    # REPRODUCIBILITY & ACCESSIBILITY (always visible)
    # =========================================================================
    st.divider()
    with st.expander("🔬 Reproducibility & Accessibility Information", expanded=False):
        
        # --- Audit Trail ---
        st.markdown("#### 📋 Audit Trail")
        
        loaded_counts = st.session_state.get("preprocessing_loaded_counts", {})
        repro = {
            "stage": "preprocessing",
            "timestamp": datetime.now().isoformat(),
            "input_records": {
                "web_of_science": loaded_counts.get("wos", "N/A"),
                "scopus": loaded_counts.get("scopus", "N/A"),
                "sjr_rankings": loaded_counts.get("sjr", "N/A"),
                "jcr_impact_factors": loaded_counts.get("jcr", "N/A"),
            },
            "after_deduplication": st.session_state.get("preprocessing_after_dedup", "N/A"),
            "final_corpus_size": len(st.session_state["preprocessing_final"]) if "preprocessing_final" in st.session_state else "N/A",
            "removed_records": len(st.session_state["preprocessing_removed"]) if "preprocessing_removed" in st.session_state else "N/A",
            "filter_criteria": st.session_state.get("preprocessing_criteria", "Not yet run"),
            "sjr_quartiles_selected": st.session_state.get("preprocessing_sjr_quartiles", []),
            "jcr_quartiles_selected": st.session_state.get("preprocessing_jcr_quartiles", []),
            "output_schema_columns": [
                "Authors", "Title", "Abstract", "Author Keywords", "Index Keywords", "Source Title",
                "Document Type", "Publisher", "DOI", "ISSN", "ISBN",
                "Book Authors", "Group Authors", "Book Group Authors", "Book Editors",
                "Book Series Title", "Book Series Subtitle", "Volume", "Issue",
                "Special Issue", "Start Page", "End Page", "Book DOI", "Early Access Date",
                "Publication Date", "Publication Year", "Times Cited, All Databases",
                "180 Day Usage Count", "Since 2013 Usage Count", "Indexed Date",
                "Cited by", "Link", "PubMed ID", "SJR_Q", "JCR_Q", "SJR_found", "JCR_found"
            ],
            "deduplication_method": "DOI-normalized primary key; title-normalized fallback; row-level value merging for multi-source records",
            "journal_matching_method": "Normalized journal name + ISSN cross-reference against SJR and JCR databases",
        }
        
        st.json(repro)
        st.code(json.dumps(repro, indent=2), language="json")
        
        st.divider()
        
        # --- Manual Verification Reminder ---
        st.markdown("#### ⚠️ Post-Processing: Manual Verification of Book Chapters")
        st.warning("""
        The pipeline automatically separates book chapters from journal articles because 
        book chapters follow different indexing and quality-assessment conventions 
        (they are not indexed in JCR/SJR in the same way as journal articles).
        
        **Required manual step:** Book chapters included in the final corpus have NOT been 
        quality-filtered. You must manually verify them against the **Scholarly Publishers 
        Indicators (SPI)** database or an equivalent regional/national quality index for 
        book publishers before proceeding to screening.
        
        In the original case study, 115 book chapters were verified against the 2022 SPI 
        database for linguistics and philology, and 56 from non-indexed publishers were 
        manually excluded.
        """)
        
        st.divider()
        
        # --- Adaptation Instructions ---
        st.markdown("#### 🔧 Adaptation & Reproducibility Instructions")
        st.markdown("""
        To reproduce or adapt this pipeline for your own systematic review:
        
        1. **Update input files** — Point to your local WoS/Scopus exports and the most recent SJR/JCR/SPI database files
        2. **Apply quartile filtering** — Choose your own quartile thresholds for SJR and JCR, or disable filtering entirely
        """)