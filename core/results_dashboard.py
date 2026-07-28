"""
SARSP-LangEd - Stage 8: Results Dashboard & Reproducibility Package
Centralized reporting, cross-stage synthesis, and audit trail generation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import os
import zipfile
import time
from pathlib import Path
from datetime import datetime

from core.utils import get_session_temp_dir
TEMP_DIR = get_session_temp_dir()

# Stage-specific paths matching updated folder structure
STAGE1_DIR = os.path.join(TEMP_DIR, "stage1_preprocessing")
STAGE2_DIR = os.path.join(TEMP_DIR, "stage2_screening")
STAGE3_DIR = os.path.join(TEMP_DIR, "stage3_extraction")
STAGE4_DIR = os.path.join(TEMP_DIR, "stage4_validation")
STAGE5_DIR = os.path.join(TEMP_DIR, "stage5_normalization")
STAGE6_DIR = os.path.join(TEMP_DIR, "stage6_prompt_analysis")
STAGE7_DIR = os.path.join(TEMP_DIR, "stage7_topic_discovery")
STAGE8_DIR = os.path.join(TEMP_DIR, "stage8_results")

# Key artifact paths
PREPROCESSING_FINAL = os.path.join(STAGE1_DIR, "final_filtered_refs.csv")
SCREENING_AUTO_SAVE = os.path.join(STAGE2_DIR, "screening.xlsx")
EXTRACTION_OUTPUT = os.path.join(STAGE3_DIR, "structured_output.json")
VALIDATION_AUTO_SAVE = os.path.join(STAGE4_DIR, "auto_save", "agreement_summary.xlsx")
NORMALIZATION_OUTPUT = os.path.join(STAGE5_DIR, "auto_save", "normalized_output.json")
NORMALIZATION_LIGHT_OUTPUT = os.path.join(STAGE5_DIR, "auto_save", "normalized_output_light.json")
PROMPT_ANALYSIS_OUTPUT = os.path.join(STAGE6_DIR, "auto_save", "analyzed_prompts.csv")
TOPIC_OUTPUT_DIR = os.path.join(STAGE7_DIR, "results")
TOPIC_AUTO_SAVE_DIR = os.path.join(STAGE7_DIR, "auto_save")
NORMALIZATION_FIGURES_DIR = os.path.join(STAGE5_DIR, "figures")
PROMPT_FIGURES_DIR = os.path.join(STAGE6_DIR, "figures")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_json_safe(filepath):
    """Load JSON file safely, returning None if missing or invalid."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def load_session_results():
    """
    Aggregate results from disk artifacts (primary) and session state (fallback).
    In replication mode, reads from replication_data/ instead of session temp.
    Resilient to session loss / browser refresh.
    """
    from core.utils import REPLICATION_DIR

    is_repl = st.session_state.get("replication_mode", False)
    results = {}

    # === STAGE 1: PREPROCESSING ===
    if is_repl:
        # Check verified corpus first (canonical post-preprocessing artifact),
        # then fall back to pre-verification filtered refs
        repl_final_verified = os.path.join(REPLICATION_DIR, "stage1_final_filtered_after_manual_verification_book_chapters.csv")
        repl_final_raw = os.path.join(REPLICATION_DIR, "stage1_final_filtered_refs.csv")
        repl_final = repl_final_verified if os.path.exists(repl_final_verified) else repl_final_raw

        if os.path.exists(repl_final):
            try:
                with open(repl_final, "r", encoding="utf-8") as f:
                    n_records = sum(1 for _ in f) - 1
                label = "verified" if repl_final == repl_final_verified else "filtered"
                results["preprocessing"] = {"records": max(0, n_records), "status": f"✅ Complete (replication, {label})"}
            except Exception:
                results["preprocessing"] = {"records": 0, "status": "⚠️ Error reading replication file"}
        else:
            results["preprocessing"] = {"records": 0, "status": "⏳ Not in replication data"}
    elif os.path.exists(PREPROCESSING_FINAL):
        try:
            with open(PREPROCESSING_FINAL, "r", encoding="utf-8") as f:
                n_records = sum(1 for _ in f) - 1
            results["preprocessing"] = {"records": max(0, n_records), "status": "✅ Complete"}
        except Exception:
            results["preprocessing"] = {"records": 0, "status": "⚠️ Error reading file"}
    elif "preprocessing_final" in st.session_state:
        results["preprocessing"] = {
            "records": len(st.session_state["preprocessing_final"]),
            "status": "✅ Complete (session)"
        }
    else:
        results["preprocessing"] = {"records": 0, "status": "⏳ Not run"}

    # === STAGE 2: SCREENING ===
    if is_repl:
        repl_screening = os.path.join(REPLICATION_DIR, "stage2_screening_after_review.xlsx")
        if os.path.exists(repl_screening):
            try:
                screening_df = pd.read_excel(repl_screening)
                results["screening"] = {
                    "records": len(screening_df),
                    "excluded": int((screening_df.get("Final decision", "") == "EXCLUDE").sum()),
                    "disagreements": int((screening_df.get("Final decision", "") == "DISAGREE").sum()),
                    "status": "✅ Complete (replication)"
                }
            except Exception:
                results["screening"] = {"records": 0, "status": "⚠️ Error reading replication file"}
        else:
            results["screening"] = {"records": 0, "excluded": 0, "disagreements": 0, "status": "⏳ Not in replication data"}
    elif os.path.exists(SCREENING_AUTO_SAVE):
        try:
            screening_df = pd.read_excel(SCREENING_AUTO_SAVE)
            results["screening"] = {
                "records": len(screening_df),
                "excluded": int((screening_df.get("Final decision", "") == "EXCLUDE").sum()),
                "disagreements": int((screening_df.get("Final decision", "") == "DISAGREE").sum()),
                "status": "✅ Complete"
            }
        except Exception:
            results["screening"] = {"records": 0, "status": "⚠️ Error reading file"}
    elif "screening_complete" in st.session_state:
        screened_df = st.session_state.get("screening_df")
        results["screening"] = {
            "records": len(screened_df) if screened_df is not None else 0,
            "excluded": int((screened_df.get("Final decision", "") == "EXCLUDE").sum()) if screened_df is not None else 0,
            "disagreements": int((screened_df.get("Final decision", "") == "DISAGREE").sum()) if screened_df is not None else 0,
            "status": "✅ Complete (session)"
        }
    else:
        results["screening"] = {"records": 0, "excluded": 0, "disagreements": 0, "status": "⏳ Not run"}

    # === STAGE 3: EXTRACTION ===
    if is_repl:
        repl_extraction = os.path.join(REPLICATION_DIR, "stage3_structured_output.json")
        repl_data = load_json_safe(repl_extraction)
        if repl_data:
            results["extraction"] = {"records": len(repl_data), "status": "✅ Complete (replication)"}
        else:
            results["extraction"] = {"records": 0, "status": "⏳ Not in replication data"}
    else:
        extraction_data = load_json_safe(EXTRACTION_OUTPUT)
        if extraction_data:
            results["extraction"] = {"records": len(extraction_data), "status": "✅ Complete"}
        elif "extraction_results_count" in st.session_state:
            results["extraction"] = {
                "records": st.session_state["extraction_results_count"],
                "status": "✅ Complete (session)"
            }
        else:
            results["extraction"] = {"records": 0, "status": "⏳ Not run"}

    # === STAGE 4: VALIDATION ===
    if is_repl:
        repl_metrics = os.path.join(REPLICATION_DIR, "stage4_validation_metrics.xlsx")
        if os.path.exists(repl_metrics):
            try:
                val_df = pd.read_excel(repl_metrics, sheet_name="OVERALL")
                if not val_df.empty:
                    ov = val_df.iloc[0]
                    results["validation"] = {
                        "agreement_rate": ov.get("overall_agreement_rate", None),
                        "records_compared": int(ov.get("total_valid_cells", 0)),
                        "status": "✅ Complete (replication)"
                    }
                else:
                    results["validation"] = {"agreement_rate": None, "records_compared": 0, "status": "⚠️ Empty replication file"}
            except Exception:
                results["validation"] = {"agreement_rate": None, "records_compared": 0, "status": "⚠️ Error reading replication file"}
        else:
            results["validation"] = {"agreement_rate": None, "records_compared": 0, "status": "⏳ Not in replication data"}
    elif os.path.exists(VALIDATION_AUTO_SAVE):
        try:
            val_df = pd.read_excel(VALIDATION_AUTO_SAVE, sheet_name="OVERALL")
            if not val_df.empty:
                ov = val_df.iloc[0]
                results["validation"] = {
                    "agreement_rate": ov.get("overall_agreement_rate", None),
                    "records_compared": int(ov.get("total_valid_cells", 0)),
                    "status": "✅ Complete"
                }
            else:
                results["validation"] = {"agreement_rate": None, "records_compared": 0, "status": "⚠️ Empty file"}
        except Exception:
            results["validation"] = {"agreement_rate": None, "records_compared": 0, "status": "⚠️ Error reading file"}
    elif "val_overall" in st.session_state:
        ov = st.session_state["val_overall"].iloc[0] if not st.session_state["val_overall"].empty else {}
        results["validation"] = {
            "agreement_rate": ov.get("overall_agreement_rate", None),
            "records_compared": int(ov.get("total_valid_cells", 0)),
            "status": "✅ Complete (session)"
        }
    else:
        results["validation"] = {"agreement_rate": None, "records_compared": 0, "status": "⏳ Not run"}

    # === STAGE 5: NORMALIZATION ===
    if is_repl:
        repl_norm = os.path.join(REPLICATION_DIR, "stage5_normalized_output.json")
        repl_norm_data = load_json_safe(repl_norm)
        if repl_norm_data:
            results["normalization"] = {"records": len(repl_norm_data), "mode": "default (replication)", "status": "✅ Complete (replication)"}
        else:
            results["normalization"] = {"records": 0, "mode": "N/A", "status": "⏳ Not in replication data"}
    else:
        norm_data = load_json_safe(NORMALIZATION_OUTPUT) or load_json_safe(NORMALIZATION_LIGHT_OUTPUT)
        if norm_data:
            mode = "default" if os.path.exists(NORMALIZATION_OUTPUT) else "custom_light"
            results["normalization"] = {"records": len(norm_data), "mode": mode, "status": "✅ Complete"}
        elif "norm_result_df" in st.session_state:
            results["normalization"] = {
                "records": len(st.session_state["norm_result_df"]),
                "mode": st.session_state.get("norm_mode_used", "unknown"),
                "status": "✅ Complete (session)"
            }
        else:
            results["normalization"] = {"records": 0, "mode": "N/A", "status": "⏳ Not run"}

    # === STAGE 6: PROMPT ANALYSIS ===
    if is_repl:
        repl_prompts = os.path.join(REPLICATION_DIR, "stage6_analyzed_prompts.csv")
        if os.path.exists(repl_prompts):
            try:
                # Use pandas to correctly handle multiline CSV fields
                prompt_df = pd.read_csv(repl_prompts, dtype=str, usecols=[0])
                n_prompts = len(prompt_df)
                results["prompt_analysis"] = {"prompts_analyzed": n_prompts, "status": "✅ Complete (replication)"}
            except Exception:
                results["prompt_analysis"] = {"prompts_analyzed": 0, "status": "⚠️ Error reading replication file"}
        else:
            results["prompt_analysis"] = {"prompts_analyzed": 0, "status": "⏳ Not in replication data"}
    elif os.path.exists(PROMPT_ANALYSIS_OUTPUT):
        try:
            prompt_df = pd.read_csv(PROMPT_ANALYSIS_OUTPUT, dtype=str, usecols=[0])
            n_prompts = len(prompt_df)
            results["prompt_analysis"] = {"prompts_analyzed": n_prompts, "status": "✅ Complete"}
        except Exception:
            results["prompt_analysis"] = {"prompts_analyzed": 0, "status": "⚠️ Error reading file"}
    elif "pa_analyzed_df" in st.session_state:
        results["prompt_analysis"] = {
            "prompts_analyzed": len(st.session_state["pa_analyzed_df"]),
            "status": "✅ Complete (session)"
        }
    else:
        results["prompt_analysis"] = {"prompts_analyzed": 0, "status": "⏳ Not run"}

    # === STAGE 7: TOPIC DISCOVERY ===
    if is_repl:
        repl_topic_dir = os.path.join(REPLICATION_DIR, "stage7_topic_discovery")
        topic_sections = {}
        if os.path.exists(repl_topic_dir):
            for sec_dir in Path(repl_topic_dir).iterdir():
                if sec_dir.is_dir():
                    themes_csv = sec_dir / "themes_merged.csv"
                    texts_csv = sec_dir / "texts_with_themes.csv"
                    if texts_csv.exists():
                        try:
                            n_texts = sum(1 for _ in open(texts_csv, "r", encoding="utf-8")) - 1
                            n_themes = 0
                            if themes_csv.exists():
                                try:
                                    themes_df = pd.read_csv(themes_csv, dtype=str)
                                    n_themes = len(themes_df)
                                except Exception:
                                    pass
                            topic_sections[sec_dir.name] = {"n_texts": n_texts, "n_themes": n_themes}
                        except Exception:
                            continue
        if topic_sections:
            total_texts = sum(v.get("n_texts", 0) for v in topic_sections.values())
            total_themes = sum(v.get("n_themes", 0) for v in topic_sections.values())
            results["topic_discovery"] = {
                "sections": len(topic_sections),
                "total_texts": total_texts,
                "total_themes": total_themes,
                "status": "✅ Complete (replication)"
            }
        else:
            results["topic_discovery"] = {"sections": 0, "total_texts": 0, "total_themes": 0, "status": "⏳ Not in replication data"}
    else:
        topic_sections = {}
        if os.path.exists(TOPIC_AUTO_SAVE_DIR):
            for meta_file in Path(TOPIC_AUTO_SAVE_DIR).glob("*_metadata.json"):
                try:
                    with open(meta_file, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                    sec_key = meta.get("section", meta_file.stem.replace("_metadata", ""))
                    topic_sections[sec_key] = meta
                except Exception:
                    continue
        if topic_sections:
            total_texts = sum(v.get("n_texts", 0) for v in topic_sections.values())
            total_themes = sum(v.get("n_themes", 0) for v in topic_sections.values())
            results["topic_discovery"] = {
                "sections": len(topic_sections),
                "total_texts": total_texts,
                "total_themes": total_themes,
                "status": "✅ Complete"
            }
        elif "td_results" in st.session_state:
            td = st.session_state["td_results"]
            results["topic_discovery"] = {
                "sections": len(td),
                "total_texts": sum(v["n_texts"] for v in td.values()),
                "total_themes": sum(v["n_themes"] for v in td.values()),
                "status": "✅ Complete (session)"
            }
        else:
            results["topic_discovery"] = {"sections": 0, "total_texts": 0, "total_themes": 0, "status": "⏳ Not run"}

    return results

def build_reproducibility_package(results):
    """Create a ZIP archive containing all pipeline artifacts and metadata.
    In replication mode, bundles files from replication_data/."""
    from core.utils import REPLICATION_DIR

    is_repl = st.session_state.get("replication_mode", False)
    zip_buffer = io.BytesIO()

    manifest = {
        "pipeline": "SARSP-LangEd",
        "generated_at": datetime.now().isoformat(),
        "mode": "replication" if is_repl else "live",
        "stages_completed": [k for k, v in results.items() if v.get("status", "").startswith("✅")],
        "stage_summaries": results,
        "session_metadata": {
            "extraction_model": st.session_state.get("extraction_primary_model", "N/A"),
            "extraction_prompt_mode": st.session_state.get("extraction_prompt_mode", "N/A"),
            "normalization_mode": st.session_state.get("norm_mode_used", "N/A"),
            "topic_discovery_provider": st.session_state.get("td_provider", "N/A"),
            "topic_discovery_model": st.session_state.get("td_model_name", st.session_state.get("td_model_name_manual", "N/A")),
            "topic_embedding_model": st.session_state.get("td_embedding_model", "N/A"),
            "screening_provider": st.session_state.get("screening_provider_used", "N/A"),
        }
    }

    # Auto-save manifest to disk
    os.makedirs(STAGE8_DIR, exist_ok=True)
    manifest_path = os.path.join(STAGE8_DIR, "REPRODUCIBILITY_MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("REPRODUCIBILITY_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))

        if is_repl:
            # Bundle replication_data/ files
            repl_files = {
                "stage1_preprocessing/final_filtered_refs.csv": "stage1_final_filtered_refs.csv",
                "stage1_preprocessing/removed_refs.csv": "stage1_removed_refs.csv",
                "stage2_screening/screened_refs.xlsx": "stage2_screening_after_review.xlsx",
                "stage3_extraction/structured_output.json": "stage3_structured_output.json",
                "stage4_validation/agreement_summary.xlsx": "stage4_validation_metrics.xlsx",
                "stage4_validation/comparison_table.xlsx": "stage4_validation_comparison.xlsx",
                "stage5_normalization/normalized_output.json": "stage5_normalized_output.json",
                "stage6_prompt_analysis/analyzed_prompts.csv": "stage6_analyzed_prompts.csv",
            }
            for arcname, repl_filename in repl_files.items():
                src = os.path.join(REPLICATION_DIR, repl_filename)
                if os.path.exists(src):
                    zf.write(src, arcname)

            # Topic discovery subdirectories
            repl_topic_dir = os.path.join(REPLICATION_DIR, "stage7_topic_discovery")
            if os.path.exists(repl_topic_dir):
                for section_dir in Path(repl_topic_dir).iterdir():
                    if section_dir.is_dir():
                        for csv_file in section_dir.glob("*.csv"):
                            arcname = f"stage7_topic_discovery/{section_dir.name}/{csv_file.name}"
                            zf.write(str(csv_file), arcname)
        else:
            # Bundle live session temp files (original logic)
            if os.path.exists(PREPROCESSING_FINAL):
                zf.write(PREPROCESSING_FINAL, "stage1_preprocessing/final_filtered_refs.csv")
            removed_refs = os.path.join(STAGE1_DIR, "removed_refs.csv")
            if os.path.exists(removed_refs):
                zf.write(removed_refs, "stage1_preprocessing/removed_refs.csv")

            if os.path.exists(SCREENING_AUTO_SAVE):
                zf.write(SCREENING_AUTO_SAVE, "stage2_screening/screened_refs.xlsx")

            if os.path.exists(EXTRACTION_OUTPUT):
                zf.write(EXTRACTION_OUTPUT, "stage3_extraction/structured_output.json")

            if os.path.exists(VALIDATION_AUTO_SAVE):
                zf.write(VALIDATION_AUTO_SAVE, "stage4_validation/agreement_summary.xlsx")
            val_comparison = os.path.join(STAGE4_DIR, "auto_save", "comparison_table.xlsx")
            if os.path.exists(val_comparison):
                zf.write(val_comparison, "stage4_validation/comparison_table.xlsx")

            for norm_file in [NORMALIZATION_OUTPUT, NORMALIZATION_LIGHT_OUTPUT]:
                if os.path.exists(norm_file):
                    zf.write(norm_file, f"stage5_normalization/{os.path.basename(norm_file)}")
            if os.path.exists(NORMALIZATION_FIGURES_DIR):
                for fig_file in Path(NORMALIZATION_FIGURES_DIR).glob("*.png"):
                    zf.write(str(fig_file), f"figures/normalization/{fig_file.name}")

            if os.path.exists(PROMPT_ANALYSIS_OUTPUT):
                zf.write(PROMPT_ANALYSIS_OUTPUT, "stage6_prompt_analysis/analyzed_prompts.csv")
            prompt_meta = os.path.join(STAGE6_DIR, "auto_save", "analysis_metadata.json")
            if os.path.exists(prompt_meta):
                zf.write(prompt_meta, "stage6_prompt_analysis/analysis_metadata.json")
            if os.path.exists(PROMPT_FIGURES_DIR):
                for fig_file in Path(PROMPT_FIGURES_DIR).glob("*.png"):
                    zf.write(str(fig_file), f"figures/prompt_analysis/{fig_file.name}")

            if os.path.exists(TOPIC_OUTPUT_DIR):
                for section_dir in Path(TOPIC_OUTPUT_DIR).iterdir():
                    if section_dir.is_dir():
                        for csv_file in section_dir.glob("*.csv"):
                            arcname = f"stage7_topic_discovery/{section_dir.name}/{csv_file.name}"
                            zf.write(str(csv_file), arcname)
            if os.path.exists(TOPIC_AUTO_SAVE_DIR):
                for meta_file in Path(TOPIC_AUTO_SAVE_DIR).glob("*_metadata.json"):
                    zf.write(str(meta_file), f"stage7_topic_discovery/auto_save/{meta_file.name}")

    zip_buffer.seek(0)
    return zip_buffer, manifest


# =============================================================================
# STREAMLIT RENDER FUNCTION
# =============================================================================

def render_results_page():
    st.title("8️⃣ Results Dashboard & Reproducibility Package")
    st.markdown("""
    Centralized overview of all pipeline stages, key metrics, and a downloadable 
    reproducibility package containing all artifacts, metadata, and audit trails.
    """)
        # Show replication mode banner
    if st.session_state.get("replication_mode"):
        st.info("🔬 **Replication Mode:** Dashboard shows metrics from the original published experiment. Toggle off in sidebar to view live pipeline results.")

    tab1, tab2, tab3 = st.tabs([
        "📊 Pipeline Overview",
        "📈 Key Metrics",
        "📦 Reproducibility Package"
    ])

    results = load_session_results()

    # =========================================================================
    # TAB 1: PIPELINE OVERVIEW
    # =========================================================================
    with tab1:
        st.subheader("Pipeline Status Summary")

        status_rows = []
        stage_labels = {
            "preprocessing": "1️⃣ Preprocessing",
            "screening": "2️⃣ Screening",
            "extraction": "3️⃣ Extraction",
            "validation": "4️⃣ Validation",
            "normalization": "5️⃣ Normalization",
            "prompt_analysis": "6️⃣ Prompt Analysis",
            "topic_discovery": "7️⃣ Topic Discovery",
        }

        for key, label in stage_labels.items():
            r = results.get(key, {})
            status_rows.append({
                "Stage": label,
                "Status": r.get("status", "❓ Unknown"),
                "Records / Items": r.get("records", r.get("prompts_analyzed", r.get("total_texts", r.get("sections", "—")))),
                "Key Metric": (
                    f"Agreement: {r['agreement_rate']:.1%}" if r.get("agreement_rate") is not None
                    else f"Excluded: {r.get('excluded', '—')} | Disagree: {r.get('disagreements', '—')}" if key == "screening"
                    else f"Themes: {r.get('total_themes', '—')}" if key == "topic_discovery"
                    else f"Mode: {r.get('mode', '—')}" if key == "normalization"
                    else "—"
                )
            })

        status_df = pd.DataFrame(status_rows)
        st.dataframe(status_df, use_container_width=True, hide_index=True)

        # Progress indicator
        completed = sum(1 for v in results.values() if v.get("status", "").startswith("✅"))
        total = len(stage_labels)
        st.progress(completed / total, text=f"Pipeline Completion: {completed}/{total} stages")

    # =========================================================================
    # TAB 2: KEY METRICS
    # =========================================================================
    with tab2:
        st.subheader("Cross-Stage Key Metrics")

        metric_cols = st.columns(4)
        metric_cols[0].metric(
            "Studies Extracted",
            results["extraction"]["records"],
            delta=None
        )
        metric_cols[1].metric(
            "Screening Exclusions",
            results["screening"].get("excluded", 0),
            delta=None
        )
        val_rate = results["validation"].get("agreement_rate")
        metric_cols[2].metric(
            "Human-LLM Agreement",
            f"{val_rate:.1%}" if val_rate is not None else "N/A",
            delta=None
        )
        metric_cols[3].metric(
            "Discovered Themes",
            results["topic_discovery"].get("total_themes", 0),
            delta=None
        )

        st.divider()

        # Detailed metrics by stage
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Extraction & Normalization")
            ext_norm_data = []
            if results["extraction"]["records"] > 0:
                ext_norm_data.append({"Metric": "PDFs Processed", "Value": results["extraction"]["records"]})
            if results["normalization"]["records"] > 0:
                ext_norm_data.append({"Metric": "Normalized Records", "Value": results["normalization"]["records"]})
                ext_norm_data.append({"Metric": "Normalization Mode", "Value": results["normalization"].get("mode", "N/A")})
            if results["prompt_analysis"]["prompts_analyzed"] > 0:
                ext_norm_data.append({"Metric": "Prompts Analyzed", "Value": results["prompt_analysis"]["prompts_analyzed"]})
            if ext_norm_data:
                st.dataframe(pd.DataFrame(ext_norm_data), use_container_width=True, hide_index=True)
            else:
                st.info("No extraction/normalization data available yet.")

        with col_b:
            st.markdown("#### Topic Discovery")
            td_data = []
            if results["topic_discovery"]["sections"] > 0:
                td_data.append({"Metric": "Sections Analyzed", "Value": results["topic_discovery"]["sections"]})
                td_data.append({"Metric": "Text Segments Clustered", "Value": results["topic_discovery"]["total_texts"]})
                td_data.append({"Metric": "Unified Themes Generated", "Value": results["topic_discovery"]["total_themes"]})
            if td_data:
                st.dataframe(pd.DataFrame(td_data), use_container_width=True, hide_index=True)
            else:
                st.info("No topic discovery data available yet.")

    # =========================================================================
    # TAB 3: REPRODUCIBILITY PACKAGE
    # =========================================================================
    with tab3:
        st.subheader("📦 Download Reproducibility Package")
        st.markdown("""
        The reproducibility package contains all pipeline artifacts, configuration metadata, 
        and an audit trail manifest. Use this to document your methodology, share with 
        collaborators, or verify results independently.
        """)

        zip_buffer, manifest = build_reproducibility_package(results)

        st.download_button(
            label="💾 Download Full Reproducibility Package (ZIP)",
            data=zip_buffer,
            file_name=f"sarsp_langed_reproducibility_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            type="primary",
            key="results_download_repro_zip"
        )

        st.divider()
        st.markdown("#### 📋 Reproducibility Manifest Preview")
        st.json(manifest)

        st.divider()
        st.markdown("""
        #### Package Contents
        | File / Directory | Source Stage | Description |
        |---|---|---|
        | `REPRODUCIBILITY_MANIFEST.json` | All | Audit trail, config metadata, timestamps |
        | `stage1_preprocessing/final_filtered_refs.csv` | Preprocessing | Deduplicated & filtered corpus |
        | `stage1_preprocessing/removed_refs.csv` | Preprocessing | Removal log with reasons |
        | `stage2_screening/screened_refs.xlsx` | Screening | Color-coded dual-rater decisions |
        | `stage3_extraction/structured_output.json` | Extraction | Raw LLM-extracted metadata |
        | `stage4_validation/agreement_summary.xlsx` | Validation | Per-column & overall agreement rates |
        | `stage4_validation/comparison_table.xlsx` | Validation | Human-LLM comparison with mismatch reasons |
        | `stage5_normalization/normalized_output*.json` | Normalization | Cleaned & harmonized dataset |
        | `stage6_prompt_analysis/analyzed_prompts.csv` | Prompt Analysis | Structural features & syntactic heuristics |
        | `stage7_topic_discovery/*/` | Topic Discovery | Per-section cluster assignments, summaries, merged themes |
        | `figures/normalization/*.png` | Normalization | Saved visualization charts |
        | `figures/prompt_analysis/*.png` | Prompt Analysis | Saved visualization charts |
        """)