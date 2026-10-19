import streamlit as st
import os
import json
import pandas as pd
from pathlib import Path


def get_session_temp_dir():
    """Return temp directory appropriate for deployment context."""
    if os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("STREAMLIT_CLOUD"):
        # Multi-user cloud: isolate per session
        import uuid
        session_id = st.session_state.get("_session_id", str(uuid.uuid4())[:8])
        st.session_state["_session_id"] = session_id
        base = os.path.join("temp", session_id)
    else:
        # Local / single-user server: shared temp directory
        base = "temp"
    
    os.makedirs(base, exist_ok=True)
    return base



REPLICATION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "replication_data"
)

# Canonical mapping of replication artifacts
REPLICATION_FILES = {
    # Stage 1: Preprocessing
    "s1_wos": "stage1_wos_results_export.xls",
    "s1_scopus": "stage1_scopus_results_export.csv",
    "s1_sjr": "stage1_sjr_rank.csv",
    "s1_jcr": "stage1_jcr_rank.xlsx",
    "s1_final_refs": "stage1_final_filtered_refs.csv",
    "s1_removed_refs": "stage1_removed_refs.csv",
    "s1_manual_verification": "stage1_final_filtered_after_manual_verification_book_chapters.csv",
    "s1_verified_corpus": "stage1_final_filtered_after_manual_verification_book_chapters.csv",
    
    # Stage 2: Screening
    "s2_screening": "stage2_screening.xlsx",
    "s2_screening_reviewed": "stage2_screening_after_review.xlsx",
    
    # Stage 3: Extraction
    "s3_structured_output": "stage3_structured_output.json",
    
    # Stage 4: Validation
    "s4_template": "stage4_validation_template.xlsx",
    "s4_human_coding": "stage4_validation_human_coding.xlsx",
    "s4_comparison": "stage4_validation_comparison.xlsx",
    "s4_metrics": "stage4_validation_metrics.xlsx",
    
    # Stage 5: Normalization
    "s5_normalized": "stage5_normalized_output.json",
    
    # Stage 6: Prompt Analysis
    "s6_prompts": "stage6_analyzed_prompts.csv",
}


def get_replication_path(key: str) -> str | None:
    """Return full path for a replication artifact, or None if missing."""
    filename = REPLICATION_FILES.get(key)
    if not filename:
        return None
    path = os.path.join(REPLICATION_DIR, filename)
    return path if os.path.exists(path) else None


def load_replication_json(key: str) -> list | dict | None:
    """Load a JSON replication artifact safely."""
    path = get_replication_path(key)
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_replication_csv(key: str, **kwargs) -> pd.DataFrame | None:
    """Load a CSV/XLS/XLSX replication artifact safely."""
    path = get_replication_path(key)
    if not path:
        return None
    try:
        ext = Path(path).suffix.lower()
        if ext == ".xls":
            return pd.read_excel(path, **kwargs)
        elif ext == ".xlsx":
            return pd.read_excel(path, **kwargs)
        elif ext == ".csv":
            return pd.read_csv(path, **kwargs)
        return None
    except Exception:
        return None


def get_available_replication_files() -> dict:
    """Return dict of available replication artifacts with metadata."""
    available = {}
    for key, filename in REPLICATION_FILES.items():
        path = os.path.join(REPLICATION_DIR, filename)
        if os.path.exists(path):
            size_kb = round(os.path.getsize(path) / 1024, 1)
            available[key] = {"filename": filename, "size_kb": size_kb, "path": path}
    
    # Check topic discovery subfolders
    topic_dir = os.path.join(REPLICATION_DIR, "stage7_topic_discovery")
    if os.path.exists(topic_dir):
        sections = [d.name for d in Path(topic_dir).iterdir() if d.is_dir()]
        if sections:
            available["s7_topics"] = {
                "sections": sections, 
                "path": topic_dir,
                "n_sections": len(sections)
            }
    
    return available


def copy_to_session_temp(src_path: str, dest_subpath: str) -> str | None:
    """Copy a replication file into the current session's temp directory."""
    import shutil
    from core.utils import get_session_temp_dir
    
    if not src_path or not os.path.exists(src_path):
        return None
    
    dest = os.path.join(get_session_temp_dir(), dest_subpath)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(src_path, dest)
    return dest