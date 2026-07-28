# SARSP-LangEd Replication Guide

## Overview

This package contains **all original experiment artifacts** from the published 
*Elements* study, enabling full replication of Stages 1–8. The only items excluded 
are copyrighted PDF full texts; all other inputs and outputs are provided.

## Contents

### Stage 1: Preprocessing
| File | Description |
|------|-------------|
| `stage1_wos_results_export.xls` | Original Web of Science export |
| `stage1_scopus_results_export.csv` | Original Scopus export |
| `stage1_sjr_rank.csv` | SJR journal rankings used for filtering |
| `stage1_jcr_rank.xlsx` | JCR impact factors used for filtering |
| `stage1_final_filtered_refs.csv` | Final corpus after automated filtering |
| `stage1_removed_refs.csv` | Removal log with reasons |
| `stage1_final_filtered_after_manual_verification_book_chapters.csv` | Corpus after manual book chapter verification |

### Stage 2: Screening
| File | Description |
|------|-------------|
| `stage2_screening.xlsx` | Raw LLM screening decisions |
| `stage2_screening_after_review.xlsx` | Final screening after human adjudication |

### Stage 3: Extraction
| File | Description |
|------|-------------|
| `stage3_structured_output.json` | Full LLM-extracted metadata (no PDFs needed) |

### Stage 4: Validation
| File | Description |
|------|-------------|
| `stage4_validation_template.xlsx` | Blank template for human coding |
| `stage4_validation_human_coding.xlsx` | Completed human-coded validation |
| `stage4_validation_comparison.xlsx` | Human-LLM comparison table |
| `stage4_validation_metrics.xlsx` | Agreement metrics summary |

### Stage 5: Normalization
| File | Description |
|------|-------------|
| `stage5_normalized_output.json` | Fully normalized dataset |

### Stage 6: Prompt Analysis
| File | Description |
|------|-------------|
| `stage6_analyzed_prompts.csv` | Structural & syntactic prompt analysis |

### Stage 7: Topic Discovery
| Directory | Description |
|-----------|-------------|
| `stage7_topic_discovery/` | Per-section subfolders with `texts_with_themes.csv`, `cluster_summaries.csv`, `themes_merged.csv` |

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Download spaCy models: `bash post_install.sh`
3. Run: `streamlit run streamlit_app.py`
4. Toggle **🔬 Replication Mode** in the sidebar
5. Navigate through stages to verify results

## Stage-by-Stage Replication

### Stage 1: Preprocessing ✅ Fully Replicable
- Upload the bundled WoS, Scopus, SJR, and JCR files manually to re-run preprocessing end-to-end
- In Replication Mode, the **Results Dashboard** reads the verified corpus directly; Stage 1 UI is not pre-populated (re-running requires manual upload)
- Compare final corpus size and removal counts against published values
- Note: Manual book chapter verification was performed separately; the verified corpus is provided as `stage1_final_filtered_after_manual_verification_book_chapters.csv`

### Stage 2: Screening ⚠️ Partially Replicable
- Original screening Excel is loaded in Replication Mode
- Re-running screening requires an LLM API key and will produce different results due to API non-determinism
- Use the provided `stage2_screening_after_review.xlsx` as the ground truth

### Stage 3: Extraction ✅ Fully Verifiable
- `stage3_structured_output.json` contains all extracted metadata
- No PDFs or API calls needed to proceed to downstream stages
- Re-extraction from PDFs is not possible without the original copyrighted files

### Stage 4: Validation ✅ Fully Replicable
- All four validation artifacts are provided
- Upload `stage4_validation_comparison.xlsx` in Tab 3 to recompute agreement metrics
- Compare against `stage4_validation_metrics.xlsx`

### Stage 5: Normalization ✅ Fully Replicable & Deterministic
- Load `stage3_structured_output.json` and run Default Pipeline normalization
- Output should be identical to `stage5_normalized_output.json`
- Any differences indicate a code change since publication

### Stage 6: Prompt Analysis ✅ Fully Replicable & Deterministic
- Load `stage3_structured_output.json` and run analysis
- The replication CSV contains **1,160 analyzed prompts** (individual prompt records, not studies)
- Compare structural feature percentages and prompt count against `stage6_analyzed_prompts.csv`
- Note: Prompt CSV uses multiline fields; always use `pd.read_csv()` for accurate record counting (raw line counting will overcount)

### Stage 7: Topic Discovery ✅ Verifiable; Re-run Requires API
- Original cluster assignments and themes are in `stage7_topic_discovery/`
- Re-running requires an LLM API and will produce different labels
- Cluster structure (which texts belong together) can be verified against provided CSVs

### Stage 8: Results Dashboard ✅ Auto-Computed
- All metrics computed from disk artifacts
- Serves as final verification checkpoint

## Verification Checklist

| Metric | Published Value | Your Value | Match? |
|--------|----------------|------------|--------|
| WoS records loaded | ___ | | ☐ |
| Scopus records loaded | ___ | | ☐ |
| After deduplication | ___ | | ☐ |
| After quartile filtering | ___ | | ☐ |
| After manual book verification | ___ | | ☐ |
| After screening | ___ | | ☐ |
| Extracted studies | ___ | | ☐ |
| Validation agreement rate | ___% | | ☐ |
| Normalized records | ___ | | ☐ |
| Prompts analyzed | ___ | | ☐ |
| Topic sections | ___ | | ☐ |
| Total unified themes | ___ | | ☐ |

*(Fill in published values from the Elements book)*

## Notes

- **Deterministic stages** (1, 4, 5, 6): Should produce byte-identical results when re-run
- **Non-deterministic stages** (2, 3, 7): LLM outputs vary across API providers and time; provided files represent the exact outputs used in publication
- **PDFs**: Not included due to copyright. Stage 3 output JSON serves as the authoritative extraction record