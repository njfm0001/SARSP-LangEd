# SARSP-LangEd

**Semi-Automated Research Synthesis Protocol for Language Education**

An open-source, browser-based application for conducting semi-automated systematic reviews in fast-moving, data-intensive educational domains. SARSP-LangEd integrates LLM-assisted screening and extraction with human-in-the-loop validation, rule-based normalisation, prompt analysis, and datag-driven topic discovery—packaging every intermediate artifact into a single downloadable reproducibility archive.

> 📖 **Companion publication:** Fernández-Martínez, N.J. & Pérez-Paredes, P. (in preparation). *Semi-Automated Research Synthesis in Language Education*. Cambridge Elements in Research Methods in Applied Linguistics. Cambridge University Press.

---

## Overview

SARSP-LangEd operationalises a modular, eight-stage analytical pipeline that distributes cognitive labour between computational automation and human expertise:

| Stage | Function | API Required? |
|:---:|:---|:---:|
| 1️⃣ | **Preprocessing** — Deduplication, configurable quartile filtering, schema unification | ❌ |
| 2️⃣ | **Screening** — Dual-rater LLM classification with human adjudication of disagreements | ✅ |
| 3️⃣ | **Extraction** — Schema-constrained PDF metadata extraction via Mistral OCR | ✅ |
| 4️⃣ | **Validation** — Double-blind human–LLM comparison with weighted agreement metrics | ❌ |
| 5️⃣ | **Normalisation** — Rule-based harmonisation with dual-mode architecture (Default / Custom Light) | ❌ |
| 6️⃣ | **Prompt Analysis** — Structural profiling, functional feature detection, syntactic heuristics | ❌ |
| 7️⃣ | **Topic Discovery** — Embedding-based clustering with constrained LLM-assisted labelling | ✅ (labelling only) |
| 8️⃣ | **Results Dashboard** — Cross-stage metrics and one-click reproducibility package | ❌ |

Stages marked ❌ run entirely offline with zero API costs.

---

## Key Features

- **🖥️ Browser-based GUI** — No command-line expertise required; guided workflows with sliders, dropdowns, and editable text areas
- **🔒 Locked + Custom modes** — Default prompts/schemas guarantee downstream compatibility; Custom mode enables domain adaptation with auto-generated JSON schemas
- **👥 Human-in-the-loop** — Dual-rater screening, double-blind validation, and expert adjudication at every decision point
- **🔄 Replication Mode** — Load the original case study artifacts instantly to verify published results without re-running API calls
- **📦 Reproducibility Package** — One-click ZIP download containing all pipeline artifacts, configuration metadata, and a machine-readable manifest
- **🌐 Model-agnostic** — Supports Cerebras, Google Gemini, Mistral AI, OpenAI, and local endpoints (Ollama, LM Studio)
- **♿ Accessible** — Stages 1, 4, 5, and 6 require zero API costs and run entirely offline

---

## Case Study

The application ships with a built-in replication dataset from the companion Elements volume: a systematic review of **427 empirical studies** on text-based LLMs in language education (December 2022–October 2025), demonstrating the pipeline across all eight stages.

---

## Installation

### Prerequisites

- Python 3.10+
- (Optional) [Ollama](https://ollama.ai/) or [LM Studio](https://lmstudio.ai/) for local model inference

### Setup

```bash
# Clone the repository
git clone https://github.com/[USERNAME]/sarsp_langed_gui.git
cd sarsp_langed_gui

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model (for prompt analysis)
python -m spacy download en_core_web_sm

# (Optional) Download multilingual spaCy model
python -m spacy download xx_sent_ud_sm
