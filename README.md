# SARSP-LangEd

**Semi-Automated Research Synthesis Protocol for Language Education**

An open-source, browser-based application for conducting semi-automated systematic reviews in fast-moving, data-intensive educational domains. SARSP-LangEd integrates LLM-assisted screening and extraction with human-in-the-loop validation, rule-based normalisation, prompt analysis, and datag-driven topic discovery—packaging every intermediate artifact into a single downloadable reproducibility archive.

> 📖 **Companion publication:** Fernández-Martínez, N.J. & Pérez-Paredes, P. (in preparation). *Semi-Automated Research Synthesis in Language Education*. Cambridge Elements in Research Methods in Education. Cambridge University Press.

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

---

## Running the Application

streamlit run app.py
```

The application opens automatically in your default browser at:

```
http://localhost:8501
```

---

## API Configuration

Stages requiring API access (Screening, Extraction, and Topic Discovery labelling) support multiple providers through an OpenAI-compatible interface.

| Provider | Base URL | Notes |
|:---|:---|:---|
| **Cerebras** | `https://api.cerebras.ai/v1` | Free tier available |
| **Google Gemini** | `https://generativelanguage.googleapis.com/v1beta/openai` | Requires API key |
| **Mistral AI** | `https://api.mistral.ai/v1` | Required for Stage 3 (OCR extraction) |
| **OpenAI** | `https://api.openai.com/v1` | Requires API key |
| **Ollama (Local)** | `http://localhost:11434/v1` | Zero cost; local only |
| **LM Studio (Local)** | `http://localhost:1234/v1` | Zero cost; local only |

> 💡 **Tip:** Local providers (Ollama and LM Studio) are available only when running the application on your own machine. The cloud-hosted version restricts provider selection to commercial APIs.

---

## Project Structure

```text
sarsp_langed_gui/
├── app.py                    # Streamlit entry point
├── stages/                   # Eight pipeline stages
├── components/               # Shared GUI components
├── utils/                    # Helper functions and API clients
├── prompts/                  # Default prompts and schemas
├── data/                     # Replication dataset
├── assets/                   # Images and static resources
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Reproducibility

Every stage automatically saves its primary artifacts to a structured temporary directory. The **Stage 8 Results Dashboard** aggregates these into a single downloadable ZIP archive containing:

- **`REPRODUCIBILITY_MANIFEST.json`** — Complete audit trail with timestamps, model identifiers, prompts, and execution parameters
- Per-stage outputs (CSV, JSON, Excel, PNG figures)
- Complete configuration metadata

The **Replication Mode** (available from the sidebar) loads the original case-study artifacts, allowing independent verification of published results without re-running any computationally expensive API stages.

---

## Adapting the Pipeline

SARSP-LangEd is designed to be transferable across research domains. Researchers can adapt the pipeline without modifying the source code by:

- Modifying eligibility criteria and screening prompts (Stage 2)
- Editing the extraction schema or switching to **Custom Mode** with automatically generated JSON schemas (Stage 3)
- Adjusting journal quartile thresholds or disabling filtering entirely (Stage 1)
- Editing regular expressions and adding custom prompt-analysis markers (Stage 6)
- Selecting different embedding models, clustering parameters, and topic-labelling prompts (Stage 7)

Standard domain adaptation therefore requires configuration changes rather than software development.

---

## Citation

If you use SARSP-LangEd in your research, please cite the companion publication:

```bibtex
@misc{fernandezmartinez2026sarsp,
  author = {Fernández-Martínez, Nicolás J. and Pérez-Paredes, Pascual},
  title = {Semi-Automated Research Synthesis in Language Education},
  year = {2026},
  note = {Cambridge Elements in Research Methods in Education. In preparation.}
}
```

---

## License

This project is released under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this software for academic or commercial purposes, provided that the original copyright notice and license text are included in all copies or substantial portions of the software.

---

## Acknowledgements

SARSP-LangEd was developed as the methodological infrastructure for the Cambridge Elements volume **Semi-Automated Research Synthesis in Language Education**.

The accompanying case study comprises **427 empirical studies** on text-based large language models in language education published between **December 2022 and October 2025**.

---

## Contact

Questions, bug reports, feature requests, and collaboration proposals are welcome.

Please open an issue in this repository or contact the corresponding author at njfernan@ujaen.es

---

## License (MIT)

```text
MIT License

Copyright (c) 2026 Nicolás José Fernández-Martínez & Pascual Pérez-Paredes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Keywords

```
systematic-review
research-synthesis
llm
applied-linguistics
language-education
streamlit
nlp
reproducibility
semi-automated
prisma
meta-research
```
