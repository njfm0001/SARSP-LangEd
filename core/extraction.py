"""
SARSP-LangEd - Stage 3: Schema-Constrained PDF Extraction (Mistral AI)
Mistral OCR-based PDF extraction with resumable checkpointing,
multi-model fallback, and strict JSON schema enforcement.
"""

import streamlit as st
import os
import io
import json
import time
import random
from pathlib import Path
from mistralai import Mistral
from mistralai.models import SDKError, TextChunk, FileChunk


# =============================================================================
# CONSTANTS
# =============================================================================

from core.utils import get_session_temp_dir
TEMP_DIR = get_session_temp_dir()
STAGE_DIR = os.path.join(TEMP_DIR, "stage3_extraction")
EXTRACTION_OUTPUT = os.path.join(STAGE_DIR, "structured_output.json")
PDF_UPLOAD_DIR = os.path.join(STAGE_DIR, "uploaded_pdfs")

# Ensure directories exist
for d in [STAGE_DIR, PDF_UPLOAD_DIR]:
    os.makedirs(d, exist_ok=True)

# Available Mistral models with strong structured output + OCR support
MISTRAL_MODELS = [
    "mistral-large-latest",
    "mistral-medium-latest",
    "mistral-small-latest",
    "open-mixtral-8x22b",
    "open-mixtral-8x7b",
]

MAX_RETRIES = 10
BASE_DELAY = 5

# Locked prompt — NOT editable by users to preserve schema integrity
LOCKED_PROMPT = """As an expert scholar in applied linguistics, you specialise in the application of text-based large language models (LLMs) in language education contexts.

### TASK ###
Thoroughly and carefully examine the full content of the attached paper to extract the following information items, outputting them as JSON that conforms STRICTLY to the structure and constraints below:

- **Title of the study** (string)
- **Study author(s)** (string)
- **Study summary** (string)
- **APA reference** (string)
- **Research questions or hypotheses**: return as an ARRAY of items (each item is one research question or hypothesis). If none, return ["N/A"].

- **Study location**: return EXACTLY THREE elements in an ARRAY, each being an ARRAY of strings:  
[[region_or_city_or_province_or_state], [country], [continent]]
  If an element is missing, return `"N/A"` in that slot.

- **Educational setting**: ARRAY containing any of allowed values:  
  `"pre-school", "primary school", "k-12/secondary education", "higher education", "adult", "other", "N/A"`

- **Participant type**: ARRAY containing any of allowed values:  
  `"learners", "teachers", "policymakers", "administrators", "other", "N/A"`

- **Participant demographics**: return an ARRAY, where each element corresponds to a participant group listed in participant_type, with the following fields (string for each; return "N/A" if absent):  
  - `"participant_type"`
  - `"age"`  
  - `"gender"`  
  - `"first_language"`  
  - `"target_language"`  
  - `"language_status"` (one of: First Language (L1), Second Language (L2), Foreign Language (FL), Language for Specific Purposes (LSP), other, N/A)  
  - `"CEFR_level"` (A1, A2, B1, B2, C1, C2, N/A)  
  - `"AI_literacy"` (low, medium, high, N/A)

- **Language skills targeted**: ARRAY containing any of allowed values:  
  `"writing", "reading", "speaking", "listening", "N/A"`

- **Task types**: return an OBJECT with these fields, each as an ARRAY:  
  - `"students"`  
  - `"teachers"`
  - `"policymakers_administrators"`  
  - `"others"` (if none, return ["N/A"])  
  Allowed values include:  
  `"writing", "reading", "speaking", "pronunciation", "listening", "grammar", "vocabulary", "error correction", "translation", "summarization", "curriculum development", "lesson planning", "material generation", "feedback", "assessment/grading", "policy analysis", "cultural understanding", "other (specify)", "N/A"`.

- **Large Language Model(s) used**: return an ARRAY of text-based large language model names only (e.g., "ChatGPT", "Gemini", "Copilot", "LlaMa", "DeepSeek", "Qwen", other (specify)). No specific versions or descriptions.

- **Prompts used**: return an ARRAY of all distinct prompts (verbatim). If none, return ["N/A"].

- **Prompting techniques**: ARRAY containing any of allowed values:
  `"structured prompting", "unstructured prompting", "role-based prompting", "contextual prompting", "chain-of-thought prompting", "N/A"`

- **Prompting strategies**: ARRAY containing any of allowed values:
  `"explicit instructions", "examples", "context", "role assignment", "constraints", "reasoned steps", "iterative prompting", "other (specify)", "N/A"`

- **Research methodology**: ARRAY with one of allowed values:  
  `"quantitative", "qualitative", "mixed method (concurrent)", "mixed method (sequential)", "N/A"`

- **Data gathering methods**: ARRAY containing any of allowed values:  
  `"case study", "corpus analysis", "diaries", "elicitation tasks", "ethnographies", "experiments", "interviews", "journals", "observations", "questionnaires", "tests", "surveys", "other", "N/A"`

- **Research design**: ARRAY containing any of allowed values:  
  `"experimental", "quasi-experimental", "correlational", "ethnographic", "case study", "narrative inquiry", "triangulation design", "embedded design", "explanatory sequential design", "exploratory sequential design", "other", "N/A"`

- **Sample size** (string)

- **Duration** (string)

- **Conceptual/theoretical frameworks**: return as an ARRAY of items (each item is one named framework such as CALL, TPACK, AI literacy, Technology Acceptance Model, other (specify)). If none, return ["N/A"].

- **Learning perceptions**: return as an ARRAY of items (each item is one perception such as attitude, satisfaction, motivation, engagement, autonomy, confidence, anxiety, performance, trust, other (specify)) and a brief explanation of each. If none, return ["N/A"].

- **Outcomes**: return as an OBJECT with these fields, each as an ARRAY of items:  
  - `"benefits_affordances"`  
  - `"drawbacks"`  
  - `"study_limitations"`   
  If none for any field, return ["N/A"].
  
- **Stakeholder impact**: return as an OBJECT with these fields, each as an ARRAY of items:  
  - `"students"`  
  - `"teachers"`  
  - `"institutions"`  
  - `"other"`  
  If none for any field, return ["N/A"].

- **Policy/guidance**: return as an ARRAY of items (each item is one policy mention, guideline, or recommendation). If none, return ["N/A"].

- **Emergent themes**: return as an ARRAY of items (each item is one theme or subtheme that has NOT been mentioned in 'outcomes', 'stakeholder_impact', and 'policy_guidance'). If none, return ["N/A"].

- **Challenges, concerns, limitations of LLMs in language education**: ARRAY containing any of allowed values:
  `"accuracy", "reliability", "model hallucination", "integration and scalability issues", "overreliance", "transparency", "cognitive laziness", "lack of metalinguistic awareness", "lack of AI literacy", "academic integrity", "authorship/copyright issues", "privacy and data security", "fairness and bias", "digital divide", "accesibility", "other", "N/A"`

- **Emergent issues**: return as an ARRAY of items (each item is one new or unexpected concern, challenge, or limitation that has NOT been mentioned in 'challenges_concerns_limitations_of_LLMs_in_language_education'). If none, return ["N/A"].

If you infer information, append "(inferred)".  
If not present in the paper, return `"N/A"`.
IMPORTANT RULE: Wherever the schema includes the option "other (specify)", you MUST NOT return the literal string "other (specify)". Instead, ALWAYS replace it with a descriptive string explaining what the "other" category refers to.

### OUTPUT ###
Return a single JSON object that strictly conforms to the schema."""

# Strict JSON schema — locked to match LOCKED_PROMPT exactly
# Define the JSON schema for structured data extraction.
# This strictly enforces the output format, preventing hallucination 
# and ensuring the output can be parsed programmatically.
JSON_SCHEMA = {
    "type": "object",
    "required": [
        "title", "authors", "summary", "APA_reference",
        "research_h_q", "study_location", "educational_settings",
        "participant_type", "participant_demographics",
        "language_skills_targeted", "task_types", "LLMs_used",
        "prompts_used", "prompting_techniques",
        "prompting_strategies", "research_methodology",
        "data_gathering_methods", "research_design",
        "sample_size", "duration", "frameworks", "learning_perceptions",
        "outcomes", "stakeholder_impact", "policy_guidance", "emergent_themes",
        "challenges_concerns_limitations_of_LLMs_in_language_education", "emergent_issues"
    ],
    "properties": {
        "title": {"type": "string"},
        "authors": {"type": "string"},
        "summary": {"type": "string"},
        "APA_reference": {"type": "string"},

        "research_h_q": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
        },

      "study_location": {
      "type": "array",
      "minItems": 3,
      "maxItems": 3,
      "items": {
      "type": "array",
      "description": "Geographic setting of the study. Must include region/city/province/state, country, and continent.",
      "minItems": 1,
      "items": { "type": "string" }
        }
      },

      "educational_settings": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                "pre-school", "primary school", "k-12/secondary education",
                "higher education", "adult", "other", "N/A"
                ]
            }
        },      

    "participant_type": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": [
          "learners",
          "teachers",
          "policymakers",
          "administrators",
          "other"
        ]
      }
    },

    "participant_demographics": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "participant_type",
          "age",
          "gender",
          "first_language",
          "target_language",
          "language_status",
          "CEFR",
          "AI_literacy"
        ],
        "properties": {
          "participant_type": {
            "type": "string",
            "enum": [
              "learners",
              "teachers",
              "policymakers",
              "administrators",
              "other"
            ]
          },
          "age": {"type": "string"},
          "gender": {"type": "string"},
          "first_language": {"type": "string"},
          "target_language": {"type": "string"},
          "language_status": {
            "type": "string",
            "enum": ["First Language (L1)", "Second Language (L2)", "Foreign Language (FL)", "Language for Specific Purposes (LSP)", "other", "N/A"]
          },
          "CEFR": {
            "type": "string",
            "enum": ["A1", "A2", "B1", "B2", "C1", "C2", "N/A"]
          },
          "AI_literacy": {
            "type": "string",
            "enum": ["low", "medium", "high", "N/A"]
          }
        }
      }
    },

"language_skills_targeted": {
  "type": "array",
  "items": {
    "type": "string",
    "enum": ["speaking", "writing", "reading", "listening", "N/A"]
  }
},

        "task_types": {
            "type": "object",
            "required": ["students", "teachers", "policymakers_administrators", "others"],
            "properties": {
                "students": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "teachers": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "policymakers_administrators": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "others": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        },
        
        "LLMs_used": {
            "type": "array",
            "description": "List only the names of the text-based Large Language Models used in the study (e.g., ChatGPT, Gemini, LLaMA, Qwen, Deepseek, etc.)",
            "items": {"type": "string"}
        },

        "prompts_used": {
        "type": "array",
            "items": {"type": "string"}
        },

        "prompting_techniques": {
                "type": "array",
            "items": {
                "type": "string",
                "enum": [
                "structured prompting", "unstructured prompting", "role-based prompting", "contextual prompting",
                 "chain-of-thought prompting", "N/A"
                ]
            }
        },

        "prompting_strategies": {
            "type": "array",
            "items": {"type": "string"}
        },

"research_methodology": {
  "type": "string",
  "enum": [
    "quantitative",
    "qualitative",
    "mixed method (concurrent)", 
    "mixed method (sequential)",
    "N/A"
  ],
  "description": "quantitative: numerical data & statistical analysis; qualitative: understanding experiences & attitudes; mixed method (concurrent): simultaneous data collection; mixed method (sequential): phased approach; N/A: not applicable/available"
},

    "data_gathering_methods": {
  "type": "array",
  "items": {
    "type": "string",
    "enum": [
      "case study",
      "corpus analysis",
      "diaries",
      "elicitation tasks",
      "ethnographies",
      "experiments",
      "interviews",
      "journals",
      "observations",
      "questionnaires",
      "tests",
      "surveys",
      "other",
      "N/A"
    ]
  },
  "description": "case study: in-depth investigation of individual/group; corpus analysis: computational study of large text collections; diaries: learner-recorded reflections over time; elicitation tasks: controlled prompts to elicit language; ethnographies: immersive observation in cultural/social contexts; experiments: controlled testing of interventions; interviews: structured or unstructured exploration of attitudes/experiences; journals: ongoing documentation of learning processes; observations: classroom/naturalistic recording of interactions; questionnaires: structured data collection on attitudes/background; tests: standardized proficiency measures (e.g., TOEFL, IELTS); surveys: large-scale quantitative data collection; other: alternative or mixed methods; N/A: not applicable"
},


    "research_design": {
  "type": "array", 
  "items": {
    "type": "string",
    "enum": [
      "experimental",
      "quasi-experimental",
      "correlational",
      "ethnographic",
      "case study", 
      "narrative inquiry",
      "triangulation design",
      "embedded design",
      "explanatory sequential design",
      "exploratory sequential design",
      "other",
      "N/A"
    ]
  },
  "description": "experimental: manipulates variables for causality; quasi-experimental: experimental without random assignment; correlational: examines variable relationships; ethnographic: immersive cultural study; case study: detailed single case; narrative inquiry: personal stories analysis; triangulation design: concurrent mixed-methods validation; embedded design: one data type within another; explanatory sequential design: quantitative → qualitative sequence; exploratory sequential design: qualitative → quantitative sequence"
},

        "sample_size": {"type": "string"},
        "duration": {"type": "string"},
        
        "frameworks": {
            "type": "array",
            "items": {"type": "string"}
        },
        
        "learning_perceptions": {
            "type": "array",
            "items": {"type": "string"}
        },
        
        "outcomes": {
            "type": "object",
            "required": ["benefits_affordances", "drawbacks", "study_limitations"],
            "properties": {
                "benefits_affordances": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "drawbacks": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "study_limitations": {
                    "type": "array",
                    "items": {"type": "string"}
                },
            }
        },

        "stakeholder_impact": {
            "type": "object",
            "required": ["students", "teachers", "institutions", "other"],
            "properties": {
                "students": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "teachers": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "institutions": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "other": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        },

        "policy_guidance": {
            "type": "array",
            "items": {"type": "string"}
        },

        "emergent_themes": {
            "type": "array",
            "items": {"type": "string"}
        },
        
  "challenges_concerns_limitations_of_LLMs_in_language_education": {
  "type": "array",
  "items": {
    "type": "string",
    "enum": [
      "accuracy", "reliability", "model hallucination", "integration and scalability issues",
      "overreliance", "transparency", "cognitive laziness", "lack of metalinguistic awareness",
      "lack of AI literacy", "academic integrity", "authorship/copyright issues",
      "privacy and data security", "fairness and bias", "digital divide", "accesibility",
      "other", "N/A"
    ]
  }
},

        "emergent_issues": {
            "type": "array",
            "items": {"type": "string"}
        },
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def reorder_with_source_filename_first(data_json: dict, fname: str) -> dict:
    """Ensure source_filename is always the first key."""
    ordered = {"source_filename": fname}
    for k, v in data_json.items():
        if k != "source_filename":
            ordered[k] = v
    return ordered


def load_extraction_progress() -> list:
    """Load existing extraction results from checkpoint file."""
    if os.path.exists(EXTRACTION_OUTPUT):
        try:
            with open(EXTRACTION_OUTPUT, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def save_extraction_progress(results: list):
    """Save extraction results to checkpoint file."""
    with open(EXTRACTION_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

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

def generate_schema_from_prompt(client: Mistral, model_name: str, prompt_text: str, log_container) -> dict | None:
    """
    Use an LLM to generate a JSON Schema from a user-edited extraction prompt.
    Handles mistralai SDK internal KeyError on malformed error responses.
    Only processes content inside ### TASK ### markers from the user prompt.
    """
    import re

    raw = ""
    json_str = ""

    # --- Extract only the ### TASK ### section from user prompt ---
    task_match = re.search(r'###\s*TASK\s*###\s*\n([\s\S]*?)(?=\n###\s|\Z)', prompt_text)
    if task_match:
        task_content = task_match.group(1).strip()
        log_container.info(f"ℹ️ Extracted TASK section ({len(task_content)} chars) for schema generation")
    else:
        task_content = prompt_text.strip()
        log_container.warning("⚠️ No ### TASK ### section found. Using full prompt for schema generation.")

    # Truncate to prevent token overflow
    MAX_SCHEMA_PROMPT_CHARS = 12000
    if len(task_content) > MAX_SCHEMA_PROMPT_CHARS:
        task_content = task_content[:MAX_SCHEMA_PROMPT_CHARS] + "\n\n[TRUNCATED]"
        log_container.warning(f"⚠️ Task content truncated to {MAX_SCHEMA_PROMPT_CHARS} chars")

    formatted_prompt = SCHEMA_GENERATION_PROMPT.format(prompt=task_content)

    # --- PHASE 1: API CALL with retry for SDK internal errors ---
    max_api_retries = 3
    for api_attempt in range(max_api_retries):
        try:
            response = client.chat.complete(
                model=model_name,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0,
            )
            break  # Success — exit retry loop

        except KeyError as ke:
            # This is the SDK's internal KeyError on malformed error responses
            if api_attempt < max_api_retries - 1:
                delay = BASE_DELAY * (2 ** api_attempt)
                log_container.warning(
                    f"⚠️ SDK internal KeyError (attempt {api_attempt+1}/{max_api_retries}): {ke}\n"
                    f"Retrying in {delay}s... This is usually a transient API/Cloudflare issue."
                )
                time.sleep(delay)
            else:
                log_container.error(
                    f"❌ SDK internal KeyError persisted after {max_api_retries} attempts: {ke}\n"
                    f"This indicates a persistent network/DNS/Cloudflare block.\n"
                    f"Try: (1) Switch DNS to 8.8.8.8, (2) Use VPN, (3) Try again later."
                )
                return None

        except SDKError as sdk_err:
            status_code = getattr(sdk_err, "status_code", 0)
            log_container.error(
                f"❌ Mistral API error (HTTP {status_code}): {sdk_err}\n"
                f"Possible causes: prompt too long, content filter, or quota exceeded."
            )
            return None

        except Exception as api_err:
            log_container.error(
                f"❌ API call failed ({type(api_err).__name__}): {api_err}"
            )
            return None

    # --- PHASE 2: RESPONSE VALIDATION ---
    if not response or not response.choices:
        log_container.error("❌ API returned empty response (no choices).")
        return None

    raw = response.choices[0].message.content
    if not raw or not raw.strip():
        finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
        log_container.error(f"❌ Empty message content (finish_reason={finish_reason}).")
        return None

    raw = raw.strip()
    log_container.info(f"📥 Raw LLM response ({len(raw)} chars): {raw[:300]}...")

    # --- PHASE 3: MARKDOWN FENCE STRIPPING ---
    cleaned = raw
    fence_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?\s*```', cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()
        log_container.info("ℹ️ Stripped markdown code fences")

    # --- PHASE 4: JSON EXTRACTION ---
    json_match = re.search(r'\{[\s\S]*\}', cleaned)
    if not json_match:
        log_container.error(f"❌ No JSON object found.\nCleaned:\n{cleaned[:500]}")
        return None

    json_str = json_match.group(0)

    # --- PHASE 5: JSON PARSING ---
    parsed = None
    for attempt in range(2):
        try:
            parsed = json.loads(json_str)
            break
        except json.JSONDecodeError:
            if attempt == 0:
                json_str = json_str.strip('"').replace('\\"', '"')
            else:
                log_container.error(f"❌ JSON parse failed.\nExtracted:\n{json_str[:500]}")
                return None

    if parsed is None or not isinstance(parsed, dict):
        log_container.error(f"❌ Not a dict. Got {type(parsed).__name__}: {str(parsed)[:300]}")
        return None

    # --- PHASE 6: UNWRAP NESTED SCHEMAS ---
    schema = parsed
    if "type" not in schema:
        for wrapper_key in ("schema", "json_schema", "output", "result", "data"):
            if wrapper_key in schema and isinstance(schema[wrapper_key], dict) and "type" in schema[wrapper_key]:
                schema = schema[wrapper_key]
                log_container.info(f"ℹ️ Unwrapped from '{wrapper_key}'")
                break

    # --- PHASE 7: VALIDATE ---
    if "type" not in schema:
        log_container.error(f"❌ Missing 'type'. Keys: {list(schema.keys())}")
        return None
    if "properties" not in schema:
        log_container.error(f"❌ Missing 'properties'. Keys: {list(schema.keys())}")
        return None

    n_fields = len(schema.get("properties", {}))
    log_container.success(f"✅ Schema generated with {n_fields} fields")
    return schema


def upload_pdf_to_mistral(client: Mistral, pdf_bytes: bytes, fname: str):
    """
    Upload a PDF to Mistral's file engine for OCR processing.
    Returns the file_id string or None on failure.
    """
    try:
        uploaded_file = client.files.upload(
            file={"file_name": fname, "content": pdf_bytes},
            purpose="ocr"
        )
        return uploaded_file.id
    except Exception as e:
        return None


def delete_mistral_file(client: Mistral, file_id: str):
    """Best-effort cleanup of uploaded file from Mistral storage."""
    try:
        client.files.delete(file_id=file_id)
    except Exception:
        pass


def extract_single_pdf(client: Mistral, model_name: str, pdf_bytes: bytes, fname: str,
                       log_container, prompt_text: str, json_schema: dict,
                       max_retries=MAX_RETRIES, base_delay=BASE_DELAY,
                       fallback_models=None, use_fallback=False):
    """
    Extract structured data from a single PDF using Mistral OCR + Chat.
    Returns (data_dict, success_bool).
    Handles file upload, retries, exponential backoff, multi-model fallback, and cleanup.
    """
    # Step 1: Upload PDF to Mistral
    file_id = upload_pdf_to_mistral(client, pdf_bytes, fname)
    if not file_id:
        log_container.error(f"❌ Failed to upload {fname} to Mistral. Skipping.")
        return None, False

    # Build message content blocks using the active prompt (locked or custom)
    messages = [
        {
            "role": "user",
            "content": [
                TextChunk(text=prompt_text),
                FileChunk(file_id=file_id)
            ]
        }
    ]

    models_to_try = [model_name]
    if use_fallback and fallback_models:
        models_to_try += [m for m in fallback_models if m != model_name]

    result = None
    success = False

    for current_model in models_to_try:
        for attempt in range(max_retries):
            try:
                response = client.chat.complete(
                    model=current_model,
                    messages=messages,
                    temperature=0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "ExtractionSchema",
                            "schema": json_schema,
                            "strict": True
                        }
                    }
                )

                raw_text = response.choices[0].message.content
                data_json = json.loads(raw_text)
                data_json = reorder_with_source_filename_first(data_json, fname)

                if current_model != model_name:
                    log_container.info(f"✅ Extracted via fallback model `{current_model}`: {fname}")
                else:
                    log_container.info(f"✅ Extracted: {fname}")

                # Display full extracted JSON in live log
                log_container.json(data_json)

                result = data_json
                success = True
                break  # Success → stop retrying this model

            except SDKError as sdk_err:
                status_code = getattr(sdk_err, "status_code", 0)
                err_str = str(sdk_err).lower()

                if status_code == 429 or "rate_limit" in err_str or "quota" in err_str:
                    if use_fallback and current_model != models_to_try[-1]:
                        log_container.warning(f"⚠️ Quota exhausted on `{current_model}`, switching to next model...")
                        break  # Move to next fallback model
                    else:
                        log_container.error(f"❌ All models quota exhausted. Stopping.")
                        break
                elif status_code in (408, 500, 503, 504) or "timeout" in err_str:
                    if attempt == max_retries - 1:
                        log_container.error(f"❌ Server failure after {max_retries} retries for {fname}. Skipping.")
                        break
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 5)
                    log_container.warning(f"⚠️ Server error ({status_code}) — retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(delay)
                else:
                    log_container.warning(f"⚠️ Non-recoverable error on {fname}: {sdk_err}")
                    break

            except json.JSONDecodeError:
                log_container.error(f"❌ Invalid JSON response from Mistral for {fname}. Skipping.")
                break

            except Exception as e:
                log_container.warning(f"⚠️ Unexpected error on {fname}: {e}")
                break

        if success:
            break  # Stop trying fallback models if we succeeded

    # Cleanup: always delete uploaded file regardless of success/failure
    delete_mistral_file(client, file_id)

    return result, success


# =============================================================================
# STREAMLIT RENDER FUNCTION
# =============================================================================

def render_extraction_page():
    """Render the full Extraction stage UI."""
    st.title("3️⃣ Schema-Constrained PDF Extraction")
    st.markdown("""
    Upload PDFs and extract structured metadata using **Mistral AI's OCR-powered document understanding**. 
    The extraction prompt and JSON schema are **locked** to ensure compatibility with downstream 
    normalization and topic discovery stages. Results are checkpointed after every file for full resumability.
    """)

    st.info("💡 **Prompt Modes:** Use **Default** for pipeline-compatible extraction (Stages 4–7). Use **Custom** to adapt extraction to your own research goals — the app will auto-generate a matching JSON schema via LLM. Custom outputs may need manual post-processing before downstream stages.")
    st.divider()

        # === REPLICATION MODE INJECTION ===
    if st.session_state.get("replication_mode"):
        from core.utils import get_replication_path
        
        st.info("🔬 **Replication Mode:** Original extraction results loaded. PDFs are not included due to copyright. Toggle off to run your own extraction.")
        
        repl_path = get_replication_path("s3_structured_output")
        if repl_path and os.path.exists(repl_path):
            try:
                with open(repl_path, "r", encoding="utf-8") as f:
                    repl_results = json.load(f)
                
                st.session_state["extraction_results_count"] = len(repl_results)
                st.session_state["extraction_complete"] = True
                
                # Copy to session temp so downstream stages can find it
                import shutil
                temp_out = os.path.join(TEMP_DIR, "structured_output.json")
                shutil.copy2(repl_path, temp_out)
                
                st.success(f"✅ Loaded {len(repl_results)} extracted records from replication data.")
            except Exception as e:
                st.error(f"❌ Failed to load replication extraction file: {e}")
        else:
            st.warning("⚠️ Replication extraction file not found. Check `replication_data/stage3_structured_output.json`.")
        
        # In replication mode, skip config/upload/processing — fall through to RESULTS DISPLAY
        if not st.session_state.get("extraction_complete"):
            return
    else:
        # Normal mode: show configuration panel
        pass  # Existing configuration code continues below

    # =========================================================================
    # CONFIGURATION (skipped in replication mode)
    # =========================================================================
    if not st.session_state.get("replication_mode"):
      with st.expander("⚙️ Extraction Configuration", expanded=True):
        # --- API Key ---
        api_key = st.text_input(
            "Mistral API Key",
            type="password",
            value="",
            help="Get your key at https://console.mistral.ai/api-keys",
            key="extraction_api_key"
        )

        # --- Model Selection ---
        col1, col2 = st.columns([1, 1])
        with col1:
            primary_model = st.selectbox(
                "Primary Mistral Model",
                options=MISTRAL_MODELS,
                index=0,
                help="The primary model used for extraction AND schema generation. mistral-large-latest recommended.",
                key="extraction_primary_model"
            )
        with col2:
            use_fallback = st.toggle(
                "🔄 Smart Multi-Model Fallback",
                value=True,
                help="Automatically switches to other Mistral models on quota errors during extraction.",
                key="extraction_use_fallback"
            )

        if use_fallback:
            fallback_models = st.multiselect(
                "Fallback Models (ordered by preference)",
                options=[m for m in MISTRAL_MODELS if m != primary_model],
                default=[m for m in MISTRAL_MODELS if m != primary_model][:3],
                help="Models to try when the primary model is quota-exhausted.",
                key="extraction_fallback_models"
            )
        else:
            fallback_models = []

        st.divider()

        # --- Prompt Mode Selection ---
        prompt_mode = st.radio(
            "Prompt Mode",
            ["🔒 Default (Locked)", "✏️ Custom (Editable)"],
            horizontal=True,
            help="Default uses the validated pipeline prompt. Custom lets you edit the prompt and auto-generate a matching JSON schema.",
            key="extraction_prompt_mode"
        )

        if prompt_mode == "🔒 Default (Locked)":
            active_prompt = LOCKED_PROMPT
            active_schema = JSON_SCHEMA
            st.info("Using the default locked prompt and schema. These are validated for compatibility with Stages 4–7.")
            with st.expander("👁️ Preview Default Prompt (read-only)"):
                st.code(LOCKED_PROMPT, language="text")
        else:
            st.warning("⚠️ **Custom mode:** Downstream stages (Normalization, Topic Discovery) expect the default schema. Custom extractions may require manual post-processing.")

            custom_prompt = st.text_area(
                "Edit Extraction Prompt",
                value=st.session_state.get("extraction_custom_prompt", LOCKED_PROMPT),
                height=400,
                help="Describe what to extract. Be explicit about field names, types, allowed values, and array/object structures.",
                key="extraction_custom_prompt_editor"
            )
            st.session_state["extraction_custom_prompt"] = custom_prompt

            if st.button("🧠 Auto-Generate JSON Schema from Prompt", key="extraction_gen_schema_btn"):
                if not api_key:
                    st.error("❌ API key required for schema generation.")
                elif not custom_prompt.strip():
                    st.error("❌ Prompt cannot be empty.")
                else:
                    # Test connectivity first
                    with st.spinner("🔍 Testing API connectivity..."):
                        try:
                            test_client = Mistral(api_key=api_key)
                            test_resp = test_client.chat.complete(
                                model=primary_model,
                                messages=[{"role": "user", "content": 'Reply with exactly: {"status": "ok"}'}],
                                temperature=0,
                            )
                            test_content = test_resp.choices[0].message.content
                            st.success(f"✅ API reachable. Test response: {test_content[:100]}")
                        except Exception as test_err:
                            st.error(f"❌ API UNREACHABLE: {type(test_err).__name__}: {test_err}")
                            st.stop()

                    # Generate schema (inside its own spinner)
                    with st.spinner("Generating JSON schema from your prompt..."):
                        schema_client = Mistral(api_key=api_key)
                        gen_log = st.container()
                        generated = generate_schema_from_prompt(schema_client, primary_model, custom_prompt, gen_log)
                        if generated:
                            st.session_state["extraction_custom_schema"] = generated
                            st.success("✅ Schema generated successfully!")

            # Show generated or cached custom schema
            if "extraction_custom_schema" in st.session_state:
                active_prompt = custom_prompt
                active_schema = st.session_state["extraction_custom_schema"]
                with st.expander("👁️ Preview Generated JSON Schema", expanded=True):
                    st.json(active_schema)
                    st.code(json.dumps(active_schema, indent=2), language="json")
            else:
                active_prompt = None
                active_schema = None
                st.info("👆 Edit the prompt above and click 'Auto-Generate JSON Schema' to proceed.")

        # Store active prompt/schema for the processing loop
        st.session_state["extraction_active_prompt"] = active_prompt
        st.session_state["extraction_active_schema"] = active_schema

        st.divider()

        # =========================================================================
        # PDF UPLOAD
        # =========================================================================
        st.subheader("📄 Upload PDFs")

        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="All uploaded PDFs will be saved temporarily and processed sequentially.",
            key="extraction_pdf_upload"
        )

        # Also check for previously uploaded PDFs in temp folder
        existing_pdfs = [f for f in os.listdir(PDF_UPLOAD_DIR) if f.lower().endswith(".pdf")]

        # Save newly uploaded files to temp
        if uploaded_files:
            for uf in uploaded_files:
                dest = os.path.join(PDF_UPLOAD_DIR, uf.name)
                if not os.path.exists(dest):
                    with open(dest, "wb") as f:
                        f.write(uf.getbuffer())

        # Refresh file list
        all_pdfs = sorted(set(os.listdir(PDF_UPLOAD_DIR)))
        all_pdfs = [f for f in all_pdfs if f.lower().endswith(".pdf")]

        if all_pdfs:
            st.success(f"📂 {len(all_pdfs)} PDF(s) ready for extraction in `{PDF_UPLOAD_DIR}`")
        else:
            st.warning("⚠️ No PDFs found. Upload files above.")
            return

        st.divider()

        # =========================================================================
        # RUN EXTRACTION
        # =========================================================================
        can_run = bool(api_key) and len(all_pdfs) > 0
        is_running = st.session_state.get("extraction_is_running", False)

        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if is_running:
                if st.button("🛑 Stop Extraction", type="secondary", key="extraction_stop_btn"):
                    st.session_state["extraction_is_running"] = False
                    st.rerun()
            else:
                if st.button("🚀 Run Extraction", type="primary", disabled=not can_run, key="extraction_run_btn"):
                    st.session_state["extraction_is_running"] = True
                    st.rerun()

        # PROCESSING LOOP
        if st.session_state.get("extraction_is_running", False):
            if not api_key:
                st.session_state["extraction_is_running"] = False
                st.error("❌ API key required.")
                st.rerun()

            client = Mistral(api_key=api_key)

            # Load existing progress
            results = load_extraction_progress()
            processed_files = {item.get("source_filename") for item in results if "source_filename" in item}
            to_process = [f for f in all_pdfs if f not in processed_files]

            total = len(to_process)
            completed = 0
            errors = 0

            progress_bar = st.progress(0, text=f"Progress: 0/{total}")
            status_text = st.empty()
            log_container = st.container()

            if total == 0:
                st.session_state["extraction_is_running"] = False
                st.success(f"✅ All {len(all_pdfs)} PDFs already extracted!")
                st.rerun()

            for fname in to_process:
                if not st.session_state.get("extraction_is_running", True):
                    break

                pdf_path = os.path.join(PDF_UPLOAD_DIR, fname)
                status_text.text(f"🔄 Processing: **{fname}**")

                try:
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                except Exception as e:
                    log_container.error(f"❌ Cannot read {fname}: {e}")
                    errors += 1
                    completed += 1
                    progress_bar.progress(completed / total, text=f"Progress: {completed}/{total}")
                    continue

                _active_prompt = st.session_state.get("extraction_active_prompt")
                _active_schema = st.session_state.get("extraction_active_schema")

                if not _active_prompt or not _active_schema:
                    log_container.error("❌ No active prompt/schema configured. Stop and configure in Settings.")
                    st.session_state["extraction_is_running"] = False
                    break

                data_json, success = extract_single_pdf(
                    client=client,
                    model_name=primary_model,
                    pdf_bytes=pdf_bytes,
                    fname=fname,
                    log_container=log_container,
                    prompt_text=_active_prompt,
                    json_schema=_active_schema,
                    fallback_models=fallback_models,
                    use_fallback=use_fallback
                )

                if success and data_json:
                    results.append(data_json)
                    save_extraction_progress(results)
                    # AUTO-SAVE: Already handled by save_extraction_progress() 
                    # which writes to temp/structured_output.json on every record
                else:
                    errors += 1

                completed += 1
                progress_bar.progress(completed / total, text=f"Progress: {completed}/{total}")

            # Finalize
            st.session_state["extraction_is_running"] = False
            st.session_state["extraction_complete"] = True
            st.session_state["extraction_results_count"] = len(results)

            progress_bar.progress(1.0, text="✅ Extraction complete!")
            status_text.empty()
            st.success(f"✅ **Extraction Complete!** Extracted: {len(results)} | Errors: {errors} | Total PDFs: {len(all_pdfs)}")
            st.rerun()

    # =========================================================================
    # RESULTS DISPLAY & DOWNLOAD
    # =========================================================================
    results = load_extraction_progress()

    if results:
        st.divider()
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("Extracted Records", len(results))
        with metric_cols[1]:
            st.metric("Output File", EXTRACTION_OUTPUT)
        with metric_cols[2]:
            all_pdfs_count = len([f for f in os.listdir(PDF_UPLOAD_DIR) if f.lower().endswith(".pdf")])
            remaining = all_pdfs_count - len(results)
            st.metric("Remaining PDFs", max(0, remaining))

        with st.expander("👀 Preview Extracted Data (first 5 records)"):
            # Show ALL fields for each record; serialize complex objects to JSON strings for display
            preview_data = []
            for r in results[:5]:
                row = {}
                for key, value in r.items():
                    if isinstance(value, (dict, list)):
                        row[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        row[key] = value
                preview_data.append(row)
            st.dataframe(preview_data, use_container_width=True)

        st.divider()
        st.subheader("📥 Download Results")

        json_str = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            label="📊 Download Structured Output (JSON)",
            data=json_str,
            file_name="structured_output.json",
            mime="application/json",
            type="primary",
            key="extraction_download_btn"
        )

        with st.expander("🔬 Reproducibility & Accessibility Information", expanded=False):
            repro = {
                "stage": "extraction",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "audit_trail": {
                    "primary_model": st.session_state.get("extraction_primary_model", "N/A"),
                    "fallback_enabled": st.session_state.get("extraction_use_fallback", False),
                    "fallback_models": st.session_state.get("extraction_fallback_models", []),
                    "prompt_mode": st.session_state.get("extraction_prompt_mode", "Default"),
                    "active_prompt": st.session_state.get("extraction_active_prompt", LOCKED_PROMPT),
                    "json_schema": st.session_state.get("extraction_active_schema", JSON_SCHEMA),
                    "temperature": 0,
                    "max_retries": MAX_RETRIES,
                    "base_delay": BASE_DELAY,
                },
                "results_summary": {
                    "total_extracted": len(results),
                    "output_file": EXTRACTION_OUTPUT,
                    "pdf_directory": PDF_UPLOAD_DIR,
                    "source_filename_tracking": "Each JSON object has 'source_filename' as first key for traceability",
                    "missing_data_handling": "'N/A' for absent fields; '(inferred)' appended to inferred values",
                },
            }
            st.json(repro)
            st.code(json.dumps(repro, indent=2), language="json")

            st.divider()

            # Reproducibility Checklist
            st.markdown("#### ✅ Reproducibility Checklist")
            st.markdown("""
            - [x] Exact prompt and JSON schema archived above
            - [x] Model ID, API endpoint, and temperature (0) recorded
            - [x] Input PDFs preserved in `{pdf_dir}`
            - [x] Structured output saved to `{output_file}`
            - [x] Source filename tracking enabled for every record
            - [ ] Document any manual interventions or schema adjustments post-extraction
            """.format(pdf_dir=PDF_UPLOAD_DIR, output_file=EXTRACTION_OUTPUT))

            st.divider()

            # Accessibility Alternatives
            st.markdown("#### ♿ Accessibility Alternatives")
            st.markdown("""
            | Scenario | Alternative |
            |---|---|
            | **API rate limits / quota exhaustion** | Process PDFs in smaller batches; enable multi-model fallback |
            | **Non-programmers** | Use the downloaded `structured_output.json` directly; all downstream stages accept this format |
            """)
    else:
        st.info("ℹ️ No extraction results yet. Upload PDFs and run extraction above.")