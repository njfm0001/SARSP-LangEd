"""
SARSP-LangEd - Stage 5: Rule-Based Normalisation
Dual-mode: Default pipeline normalisation OR custom light normalisation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import zipfile
from datetime import datetime

# Set style once at module level (not inside render function)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Consistent temp directory structure
from core.utils import get_session_temp_dir
TEMP_DIR = get_session_temp_dir()
STAGE_DIR = os.path.join(TEMP_DIR, "stage5_normalization")
AUTO_SAVE_DIR = os.path.join(STAGE_DIR, "auto_save")
FIGURES_DIR = os.path.join(STAGE_DIR, "figures")

# Ensure directories exist at module load
for d in [STAGE_DIR, AUTO_SAVE_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)


# =============================================================================
# DEFAULT NORMALIZATION FUNCTIONS (from original pipeline)
# =============================================================================
# They are ONLY used when mode == "default".


# ----------------------------
# Human / non-human terms
# ----------------------------
HUMAN_TERMS = (
    r'(participants?|students?|learners?|teachers?|instructors?|'
    r'respondents?|interviewees?|writers?|speakers?|people|individuals|'
    r'pupils?|tutors?|educators?|faculty|lecturers?|professors?|'
    r'raters?|graders?|assessors?|evaluators?|researchers?|'
    r'children|parents?|dyads?|pairs?|casestud(?:y|ies)|'
    r'pre[- ]service|in[- ]service|preservice|inservice|'
    r'undergraduates?|graduates?|postgraduates?|majors?|'
    r'freshmen|sophomores?|juniors?|seniors?|'
    r'efl|esl|eal|elt|tesol|tefl|'
    r'non[- ]english|english[- ]major|english[- ]language|'
    r'engineers?|doctors?|phds?|masters?|bachelors?|'
    r'females?|males?|internationals?|chinese|korean|japanese|'
    r'novices?|experts?|experienced|'
    r'focus[- ]?groups?|sessions?|workshops?|'
    r'classes?|groups?|samples?|cohorts?|'
    r'survey|questionnaire|interview|test|exam|phase|'
    r'data|study|analysis|response|valid|final|'
    r'\betc\.|\bincluding\b|\bwith\b|\bfrom\b|\band\b|\bor\b|\bthe\b|\ba\b|\ban\b|\bfor\b|\bon\b|\bby\b|\bin\b|\bat\b)'
)

NON_HUMAN_TERMS = (
    r'(texts?|essays?|tasks?|responses?|sentences?|tokens?|videos?|'
    r'tools?|prompts?|comments?|corpus|dataset|questions?|items?|'
    r'documents?|reports?|statements?|sections?|articles?|papers?|'
    r'writings?|translations?|generations?|annotations?|ratings?|'
    r'scores?|marks?|grades?|results?|findings?|'
    r'chats?|sessions?|conversations?|interactions?|dialogues?|'
    r'stories?|narratives?|reflections?|journals?|logs?|'
    r'corpora|datasets|examples?|samples?|instances?|'
    r'words?|phrases?|paragraphs?|chapters?|books?|'
    r'lessons?|plans?|materials?|resources?|contents?|'
    r'posts?|threads?|comments?|videos?|youtubers?|'
    r'models?|systems?|algorithms?|llms?|chatgpt|ai|genai|'
    r'versions?|variants?|types?|forms?|kinds?|'
    r'conditions?|settings?|environments?|contexts?|'
    r'tests?|exams?|quizzes?|assessments?|evaluations?|'
    r'analyses?|studies?|research|experiments?|trials?|'
    r'data|information|details|specifications?|'
    r'\bwrit\b|\bgen\b|\bai\b|\bllm\b|\bchat\b|\bgpt\b)'
)

# ----------------------------
# Word numbers (with capitals)
# ----------------------------
WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100, "thousand": 1000,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10
}

# Add capitalized versions
WORD_NUMS.update({k.capitalize(): v for k, v in WORD_NUMS.items()})

# ----------------------------
# Helper function to detect if a number is part of a non-human term
# ----------------------------
def is_human_context(text, position, number_str):
    """Check if the number is in a human context"""
    # Look backwards for human terms within 10 words
    start = max(0, position - 100)  # Look back up to 100 chars
    preceding_text = text[start:position].lower()
    
    # Check if preceded by human indicators
    human_indicators = [
        r'\b(participants?|students?|learners?|teachers?|instructors?|'
        r'respondents?|interviewees?|writers?|speakers?|people|individuals|'
        r'pupils?|tutors?|educators?|faculty|lecturers?|professors?|'
        r'raters?|graders?|assessors?|evaluators?|researchers?|'
        r'children|parents?|dyads?|pairs?|casestud(?:y|ies)|'
        r'undergraduates?|graduates?|postgraduates?|majors?|'
        r'females?|males?|internationals?|chinese|korean|japanese)'
    ]
    
    for pattern in human_indicators:
        if re.search(pattern, preceding_text[-50:]):  # Check last 50 chars
            return True
    
    # Look forwards for human terms within 5 words
    end = min(len(text), position + len(number_str) + 50)
    following_text = text[position + len(number_str):end].lower()
    
    # Remove common false positives like "L2", "B1", etc.
    following_text = re.sub(r'\b[a-z]\d\b', ' ', following_text)  # Remove L2, B1, etc.
    following_text = re.sub(r'\b\d+[a-z]+\b', ' ', following_text)  # Remove 10th, 2nd, etc.
    
    # Check if followed by human terms
    human_patterns = [
        r'^\s*(participants?|students?|learners?|teachers?|instructors?|'
        r'respondents?|interviewees?|writers?|speakers?|people|individuals)',
        r'\sof\s+(?:the\s+)?(participants?|students?|learners?|teachers?)',
        r'\s+(participants?|students?|learners?|teachers?)'
    ]
    
    for pattern in human_patterns:
        if re.search(pattern, following_text):
            return True
    
    return False

def normalize_sample_size(entry):
    if not entry or not isinstance(entry, str):
        return None

    original_text = entry
    text = entry.lower()
    
    # ---------- STEP 0: Handle special cases first ----------
    # Remove "N = " pattern and extract number
    text = re.sub(r'\bn\s*[=:]\s*(\d+)', r'\1', text)
    
    # Handle "N/A" cases
    if re.search(r'\b(n/?a|no human participants|not applicable|design and development study)', text):
        return None
    
    # ---------- STEP 1: Clean and preprocess ----------
    # Remove thousands separators
    text = re.sub(r'(\d),(?=\d{3}\b)', r'\1', text)
    
    # Remove content in parentheses that might confuse things
    text = re.sub(r'\([^)]*\)', ' ', text)
    
    # Remove percentage signs and other non-relevant symbols
    text = re.sub(r'%\s*', ' ', text)
    
    # ---------- STEP 2: Convert word numbers ----------
    # First, protect L2, B1, etc. by replacing with placeholders
    protected_terms = {}
    l2_matches = list(re.finditer(r'\b[a-z]\d\b', text, re.IGNORECASE))
    for i, match in enumerate(l2_matches):
        placeholder = f"__PROTECTED_{i}__"
        protected_terms[placeholder] = match.group()
        text = text.replace(match.group(), placeholder)
    
    # Convert word numbers to digits
    for word, num in WORD_NUMS.items():
        word_pattern = rf'\b{word}\b'
        # Replace standalone word numbers
        text = re.sub(word_pattern, str(num), text)
    
    # Restore protected terms
    for placeholder, original in protected_terms.items():
        text = text.replace(placeholder, original)
    
    # ---------- STEP 3: Extract all potential human numbers ----------
    human_numbers = []
    
    # Pattern 1: Direct numbers followed by human terms
    direct_pattern = r'(\d+)\s+(?:[a-z]+\s+){0,3}(participants?|students?|learners?|teachers?|instructors?|respondents?|interviewees?|writers?|speakers?|people|individuals|pupils?|tutors?|educators?|faculty|lecturers?|professors?|raters?|graders?|assessors?|evaluators?|researchers?|children|parents?|undergraduates?|graduates?|postgraduates?|majors?)'
    
    for match in re.finditer(direct_pattern, text, re.IGNORECASE):
        num = int(match.group(1))
        # Check if this is really a human count (not part of "Grade 10" etc.)
        context_check = text[max(0, match.start()-20):match.end()+20]
        if not re.search(r'\bgrade\s+\d+', context_check, re.IGNORECASE):
            human_numbers.append(num)
    
    # Pattern 2: Numbers in phrases like "X for surveys, Y for interviews"
    survey_patterns = [
        r'(\d+)\s+(?:for\s+)?(?:the\s+)?(?:quantitative|qualitative|survey|questionnaire)',
        r'(\d+)\s+(?:for\s+)?(?:interviews?|focus\s+groups?)',
        r'(\d+)\s+(?:participants?|students?|learners?|teachers?)'
    ]
    
    for pattern in survey_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            num = int(match.group(1))
            # Check context to avoid non-human counts
            if is_human_context(text, match.start(), match.group(1)):
                human_numbers.append(num)
    
    # Pattern 3: Handle "X students and Y teachers" patterns
    and_pattern = r'(\d+)\s+(?:[a-z]+\s+){0,2}(?:students?|learners?|teachers?|instructors?)(?:\s+and\s+|\s*,\s*)(\d+)\s+(?:[a-z]+\s+){0,2}(?:students?|learners?|teachers?|instructors?)'
    for match in re.finditer(and_pattern, text, re.IGNORECASE):
        human_numbers.append(int(match.group(1)))
        human_numbers.append(int(match.group(2)))
    
    # ---------- STEP 4: Extract standalone numbers in human context ----------
    # Find all numbers in the text
    all_numbers = re.findall(r'\b\d+\b', text)
    
    for num_str in all_numbers:
        position = text.find(num_str)
        if position != -1:
            if is_human_context(text, position, num_str):
                num = int(num_str)
                if num not in human_numbers:
                    human_numbers.append(num)
    
    # ---------- STEP 5: Handle special patterns ----------
    # Pattern for "X (Y for A, Z for B)"
    split_pattern = r'(\d+)\s*\([^)]*\)'
    for match in re.finditer(split_pattern, text):
        # Take the number before parentheses
        human_numbers.append(int(match.group(1)))
    
    # Pattern for ranges like "X-Y"
    range_pattern = r'(\d+)\s*[-–]\s*(\d+)'
    for match in re.finditer(range_pattern, text):
        # Take the larger number in ranges
        human_numbers.append(max(int(match.group(1)), int(match.group(2))))
    
    # ---------- STEP 6: Filter out obviously non-human numbers ----------
    filtered_numbers = []
    for num in human_numbers:
        # Check if number appears in non-human context
        num_pattern = rf'\b{num}\b'
        num_matches = list(re.finditer(num_pattern, text))
        
        human_context_found = False
        for match in num_matches:
            if is_human_context(text, match.start(), str(num)):
                human_context_found = True
                break
        
        if human_context_found:
            filtered_numbers.append(num)
    
    human_numbers = filtered_numbers
    
    # ---------- STEP 7: Remove duplicates and very small numbers from large sets ----------
    if human_numbers:
        # Remove duplicates while preserving order
        seen = set()
        unique_numbers = []
        for num in human_numbers:
            if num not in seen:
                seen.add(num)
                unique_numbers.append(num)
        
        human_numbers = unique_numbers
        
        # If we have both small and large numbers, prioritize larger ones
        # (e.g., to avoid counting raters when there are many students)
        if len(human_numbers) > 1:
            max_num = max(human_numbers)
            min_num = min(human_numbers)
            
            # If max is at least 10x larger than min, and min < 50, likely min is raters/teachers
            if max_num >= min_num * 10 and min_num < 50:
                human_numbers = [max_num]
        
        # Return sum if multiple reasonable numbers found
        if len(human_numbers) > 1:
            # Check if numbers are likely complements (e.g., survey + interview)
            total = sum(human_numbers)
            avg = total / len(human_numbers)
            
            # If numbers are of similar magnitude, they might be groups
            # If one is much larger, it's probably the main sample
            if max(human_numbers) <= min(human_numbers) * 5:
                return total
            else:
                return max(human_numbers)
        elif len(human_numbers) == 1:
            return human_numbers[0]
    
    # ---------- STEP 8: Check for obvious non-human studies ----------
    non_human_keywords = [
        r'\b(corpus|corpora|dataset|texts?|essays?|sentences?|tokens?|words?|'
        r'videos?|posts?|comments?|questions?|items?|prompts?|'
        r'lessons?|plans?|materials?|tools?|systems?|models?|'
        r'responses?|generations?|outputs?|translations?|'
        r'analysis|evaluation|assessment|rating|scoring|grading)\b'
    ]
    
    has_non_human = False
    for pattern in non_human_keywords:
        if re.search(pattern, text, re.IGNORECASE):
            # Check if it's ONLY non-human (no human terms at all)
            human_terms_present = re.search(
                r'\b(participants?|students?|learners?|teachers?|'
                r'people|individuals|respondents?|interviewees?)\b',
                text, re.IGNORECASE
            )
            if not human_terms_present:
                has_non_human = True
                break
    
    if has_non_human:
        return None
    
    # ---------- STEP 9: Fallback to largest number in text ----------
    # But only if it's reasonable (not a year, page number, etc.)
    all_nums = [int(n) for n in re.findall(r'\b\d+\b', text)]
    
    # Filter out unlikely sample sizes (years, small counts for non-human)
    reasonable_nums = []
    for num in all_nums:
        # Sample sizes are usually > 0 and < 100000 for linguistics studies
        # Also filter out years (1900-2100)
        if 0 < num < 100000 and not (1900 <= num <= 2100):
            reasonable_nums.append(num)
    
    if reasonable_nums:
        # Return the largest reasonable number
        return max(reasonable_nums)
    
    return None

def categorize_sample_size(sample_size):
    """Categorize sample sizes into bands similar to age normalization"""
    if sample_size is None or sample_size == "N/A":
        return ["N/A"]
    sample_size = normalize_sample_size(sample_size)
    try:
        n = int(sample_size)
    except (ValueError, TypeError):
        return ["N/A"]
    
    # Define sample size bands (similar to age bands structure)
    SAMPLE_SIZE_BANDS = [
        (1, 10, "1-10"),
        (11, 30, "11-30"),
        (31, 100, "31-100"),
        (101, 300, "101-300"),
        (301, 1000, "301-1000"),
        (1001, 5000, "1001-5000"),
        (5001, float('inf'), "5001+")
    ]
    
    # Find the appropriate band
    for low, high, label in SAMPLE_SIZE_BANDS:
        if low <= n <= high:
            return [label]
    
    return ["N/A"]

def normalize_age(raw, debug=False):
    if not raw or not isinstance(raw, str):
        return ["N/A"]
    AGE_BANDS = [
        (0, 12, "0-12"),
        (13, 17, "13-17"),
        (18, 25, "18-25"),
        (26, 35, "26-35"),
        (36, 45, "36-45"),
        (46, 55, "46-55"),
        (56, 65, "56-65"),
        (66, 100, "66+")
    ]

    text = raw.lower().strip()

    # ---------- STEP 0: stats short-circuit (TYPE 1 FIX) ----------
    STATS_PAT = r'\b(mean|sd|average|m\s*=|sd\s*=)\b'
    EXPLICIT_RANGE_PAT = (
        r'\b\d{2}\s*[-–]\s*\d{2}\b'
        r'|\b\d{2}\s*(?:to|between)\s*\d{2}\b'
        r'|\baged\s+\d{2}\s*[-–]\s*\d{2}\b'
        r'|\b\d{2}\+\b'
        r'|>\s*\d{2}'
    )

    if re.search(STATS_PAT, text) and not re.search(EXPLICIT_RANGE_PAT, text):
        if debug:
            print(f"EXCLUDED (stats, no explicit age range): {raw}")
        return ["N/A"]

    # ---------- STEP 1: explicit N/A ----------
    if re.search(r'\b(n/?a|not specified|unspecified)\b', text):
        if debug:
            print(f"EXCLUDED (N/A): {raw}")
        return ["N/A"]

    # ---------- STEP 2: strip metadata ----------
    text = re.sub(r'\([^)]*(mean|sd)[^)]*\)', '', text)
    text = re.sub(r'\d+(?:\.\d+)?\s*%', '', text)

    # ---------- STEP 3: experience rejection ----------
    if re.search(r'\b\d+(?:\.\d+)?\s+years?\s+of\s+experience\b', text):
        if debug:
            print(f"EXCLUDED (experience): {raw}")
        return ["N/A"]

    if re.search(r'\bexperience\b', text) and not re.search(r'\bage|aged|years old\b', text):
        if debug:
            print(f"EXCLUDED (experience): {raw}")
        return ["N/A"]

    age_ranges = []

    # ---------- STEP 4: extract explicit ranges ----------
    for m in re.finditer(r'(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})', text):
        lo, hi = int(m.group(1)), int(m.group(2))
        if 5 <= lo <= hi <= 100:
            age_ranges.append((lo, hi))

    # ---------- STEP 5: under / over ----------
    for m in re.finditer(r'(?:under|<)\s*(\d{1,2})', text):
        age_ranges.append((0, int(m.group(1)) - 1))

    for m in re.finditer(r'(?:over|>|above)\s*(\d{1,2})', text):
        age_ranges.append((int(m.group(1)) + 1, 200))

    # ---------- STEP 6: isolated ages (safe fallback) ----------
    if not age_ranges:
        for m in re.finditer(r'\b(\d{1,2})\b', text):
            age = int(m.group(1))
            if 5 <= age <= 100:
                age_ranges.append((age, age))

    if not age_ranges:
        if debug:
            print(f"NO AGE FOUND: {raw}")
        return ["N/A"]

    # ---------- STEP 7: map to bands ----------
    bands = set()
    for lo, hi in age_ranges:
        for b_lo, b_hi, label in AGE_BANDS:
            if lo <= b_hi and hi >= b_lo:
                bands.add(label)

    bands = sorted(
        bands,
        key=lambda x: [b[2] for b in AGE_BANDS].index(x)
    )

    if debug:
        print(f"OK: '{raw}' → ranges={age_ranges} → bands={bands}")

    return bands

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}


def normalize_gender(value, debug=False):
    """
    Normalize gender to a LIST of labels (pipeline-compatible).

    Each occurrence in the list represents ONE participant.
    """
    if not value or not isinstance(value, str):
        return ["N/A"]

    raw = value.strip()
    if not raw:
        return ["N/A"]

    text = raw.lower()

    # ---------- STEP 0: explicit N/A ----------
    if re.fullmatch(r'n/a(\s*\(.*\))?', text):
        if debug:
            print(f"EXPLICIT N/A: {raw}")
        return ["N/A"]

    vague_phrases = [
        "various genders", "mixed genders", "diverse genders",
        "participants of all genders", "all genders",
        "mixed-gender composition", "names suggest male and female"
    ]
    if any(p in text for p in vague_phrases):
        if debug:
            print(f"VAGUE GENDER: {raw}")
        return ["N/A"]

    # ---------- helpers ----------
    def normalize_token(tok):
        tok = tok.strip().lower()
        if tok in ["female", "females", "woman", "women", "girl", "girls"]:
            return "female"
        if tok in ["male", "males", "man", "men", "boy", "boys"]:
            return "male"
        if tok in ["non-binary", "nonbinary"]:
            return "non-binary"
        if tok in ["other", "transgender", "trans"]:
            return "other"
        if tok in [
            "prefer not to say", "preferred not to respond",
            "unknown", "undisclosed", "not disclosed",
            "unspecified", "not specified"
        ]:
            return "N/A"
        return None

    counts = defaultdict(int)

    # ---------- STEP 1: split major groups ----------
    groups = re.split(r';\s*|\s+and\s+for\s+the\s+', text)

    for group in groups:
        group = group.strip()
        if not group:
            continue

        if debug:
            print(f"\nDEBUG: Processing group: '{group}'")

        # ---------- STEP 2a: parenthetical counts ----------
        for g, n in re.findall(
            r'\b(female|male|non[- ]?binary|other|transgender|'
            r'prefer not to say|preferred not to respond|unknown|'
            r'undisclosed|not disclosed|unspecified)\s*\(\s*(\d+)\s*\)',
            group
        ):
            norm = normalize_token(g)
            if norm:
                counts[norm] += int(n)
                if debug:
                    print(f"DEBUG: Parenthetical count → {norm}: {n}")
            group = re.sub(rf'\b{re.escape(g)}\s*\(\s*{n}\s*\)', '', group)

        # ---------- STEP 2a-b: semantic parentheticals (BLOCKED for %) ----------
        if '%' not in group:
            for g, content in re.findall(
                r'\b(female|male|non[- ]?binary|other|transgender)\s*\(([^)]*)\)',
                group
            ):
                if re.search(r'\d|%', content):
                    continue
                norm = normalize_token(g)
                if norm and counts[norm] == 0:
                    counts[norm] += 1
                    if debug:
                        print(f"DEBUG: Semantic inference → {norm}: +1")
                group = re.sub(rf'\b{re.escape(g)}\s*\([^)]*\)', '', group)

        # ---------- STEP 2b: explicit counts ----------
        for n, g in re.findall(
            r'(\d+)\s*(?:\w+\s+)?'
            r'(female|male|non[- ]?binary|other|transgender|'
            r'prefer not to say|preferred not to respond|unknown|'
            r'undisclosed|not disclosed|unspecified)s?\b',
            group
        ):
            norm = normalize_token(g)
            if norm:
                counts[norm] += int(n)
                if debug:
                    print(f"DEBUG: Explicit count → {norm}: {n}")
            group = re.sub(rf'\b{n}\s*(?:\w+\s+)?{re.escape(g)}s?\b', '', group)

        # ---------- STEP 2c: percentage-only groups ----------
        if '%' in group and not counts:
            if debug:
                print("DEBUG: Percentage-only group → ignored")
            continue

        # ---------- STEP 2d: bare tokens ----------
        tokens = re.split(r'[,&;]|\s+and\s+|\s+or\s+', group)
        for t in tokens:
            norm = normalize_token(t)
            if norm and counts[norm] == 0 and norm != "N/A":
                counts[norm] += 1
                if debug:
                    print(f"DEBUG: Bare token → {norm}: +1")

    # ---------- STEP 3: final unspecified clause ----------
    m = re.search(r'final\s+(\d+)\s+participants\s+is\s+not\s+specified', text)
    if m:
        counts["N/A"] += int(m.group(1))
        if debug:
            print(f"DEBUG: Final unspecified → N/A: {m.group(1)}")

    # ---------- STEP 4: materialise LIST ----------
    output = []
    for gender, n in counts.items():
        output.extend([gender] * n)

    if not output:
        return ["N/A"]

    if debug:
        print(f"\nFINAL OUTPUT LIST: {output}")

    return output


# print("TESTING PROBLEMATIC CASES:")
# for ex in gender_data:
#     genders, counts = normalize_gender(ex)
#     print(f"{ex} -> {genders}, counts: {counts}\n")


def normalise_region_label_simple(s):
    """
    Simplified region normalization that handles key edge cases and capitalizes.
    """
    if not isinstance(s, str) or not s.strip():
        return None
    
    original = s.strip()
    key = original.lower()
    
    # Handle N/A cases first
    if key in ['n/a', 'na', 'none', 'unknown', 'not specified', '']:
        return "N/A"
    
    # Only keep essential equivalences for critical standardization
    essential_equivalences = {
        # China administrative regions
        "hong kong sar": "Hong Kong",
        "macao": "Macau", 
        
        # Critical China directional standardization
        "east china": "Eastern China",
        "shangai": "Eastern China",
        "south china": "Southern China", 
        "north china": "Northern China",
        "west china": "Western China",
        "central china": "Central China",
        "southeast china": "Southeastern China",
        "southwest china": "Southwestern China", 
        "northeast china": "Northeastern China",
        "northwest china": "Northwestern China",
        "south-eastern china": "Southeastern China",
        "a third-tier city in northern china": "Northern China",
        "nanjian yi autonomous county": "Yunnan",
        "nanjian yi autonomous , dali bai autonomous , yunnan": "Yunnan",
        
        # Critical US region standardization
        "midwest": "Midwestern US",
        "midwestern u.s.": "Midwestern US", 
        "midwestern us": "Midwestern US",
        "eastern us": "Eastern US",
        "western us": "Western US", 
        "southern us": "Southern US",
        "northern us": "Northern US",
        "rural northeast u.s.": "Rural Northeast US",
        
        
        # Other patterns
        "centre of vietnam": "Central Vietnam",
        "souss massa region": "Souss Massa",
        "kansai region": "Kansai",
        "middle east and north africa (mena)": "Middle East and North Africa",  
        "northern border university": "Arar",
        "saudi college of applied technology": "N/A",
        "western region" : "N/A",
        'remote first nation community': "N/A",
        "multinational (nato context)": "N/A",
    }
    
    # Check essential equivalences first
    if key in essential_equivalences:
        return essential_equivalences[key]

    # Filter out overly vague or non-geographic terms
    vague_terms = ['various', 'different', 'multiple', 'diverse', 'including', 
                   'unnamed', 'colleges', 'universities', 'regions', "provinces"]
    non_geo_terms = ['online', 'global', 'remote', 'community', 'inferred']
    
    if (any(vague in key for vague in vague_terms) or 
        any(non_geo in key for non_geo in non_geo_terms)):
        return "N/A"
    
    # Handle "near" cases
    if key.startswith('near '):
        main_loc = key[5:].strip()
        normalized_main = normalise_region_label_simple(main_loc)
        return normalized_main
        # Handle "City, State" patterns (e.g., "Atlanta, GA")
    if re.search(r', [a-z]{2}$', key):  # Matches ", xx" where xx are two letters
        city_part = key.split(',')[0].strip()
        return city_part.title()
    
    # Remove common suffixes but be conservative
    suffixes_to_remove = r'\b(province|city|region|area|district|prefecture|county)\b'
    clean_key = re.sub(suffixes_to_remove, '', key, flags=re.IGNORECASE)
    clean_key = re.sub(r'\s+', ' ', clean_key).strip()
        # Only use cleaned version if it's clearly better
    if clean_key and clean_key != key and len(clean_key) >= 3:
        # Check if cleaned version is in essential equivalences
        if clean_key in essential_equivalences:
            return essential_equivalences[clean_key]
        # Don't use cleaned version if it's too vague
        if not any(vague in clean_key for vague in ['various', 'different', 'multiple']):
            return clean_key.title()
    
    # Final fallback - return title case for reasonable-looking locations
    if (len(original) >= 3 and 
        not re.search(r'\d', original) and  # No random numbers
        any(char.isalpha() for char in original)):
        return original.title()
    
    return "N/A"
def normalise_L1(value):
    """
    Simplified L1 normalization function that properly handles all cases.
    Returns a LIST of languages instead of a single string.
    """
    if not value or not isinstance(value, str):
        return ["N/A"]
    
    raw = value.strip()
    if not raw:
        return ["N/A"]

    lower = raw.lower()
    
    # Handle N/A cases first
    if lower in ["n/a", "na", "none", "not specified", "unspecified", "unclear"]:
        return ["N/A"]
    
    # Handle specific vague phrases that should be N/A
    vague_phrases = [
        "various languages from",
        "more than",
        "different first languages",
        "different languages",
        "various l1s",
        "various regional dialects",
        "diverse native languages",
        "internationally diverse",
        "varied (including",
        "varied (international"
    ]
    
    for phrase in vague_phrases:
        if phrase in lower:
            return ["N/A"]
    
    # ----- Helper function for single token normalization -----
    def normalise_single_token(v):
        if not v or not isinstance(v, str):
            return None
        
        v = v.strip().rstrip(",.;").lower()
        if not v:
            return None

        # Skip these specific non-language terms
        non_language_terms = ["territories", "countries", "regions", "dialects", 
                             "varieties", "participants", "school", "university",
                             "corpus", "corpora", "students", "learners", "speakers"]
        
        if v in non_language_terms:
            return None

        # Remove content in parentheses
        cleaned_v = re.sub(r'\([^)]*\)', '', v).strip()
        if not cleaned_v:
            cleaned_v = v

        # Chinese variants (all become "Chinese")
        chinese_terms = ["chinese", "mandarin", "putonghua", "cantonese", "hànyǔ", "hanyu", "taiwanese"]
        if any(term in cleaned_v for term in chinese_terms):
            return "Chinese"

        # Persian variants
        if any(term in cleaned_v for term in ["persian", "farsi", "iranian"]):
            return "Persian"

        # Turkish
        if "turkish" in cleaned_v:
            return "Turkish"

        # Arabic
        if "arabic" in cleaned_v:
            return "Arabic"

        # English (including "varieties of english")
        if "english" in cleaned_v:
            return "English"

        # Other languages - check if token equals or contains language name
        language_terms = {
            "japanese": "Japanese",
            "korean": "Korean", 
            "vietnamese": "Vietnamese",
            "thai": "Thai",
            "hindi": "Hindi",
            "urdu": "Urdu",
            "indonesian": "Indonesian",
            "bahasa": "Indonesian",
            "malay": "Malay",
            "filipino": "Filipino",
            "burmese": "Burmese",
            "mongolian": "Mongolian",
            "kazakh": "Kazakh",
            "german": "German",
            "dutch": "Dutch", 
            "spanish": "Spanish",
            "french": "French",
            "polish": "Polish",
            "italian": "Italian",
            "russian": "Russian",
            "czech": "Czech",
            "swedish": "Swedish",
            "norwegian": "Norwegian",
            "hungarian": "Hungarian",
            "finnish": "Finnish",
            "georgian": "Georgian",
            "greek": "Greek",
            "romanian": "Romanian",
            "catalan": "Catalan",
            "ukrainian": "Ukrainian",
            "amharic": "Amharic",
            "gujarati": "Gujarati",
            "ojibwe": "Ojibwe",
            "singhala": "Sinhala",
            "hebrew": "Hebrew",
            "sakha": "Sakha",
            "twi": "Twi",
            "tamil": "Tamil",
            "telugu": "Telugu",
            "malayalam": "Malayalam"
        }
        
        # First check exact matches
        if cleaned_v in language_terms:
            return language_terms[cleaned_v]
        
        # Then check if token contains language term
        for term, lang_name in language_terms.items():
            if term in cleaned_v:
                return lang_name

        # Skip vague/non-language terms
        vague_terms = ["various", "diverse", "multiple", "mixed", "different", 
                      "other", "languages", "backgrounds", "international", "varied",
                      "including", "from", "and", "or", "the", "of", "for", "with"]
        if any(term == cleaned_v for term in vague_terms):
            return None

        # Skip country names that aren't languages
        country_names = ["brazil", "france", "india", "vietnam", "china", "japan", 
                        "korea", "turkey", "germany", "spain", "italy", "russia",
                        "algeria", "bangladesh", "australia", "canada", "mexico",
                        "uk", "usa", "united states", "united kingdom"]
        if cleaned_v in country_names:
            return None

        # Single word that looks like a language
        if re.match(r'^[a-z]+$', cleaned_v) and len(cleaned_v) > 2:
            # Skip if it looks like a proper noun/country
            if cleaned_v not in country_names and cleaned_v not in vague_terms:
                return cleaned_v.title()
        
        return None

    # ----- Special handling for country lists -----
    # If it contains country percentages like "China (29.6%), South Korea (24.9%)"
    # Extract and map countries to languages
    country_to_language = {
        "brazil": "Portuguese",
        "france": "French", 
        "india": "Hindi",
        "vietnam": "Vietnamese",
        "china": "Chinese",
        "south korea": "Korean",
        "korea": "Korean",
        "türkiye": "Turkish",
        "turkey": "Turkish",
        "germany": "German",
        "spain": "Spanish",
        "italy": "Italian",
        "russia": "Russian",
        "japan": "Japanese",
        "indonesia": "Indonesian",
        "thailand": "Thai",
        "saudi arabia": "Arabic",
        "iran": "Persian",
        "egypt": "Arabic",
        "mexico": "Spanish",
        "greece": "Greek",
        "pakistan": "Urdu"
    }
    
    # Check for country percentage patterns
    country_percent_pattern = r'([a-zA-Z\s]+)\s*\((\d+\.?\d*%)\)'
    country_matches = re.findall(country_percent_pattern, lower)
    if country_matches:
        languages = []
        for country, _ in country_matches:
            country = country.strip().lower()
            if country in country_to_language:
                languages.append(country_to_language[country])
        if languages:
            return sorted(set(languages))
    
    # ----- Handle percentage patterns for languages -----
    if '(' in raw and any(x in raw for x in ['%']):
        languages = []
        pattern = r'([a-zA-Z\s]+)\s*\((\d+\.?\d*%)\)'
        matches = re.findall(pattern, raw)
        for lang, _ in matches:
            norm_lang = normalise_single_token(lang)
            if norm_lang:
                languages.append(norm_lang)
        
        if languages:
            return sorted(set(languages))
    
    # ----- Handle "varieties of X" pattern -----
    varieties_match = re.search(r'varieties of ([a-zA-Z]+)', lower)
    if varieties_match:
        language = varieties_match.group(1).strip()
        norm_lang = normalise_single_token(language)
        if norm_lang:
            return [norm_lang]
    
    # ----- Handle inferred cases -----
    if "inferred" in lower:
        # Extract language after "inferred"
        inferred_match = re.search(r'inferred\s+(?:as\s+)?([a-zA-Z]+)', lower)
        if inferred_match:
            inferred_lang = normalise_single_token(inferred_match.group(1))
            if inferred_lang:
                return [inferred_lang]
    
    # ----- Clean and split text -----
    # Remove content in parentheses but keep the text before parentheses
    cleaned = raw
    
    # Split on various separators - including "and" at the end of lists
    separators = r'[,;]|\s+and\s+|\s+or\s+|&'
    tokens = re.split(separators, cleaned)
    
    # Clean each token
    cleaned_tokens = []
    for token in tokens:
        token = token.strip()
        # Remove trailing punctuation and whitespace
        token = re.sub(r'^\W*(.*?)\W*$', r'\1', token)
        # Remove content in parentheses from individual tokens
        token = re.sub(r'\([^)]*\)', '', token).strip()
        if token and len(token) > 1:
            cleaned_tokens.append(token)

    if not cleaned_tokens:
        return ["N/A"]

    # ----- Normalize each token -----
    normalised = []
    for token in cleaned_tokens:
        norm_token = normalise_single_token(token)
        if norm_token:
            normalised.append(norm_token)

    if not normalised:
        return ["N/A"]

    # ----- Process the results -----
    # Remove duplicates while preserving order
    seen = set()
    unique_langs = []
    for lang in normalised:
        if lang not in seen:
            seen.add(lang)
            unique_langs.append(lang)
    
    # If only one language type
    if len(unique_langs) == 1:
        return [unique_langs[0]]
    
    # If Chinese variants only
    if all(lang == "Chinese" for lang in unique_langs):
        return ["Chinese"]
    
    # For reasonable number of languages, return them
    if len(unique_langs) <= 15:  # Increased limit to handle longer lists
        return sorted(unique_langs)
    else:
        # Too many languages
        return ["N/A"]


def normalise_target_language(value):
    """
    Normalize target language strings with comprehensive handling of:
    - Brackets removal
    - Comma/slash separation
    - Chinese variants consolidation
    - Vague descriptions
    - Multiple languages extraction
    """
    if not value or not isinstance(value, str):
        return ["N/A"]
    
    original = value.strip().replace(".", "")
    if not original:
        return ["N/A"]
    
    # Handle N/A cases first
    lower = original.lower()
    if any(item in lower for item in ['n/a', 'na', 'none', 'unknown', 'not specified']):
        return ["N/A"]
    
    # Handle vague descriptions
    vague_terms = [
        'unspecified non-native foreign language',
        'foreign language',
    ]
    
    if any(term in lower for term in vague_terms):
        return ["N/A"]
    
    # Remove content in brackets and parentheses
    cleaned = re.sub(r'\([^)]*\)', '', original)  # Remove (content)
    cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)  # Remove [content]
    cleaned = re.sub(r'^.*including', '', cleaned)  # Remove "...including" clauses
    
    # Split on common separators
    separators = r'[,/;]| and |&'
    tokens = re.split(separators, cleaned)
    
    # Clean and normalize each token
    normalized_languages = []
    for token in tokens:
        token = token.strip()
        if not token or len(token) < 2:
            continue
            
        token_lower = token.lower()
        
        # Skip vague terms within tokens
        if any(vague in token_lower for vague in ['various', 'different', 'multiple', 'languages']):
            continue
            
        # Chinese variants consolidation
        if any(chinese_term in token_lower for chinese_term in [
            'chinese', 'mandarin', 'cantonese', 'putonghua', 'hanyu', 
            'mandarin chinese', 'cantonese chinese', 'chinese dialect'
        ]):
            normalized_languages.append("Chinese")
            continue
        
        # Skip LCTL and similar vague terms
        if any(term in token_lower for term in ['lctl', 'less commonly', 'taught language']):
            continue

        if "others" in token_lower:
            normalized_languages.append("N/A")
            continue            
        # For all other language-like tokens, simply capitalize them
        if (len(token) > 2 and 
            token.isalpha() and 
            not any(vague in token_lower for vague in ['various', 'different', 'multiple'])):
            normalized_languages.append(token.title())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_languages = []
    for lang in normalized_languages:
        if lang not in seen:
            seen.add(lang)
            unique_languages.append(lang)
    
    # Handle empty results
    if not unique_languages:
        # Check if it's a "multiple languages" case with numbers
        if any(word in lower for word in ['different languages', 'various languages', 'multiple languages']):
            if any(str(i) in lower for i in range(10)):  # Contains numbers like "32 different languages"
                return ["N/A"]
            return ["N/A"]
        return ["N/A"]    
    # If we have a reasonable number of specific languages, return them
    return unique_languages

def normalise_continent(s):
    """
    Simplified continent normalization with essential equivalences and title case.
    """
    if not isinstance(s, str) or not s.strip():
        return "N/A"
    
    original = s.strip()
    key = original.lower()
    
    # Only map meaningful changes, not capitalization differences
    continent_equivalences = {
        "latin america": "South America",
        "central america": "North America",
        "australia": "Oceania",
        "australia/oceania": "Oceania",

        # Global/unusable labels
        "global": "Global",
        "worldwide": "Global",
        "global (online)": "Global",
        "n/a": "N/A",
        "na": "N/A",
        "none": "N/A",
        "unknown": "N/A",
        "not specified": "N/A",
        "": "N/A",
    }
    
    # Check equivalences first
    if key in continent_equivalences:
        return continent_equivalences[key]    
    # Simple title case for everything else
    return original.title()

def normalise_country(s):
    """
    Simplified country normalization with essential equivalences and title case.
    """
    if not isinstance(s, str) or not s.strip():
        return "N/A"
    
    original = s.strip()
    key = re.sub(r"\s+", " ", original.lower())
    
    # Only map meaningful changes, not capitalization differences
    country_equivalences = {
        # USA variants
        "usa": "United States",
        "us": "United States",
        "u.s.": "United States",

        # UK variants
        "uk": "United Kingdom",

        # China variants
        "mainland china": "China",
        "people's republic of china": "China",
        
        # Hong Kong (treated as country in dataset)
        "hong kong sar": "Hong Kong",

        # Korea variants
        "republic of korea": "South Korea",
        "korea": "South Korea",

        # Türkiye variants
        "türkiye": "Turkey",
        "turkiye": "Turkey",
        
        # UAE variants
        "uae": "United Arab Emirates",

        # Bad / ambiguous / unusable country-level labels
        "global": "N/A",
        "worldwide": "N/A",
        "an efl country in east asia": "N/A",
        "other anglophone countries": "N/A",
        "n/a": "N/A",
        "na": "N/A",
        "none": "N/A",
        "unknown": "N/A",
        "not specified": "N/A",
        "": "N/A",
    }
    
    # Check equivalences first
    if key in country_equivalences:
        return country_equivalences[key]
    
    # Simple title case for everything else
    return original.title()

def normalise_task_type(value):
    """
    Normalize task type to standardized categories.
    Handles lists of strings
    """
    if not value:
        return "N/A"
    
    # If it's a list, process each item
    if isinstance(value, list):
        normalized_items = []
        for item in value:
            norm = normalise_task_type_single(item)
            if norm and norm != "N/A":
                normalized_items.append(norm)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_items = []
        for item in normalized_items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        
        return unique_items if unique_items else ["N/A"]
    
    # If it's a single string
    return [normalise_task_type_single(value)]

def normalise_task_type_single(task_str):
    """
    Normalize task type strings, handling all edge cases from the data.
    """
    if not isinstance(task_str, str):
        return "N/A"
    
    original = task_str.strip()
    if not original:
        return "N/A"
    
    lower = original.lower()
    
    # Handle N/A cases first
    if lower in ['n/a', 'na', 'none', 'not applicable', 'not specified']:
        return "N/A"
    
    # Remove trailing spaces
    lower = lower.rstrip()
   
   # ----- Critical specific mappings (not just capitalization) -----
    critical_mappings = {
    # ========== LANGUAGE SKILLS (Core competencies) ==========
    # Writing skills
    "writing": "Writing",
    "academic writing": "Writing",
    "essay writing": "Writing",
    "writing (for chatgpt)": "Writing",
    "writing (research proposals)": "Writing",
    "writing (academic essays, personal emails, social media posts, academic journal articles, course essays, laboratory reports, course assignments, working documents, reports, resumes, posters, slogans, presentation scripts, diaries, personal reflections)": "Writing",
    "business communication tasks": "Writing",
    "drafting corporate reports": "Writing",
    "generating persuasive emails or proposals": "Writing",
    "other (composing messages or formal responses)": "Writing",
    "other (brainstorming, revising drafts, writing formal emails, preparing reports, composing statements of purpose, drafting cover letters, learning genre-specific conventions)": "Writing",
    "other (brainstorming, drafting, revising, and editing)": "Writing",
    "other (drafting and revising)": "Writing",
    "other (english polishing, generating outlines, format editing, literature retrieval, knowledge gathering)": "Writing",
    "other (argumentative essay writing)": "Writing",
    "other (writing inquiry reports)": "Writing",
    "academic work": "Writing",
    "assignments and essays": "Writing",
    "other (consulting on writing issues)": "Writing",

        # ========== WRITING SUBCATEGORIES ==========
    "brainstorming": "Writing",
    "outlining": "Writing",
    "drafting": "Writing",
    "composing": "Writing",
    "editing": "Writing",
    "revising": "Writing",
    "proofreading": "Writing",
    "other (using ai for proofreading)": "Writing",
    "idea generation": "Writing",
    "idea development": "Writing",
    "other (idea generation)": "Writing",
    "other (brainstorming and outlining)": "Writing",
    "other (brainstorming, idea generation, paraphrasing)": "Writing",
    "other (brainstorming activities to foster critical thinking)": "Writing",
    "other (brainstorming argument elements)": "Writing",
    "other (brainstorming and idea generation)": "Writing",
    "other (idea generation/brainstorming)": "Writing",
    "other (idea generation and stylistic refinement)": "Writing",
    "other (idea generation and refinement)": "Writing",
    "generating ideas": "Writing",
    "paraphrasing": "Writing",
    "other (paraphrasing)": "Writing",
    "other (paraphrasing and synthesizing texts)": "Writing",
    "writing improvement": "Writing",
    "other (idea generation, paraphrasing, improving structure and flow)": "Writing",
    "other (using ai for full text generation)": "Writing",
    "other (argument mapping)": "Writing",
    "other (text formatting)": "Writing",
    "other (reflective writing)": "Writing",
    
    # Speaking/Oral skills
    "speaking": "Speaking",
    "conversational practice": "Speaking",
    "conversation practice": "Speaking",
    "other: conversation practice": "Speaking",
    "other (conversation practice)": "Speaking",
    "practicing conversational skills": "Speaking",
    "practicing speaking with immediate feedback": "Speaking",
    "speaking activities": "Speaking",
    "other (oral and text interactions)": "Speaking",
    "other: dialogue practice": "Speaking",
    "other (role-playing conversations)": "Speaking",
    "other (q&a exercises)": "Speaking", 
    "communication skills practice": "Speaking",
    "other (debate)": "Speaking",
    "other (role-playing conversations)": "Speaking",
    "conversation simulations": "Speaking",
    "presentation": "Speaking",
    "presentation preparation": "Speaking",
    # Role-playing & simulations
    "role-playing scenarios":  "Speaking",
    "simulating real-life language usage scenarios": "Speaking",
    "other (creating teacher-student scenarios)": "Speaking",
    
    
    
    # Reading skills
    "reading": "Reading",
    "other (comprehension exercises, rhyming exercises, cloze tests)": "Reading",
    "other (identifying main ideas, sentence structures, and counterarguments)": "Reading",
    "other (instruction and implementation of reading scheme)": "Reading",
    "other (inference tasks)": "Reading",
    "other (reapplying and refining targeted reading skills)": "Reading",
    
    # Listening skills
    "listening": "Listening",
    "answering multiple-choice questions on spoken language knowledge": "Listening",
    
    # Grammar
    "grammar": "Grammar",
    "other (writing grammatical explanations)": "Grammar",
    
    # Vocabulary
    "vocabulary": "Vocabulary",
    "vocabulary acquisition through games": "Vocabulary",
    "game-based vocabulary": "Vocabulary",
    "providing synonyms": "Vocabulary", 
    
    
    # Pronunciation
    "pronunciation": "Pronunciation",
    
    # Translation
    "translation": "Translation",
    
    # Summarization
    "summarization": "Summarization",    
    
    # ========== MULTIMODAL/TECH-ENHANCED WRITING ==========
    "writing (image-based storytelling)": "Multimodal Writing",
    "other (digital multimodal composing)": "Multimodal Writing",
    "other (digital multimodal composing, including text-to-image and text-to-video generation for brainstorming and idea creation)": "Multimodal Writing",
    "other (multimodal composition)": "Multimodal Writing",
    "other (multimodal composing)": "Multimodal Writing",
    
    # ========== ASSESSMENT & FEEDBACK ==========
    "assessment/grading": "Assessment",
    "assessment/grading (by ai)": "Assessment",
    "assessment/grading (by chatbot)": "Assessment",
    "assessment/grading (inferred from interviews)": "Assessment",
    "assessment/grading (performed by ai tools)": "Assessment",
    "assessment": "Assessment",
    "grading": "Assessment",
    "interactive quizzes": "Assessment", 
    "observations": "Assessment", 
    "other (evaluating the game's effectiveness and usability)": "Assessment",
    "feedback": "Feedback",
    "the researcher provided chatbot-generated feedback to the experimental group": "Feedback",
    "writing feedback": "Feedback",
    "feedback on essays": "Feedback",
    "providing feedback": "Feedback",
    "peer feedback": "Feedback",
    "ai models performed assessment/grading and feedback": "Assessment & Feedback",
    "error correction": "Feedback",
    "other (checking coherence and content)": "Feedback",
    "other (refining teacher talk)": "Feedback",
    
    
    # ========== PEDAGOGICAL PRACTICES ==========
    # Collaborative learning
    "collaborative writing": "Collaborative Learning",
    "group discussion": "Collaborative Learning",
    "other (collaborative learning)": "Collaborative Learning",
    "other (collaborative argumentation)": "Collaborative Learning",
    "other (group and in-class discussions)": "Collaborative Learning",
    "other (co-creating stories by selecting plot, setting, character, and theme)": "Collaborative Learning",
    "other (collaborative inquiry and discussion)": "Collaborative Learning",
    "other (collaborative reflection)": "Collaborative Learning",
    "real-time discussions": "Collaborative Learning",
    

    # Problem-based learning
    "other (solving a detective case)": "Problem-Based Learning",
    "other (conducting inquiries with npcs)": "Problem-Based Learning",
    "other (interacting with ai characters to conduct an investigation)": "Problem-Based Learning",
    "other (problem-solving assignments)": "Problem-Based Learning",
    
    # Game-based learning
    "language games": "Game-Based Learning",
    
    # Project-based learning
    "other (project-based learning)": "Project-Based Learning",    
    
    # ========== TEACHER TASKS ==========
    "lesson planning": "Lesson Planning",
    "material generation": "Material Generation",
    "curriculum development": "Curriculum Development",
    "other (microteaching, reflection)": "Teaching Practice",
    "other (material selection)": "Material Selection",
    "the ai model's task was material generation (lesson plans)": "Material Generation",    
    "other (designing instructional intervention)": "Lesson Planning",
    "other (assignment design)": "Material Generation",
    "other (creating visuals)": "Material Generation",
    "other (instructional guidance on tool use)": "Teacher Guidance",
    "other (facilitating and guiding activities)": "Teacher Guidance",
    "other (facilitating discussions)": "Teaching Practice",
    "other (facilitating learning activities)": "Teaching Practice",
    "other (instructor-led writing workshops)": "Teaching Practice",
    "other (facilitating discussions and research)": "Teaching Practice",
    "other (demonstrating chatbot use, leading discussions, encouraging reflection)": "Teaching Practice",
    "other (delivering lectures)": "Teaching Practice",
    "other (facilitating inquiry sessions)": "Teaching Practice",
    "other (demonstrating ai use)": "Teaching Practice",
    "facilitating ai-assisted writing workshops": "Teaching Practice",
    "other (specify: demonstrating ai tool usage)": "Teaching Practice",
    "other (instruction and live demonstration on chatgpt usage)": "Teaching Practice",
    "other (facilitating discussion)": "Teaching Practice",
    "other (addressing student issues)": "Teaching Practice",
    "other (overseeing technology integration)": "Teaching Practice",
    "other (course instruction)": "Teaching Practice",
    "other (instruction and guidance on ai tool use)": "Teaching Practice",
    "other (supervising ai tool use)": "Teaching Practice",
    "other (observation of student process)": "Teaching Practice",
    "other (provided initial guidance and technical support)": "Teaching Practice",
    "other (participated in interviews and curriculum consultation)": "Teaching Practice",
    "creative activities": "Material Generation",  
    
    # ========== PROFESSIONAL DEVELOPMENT ==========
    "other (participating in professional development activities)": "Professional Development",
    "other (professional development)": "Professional Development",

   
    # ========== RESEARCH & ANALYSIS ==========
    "other (concept clarifying, literature searching, data validating, format adjusting)": "Research Skills",
    "other (brainstorming, information retrieval, note organisation)": "Research Skills",
    "other (planning, drafting, revision, brainstorming, idea generation, text refinement, citing sources)": "Research Skills",
    "other (research support)": "Research Skills",
    "other (assisting with research and data analysis)": "Research Skills",
    "other (refining research scope, organizing and presenting results, interpreting data)": "Research Skills",
    "other (information gathering and synthesizing for research)": "Research Skills",
    "other (conducting research)": "Research Skills",
    "other (academic research tasks including literature review, data analysis, and text polishing)": "Research Skills",
    "other (small-scale research project support)": "Research Skills",
    "other (facilitating discussions and research)": "Research Skills",
    "other (developing research competency skills like topic selection, question formulation, and citation)": "Research Skills",
    "q-sort exercise to rank research priorities": "Research Skills",
    "other (researchers used the system for comparative analysis of llms)": "Research Skills",
    "other (information retrieval and discussion preparation)": "Research Skills",
    "other (information gathering)": "Research Skills",
    "other (information retrieval)": "Research Skills",
    
    # Corpus analysis
    "other (corpus annotation)": "Corpus Analysis",
    "other (corpus building)": "Corpus Analysis",
    "other (analysis of learner errors)": "Corpus Analysis",
    "other (register analysis, lexico-grammatical analysis, paraphrasing, text evaluation, data analysis and reporting)": "Corpus Analysis",
    "other (cross-linguistic analysis)": "Corpus Analysis",
    "corpus analysis": "Corpus Analysis",
    "other (corpus search and analysis)": "Corpus Analysis",
    "other (genre analysis)": "Corpus Analysis",
    
    # ========== AI-SPECIFIC TASKS ==========
    "other (chatbot creation)": "AI Development",
    "other (chatbot building and evaluation)": "AI Development",
    "other (training participants on chatbot development)": "AI Development",
    "other (it specialists training the ai language model)": "AI Development",
    "other (system development and evaluation)": "AI Development",
    "other (ai model design)": "AI Development",
    "ai system development and maintenance": "AI Development",    

    "other (training students on ai usage)": "AI Literacy Devevelopment",
    "other (customizing ai tools)": "AI Literacy Devevelopment",
    "other (prompt crafting)": "AI Literacy Devevelopment",
    "other (crafting prompts for ai)": "AI Literacy Devevelopment",
    "other (prompt development training)": "AI Literacy Devevelopment",
    "other (prompt engineering)": "AI Literacy Devevelopment",
    "other (prompt engineering practice)": "AI Literacy Devevelopment",
    "other (prompt writing)": "AI Literacy Devevelopment",
    "speaking prompts": "AI Literacy Devevelopment",
    "writing prompts": "AI Literacy Devevelopment",    
    "other (distinguishing ai-generated from human-written text)": "AI Literacy Devevelopment",
    "other (ai text detection)": "AI Literacy Devevelopment",
    "other (evaluating and comparing human vs. ai-generated texts)": "AI Literacy Devevelopment",
    "other (critical analysis of ai output)": "AI Literacy Devevelopment",
    "other (critical analysis of ai tools)": "AI Literacy Devevelopment",
    "other (critical evaluation of ai responses)": "AI Literacy Devevelopment",
    "other (engaging in dialogue with ai to clarify concepts)": "AI Literacy Devevelopment",
    "other (inquiring, clarifying, reflecting, and synthesizing data with a chatbot)": "AI Literacy Devevelopment",
    "other (specify: analyzing and reporting on ai-generated revisions)": "AI Literacy Devevelopment",

    
    "other (researching and using genai tools for creative tasks like image generation and story writing)": "Content Creation",
    "other (image generation)": "Content Creation",
    "other (creating storybooks)": "Content Creation",
    "other (storybook creation)": "Content Creation",
    
    # ========== LEARNER STRATEGIES & AUTONOMY ==========
    "other (self-regulated learning strategy use)": "Learning Strategies",
    "other (srl training delivery)": "Learning Strategies",
    "other (self-study and clarification)": "Learning Strategies",
    "other (understanding and reviewing study materials)": "Learning Strategies",
    "other (getting study tips and creating practice questions)": "Learning Strategies",
    "other (self-directed learning planning)": "Learning Strategies",
    "other (self-study and collaborative activities)": "Learning Strategies",
    "other (self-directed informal learning using genai)": "Learning Strategies",
    "other (mind mapping for conceptual organization)": "Learning Strategies",
    "other (organizing learning materials)": "Learning Strategies",
    
    "goal setting": "Autonomous Learning",
    "planning": "Autonomous Learning",
    "monitoring learning": "Autonomous Learning",
    "evaluating learning": "Autonomous Learning",
    "personalised learning": "Autonomous Learning",
    "other (personalized learning design)": "Autonomous Learning",
    "other (acquiring personalised learning tips and strategies)": "Autonomous Learning",
    "getting personalized learning tips and strategies": "Autonomous Learning",
    "personalized assignments": "Autonomous Learning",
    "individualized learning activities": "Autonomous Learning",
    
    # ========== AFFECTIVE DOMAIN ==========
    "other (emotional support and motivation)": "Affective Support",
    "other (developing coping strategies and promoting ai ethics)": "Affective Support",
    "other (social skills practice)": "Affective Support",
    
    # ========== CULTURAL COMPETENCE ==========
    "cultural understanding": "Cultural Competence",
    "cultural understanding through discussions": "Cultural Competence",
    "other (elder participated in interviews and provided cultural guidance)": "Cultural Competence",
    
    # ========== TEST PREPARATION ==========
    "ielts test preparation": "Test Preparation",
    "exam preparation": "Test Preparation",
    "other (answering multiple-choice questions on spoken language knowledge)": "Test Preparation",
    "answering short-answer questions": "Test Preparation",
    
    # ========== REAL-WORLD TASKS ==========
    "purchasing an airline ticket": "Real-world Tasks",
    "seeking advice about insomnia": "Real-world Tasks",
    "other (work-related and study-related tasks)": "Real-world Tasks",
    "other (understanding concepts and daily life tasks)": "Real-world Tasks",
    
    # ========== RESOURCE MANAGEMENT ==========
    "other: accessing learning materials": "Resource Management",
    "accessing learning materials": "Resource Management",
    "finding and accessing learning resources": "Resource Management",
    "other (collecting and integrating english learning information and resources online)": "Resource Management",
    "information acquisition": "Resource Management",
    
    # ========== ADMINISTRATIVE TASKS ==========
    "other (lightening administrative loads)": "Administrative Tasks",

    # ========== GENERAL CATEGORIES ==========
    "general language learning": "General Language Practice",
    "general l2 learning": "General Language Practice",
    "learning english knowledge and skills": "General Language Practice",
    "other (general language skill improvement)": "General Language Practice",
    "other (general english learning tasks)": "General Language Practice",
    "improving linguistic skills": "General Language Practice",
    "improving communication skills": "General Language Practice",
    "other (general academic tasks such as weekly assignments, studying for exams, and administrative tasks)": "General Language Practice",
    
    "other (creative ideation and brainstorming)": "Creative Thinking",
    "other (engaging in critical thinking exercises)": "Creative Thinking",
    "other (learning principles of logic and reasoning)": "Creative Thinking",
    "developing critical and creative thinking": "Creative Thinking",
    
    "content creation about language learning": "Content Creation",
    "other (keyword selection for content generation)": "Content Creation",  
    "other (generating questions using qft)": "Content Creation",  
    "other (generating questions for clarification)": "Content Creation",  
        
    "homework assignments": "Homework",

    "other (coding assistance)": "Technical Support",
    "other (programming, cybersecurity analysis, art generation)": "Technical Support",
    
    "other (requesting model answers)": "Model-based Learning",
    "other (comparing their text with a model text)": "Model-based Learning",
    
    "other (validation of content and pedagogical sequencing)": "Quality Assurance",
    
    "other (data analysis)": "Data Analysis",
    
    "other (specify: assigning writing topics)": "Task Assignment",

    "other (reflection)": "Reflection",
    "other (reflection on learning)": "Reflection",
    "other (collaborative reflective practice)": "Reflection",
    
    "other (ai-mediated inquiry)": "Inquiry-based Learning",
    
    "other (policy development)": "Policy Development",
    
    "other (citing ai-generated content)": "Academic Integrity",
    
    "other (parents supporting language learning at home)": "Parental Involvement",
    
    "other (scaffolding)": "Scaffolding",
    "other (scaffolding students on essay structure and use of chatgpt)": "Scaffolding",
    "other (scaffolding informal l2 writing practicum)": "Scaffolding",
    "other (providing scaffolding and modeling)": "Scaffolding",
    
    "other: metaphor elicitation to understand perceptions of genai": "Metaphor Analysis",
    }

    
    # Check for exact matches in critical mappings
    for key, value in critical_mappings.items():
        if key == lower:
            return value
    
    # Check for partial matches in critical mappings
    for key, value in critical_mappings.items():
        if key in lower:
           return value
    
    # ----- Extract content from "other" if not already matched -----
    if lower.startswith('other') and '(' in lower:
        # Try to extract and categorize
        match = re.search(r'other\s*[:\s(]+(.*)', lower)
        if match:
            content = match.group(1).strip(' )')
            # Remove any remaining parentheses
            content = re.sub(r'\([^)]*\)', '', content).strip()
            if content:
                return f"Other: {content.title()}"
    
    # ----- Common prefixes to remove -----
    prefixes_to_remove = [
        'other',
        'other:',
        'other (',
        'specify:',
        'the researcher ',
        'the ai model',
        'ai models ',
        'learning ',
        'practicing ',
        'improving ',
        'developing ',
        'getting ',
        'finding and ',
        'accessing ',
        'providing ',
        'facilitating ',
        'generating ',
        'answering ',
    ]
    
    cleaned = lower
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    
    # Remove parenthetical content for further processing
    cleaned = re.sub(r'\([^)]*\)', '', cleaned).strip()
    cleaned = re.sub(r'[:\-]\s*$', '', cleaned).strip()
    
    # If cleaned is empty or too short, return original capitalized
    if not cleaned or len(cleaned) < 3:
        if len(original) > 2:
            return original.title()
        return "N/A"
    
    # Return cleaned and capitalized
    return cleaned.title()

def normalise_llm_model(value):
    """
    Normalize LLM model names to standardized categories.
    Handles lists of strings
    """
    if not value:
        return "N/A"
    
    # If it's a list, process each item
    if isinstance(value, list):
        normalized_items = []
        for item in value:
            norm = normalise_each_llm(item)
            if norm and norm != "N/A":
                normalized_items.append(norm)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_items = []
        for item in normalized_items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        
        return unique_items if unique_items else ["N/A"]
    
    # If it's a single string
    return [normalise_each_llm(value)]

def normalise_each_llm(model_str):
    """
    Normalize LLM model names to standardized categories.
    """
    if not isinstance(model_str, str):
        return "N/A"
    
    original = model_str.strip()
    if not original:
        return "N/A"
    
    lower = original.lower()
    
    # Handle N/A and unspecified cases
    if lower in ['n/a', 'na', 'none', 'not specified', 'unspecified']:
        return "N/A"
    
    # Handle generic terms
    if any(term in lower for term in [
        'unspecified', 'not named', 'not specified', 'similar ai tools', 'other',
        'generative ai', 'genai', 'large language models', 'llm', 'chatbot', 'ai chat'
    ]):
        return "N/A"
    
    # ----- OpenAI GPT family -----
    if 'gpt' in lower:
        return "ChatGPT"  # Default for plain ChatGPT
    
    # ----- Google models -----
    if any(term in lower for term in ['gemini', 'bard', 'google bard', 'palm']):
        return "Google AI (Gemini, Bard, PaLM)"
    
    # ----- Anthropic models -----
    if 'claude' in lower:
        return "Claude"
    
    # ----- Microsoft models -----
    if any(term in lower for term in ['copilot', 'bing', 'bing chat', 'bing ai']):
        return "Microsoft AI (Copilot, Bing)"
        # ----- Meta models -----
    if 'llama' in lower:
        return "LlaMa"
    if 'deepseek' in lower:
        return "DeepSeek"
    if 'qwen' in lower:
        return "Qwen"    
    if 'bart' in lower:
        return "BART"    
    # ----- Other foundation models -----
    if 'mistral' in lower or 'mixtral' in lower:
        return "Mistral"    
    if 't5' in lower:
        return "T5"    
    if 'bert' in lower or 'roberta' in lower:
        if 'roberta' in lower:
            return "RoBERTa"
        elif 'electra' in lower:
            return "ELECTRA"
        else:
            return "BERT"
    
    if 'blenderbot' in lower:
        return "BlenderBot"
    
    if 'vicuna' in lower:
        return "Vicuna"
    
    if 'alpaca' in lower:
        return "Alpaca"
    
    if 'pythia' in lower:
        return "Pythia"
    
    if 'zephyr' in lower:
        return "Zephyr"
    
    # ----- Chinese models -----
    chinese_models = {
        'ernie': "ERNIE Bot",
        'wenxin': "Wenxin Yiyan",
        'wenxinyiyan': "Wenxin Yiyan",
        'zhipu': "Zhipu AI",
        'zhipuqingyan': "Zhipu AI",
        'chatglm': "ChatGLM",
        'doubao': "Doubao",
        'deepseek': "DeepSeek",
        'qwen': "Qwen",
        'tongyi': "Tongyi Qianwen",
        'kimi': "Kimi",
        'kimichat': "Kimi",
        'spark': "iFlytek Spark",
        'spark desk': "iFlytek Spark",
        'baidu': "Baidu AI",
        'yiyan': "Wenxin Yiyan"
    }
    
    for term, normalized in chinese_models.items():
        if term in lower:
            return normalized    

    
    # ----- Writing/translation tools -----
    writing_tools = {
        'grammarly': "Grammarly",
        'quillbot': "Quillbot",
        'deepl': "DeepL",
        'reverso': "Reverso",
        'google translate': "Google Translate",
        'wordtune': "Wordtune",
        'copy.ai': "Copy.ai",
        'paperpal': "Paperpal",
        'jenni': "Jenni AI",
        'essay writer': "Essay Writer",
        'perplexity': "Perplexity"
    }
    
    for term, normalized in writing_tools.items():
        if term in lower:
            return normalized
    
    # ----- Platform/Aggregator tools -----
    platforms = {
        'poe': "Poe",
        'brainly': "Brainly",
        'yoodli': "Yoodli",
        'duolingo': "Duolingo",
        'chatpdf': "ChatPDF",
        'call annie': "Call Annie",
        'consensus': "Consensus",
        'character ai': "Character AI",
        'snapchat my ai': "Snapchat My AI",
        'pi': "Pi",
        'monica': "Monica",
        'blackbox': "Blackbox AI",
        'magicschool': "MagicSchool",
        'diffit': "Diffit",
        'elicit': "Elicit",
        'twee': "Twee",
        'mizou': "Mizou",
        'alayna': "Alayna.us",
        'quizbot': "Quizbot",
        'quill': "Quill",
        'visla': "Visla",
        'notebooklm': "NotebookLM",
        'dialogflow': "Dialogflow",
        'smry': "SMMRY",
        'smmry': "SMMRY"
    }
    
    for term, normalized in platforms.items():
        if term in lower:
            return normalized
    
    # ----- Image/Video AI -----
    if any(term in lower for term in ['dall-e', 'pika', 'sora', 'fliki', 'dall·e']):
        if 'dall-e' in lower or 'dall·e' in lower:
            return "DALL-E"
        elif 'sora' in lower:
            return "Sora"
        elif 'pika' in lower:
            return "Pika"
        elif 'fliki' in lower:
            return "Fliki"
    
    # ----- Specialized models -----
    if 'open-calm' in lower:
        return "Open-Calm"
    
    if 'jais' in lower:
        return "JAIS"
    
    if 'flan' in lower:
        return "Flan-T5"
    
    if 'neural-chat' in lower:
        return "Neural Chat"
    
    if 'aya' in lower:
        return "Aya"
    
    if 'dolphin' in lower:
        return "Dolphin"
    
    if 'nemotron' in lower:
        return "Nemotron"
    
    if 'opencoder' in lower:
        return "OpenCoder"
    
    if 'phi' in lower:
        return "Phi"
    
    if 'smollm' in lower:
        return "SmolLM"
    
    if 'wizardlm' in lower:
        return "WizardLM"
    
    if 'yi' in lower:
        return "Yi"
    
    if 'granite' in lower:
        return "Granite"
    
    if 'gemma' in lower:
        return "Gemma"
    
    # # ----- Handle "other" or "similar" -----
    # if lower.startswith('other') or lower.startswith('similar'):
    #     # Try to extract the type from parentheses
    #     match = re.search(r'\((.+)\)', lower)
    #     if match:
    #         content = match.group(1)
    #         if 'llm' in content or 'large language' in content:
    #             return "Other LLM"
    #         elif 'ai tool' in content:
    #             return "Other AI Tool"
    #         else:
    #             return "Other"
    #     return "Other"
    
    # ----- Applications with AI -----
    if 'talkfriend' in lower:
        return "TalkFriend"
    
    if 'learnalytics' in lower:
        return "Learnalytics"
    
    if 'elsa' in lower:
        return "ELSA Speak"
    
    if 'mondly' in lower:
        return "Mondly VR"
    
    # ----- Russian models -----
    if 'rugpt' in lower:
        return "RuGPT"
    
    if 'rut5' in lower:
        return "RuT5"
    
    # ----- Fallback: return title case -----
    # Remove parenthetical content for cleaner output
    clean_str = re.sub(r'\([^)]*\)', '', original).strip()
    if clean_str:
        return clean_str.title()
    
    return original.title()

def normalise_prompting_strategy(value):
    """
    Normalize prompt strategies to standardized categories.
    Handles lists of strings
    """
    if not value:
        return "N/A"
    
    # If it's a list, process each item
    if isinstance(value, list):
        normalized_items = []
        for item in value:
            norm = normalise_each_prompting_strategy(item)
            if norm and norm != "N/A":
                normalized_items.append(norm)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_items = []
        for item in normalized_items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        
        return unique_items if unique_items else ["N/A"]
    
    # If it's a single string
    return [normalise_each_prompting_strategy(value)]

def normalise_each_prompting_strategy(value):
    """
    Normalize prompting strategy strings.
    Returns a list of normalized strategy categories.
    """
    if not isinstance(value, str):
        return "N/A"
    
    original = value.strip()
    if not original:
        return "N/A"
    
    lower = original.lower()
    
    # Handle N/A cases
    if lower in ['n/a', 'na', 'none', 'not specified', 'not applicable']:
        return "N/A"
    
    # Predefined categories (core set)
    core_categories = {
        "explicit instructions": "Explicit Instructions",
        "examples": "Examples",
        "context": "Context",
        "role assignment": "Role Assignment", 
        "constraints": "Constraints",
        "reasoned steps": "Reasoned Steps",
        "iterative prompting": "Iterative Prompting",
    }
    
    # Common synonyms and variations
    synonyms = {
        "zero-shot": "Explicit Instructions",
        "ask": "Explicit Instructions",
        "specif": "Explicit Instructions",
        "request": "Explicit Instructions",
        "single-step": "Explicit Instructions",
        "instruct": "Explicit Instructions",
        "direct": "Explicit Instructions",
        "few-shot": "Examples",
        "example": "Examples",
        "retrieval augmented generation": "Context",
        "rag": "Context",
        "demonstrat": "Context",
        "detail": "Context",
        "description": "Context",
        "step": "Reasoned Steps",
        "parts": "Reasoned Steps",
        "chain-of-thought": "Reasoned Steps",
        "reasoning": "Reasoned Steps",
        "breaking down": "Reasoned Steps",
        "role": "Role Assignment",
        "persona": "Role Assignment",
        "optimization": "Iterative Prompting",
        "iterative": "Iterative Prompting",
        "follow-up": "Iterative Prompting",
        "constraint": "Constraints",
        "multi-turn": "Dialogue",
        "dialog": "Dialogue",
        "emotional": "Emotional Stimuli",
    }
    
    
    # Handle parenthetical content
    if '(' in lower and ')' in lower:
        # Extract main strategy before parentheses
        main_part = lower.split('(')[0].strip()
        # Extract content in parentheses
        parenthetical = re.findall(r'\(([^)]+)\)', lower)        
        # Process main part
        if main_part:
            # Check core categories
            for core_key, core_label in core_categories.items():
                if core_key in main_part or core_key in main_part:
                    return core_label
            for syn_key, syn_label in synonyms.items():
                if syn_key in main_part:
                    return syn_label
                # Check if it's an "other" pattern
            if main_part.startswith('other'):
                # Process parenthetical content
                for content in parenthetical:
                    content_lower = content.lower()
                    # Check other mappings
                    for core_key, core_label in core_categories.items():
                        if core_key in content_lower or core_key in content_lower:
                            return core_label
                    for syn_key, syn_label in synonyms.items():
                        if syn_key in content_lower:
                            return syn_label  
                return "Other"     
        # Also check parenthetical content for additional strategies
        for content in parenthetical:
            content_lower = content.lower()
            # Check core categories in parentheses
            for core_key, core_label in core_categories.items():
                if core_key == lower or core_key in content_lower:
                    return core_label                       
            # Check synonyms in parentheses
            for syn_key, syn_label in synonyms.items():
                if syn_key in content_lower:
                    return syn_label
    else:
        # No parentheses, check whole string:                
        # Check core categories
        for core_key, core_label in core_categories.items():
            if core_key == lower or core_key in lower:
                return core_label               
        for syn_key, syn_label in synonyms.items():
            if syn_key in lower:
                return syn_label        
    return "Other" 

def normalise_frameworks(value):
    """
    Normalize frameworks to standardized categories.
    Handles lists of strings
    """
    if not value:
        return "N/A"
    
    # If it's a list, process each item
    if isinstance(value, list):
        normalized_items = []
        for item in value:
            norm = normalise_each_theoretical_framework(item)
            if norm and norm != "N/A":
                normalized_items.append(norm)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_items = []
        for item in normalized_items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        
        return unique_items if unique_items else ["N/A"]
    
    # If it's a single string
    return [normalise_each_theoretical_framework(value)]

def normalise_each_theoretical_framework(value):
    """
    Normalize individual theoretical framework strings.
    Returns a string or list of normalized framework categories.
    """
    if not isinstance(value, str):
        return "N/A"
    
    original = value.strip()
    if not original:
        return "N/A"
    
    lower = original.lower()
    
    # Handle N/A cases
    if lower in ['n/a', 'na', 'none', 'not specified', 'not applicable']:
        return "N/A"    
    
    
    # Major theoretical framework categories
    core_frameworks = {
    # Sociocultural Theories (grouped SCT variations)
    "sociocultural theory": "Sociocultural Theory (SCT)",
    "sociocultural perspective": "Sociocultural Theory (SCT)",
    "vygotsky": "Sociocultural Theory (SCT)",
    "vygotskian": "Sociocultural Theory (SCT)",
    "vygotskian framework": "Sociocultural Theory (SCT)",
    "socio-cultural theory": "Sociocultural Theory (SCT)",
    "socio-cultural theories": "Sociocultural Theory (SCT)",
    "sociocultural learning theory": "Sociocultural Theory (SCT)",
    "sct": "Sociocultural Theory (SCT)",
    
    # Zone of Proximal Development (grouped ZPD variations)
    "zone of proximal development": "Zone of Proximal Development (ZPD)",
    "zpd": "Zone of Proximal Development (ZPD)",
    
    # Activity Theory (grouped AT variations)
    "activity theory": "Activity Theory (AT)",
    "cultural-historical activity theory": "Activity Theory (AT)",
    "chat": "Activity Theory (AT)",
    "at": "Activity Theory (AT)",
    
    # Technology Acceptance Models (grouped TAM variations)
    "technology acceptance model": "Technology Acceptance Model (TAM)",
    "tam": "Technology Acceptance Model (TAM)",
    "extended technology acceptance model": "Technology Acceptance Model (TAM)",
    
    # UTAUT (grouped variations)
    "unified theory of acceptance and use of technology": "Unified Theory of Acceptance and Use of Technology (UTAUT)",
    "utaut": "Unified Theory of Acceptance and Use of Technology (UTAUT)",
    "utaut2": "Unified Theory of Acceptance and Use of Technology (UTAUT)",
    
    # Theory of Planned Behavior (grouped TPB variations)
    "theory of planned behavior": "Theory of Planned Behavior (TPB)",
    "tpb": "Theory of Planned Behavior (TPB)",
    "theory of planned behaviour": "Theory of Planned Behavior (TPB)",
    
    # Theory of Reasoned Action (grouped TRA variations)
    "theory of reasoned action": "Theory of Reasoned Action (TRA)",
    "tra": "Theory of Reasoned Action (TRA)",
    
    # Self-Determination Theory (grouped SDT variations)
    "self-determination theory": "Self-Determination Theory (SDT)",
    "sdt": "Self-Determination Theory (SDT)",
    
    # Self-Regulated Learning (grouped SRL variations)
    "self-regulated learning": "Self-Regulated Learning (SRL)",
    "srl": "Self-Regulated Learning (SRL)",
    
    # Self-Directed Learning (grouped SDL variations)
    "self-directed learning": "Self-Directed Learning (SDL)",
    "sdl": "Self-Directed Learning (SDL)",
    
    # Self-Efficacy Theory
    "self-efficacy": "Self-Efficacy Theory",
    "self-efficacy theory": "Self-Efficacy Theory",
    
    # Control-Value Theory (grouped CVT variations)
    "control-value": "Control-Value Theory (CVT)",
    "control-value theory": "Control-Value Theory (CVT)",
    "cvt": "Control-Value Theory (CVT)",
    
    # L2 Motivational Self System (grouped L2MSS variations)
    "l2 motivational self system": "L2 Motivational Self System (L2MSS)",
    "l2mss": "L2 Motivational Self System (L2MSS)",
    
    # Cognitive Load Theory (grouped CLT variations)
    "cognitive load": "Cognitive Load Theory (CLT)",
    "cognitive load theory": "Cognitive Load Theory (CLT)",
    "clt": "Cognitive Load Theory (CLT)",
    
    # Cognitive-Affective Models (grouped CAMIL variations)
    "cognitive-affective": "Cognitive-Affective Model",
    "camil": "Cognitive-Affective Model of Immersive Learning (CAMIL)",
    "cognitive-affective model": "Cognitive-Affective Model",
    "cognitive-affective model of immersive learning": "Cognitive-Affective Model of Immersive Learning (CAMIL)",
    
    # Cognitive Theory of Multimedia Learning (grouped CTML variations)
    "multimedia learning": "Cognitive Theory of Multimedia Learning (CTML)",
    "cognitive theory of multimedia learning": "Cognitive Theory of Multimedia Learning (CTML)",
    "ctml": "Cognitive Theory of Multimedia Learning (CTML)",
    
    # SLA Theories (grouped variations)
    "second language acquisition": "Second Language Acquisition (SLA) Theory",
    "sla": "Second Language Acquisition (SLA) Theory",
    "second language acquisition theory": "Second Language Acquisition (SLA) Theory",
    
    # TBLT (grouped variations)
    "task-based language": "Task-Based Language Teaching (TBLT)",
    "tblt": "Task-Based Language Teaching (TBLT)",
    "task-based language teaching": "Task-Based Language Teaching (TBLT)",
    
    # Communicative Language Teaching (grouped CLT variations - careful, CLT also used for Cognitive Load Theory)
    "communicative language": "Communicative Language Teaching (CLT)",
    "communicative language teaching": "Communicative Language Teaching (CLT)",
    
    # Genre-Based Instruction (grouped GBI variations)
    "genre-based": "Genre-Based Instruction (GBI)",
    "gbi": "Genre-Based Instruction (GBI)",
    "genre-based instruction": "Genre-Based Instruction (GBI)",
    
    # Corpus-Based Language Pedagogy (grouped CBLP variations)
    "corpus-based": "Corpus-Based Language Pedagogy (CBLP)",
    "cblp": "Corpus-Based Language Pedagogy (CBLP)",
    "corpus-based language pedagogy": "Corpus-Based Language Pedagogy (CBLP)",
    
    # Data-Driven Learning (grouped DDL variations)
    "data-driven learning": "Data-Driven Learning (DDL)",
    "ddl": "Data-Driven Learning (DDL)",
    
    # Systemic Functional Linguistics (grouped SFL variations)
    "systemic functional": "Systemic Functional Linguistics (SFL)",
    "sfl": "Systemic Functional Linguistics (SFL)",
    "systemic functional linguistics": "Systemic Functional Linguistics (SFL)",
    
    # CALL (grouped variations)
    "computer-assisted language learning": "Computer-Assisted Language Learning (CALL)",
    "call": "Computer-Assisted Language Learning (CALL)",
    
    # Intelligent CALL (grouped variations)
    "intelligent computer-assisted language learning": "Intelligent Computer-Assisted Language Learning (ICALL)",
    "icall": "Intelligent Computer-Assisted Language Learning (ICALL)",
    
    # Writing and Feedback (grouped variations)
    "writing process": "Writing Process Model",
    "flower and hayes": "Writing Process Model",
    "process-oriented writing": "Writing Process Model",
    "process-oriented writing theory": "Writing Process Model",
    "hayes": "Writing Process Model",
    
    # Feedback Theories (grouped variations)
    "feedback": "Feedback Theory",
    "feedback theory": "Feedback Theory",
    "student-feedback interaction model": "Student-Feedback Interaction Model",
    
    # WCF (grouped variations)
    "written corrective feedback": "Written Corrective Feedback (WCF)",
    "wcf": "Written Corrective Feedback (WCF)",
    
    # AWE (grouped variations)
    "automated writing evaluation": "Automated Writing Evaluation (AWE)",
    "awe": "Automated Writing Evaluation (AWE)",
    
    # AES (grouped variations)
    "automated essay scoring": "Automated Essay Scoring (AES)",
    "aes": "Automated Essay Scoring (AES)",
    
    # AI Literacy (grouped variations)
    "ai literacy": "AI Literacy",
    "genai literacy": "AI Literacy",
    "ai literacies": "AI Literacy",
    "ai competency": "AI Literacy",
    "critical ai literacy": "AI Literacy",
    "cail": "AI Literacy",
    
    # Digital Literacy (grouped variations)
    "digital literacy": "Digital Literacy",
    "critical digital literacy": "Digital Literacy",
    "cdl": "Digital Literacy",
    
    # TPACK (grouped variations)
    "tpack": "Technological Pedagogical Content Knowledge (TPACK)",
    "technological pedagogical content knowledge": "Technological Pedagogical Content Knowledge (TPACK)",
    "technological pedagogy": "Technological Pedagogical Content Knowledge (TPACK)",
    "ai-tpack": "Technological Pedagogical Content Knowledge (TPACK)",
    "technological pedagogical and content knowledge": "Technological Pedagogical Content Knowledge (TPACK)",
    
    # Engagement Theories (grouped variations)
    "engagement theory": "Engagement Theory",
    "learner engagement": "Engagement Theory",
    "student engagement": "Engagement Theory",
    "learning engagement": "Engagement Theory",
    "fredricks": "Engagement Theory",
    "tripartite model of engagement": "Engagement Theory",
    "multidimensional engagement": "Engagement Theory",
    
    # Global Englishes (grouped variations)
    "global englishes": "Global Englishes (GE)",
    "ge": "Global Englishes (GE)",
    "gel": "Global Englishes Language Teaching (GELT)",
    "global englishes language teaching": "Global Englishes Language Teaching (GELT)",
    "gelt": "Global Englishes Language Teaching (GELT)",
    
    # Design-Based Research (grouped variations)
    "design-based research": "Design-Based Research (DBR)",
    "dbr": "Design-Based Research (DBR)",
    
    # ADDIE Model
    "addie": "ADDIE Model",
    "addie model": "ADDIE Model",
    
    # Community of Practice (grouped variations)
    "community of practice": "Community of Practice (CoP)",
    "cop": "Community of Practice (CoP)",
    
    # CEFR (grouped variations)
    "common european framework": "Common European Framework of Reference (CEFR)",
    "cefr": "Common European Framework of Reference (CEFR)",
    "common european framework of reference": "Common European Framework of Reference (CEFR)",
    
    # CAF Framework (grouped variations)
    "complexity, accuracy, and fluency": "Complexity, Accuracy, and Fluency (CAF) Framework",
    "caf": "Complexity, Accuracy, and Fluency (CAF) Framework",
    "complexity accuracy fluency": "Complexity, Accuracy, and Fluency (CAF) Framework",
    
    # Rasch Models (grouped variations)
    "rasch": "Rasch Measurement Model",
    "many-facet rasch": "Many-Facet Rasch Model (MFRM)",
    "mfr": "Many-Facet Rasch Model (MFRM)",
    "mfr": "Many-Facet Rasch Model (MFRM)",
    
    # Scaffolding (grouped variations)
    "scaffolding": "Scaffolding Theory",
    "scaffolding theory": "Scaffolding Theory",
    
    # Constructivism (grouped variations)
    "constructivism": "Constructivism",
    "social constructivism": "Constructivism",
    "cognitive constructivism": "Constructivism",
    "constructivist learning theory": "Constructivism",
    "constructivist theory": "Constructivism",
    "constructivist learning principles": "Constructivism",
    
    # Social Cognitive Theory (grouped variations)
    "social cognitive theory": "Social Cognitive Theory (SCT)",
    "social learning": "Social Cognitive Theory (SCT)",
    "bandura": "Social Cognitive Theory (SCT)",
    "social-cognitive theory": "Social Cognitive Theory (SCT)",
    "socio-cognitive theory": "Social Cognitive Theory (SCT)",
    "socio-cognitive": "Social Cognitive Theory (SCT)",
    
    # Language Hypotheses (grouped variations)
    "input hypothesis": "Input Hypothesis",
    "output hypothesis": "Output Hypothesis",
    "interaction hypothesis": "Interaction Hypothesis",
    "interaction theory": "Interaction Hypothesis",
    "interactionist approach": "Interaction Hypothesis",
    "noticing hypothesis": "Noticing Hypothesis",
    
    # Positive Psychology
    "positive psychology": "Positive Psychology",
    
    # Post-humanism (grouped variations)
    "post-human": "Post-humanism",
    "posthuman": "Post-humanism",
    "posthumanism": "Post-humanism",
    
    # Assemblage Theory
    "assemblage": "Assemblage Theory",
    "assemblage framework": "Assemblage Theory",
    
    # Translanguaging Theory
    "translanguaging": "Translanguaging Theory",
    "translanguaging theory": "Translanguaging Theory",
    
    # Inquiry-Based Learning (grouped variations)
    "inquiry-based learning": "Inquiry-Based Learning (IBL)",
    "ibl": "Inquiry-Based Learning (IBL)",
    
    # Learner Autonomy
    "learner autonomy": "Learner Autonomy",
    
    # Critical Thinking Frameworks
    "critical thinking": "Critical Thinking Framework",
    "paul and elder": "Paul and Elder's Critical Thinking Framework",
    
    # Attachment Theory
    "attachment theory": "Attachment Theory",
    
    # Interpretivist Paradigm
    "interpretivist": "Interpretivist Paradigm",
    "interpretive paradigm": "Interpretivist Paradigm",
    
    # Retrieval Augmented Generation
    "retrieval augmented generation": "Retrieval Augmented Generation (RAG)",
    "rag": "Retrieval Augmented Generation (RAG)",
    
    # Professional Competence Frameworks
    "professional competence": "Professional Competence Framework",
    "digital competence": "Digital Competence Framework",
    "p-genai-c": "Professional GenAI Competence Framework",
    "pdc": "Professional Digital Competence Framework",
    
    # Ecological Perspectives
    "critical ecological perspective": "Critical Ecological Perspective",
    "ecological perspective": "Ecological Perspective",
    "ecological systems": "Ecological Systems Theory",
    
    # Notional-Functional Syllabus
    "notional-functional": "Notional-Functional Syllabus",
    "functional syllabus": "Notional-Functional Syllabus",
    
    # Logic Learning
    "logic learning": "Logic Learning",
    
    # Peer Leadership Theory
    "peer leadership": "Peer Leadership Theory",
    
    # Agency Theories
    "agency": "Agency Theory",
    "empowerment": "Empowerment Theory",
    "distributed agency": "Distributed Agency Theory",
    "teacher agency": "Teacher Agency Theory",
    
    # Bloom's Taxonomy
    "bloom's taxonomy": "Bloom's Taxonomy",
    "bloom taxonomy": "Bloom's Taxonomy",
    
    # Investment Models
    "investment model": "Investment Model",
    "darvin and norton": "Darvin and Norton's Investment Model",
    "model of investment": "Investment Model",
    
    # Writer's Community Models
    "writer within community": "Writer Within Community Model",
    "wwc": "Writer Within Community Model",
    
    # Cue Utilization
    "cue-utilization": "Cue-Utilization Framework",
    
    # Reflective Models
    "reflective model": "Reflective Model",
    "schön": "Reflective Model",
    "reflective practice": "Reflective Practice",
    "reflective theory": "Reflective Theory",
    "reflective learning": "Reflective Learning",
    
    # Emotion Frameworks
    "emotion categorization": "Emotion Categorization Framework",
    "emotion regulation": "Emotion Regulation Model",
    
    # Social Presence Theory
    "social presence": "Social Presence Theory",
    
    # Mental Models
    "mental models": "Mental Models",
    
    # Uses and Gratifications Theory
    "uses and gratifications": "Uses and Gratifications Theory",
    "u&g": "Uses and Gratifications Theory",
    
    # Psycholinguistics
    "psycholinguistics": "Psycholinguistics",
    "computational psycholinguistics": "Computational Psycholinguistics",
    
    # Writing Development
    "writing development": "Writing Development Theory",
    "l2 writing": "L2 Writing Development",
    
    # Metadiscourse Frameworks
    "metadiscourse": "Metadiscourse Framework",
    "hyland": "Hyland's Metadiscourse Framework",
    
    # Metaphorical Analysis
    "metaphorical analysis": "Metaphorical Analysis",
    "metaphor analysis": "Metaphorical Analysis",
    "ma": "Metaphorical Analysis",
    "conceptual metaphor": "Conceptual Metaphor Theory",
    
    # Corpus Analysis Methods
    "register analysis": "Register Analysis",
    "multi-dimensional analysis": "Multi-Dimensional Analysis",
    "biber": "Multi-Dimensional Analysis",
    
    # Collaborative Writing
    "collaborative writing": "Collaborative Writing Model",
    "storch": "Collaborative Writing Model",
    
    # Functional Adequacy
    "functional adequacy": "Functional Adequacy Framework",
    
    # Materiality/Indexicality
    "materiality": "Materiality Theory",
    "indexicality": "Indexicality Theory",
    "ideology": "Ideological Analysis",
    
    # Genre Analysis
    "genre analysis": "Genre Analysis",
    "swales": "Genre Analysis",
    
    # Sense-Making Theory
    "sense-making": "Sense-Making Theory",
    "sensemaking": "Sense-Making Theory",
    
    # DIME Model
    "direct and inferential mediation": "Direct and Inferential Mediation (DIME) Model",
    "dime": "Direct and Inferential Mediation (DIME) Model",
    
    # Distributed Cognition
    "distributed cognition": "Distributed Cognition Theory",
    
    # Interactive Learning
    "interactive language learning": "Interactive Language Learning Theory",
    
    # Personalized Learning
    "personalized learning": "Personalized Learning Theory",
    
    # Creativity Theories
    "componential theory of creativity": "Componential Theory of Creativity",
    "creativity": "Creativity Theory",
    
    # Argumentation Frameworks
    "toulmin": "Toulmin's Argumentation Framework",
    "argumentative writing": "Argumentation Framework",
    "argument mapping": "Argumentation Framework",
    
    # Error Analysis
    "error analysis": "Error Analysis Framework",
    "ellis": "Error Analysis Framework",
    
    # Corpus Literacy
    "corpus literacy": "Corpus Literacy",
    
    # ESP Pedagogy
    "english for specific purposes": "English for Specific Purposes (ESP)",
    "esp": "English for Specific Purposes (ESP)",
    
    # Authentic Materials
    "authentic materials": "Authentic Materials Approach",
    
    # Coh-Metrix
    "coh-metrix": "Coh-Metrix Analysis",
    
    # Test-Taker Engagement
    "test-taker engagement": "Test-Taker Engagement Model",
    
    # Expectation-Value/Expectancy-Value Theories
    "expectation-value": "Expectation-Value Theory",
    "expectancy-value": "Expectation-Value Theory",
    "expectancy-value theory": "Expectation-Value Theory",
    
    # Cooperative Principle
    "cooperative principle": "Cooperative Principle",
    "grice": "Cooperative Principle",
    
    # Politeness Theory
    "politeness theory": "Politeness Theory",
    "brown and levinson": "Politeness Theory",
    
    # Thematic Analysis
    "thematic analysis": "Thematic Analysis",
    "braun and clarke": "Thematic Analysis",
    
    # Pedagogical Grammar
    "pedagogical grammar": "Pedagogical Grammar",
    
    # CALF Framework
    "calf": "CALF Framework (Complexity, Accuracy, Lexical Complexity, Fluency)",
    "complexity, accuracy, lexical": "CALF Framework",
    
    # Limited Attentional Capacity
    "limited attentional capacity": "Limited Attentional Capacity Model",
    "skehan": "Limited Attentional Capacity Model",
    
    # Constructive Alignment
    "constructive alignment": "Constructive Alignment",
    "biggs": "Constructive Alignment",
    
    # Culturally Responsive Pedagogy
    "culturally responsive pedagogy": "Culturally Responsive Pedagogy (CRP)",
    "crp": "Culturally Responsive Pedagogy (CRP)",
    "culturally responsive teaching": "Culturally Responsive Pedagogy (CRP)",
    "culturally relevant pedagogy": "Culturally Responsive Pedagogy (CRP)",
    "culturally sustaining": "Culturally Responsive Pedagogy (CRP)",
    "culturally sustaining practices": "Culturally Responsive Pedagogy (CRP)",
    
    # Pedagogical AI Integration
    "pedagogical ai integration": "Pedagogical AI Integration Model (PAIM)",
    "paim": "Pedagogical AI Integration Model (PAIM)",
    
    # Discourse Analysis
    "discourse analysis": "Discourse Analysis (DA)",
    "da": "Discourse Analysis (DA)",
    
    # Ethnography
    "connective ethnography": "Connective Ethnography",
    "ethnography": "Ethnographic Approach",
    
    # Task Engagement
    "task engagement": "Task Engagement Model",
    
    # Post-structuralist Theories
    "post-structuralist": "Post-structuralist Theory",
    "poststructural": "Post-structuralist Theory",
    
    # Spatial Repertoires
    "spatial repertoires": "Spatial Repertoires Theory",
    
    # Maslow's Hierarchy
    "maslow": "Maslow's Hierarchy of Needs",
    "hierarchy of needs": "Maslow's Hierarchy of Needs",
    
    # Escapism
    "escapism": "Escapism Theory",
    
    # Spoken Language Intelligence
    "spoken language intelligence": "Spoken Language Intelligence (SLI)",
    "sli": "Spoken Language Intelligence (SLI)",
    
    # Writing Workshop
    "writing workshop": "Writing Workshop Model",
    
    # Content Generation
    "content generation": "Content Generation Model",
    
    # Pedagogical Reasoning
    "pedagogical reasoning": "Pedagogical Reasoning Model",
    "shulman": "Pedagogical Reasoning Model",
    
    # UNESCO Frameworks
    "unesco": "UNESCO AI Competency Framework",
    "unesco ai competency framework": "UNESCO AI Competency Framework",
    
    # Situated Learning
    "situated learning": "Situated Learning Theory",
    
    # Task Complexity
    "task complexity": "Task Complexity Framework",
    
    # Narrative Inquiry
    "narrative inquiry": "Narrative Inquiry",
    
    # Teacher Cognition
    "teacher cognition": "Teacher Cognition Theory",
    "borg": "Teacher Cognition Theory",
    
    # Hybrid Human-AI Regulation
    "hybrid human-ai regulation": "Hybrid Human-AI Regulation (HHAIR)",
    "hhair": "Hybrid Human-AI Regulation (HHAIR)",
    
    # Equivalence Principle
    "equivalence principle": "Equivalence Principle",
    
    # Social Agency Theory
    "social agency": "Social Agency Theory",
    
    # Universal Design for Learning
    "universal design for learning": "Universal Design for Learning (UDL)",
    "udl": "Universal Design for Learning (UDL)",
    
    # Connectivism
    "connectivism": "Connectivism",
    
    # Broaden-and-Build Theory
    "broaden-and-build": "Broaden-and-Build Theory",
    
    # Professional Development
    "professional development": "Professional Development Framework",
    
    # Participatory Design
    "participatory design": "Participatory Design",
    
    # Joint Media Engagement
    "joint media engagement": "Joint Media Engagement",
    
    # Ethical Design
    "ethical design": "Ethical Design Framework",
    
    # Student-Centered Learning
    "student-centered learning": "Student-Centered Learning",
    
    # Active Learning
    "active learning": "Active Learning",
    
    # Jigsaw Technique
    "jigsaw": "Jigsaw Technique",
    "jigsaw technique": "Jigsaw Technique",
    
    # Assessment Frameworks
    "edtpa": "edTPA Framework",
    "actfl": "ACTFL Framework",
    "proficiency guidelines": "ACTFL Proficiency Guidelines",
    
    # Exploration-Exploitation Theory
    "exploration-exploitation": "Exploration-Exploitation Theory (EET)",
    "eet": "Exploration-Exploitation Theory (EET)",
    
    # Genre Theory
    "genre theory": "Genre Theory",
    
    # Affordance Theory
    "affordance theory": "Affordance Theory",
    
    # Corpus Linguistics
    "corpus linguistics": "Corpus Linguistics",
    
    # CARS Model
    "create a research space": "Create a Research Space (CARS) Model",
    "cars": "Create a Research Space (CARS) Model",
    
    # Word Knowledge Approach
    "word knowledge": "Word Knowledge Approach",
    
    # Developmental Knowledge Approach
    "developmental knowledge": "Developmental Knowledge Approach",
    
    # Functional Tool Framework
    "functional tool": "Functional Tool Framework",
    
    # Cyber-Social Literacy
    "cyber-social literacy": "Cyber-Social Literacy Learning",
    
    # Process-Oriented Learning
    "process-oriented learning": "Process-Oriented Learning",
    
    # ICAP Framework
    "icap": "ICAP Framework (Interactive, Constructive, Active, Passive)",
    
    # Evaluation Models
    "kirkpatrick": "Kirkpatrick's Evaluation Model",
    
    # Reflection Models
    "gibbs": "Gibbs' Reflective Model",
    
    # Pedagogical Frameworks
    "five-part pedagogical framework": "Pedagogical Framework",
    "game design framework": "Pedagogical Framework",
    "lan": "Pedagogical Framework",
    
    # Machine-in-the-loop
    "machine-in-the-loop": "Machine-in-the-Loop Framework",
    "humans in the loop": "Machine-in-the-Loop Framework",
    "humans-in-the-loop": "Machine-in-the-Loop Framework",
    
    # Informal Digital Learning
    "informal digital learning": "Informal Digital Learning of English (IDLE)",
    "idle": "Informal Digital Learning of English (IDLE)",
    
    # Needs Analysis
    "needs analysis": "Needs Analysis Model",
    "hutchinson and waters": "Needs Analysis Model",
    
    # Well-being Models
    "well-being model": "Well-being Model",
    "hedonic and eudaimonic": "Hedonic and Eudaimonic Well-being Model",
    
    # Q Methodology
    "q methodology": "Q Methodology",
    
    # EFL Teaching Models
    "ettar": "EFL Teachers' Teaching and Academic Research (ETTAR) Model",
    
    # Integrated Technology Acceptance
    "integrated model of technology acceptance": "Integrated Model of Technology Acceptance (IMTA)",
    "imta": "Integrated Model of Technology Acceptance (IMTA)",
    
    # Motivational Theory (Gardner)
    "motivational theory": "Motivational Theory",
    "gardner": "Motivational Theory",
    # Achievement Goal Theory
    "achievement goal": "Achievement Goal Theory",
    "achievement goal theory": "Achievement Goal Theory",
    "achievement goal orientation": "Achievement Goal Theory",

    # Flow Theory - add to Motivation and Self-Regulation Theories section
    "flow theory": "Flow Theory",
    "flow": "Flow Theory",

    
    # Language Attitudes
    "language attitudes": "Language Attitudes Model",
    "three-dimensional model of language attitudes": "Language Attitudes Model",
    
    # Social Identity Theory
    "social identity": "Social Identity Theory",
    
    # Holistic Learning Ecology
    "holistic learning ecology": "Holistic Learning Ecology",
    
    # Pedagogical Content Knowledge
    "pedagogical content knowledge": "Pedagogical Content Knowledge (PCK)",
    "pck": "Pedagogical Content Knowledge (PCK)",
    
    # Pedagogical Language Knowledge
    "pedagogical language knowledge": "Pedagogical Language Knowledge (PLK)",
    "plk": "Pedagogical Language Knowledge (PLK)",
    
    # Teacher Agency Models
    "three-dimensional model of teacher agency": "Three-Dimensional Teacher Agency Model",
    "ecological model of teacher agency": "Three-Dimensional Teacher Agency Model",
    "ecological perspective of teacher agency": "Three-Dimensional Teacher Agency Model",
    
    # SWOT Analysis
    "swot": "SWOT Analysis",
    
    # Human-Centered AI
    "human-centered ai": "Human-Centered AI",
    "human centered ai": "Human-Centered AI",
    
    # Complex Dynamic Systems
    "complex dynamic systems": "Complex Dynamic Systems Theory (CDST)",
    "cdst": "Complex Dynamic Systems Theory (CDST)",
    
    # Willingness to Communicate
    "willingness to communicate": "Willingness to Communicate (WTC)",
    "wtc": "Willingness to Communicate (WTC)",
    
    # AI-Enhanced Learning
    "ai-enhanced learning": "AI-Enhanced Learning Experience Theory",
    "ai-enhanced learning experience theory": "AI-Enhanced Learning Experience Theory",
    
    # Extended Technology Use Models
    "three-tier technology use": "Three-Tier Technology Use Model",
    "3-tum": "Three-Tier Technology Use Model",
    
    # Conversation Theory
    "conversation theory": "Conversation Theory",
    
    # Communication Theory
    "communication theory": "Communication Theory",
    
    # Relevance Theory
    "relevance theory": "Relevance Theory",
    
    # Technology Integration
    "technology integration matrix": "Technology Integration Matrix",
    "epp ai integration": "Technology Integration Framework",
    
    # Interactional Competence
    "interactional competence": "Interactional Competence (IC)",
    "ic": "Interactional Competence (IC)",
    
    # User Experience
    "user experience": "User Experience Framework",
    "ueq": "User Experience Framework",
    
    # Personality Frameworks
    "big five": "Big Five Personality Framework",
    "big five personality": "Big Five Personality Framework",
    
    # Academic Buoyancy
    "academic buoyancy": "Academic Buoyancy",
    
    # Generalizability Theory
    "generalizability theory": "Generalizability Theory",
    "g-theory": "Generalizability Theory",
    "generalizability (g-) theory": "Generalizability Theory",
    
    # Information Processing
    "information processing": "Information Processing Theory",
    
    # Authorial Voice
    "authorial voice": "Authorial Voice Framework",
    
    # Stance and Engagement
    "stance and engagement": "Stance and Engagement Model",
    
    # Trust Calibration
    "trust calibration": "Trust Calibration Theory",
    
    # Basic Psychological Needs
    "basic psychological needs": "Basic Psychological Needs Theory",
    "bpn": "Basic Psychological Needs Theory",
    
    # Learning-Oriented Assessment
    "learning-oriented assessment": "Learning-Oriented Assessment (LOA)",
    "loa": "Learning-Oriented Assessment (LOA)",
    
    # Formative Assessment
    "formative assessment": "Formative Assessment",
    
    # Text Processing Models
    "text processing": "Text Processing Model",
    "construction-integration": "Text Processing Model",
    "construction-integration model": "Text Processing Model",
    
    # Portfolio Assessment
    "portfolio assessment": "Portfolio Assessment",
    
    # Question Formulation
    "question formulation": "Question Formulation Technique (QFT)",
    "qft": "Question Formulation Technique (QFT)",
    
    # Socio-Technical Systems
    "socio-technical systems": "Socio-Technical Systems Theory",
    
    # Technostress
    "technostress": "Technostress Creators Framework",
    
    # World Englishes
    "world englishes": "World Englishes (WE)",
    "we": "World Englishes (WE)",
    
    # English as Lingua Franca
    "english as lingua franca": "English as a Lingua Franca (ELF)",
    "elf": "English as a Lingua Franca (ELF)",
    
    # Conversation Analysis
    "conversation analysis": "Conversation Analysis (CA)",
    "ca": "Conversation Analysis (CA)",
    
    # Automated Scoring
    "automated scoring": "Automated Scoring Framework",
    
    # Dialogue-Based CALL
    "dialogue-based call": "Dialogue-Based CALL",
    
    # Trigger-Indicator Framework
    "trigger-indicator": "Trigger-Indicator-Response-Reaction Framework",
    
    # Education 4.0/5.0
    "education 4.0": "Education 4.0",
    "education 5.0": "Education 5.0",
    
    # e-Learning
    "e-learning 2.0": "e-Learning 2.0",
    
    # SAMR Model
    "samr": "SAMR Model (Substitution, Augmentation, Modification, Redefinition)",
    "substitution augmentation": "SAMR Model",
    
    # Motivation-Engagement-Thriving
    "metux": "METUX Model (Motivation, Engagement, Thriving in User Experience)",
    
    # Skill Acquisition
    "skill acquisition": "Skill Acquisition Theory",
    
    # Mediation Models
    "mediation model": "Mediation Model",
    
    # Speaking Assessment
    "speaking assessment": "Speaking Assessment Framework",
    "automated speaking assessment": "Speaking Assessment Framework",
    "asa": "Speaking Assessment Framework",
    "toefl ibt speaking": "TOEFL iBT Speaking Evaluation Framework",
    "toefl speaking": "TOEFL iBT Speaking Evaluation Framework",
    
    # Learner-Centered Instruction
    "learner-centered instruction": "Learner-Centered Instruction",
    
    # Form-Focused Instruction
    "form-focused instruction": "Form-Focused Instruction",
    
    # Nonlinear Dynamic Learning
    "nonlinear dynamic": "Nonlinear Dynamic Language Learning Theory",
    "bahari": "Nonlinear Dynamic Language Learning Theory",
    
    # SOR Framework
    "stimulus-organism-response": "Stimulus-Organism-Response (SOR) Framework",
    "sor": "Stimulus-Organism-Response (SOR) Framework",
    
    # Expectancy Confirmation
    "expectancy confirmation": "Expectancy Confirmation Theory (ECT)",
    "ect": "Expectancy Confirmation Theory (ECT)",
    "expectancy confirmation theory": "Expectancy Confirmation Theory (ECT)",
    
    # Design and Development Research
    "design and development research": "Design and Development Research (DDR)",
    "ddr": "Design and Development Research (DDR)",
    
    # Cognition-Based Approach
    "cognition-based approach": "Cognition-Based Approach",
    
    # Territory of Information
    "territory of information": "Territory of Information Framework",
    
    # First Principles of Instruction
    "first principles of instruction": "First Principles of Instruction (FPI)",
    "fpi": "First Principles of Instruction (FPI)",
    
    # Community of Inquiry
    "community of inquiry": "Community of Inquiry (CoI)",
    "coi": "Community of Inquiry (CoI)",
    
    # Levels of Processing
    "levels of processing": "Levels of Processing Framework",
    
    # Privacy Frameworks
    "privacy framework": "Privacy Framework",
    "personalization privacy": "Privacy Framework",
    
    # Multi-Agent Systems
    "multi-agent system": "Multi-Agent System (MAS)",
    "mas": "Multi-Agent System (MAS)",
    
    # Evidence-Centered Design
    "evidence-centered design": "Evidence-Centered Design (ECD)",
    "ecd": "Evidence-Centered Design (ECD)",
    
    # Language Assessment Literacy
    "language assessment literacy": "Language Assessment Literacy (LAL)",
    "lal": "Language Assessment Literacy (LAL)",
    
    # Proactive Language Learning
    "proactive language learning": "Proactive Language Learning Theory",
    
    # PERMA Model
    "perma model": "PERMA Model",
    "perma": "PERMA Model",
    
    # Netnography
    "netnography": "Netnography",
    
    # Foreign Language Enjoyment
    "foreign language enjoyment": "Foreign Language Enjoyment (FLE)",
    "fle": "Foreign Language Enjoyment (FLE)",
    
    # Extensive Reading
    "extensive reading": "Extensive Reading Approach",
    
    # Information Literacy
    "information literacy": "Information Literacy",
    
    # Action Research
    "action research": "Action Research",
    
    # Prompt Literacy
    "prompt literacy": "Prompt Literacy",
    
    # PROMPT Models/Acronyms
    "proper": "PROPER Framework",
    "create": "CREATE Framework",
    "clear": "CLEAR Framework",
    "trust": "TRUST Framework",
    
    # Flipped Classroom
    "flipped classroom": "Flipped Classroom Model",
    "flipped classroom model": "Flipped Classroom Model",
    
    # Design Thinking
    "design thinking": "Design Thinking",
    
    # Cognitive Flexibility
    "cognitive flexibility": "Cognitive Flexibility Theory",
    "cognitive flexibility theory": "Cognitive Flexibility Theory",
    
    # Structural Equation Modeling
    "structural equation modeling": "Structural Equation Modeling (SEM)",
    "sem": "Structural Equation Modeling (SEM)",
    
    # Sociocognitive Writing Theories
    "sociocognitive theories of writing": "Sociocognitive Writing Theories",
    "sociocognitive framework": "Sociocognitive Framework",
    "kecskes": "Sociocognitive Framework",
    
    # Kolb's Experiential Learning
    "kolb": "Kolb's Experiential Learning Cycle",
    "experiential learning": "Experiential Learning Theory",
    
    # Outcome-Based Design
    "outcome-based learning": "Outcome-Based Learning Design",
    "outcome-based design": "Outcome-Based Learning Design",
    
    # Collaboration Scripts
    "collaboration scripts": "Collaboration Scripts",
    
    # Expectation Confirmation Model
    "expectation confirmation model": "Expectation Confirmation Model (ECM)",
    "ecm": "Expectation Confirmation Model (ECM)",
    
    # Task Technology Fit
    "task technology fit": "Task Technology Fit (TTF)",
    "ttf": "Task Technology Fit (TTF)",
    
    # Theory of Mind
    "theory of mind": "Theory of Mind (ToM)",
    "tom": "Theory of Mind (ToM)",
    
    # Expertise Reversal Effect
    "expertise reversal": "Expertise Reversal Effect",
    "expertise reversal effect": "Expertise Reversal Effect",
    
    # Inductive Approach
    "inductive approach": "Inductive Approach",
    "discovery learning": "Inductive Approach",
    
    # Students' Approaches to Learning
    "students' approaches to learning": "Students' Approaches to Learning (SAL)",
    "sal": "Students' Approaches to Learning (SAL)",
    
    # Personal Innovativeness
    "personal innovativeness": "Personal Innovativeness",
    
    # Teacher-AI Collaboration
    "teacher-ai collaboration": "Teacher-AI Collaboration (TAC)",
    "tac": "Teacher-AI Collaboration (TAC)",
    
    # Grabe and Kaplan's Writing Model
    "grabe and kaplan": "Grabe and Kaplan's Writing Model",
    
    # Learning Types
    "incidental learning": "Incidental Learning",
    "intentional learning": "Intentional Learning",
    
    # Repair Frameworks
    "repairs framework": "Repairs Framework",
    "schegloff": "Repairs Framework",
    
    # SUPER Framework
    "super framework": "SUPER Framework",
    
    # Reverse Searching
    "reverse searching": "Reverse Searching",
    
    # Psychology-Thinking-Style-Technology
    "elhossiny": "Elhossiny's Psychology-Thinking Style-Technology Theory",
    "psychology thinking style": "Elhossiny's Psychology-Thinking Style-Technology Theory",
    
    # Multiliteracies
    "multiliteracies": "Multiliteracies",
    
    # Revision Operations
    "revision operations": "Revision Operations Framework",
    "zhang": "Revision Operations Framework",
    
    # GACP Framework
    "gacp": "GACP Framework (GenAI-Assisted Collaborative Prewriting)",
    "genai-assisted collaborative prewriting": "GACP Framework",
    
    # GenAI Competence
    "genai competence": "GenAI Competence Framework",
    "gai competence": "GenAI Competence Framework",
    
    # ABCE Framework
    "abce": "ABCE Framework",
    "abce framework": "ABCE Framework",
    
    # APSE Model
    "apse": "APSE Model",
    "apse model": "APSE Model",
    
    # PER Approach
    "plan-enact-reflect": "Plan-Enact-Reflect (PER) Approach",
    "per": "Plan-Enact-Reflect (PER) Approach",
    "plan enact reflect": "Plan-Enact-Reflect (PER) Approach",
}
    for core_key, core_label in core_frameworks.items():
        if core_key in lower or core_key in lower:
            return core_label
    return f"Other: {lower.title()}"

def normalise_list(str_list):
    """Return cleaned strings but preserve original casing."""
    result = []
    for x in str_list:
        if isinstance(x, str):
            cleaned = x.strip().replace("(inferred)", "")
            if cleaned:
                result.append(cleaned)
    return result

def extract_demographic_field(data, field, participant_type=None):
    """
    Extract all values for a specific field across participant_demographics arrays.
    If participant_type is specified, only extracts for that participant type.
    Returns a list of strings.
    """
    results = []
    for study in data:
        # study is a list of participant demographics dictionaries
        for participant in study:
            if isinstance(participant, dict):
                # Check if we should filter by participant type
                if participant_type is not None:
                    current_type = participant.get("participant_type", "N/A")
                    if str(current_type).lower() != str(participant_type).lower():
                        continue
                
                # Extract the field value
                value = participant.get(field, "N/A")
                results.append(value)
    
    return normalise_list(results)

def extract_task_types(data, participant_type=None):
    """
    Extract task types from the task_types object structure.
    If participant_type is specified, only extracts for that participant type.
    
    Parameters:
    -----------
    data : pandas Series or list
        The task_types column from your DataFrame
    participant_type : str or None
        Optional: 'students', 'teachers', 'policymakers_administrators', 'others', or None for all
        
    Returns:
    --------
    list: All task types (flattened if participant_type is None, filtered otherwise)
    """
    results = []
    
    for study_tasks in data:
        if isinstance(study_tasks, dict):
            if participant_type is not None:
                # Extract only for specific participant type
                tasks = study_tasks.get(participant_type, [])
                if isinstance(tasks, list):
                    results.extend(tasks)
            else:
                # Extract all task types from all participant groups
                for pt in ['students', 'teachers', 'policymakers_administrators', 'others']:
                    tasks = study_tasks.get(pt, [])
                    if isinstance(tasks, list):
                        results.extend(tasks)
    
    return normalise_list(results)


def parse_study_location(loc):
    """
    loc = [
      [city/region1,...],
      [country1,...],
      [continent1,...]
    ]
    Returns: dict with region_list, country_list, continent_list
    """
    try:
        regions_raw, countries_raw, continents_raw = loc
    except Exception:
        return {"region": [], "country": [], "continent": []}
    # print (regions_raw)
    #print (countries_raw)
    return {
        "region": normalise_list(regions_raw),
        "country": normalise_list(countries_raw),
        "continent": normalise_list(continents_raw)
    }

import re
from dateutil import parser
from datetime import datetime

def normalize_study_duration(raw, debug=False):
    if not raw or not isinstance(raw, str):
        return ["N/A"]

    text = raw.lower().strip()

    # normalize ordinals (1st → 1)
    text = re.sub(r'(\d{1,2})(st|nd|rd|th)\b', r'\1', text)

    # normalize hyphens
    text = re.sub(r'[\u2010-\u2015]', '-', text)

    # ---------- STEP 0: explicit N/A ----------
    if re.search(r'\b(n/?a|not specified|unspecified)\b', text):
        return ["N/A"]

    # ---------- STEP 1: cross-sectional ----------
    if re.search(r'\b(cross[- ]sectional|single point in time|one[- ]shot)\b', text):
        return ["single-session"]

    # ---------- STEP 2: number words ----------
    WORD_NUMS = {
        "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12
    }
    for w, n in WORD_NUMS.items():
        text = re.sub(rf'\b{w}\b', str(n), text)

    weeks = []

    # ======================================================
    # HIGH PRIORITY: EXPLICIT DURATIONS
    # ======================================================

    # weeks
    for m in re.finditer(r'(\d+)\s*[- ]?\s*weeks?', text):
        weeks.append(int(m.group(1)))

    # months
    for m in re.finditer(r'(\d+)\s*[- ]?\s*months?', text):
        weeks.append(int(m.group(1)) * 4)

    # days → weeks
    for m in re.finditer(r'(\d+)\s*[- ]?\s*days?', text):
        d = int(m.group(1))
        if d >= 7:
            weeks.append(round(d / 7))

    # ======================================================
    # INTERVENTION PATTERNS
    # ======================================================

    for m in re.finditer(r'(\d+)\s*[- ]?(day|week|month)\s+intervention', text):
        n, unit = int(m.group(1)), m.group(2)
        if unit == "day" and n >= 7:
            weeks.append(round(n / 7))
        elif unit == "week":
            weeks.append(n)
        elif unit == "month":
            weeks.append(n * 4)

    # ======================================================
    # ACADEMIC CALENDAR
    # ======================================================

    if re.search(r'\bsemesters?\b', text):
        count = re.search(r'(\d+)\s+semesters?', text)
        if count:
            weeks.append(int(count.group(1)) * 15)
        else:
            weeks.append(15)

    if re.search(r'\btrimesters?\b', text):
        count = re.search(r'(\d+)\s+trimesters?', text)
        if count:
            weeks.append(int(count.group(1)) * 10)
        else:
            weeks.append(10)

    if re.search(r'\bacademic year\b', text):
        weeks.append(36)

    # ======================================================
    # QUALIFIERS
    # ======================================================

    if re.search(r'\bover\s+a\s+year\b', text):
        weeks.append(52)

    if re.search(r'at\s+least\s+(\d+)\s+weeks?', text):
        weeks.append(int(re.search(r'at\s+least\s+(\d+)', text).group(1)))

    # ======================================================
    # DATE RANGES (ROBUST)
    # ======================================================

    DATE_PAT = r'(?:[a-z]+\s+\d{1,2},?\s*\d{4}|[a-z]+\s+\d{4}|\d{1,2}\s+[a-z]+\s+\d{4})'

    patterns = [
        rf'({DATE_PAT})\s*(?:,?\s*(?:to|-|–|and)\s*)({DATE_PAT})',
        rf'between\s+({DATE_PAT})\s+and\s+({DATE_PAT})'
    ]

    for pat in patterns:
        for m in re.finditer(pat, raw, re.I):
            try:
                d1 = parser.parse(m.group(1), fuzzy=True, default=datetime(1900,1,1))
                d2 = parser.parse(m.group(2), fuzzy=True)

                # fix missing year
                if d1.year == 1900:
                    d1 = d1.replace(year=d2.year)

                delta = abs((d2 - d1).days) / 7
                if delta >= 1:
                    weeks.append(round(delta))
            except Exception:
                pass

    # ======================================================
    # MONTH-TO-MONTH FALLBACK
    # ======================================================

    if not weeks:
        if re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s*(to|and)\s*(january|february|march|april|may|june|july|august|september|october|november|december)', text):
            weeks.append(8)

    # ======================================================
    # EXCLUDE EXPERIENCE (IMPORTANT)
    # ======================================================

    if re.search(r'\b(experience using|had been using|prior experience|had used)\b', text):
        return ["N/A"]

    # ======================================================
    # SESSION DETECTION (LOW PRIORITY)
    # ======================================================

    SESSION_PAT = r'\b(minutes?|hours?|mins?|hrs?|session|class|lesson|workshop|period)\b'

    if not weeks and re.search(SESSION_PAT, text):
        return ["single-session"]

    # ======================================================
    # FINAL DECISION
    # ======================================================

    if not weeks:
        return ["N/A"]

    duration = max(weeks)

    # ======================================================
    # BANDING
    # ======================================================

    DURATION_BANDS = [
        (0, 1, "single-session(s)"),
        (2, 3, "2–3 weeks"),
        (4, 6, "4–6 weeks"),
        (7, 10, "7–10 weeks"),
        (11, 15, "11–15 weeks"),
        (16, 30, "16–30 weeks"),
        (31, 60, "31–60 weeks"),
        (61, float("inf"), "over one year"),
    ]

    for lo, hi, label in DURATION_BANDS:
        if lo <= duration <= hi:
            return [label]

    return ["N/A"]

# HELPER FUNCTIONS
def label_frequency(series):
    """
    Unified frequency counter matching the original Jupyter notebook exactly.
    Handles Series of lists, single values, and mixed types.
    Skips NaN/None (does NOT count them as N/A).
    """
    items = []
    for item in series:
        if isinstance(item, list):
            items.extend(item)
        elif pd.isna(item) or item is None:
            continue
        else:
            items.append(item)
    if not items:
        return pd.Series(dtype=int)
    return pd.Series(Counter(items)).sort_values(ascending=False)


def create_bar_chart(report_df, title, top_n=20, figsize=(14, 8)):
    """
    Create a horizontal bar chart with counts + percentages.
    Returns a matplotlib Figure object for st.pyplot().
    """
    if report_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14)
        ax.set_title(title, fontsize=14, fontweight="bold")
        return fig

    top = report_df.head(top_n).copy()
    other_count = report_df["count"].iloc[top_n:].sum() if len(report_df) > top_n else 0
    if other_count > 0:
        top = pd.concat([top, pd.DataFrame([{"label": "Other", "count": other_count}])], ignore_index=True)

    total = top["count"].sum()
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        top["label"][::-1],
        top["count"][::-1],
        color=plt.cm.viridis(np.linspace(0, 1, len(top))),
        alpha=0.8
    )

    for bar in bars:
        width = bar.get_width()
        percentage = (width / total) * 100 if total > 0 else 0
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{int(width)} ({percentage:.1f}%)',
            va='center', fontsize=9, fontweight='bold'
        )

    ax.set_xlabel("Count", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    return fig

def plot_comparative_distribution(report_dict, title="Distribution Comparison",
                                   top_n=10, figsize=(14, 10), color_map="viridis"):
    """
    Comparative horizontal bar plots across participant types.
    Returns matplotlib Figure for st.pyplot().
    """
    if not report_dict:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14)
        ax.set_title(title, fontsize=14, fontweight="bold")
        return fig

    plot_data = {}
    for participant_type, report_df in report_dict.items():
        sorted_df = report_df.sort_values("count", ascending=False)
        top_df = sorted_df.head(top_n)
        other_count = sorted_df["count"].iloc[top_n:].sum()
        if other_count > 0:
            top_df = pd.concat([top_df, pd.DataFrame([{"label": "Other", "count": other_count}])], ignore_index=True)
        plot_data[participant_type] = {
            "labels": top_df["label"].tolist(),
            "counts": top_df["count"].tolist(),
            "total": top_df["count"].sum()
        }

    n_types = len(plot_data)
    fig, axes = plt.subplots(n_types, 1, figsize=figsize)
    if n_types == 1:
        axes = [axes]

    all_labels = set()
    for d in plot_data.values():
        all_labels.update(d["labels"])
    all_labels = sorted(all_labels)
    n_labels = len(all_labels)
    cmap = plt.colormaps[color_map] if hasattr(plt, 'colormaps') else plt.cm.get_cmap(color_map)
    colors = cmap(np.linspace(0, 1, max(n_labels, 1)))
    color_mapping = {label: colors[i] for i, label in enumerate(all_labels)}

    for ax, (ptype, pdata) in zip(axes, plot_data.items()):
        labels_plot = pdata["labels"][::-1]
        counts_plot = pdata["counts"][::-1]
        total = pdata["total"]
        bar_colors = [color_mapping.get(l, "#808080") for l in labels_plot]
        bars = ax.barh(labels_plot, counts_plot, color=bar_colors, alpha=0.85)

        for bar in bars:
            width = bar.get_width()
            pct = (width / total) * 100 if total > 0 else 0
            x_text = width + 0.4
            xmax = ax.get_xlim()[1]
            if x_text > xmax * 0.98:
                x_text = width * 0.5
                text_color, ha = "white", "center"
            else:
                text_color, ha = "black", "left"
            ax.text(x_text, bar.get_y() + bar.get_height() / 2,
                    f"{int(width)} ({pct:.1f}%)", va="center", ha=ha,
                    fontsize=9, fontweight="bold", color=text_color)

        ax.set_xlabel("Count", fontsize=10, fontweight="bold")
        ax.set_title(f"{ptype.capitalize()} (n={total})", fontsize=12, fontweight="bold", pad=10)
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def compare_demographic_field(df, field, participant_types=None, normalise_func=None, title=None):
    """
    Compare a demographic field across participant types.
    Matches original notebook: raw extraction + per-value normalization.
    """
    if participant_types is None:
        participant_types = ['learners', 'teachers', 'policymakers', 'administrators', 'other']
    if title is None:
        title = f"{field.replace('_', ' ').title()} Distribution by Participant Type"

    reports = {}
    for ptype in participant_types:
        data = extract_demographic_field(df["participant_demographics"], field, participant_type=ptype)
        if normalise_func:
            data = [normalise_func(x) for x in data]
        report = label_frequency(data).rename_axis("label").reset_index(name="count")
        if len(report) > 0:
            reports[ptype] = report

    return reports, title


def compare_task_types_by_group(df, participant_groups=None, normalise_func=None, title=None):
    """
    Compare task types across participant groups.
    Matches original notebook: per-task individual normalization, N/A preserved.
    """
    if participant_groups is None:
        participant_groups = ['students', 'teachers', 'policymakers_administrators', 'others']
    if title is None:
        title = "Task Types by Participant Group"
    if normalise_func is None:
        normalise_func = normalise_task_type

    reports = {}
    for group in participant_groups:
        all_normalised = []
        for study_tasks in df["task_types"]:
            if isinstance(study_tasks, dict):
                group_tasks = study_tasks.get(group, [])
                if isinstance(group_tasks, list) and len(group_tasks) > 0:
                    # Normalize each task INDIVIDUALLY (matching notebook)
                    for task in group_tasks:
                        result = normalise_func(task)
                        if isinstance(result, list):
                            all_normalised.extend(result)
                        else:
                            all_normalised.append(result)
        if all_normalised:
            report = label_frequency(all_normalised).rename_axis("label").reset_index(name="count")
            if len(report) > 0:
                reports[group] = report

    return reports, title


def generate_all_figures(df):
    """
    Generate all standard normalization visualizations.
    Returns dict of {chart_name: matplotlib.Figure}.
    """
    figures = {}

    # Geographic
    if "region_list" in df.columns:
        figures["Regions"] = create_bar_chart(
            label_frequency(df["region_list"]).rename_axis("label").reset_index(name="count"),
            "Top 20 cities/provinces/regions in LLM language education research"
        )
    if "country_list" in df.columns:
        figures["Countries"] = create_bar_chart(
            label_frequency(df["country_list"]).rename_axis("label").reset_index(name="count"),
            "Top 15 countries in LLM language education research", top_n=15
        )
    if "continent_list" in df.columns:
        figures["Continents"] = create_bar_chart(
            label_frequency(df["continent_list"]).rename_axis("label").reset_index(name="count"),
            "Continents in LLM language education research"
        )

    # Educational settings & participants
    if "educational_settings" in df.columns:
        figures["Educational Settings"] = create_bar_chart(
            label_frequency(df["educational_settings"]).rename_axis("label").reset_index(name="count"),
            "Educational settings in LLM language education research"
        )
    if "participant_type" in df.columns:
        figures["Participant Types"] = create_bar_chart(
            label_frequency(df["participant_type"]).rename_axis("label").reset_index(name="count"),
            "Participant targets in LLM language education research"
        )

    # Languages
    if "L1_norm" in df.columns:
        figures["First Languages (L1)"] = create_bar_chart(
            label_frequency(df["L1_norm"]).rename_axis("label").reset_index(name="count"),
            "Top L1s in LLM Language Education Research", top_n=15
        )
    if "target_language_norm" in df.columns:
        figures["Target Languages"] = create_bar_chart(
            label_frequency(df["target_language_norm"]).rename_axis("label").reset_index(name="count"),
            "Top Target Languages in LLM Language Education Research", top_n=15
        )

    # Proficiency & literacy
    for col, title in [
        ("language_status_norm", "Language Status Distribution"),
        ("cefr_norm", "CEFR Levels Distribution"),
        ("ai_literacy_norm", "AI Literacy Levels Distribution"),
    ]:
        if col in df.columns:
            figures[title] = create_bar_chart(
                label_frequency(df[col]).rename_axis("label").reset_index(name="count"),
                title
            )

    # Skills, tasks, models
    if "language_skills_norm" in df.columns:
        figures["Language Skills Targeted"] = create_bar_chart(
            label_frequency(df["language_skills_norm"]).rename_axis("label").reset_index(name="count"),
            "Language skills targeted in LLM Language Education Research"
        )
    if "task_types_norm" in df.columns:
        figures["Task Types"] = create_bar_chart(
            label_frequency(df["task_types_norm"]).rename_axis("label").reset_index(name="count"),
            "Top language tasks in LLM Language Education Research", top_n=10
        )
    if "LLMs_norm" in df.columns:
        figures["LLMs Used"] = create_bar_chart(
            label_frequency(df["LLMs_norm"]).rename_axis("label").reset_index(name="count"),
            "Top 10 LLMs used in language education research", top_n=10
        )

    # Prompting
    if "prompting_techniques_norm" in df.columns:
        figures["Prompting Techniques"] = create_bar_chart(
            label_frequency(df["prompting_techniques_norm"]).rename_axis("label").reset_index(name="count"),
            "Prompting techniques in LLM language education research"
        )
    if "prompting_strategies_norm" in df.columns:
        figures["Prompting Strategies"] = create_bar_chart(
            label_frequency(df["prompting_strategies_norm"]).rename_axis("label").reset_index(name="count"),
            "Prompting strategies in LLM language education research"
        )

    # Methodology
    for col, title in [
        ("research_methodology", "Research Methodologies"),
        ("data_gathering_methods", "Data Gathering Methods"),
        ("research_design", "Research Designs"),
    ]:
        if col in df.columns:
            figures[title] = create_bar_chart(
                label_frequency(df[col]).rename_axis("label").reset_index(name="count"),
                f"{title} in LLM language education research"
            )

    # Frameworks, sample size, duration, challenges
    if "frameworks_norm" in df.columns:
        flat_frameworks = [fw for sublist in df["frameworks_norm"] if isinstance(sublist, list) for fw in sublist]
        figures["Frameworks"] = create_bar_chart(
            label_frequency(flat_frameworks).rename_axis("label").reset_index(name="count"),
            "Top 30 frameworks in LLM language education research", top_n=30
        )
    if "sample_size_norm" in df.columns:
        figures["Sample Size Distribution"] = create_bar_chart(
            label_frequency(df["sample_size_norm"]).rename_axis("label").reset_index(name="count"),
            "Sample size distribution in LLM language education research"
        )
    if "duration_norm" in df.columns:
        figures["Study Duration"] = create_bar_chart(
            label_frequency(df["duration_norm"]).rename_axis("label").reset_index(name="count"),
            "Study length distribution in LLM language education research"
        )
    if "challenges_concerns_limitations_of_LLMs_in_language_education" in df.columns:
        figures["Challenges & Limitations"] = create_bar_chart(
            label_frequency(df["challenges_concerns_limitations_of_LLMs_in_language_education"]).rename_axis("label").reset_index(name="count"),
            "Challenges, concerns, and limitations of LLMs in language education"
        )
        # === COMPARATIVE VISUALIZATIONS BY PARTICIPANT TYPE ===
    # L1 by participant type
    l1_reports, l1_title = compare_demographic_field(df, "first_language", normalise_func=normalise_L1,
                                                      title="First Language (L1) distribution by participant type")
    if l1_reports:
        figures["L1 by Participant Type"] = plot_comparative_distribution(l1_reports, title=l1_title)

    # Target language by participant type
    tgt_reports, tgt_title = compare_demographic_field(df, "target_language", normalise_func=normalise_target_language,
                                                        title="Target language distribution by participant type")
    if tgt_reports:
        figures["Target Language by Participant Type"] = plot_comparative_distribution(tgt_reports, title=tgt_title)

    # Language status by participant type
    status_reports, status_title = compare_demographic_field(df, "language_status",
                                                              title="Language status by participant type")
    if status_reports:
        figures["Language Status by Participant Type"] = plot_comparative_distribution(status_reports, title=status_title)

    # CEFR by participant type
    cefr_reports, cefr_title = compare_demographic_field(df, "CEFR", title="CEFR levels by participant type")
    if cefr_reports:
        figures["CEFR by Participant Type"] = plot_comparative_distribution(cefr_reports, title=cefr_title)

    # AI literacy by participant type
    ai_reports, ai_title = compare_demographic_field(df, "AI_literacy", title="AI literacy levels by participant type")
    if ai_reports:
        figures["AI Literacy by Participant Type"] = plot_comparative_distribution(ai_reports, title=ai_title)

    # Gender by participant type
    gender_reports, gender_title = compare_demographic_field(df, "gender", normalise_func=normalize_gender,
                                                              title="Gender distribution by participant type")
    if gender_reports:
        figures["Gender by Participant Type"] = plot_comparative_distribution(gender_reports, title=gender_title)

    # Age by participant type
    age_reports, age_title = compare_demographic_field(df, "age", normalise_func=normalize_age,
                                                        title="Age band distribution by participant type")
    if age_reports:
        figures["Age by Participant Type"] = plot_comparative_distribution(age_reports, title=age_title)

    # Task types by participant group
    task_reports, task_title = compare_task_types_by_group(df, title="Task types by participant type")
    if task_reports:
        figures["Task Types by Participant Type"] = plot_comparative_distribution(task_reports, title=task_title)

    return figures

# =============================================================================
# CUSTOM LIGHT NORMALIZATION (schema-agnostic)
# =============================================================================

def light_normalize_value(value):
    """
    Schema-agnostic normalization for a single cell value.
    Handles: list flattening, N/A standardization, whitespace cleanup,
    unicode normalization, deduplication within lists.
    Safely handles dicts and other unhashable types without crashing.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"

    if isinstance(value, dict):
        # Recursively normalize dict values; dicts themselves are not deduplicated
        return {k: light_normalize_value(v) for k, v in value.items()}

    if isinstance(value, list):
        cleaned = []
        seen = set()
        for item in value:
            normed = light_normalize_value(item)
            if normed == "N/A":
                continue
            # Only deduplicate hashable types (strings, numbers, tuples)
            # Unhashable types (dicts, lists) are always included
            try:
                if normed not in seen:
                    seen.add(normed)
                    cleaned.append(normed)
            except TypeError:
                # Unhashable type — include without deduplication
                cleaned.append(normed)
        return cleaned if cleaned else ["N/A"]

    if isinstance(value, str):
        s = unicodedata.normalize("NFKC", value).strip()
        if not s or s.lower() in ("n/a", "na", "none", "null", "", "-"):
            return "N/A"
        # Remove "(inferred)" suffix for consistency
        s = s.replace("(inferred)", "").strip()
        return s if s else "N/A"

    return value


def light_normalize_dataframe(df: pd.DataFrame, selected_columns: list) -> pd.DataFrame:
    """Apply light normalization to selected columns only."""
    result = df.copy()
    for col in selected_columns:
        if col in result.columns:
            result[col] = result[col].apply(light_normalize_value)
    return result


# =============================================================================
# STREAMLIT RENDER FUNCTION
# =============================================================================

def render_normalization_page():
    st.title("5️⃣ Rule-Based Normalization")
    st.markdown("""
    Clean and standardize extracted metadata. Choose **Default Pipeline** mode if you used 
    the locked extraction prompt (full domain-specific normalization). Choose **Custom Light** 
    mode if you used a custom extraction prompt (structural cleanup only).
    """)

    tab1, tab2, tab3 = st.tabs(["⚙️ Configuration & Run", "📊 Results & Export", "📈 Visualizations"])

    # =========================================================================
    # TAB 1
    # =========================================================================
    with tab1:
        # === REPLICATION MODE INJECTION ===
        if st.session_state.get("replication_mode"):
            from core.utils import get_replication_path

            st.info("🔬 **Replication Mode:** Original normalized dataset loaded. Toggle off in sidebar to run your own normalization.")

            repl_path = get_replication_path("s5_normalized")
            if repl_path and os.path.exists(repl_path):
                try:
                    with open(repl_path, "r", encoding="utf-8") as f:
                        repl_data = json.load(f)
                    repl_df = pd.DataFrame(repl_data)
                    st.session_state["norm_result_df"] = repl_df
                    st.session_state["norm_mode_used"] = "default"
                    # Clear cached figures so they regenerate from replication data
                    st.session_state.pop("norm_figures", None)
                    st.success(f"✅ Loaded {len(repl_df)} normalized records from replication data.")
                except Exception as e:
                    st.error(f"❌ Failed to load replication normalization file: {e}")
            else:
                st.warning("⚠️ Replication normalization file not found. Check `replication_data/stage5_normalized_output.json`.")

            st.divider()
            # Skip normal upload/config/run — fall through to results display below
            if "norm_result_df" not in st.session_state:
                return
        else:
            st.subheader("Upload Extracted Data")
            json_upload = st.file_uploader(
                "Upload structured_output.json (from Stage 3)",
                type=["json"],
                key="norm_json_upload"
            )

            df = None
            # --- AUTO-RECOVER FROM DISK IF SESSION STATE LOST ---
            if "norm_result_df" not in st.session_state:
                for candidate in ["normalized_output.json", "normalized_output_light.json"]:
                    auto_path = os.path.join(AUTO_SAVE_DIR, candidate)
                    if os.path.exists(auto_path):
                        try:
                            with open(auto_path, "r", encoding="utf-8") as f:
                                recovered_data = json.load(f)
                            st.session_state["norm_result_df"] = pd.DataFrame(recovered_data)
                            st.session_state["norm_mode_used"] = "default" if "normalized_output.json" in candidate else "custom_light"
                            st.info(f"♻️ Recovered normalized data from `{auto_path}` ({len(recovered_data)} records)")
                            break
                        except Exception as e:
                            st.warning(f"⚠️ Could not recover auto-save: {e}")
            if json_upload:
                try:
                    json_upload.seek(0)
                    data = json.loads(json_upload.read().decode("utf-8"))
                    json_upload.seek(0)
                    df = pd.DataFrame(data)
                    st.success(f"✅ Loaded {len(df)} records")
                except Exception as e:
                    st.error(f"❌ Failed to load JSON: {e}")

            if df is None:
                st.warning("⚠️ Upload a JSON file to proceed.")
                return

            st.divider()

            # --- Mode Selection ---
            norm_mode = st.radio(
                "Normalization Mode",
                ["🔒 Default Pipeline", "✏️ Custom Light"],
                horizontal=True,
                key="norm_mode",
                help="Default = full domain-specific normalization (requires default extraction schema). Custom Light = structural cleanup for any schema."
            )

            if norm_mode == "🔒 Default Pipeline":
                st.info("Applies all validated normalization functions: sample size banding, age/gender/demographics parsing, L1/target language consolidation, task type mapping, framework normalization, region/country standardization, study duration banding, and more.")
                selected_columns = list(df.columns)
            else:
                st.warning("⚠️ Custom Light mode applies only structural normalization (N/A standardization, list deduplication, whitespace cleanup, unicode normalization). Domain-specific mappings are NOT applied.")
                metadata_cols = {'source_filename', 'title', 'authors', 'summary', 'APA_reference'}
                available = sorted([c for c in df.columns if c not in metadata_cols])
                selected_columns = st.multiselect(
                    "Columns to Normalize",
                    options=available,
                    default=available,
                    key="norm_custom_columns"
                )

            st.divider()

            if st.button("🚀 Run Normalization", type="primary", key="norm_run_btn"):
                with st.spinner("Normalizing..."):
                    if norm_mode == "🔒 Default Pipeline":
                        # Apply full default normalization
                        # "Apply to dataframe" section exactly.
                        try:
                            loc_parsed = df["study_location"].apply(parse_study_location)
                            df["region_list"] = loc_parsed.apply(lambda x: x["region"]).apply(
                                lambda lst: [normalise_region_label_simple(x) for x in lst])
                            df["country_list"] = loc_parsed.apply(lambda x: x["country"]).apply(
                                lambda lst: [normalise_country(x) for x in lst])
                            df["continent_list"] = loc_parsed.apply(lambda x: x["continent"]).apply(
                                lambda lst: [normalise_continent(x) for x in lst])

                            # --- Row-safe demographic extraction (one value PER ROW) ---
                            def _row_extract_demo(demo_list, field):
                                """Extract demographic field values per-study, preserving N/A as a distinct value."""
                                vals = []
                                if isinstance(demo_list, list):
                                    for p in demo_list:
                                        if isinstance(p, dict):
                                            v = p.get(field, "")
                                            s = str(v).strip() if v else ""
                                            if not s:
                                                vals.append("N/A")
                                            elif s.lower() in ("n/a", "na", "none", "not specified", "unspecified"):
                                                vals.append("N/A")
                                            else:
                                                vals.append(s)
                                return vals if vals else ["N/A"]

                            def _row_norm_l1(demo_list):
                                """Normalize each L1 value individually, matching original notebook behavior."""
                                vals = _row_extract_demo(demo_list, "first_language")
                                if vals == ["N/A"]:
                                    return ["N/A"]
                                all_normed = []
                                for v in vals:
                                    normed = normalise_L1(v)
                                    all_normed.extend(normed)
                                return all_normed if all_normed else ["N/A"]

                            def _row_norm_target(demo_list):
                                """Normalize each target language value individually, matching original notebook behavior."""
                                vals = _row_extract_demo(demo_list, "target_language")
                                if vals == ["N/A"]:
                                    return ["N/A"]
                                all_normed = []
                                for v in vals:
                                    normed = normalise_target_language(v)
                                    all_normed.extend(normed)
                                return all_normed if all_normed else ["N/A"]

                            df["L1_norm"] = df["participant_demographics"].apply(_row_norm_l1)
                            df["target_language_norm"] = df["participant_demographics"].apply(_row_norm_target)
                            df["language_status_norm"] = df["participant_demographics"].apply(lambda d: _row_extract_demo(d, "language_status"))
                            df["cefr_norm"] = df["participant_demographics"].apply(lambda d: _row_extract_demo(d, "CEFR"))
                            df["ai_literacy_norm"] = df["participant_demographics"].apply(lambda d: _row_extract_demo(d, "AI_literacy"))
                            df["language_skills_norm"] = df["language_skills_targeted"].apply(normalise_list)

                            def _row_norm_tasks(task_obj):
                                """Normalize each task individually, matching original notebook behavior."""
                                if not isinstance(task_obj, dict):
                                    return ["N/A"]
                                all_tasks = []
                                for pt in ['students', 'teachers', 'policymakers_administrators', 'others']:
                                    tasks = task_obj.get(pt, [])
                                    if isinstance(tasks, list):
                                        all_tasks.extend(tasks)
                                if not all_tasks:
                                    return ["N/A"]
                                # Normalize each task INDIVIDUALLY (not as a batch)
                                normed = []
                                for task in all_tasks:
                                    result = normalise_task_type(task)
                                    if isinstance(result, list):
                                        normed.extend(result)
                                    else:
                                        normed.append(result)
                                return normed if normed else ["N/A"]

                            df["task_types_norm"] = df["task_types"].apply(_row_norm_tasks)
                            df["LLMs_norm"] = df["LLMs_used"].apply(lambda m: normalise_llm_model(m) if isinstance(m, list) else [normalise_llm_model(m)])
                            df["prompting_techniques_norm"] = df["prompting_techniques"].apply(normalise_list)
                            df["prompting_strategies_norm"] = df["prompting_strategies"].apply(lambda s: normalise_prompting_strategy(s) if isinstance(s, list) else [normalise_prompting_strategy(s)])
                            df["frameworks_norm"] = df["frameworks"].apply(lambda fw: normalise_frameworks(fw) if isinstance(fw, list) else [normalise_frameworks(fw)])
                            df["sample_size_norm"] = df["sample_size"].apply(categorize_sample_size)
                            df["duration_norm"] = df["duration"].apply(normalize_study_duration)

                            normalized_df = df
                            st.session_state["norm_result_df"] = normalized_df
                            st.session_state["norm_mode_used"] = "default"
                            st.success(f"✅ Default normalization complete for {len(normalized_df)} records")
                            # ---- AUTO-SAVE TO TEMP (safety net) ----
                            try:
                                normalized_df.to_json(os.path.join(AUTO_SAVE_DIR, "normalized_output.json"), orient="records", force_ascii=False, indent=2)
                                normalized_df.to_csv(os.path.join(AUTO_SAVE_DIR, "normalized_output.csv"), index=False)
                                st.caption(f"💾 Auto-saved to `{AUTO_SAVE_DIR}`")
                            except Exception as save_err:
                                st.warning(f"⚠️ Auto-save failed: {save_err}")

                        except Exception as e:
                            st.error(f"❌ Default normalization failed: {e}\nEnsure your data matches the default extraction schema.")
                            return
                    else:
                        # Custom light normalization
                        if not selected_columns:
                            st.error("❌ Select at least one column.")
                            return
                        normalized_df = light_normalize_dataframe(df, selected_columns)
                        st.session_state["norm_result_df"] = normalized_df
                        st.session_state["norm_mode_used"] = "custom_light"
                        st.session_state.pop("norm_figures", None)
                        st.session_state["norm_custom_cols_used"] = selected_columns
                        st.success(f"✅ Light normalization complete for {len(normalized_df)} records across {len(selected_columns)} columns")
                        # ---- AUTO-SAVE TO TEMP (safety net) ----
                        try:
                            normalized_df.to_json(os.path.join(AUTO_SAVE_DIR, "normalized_output_light.json"), orient="records", force_ascii=False, indent=2)
                            normalized_df.to_csv(os.path.join(AUTO_SAVE_DIR, "normalized_output_light.csv"), index=False)
                            st.caption(f"💾 Auto-saved to `{AUTO_SAVE_DIR}`")
                        except Exception as save_err:
                            st.warning(f"⚠️ Auto-save failed: {save_err}")

            # Reproducibility & Accessibility
            if "norm_result_df" in st.session_state:
                with st.expander("🔬 Reproducibility & Accessibility Information", expanded=False):
                    repro = {
                        "stage": "normalization",
                        "timestamp": datetime.now().isoformat(),
                        "audit_trail": {
                            "mode": st.session_state.get("norm_mode_used", "N/A"),
                            "records_normalized": len(st.session_state["norm_result_df"]),
                            "columns_processed": st.session_state.get("norm_custom_cols_used", "all (default pipeline)"),
                            "auto_save_directory": AUTO_SAVE_DIR,
                            "figures_directory": FIGURES_DIR,
                            "output_files": {
                                "json": os.path.join(AUTO_SAVE_DIR, "normalized_output.json"),
                                "csv": os.path.join(AUTO_SAVE_DIR, "normalized_output.csv"),
                            },
                        },
                        "normalization_logic": {
                            "default_pipeline": "Domain-specific mapping: sample size banding, age/gender parsing, L1/target language consolidation, task type mapping, framework normalization, region/country standardization, study duration banding",
                            "custom_light": "Schema-agnostic structural cleanup: N/A standardization, list deduplication, whitespace cleanup, unicode normalization, (inferred) tag removal",
                        },
                    }
                    st.json(repro)
                    st.code(json.dumps(repro, indent=2), language="json")

                    st.divider()

                    # Reproducibility Checklist
                    st.markdown("#### ✅ Reproducibility Checklist")
                    st.markdown(f"""
                    - [x] Input JSON (`structured_output.json`) preserved from Stage 3
                    - [x] Normalized output auto-saved to `{AUTO_SAVE_DIR}`
                    - [x] All normalization functions self-contained (no external API calls)
                    - [x] Mode and parameters recorded above
                    - [ ] Document any manual corrections or schema adjustments made during normalization
                    """)

                    st.divider()

                    # Accessibility Alternatives
                    st.markdown("#### ♿ Accessibility Alternatives")
                    st.markdown("""
                    | Scenario | Alternative |
                    |---|---|
                    | **Non-programmers** | Export normalization dictionaries to CSV/Excel for manual review and mapping, then re-import into Python |
                    | **Custom schemas** | Use **Custom Light** mode for structural cleanup without domain-specific mappings |
                    """)

    # =========================================================================
    # TAB 2
    # =========================================================================
    with tab2:
        if "norm_result_df" not in st.session_state:
            st.info("ℹ️ Run normalization in the Configuration tab first.")
        else:
            result_df = st.session_state["norm_result_df"]

            st.subheader("Preview Normalized Data")
            st.dataframe(result_df.head(10), use_container_width=True)

            st.divider()
            st.subheader("📥 Download Normalized Data")

            buf = io.BytesIO()
            result_df.to_json(buf, orient="records", force_ascii=False, indent=2)
            buf.seek(0)
            st.download_button(
                label="📊 Download Normalized JSON",
                data=buf,
                file_name="normalized_output.json",
                mime="application/json",
                type="primary",
                key="norm_download_json"
            )

            csv_buf = io.BytesIO()
            result_df.to_csv(csv_buf, index=False)
            csv_buf.seek(0)
            st.download_button(
                label="📄 Download Normalized CSV",
                data=csv_buf,
                file_name="normalized_output.csv",
                mime="text/csv",
                key="norm_download_csv"
            )

    # =========================================================================
    # TAB 3: VISUALIZATIONS
    # =========================================================================
    with tab3:
        if "norm_result_df" not in st.session_state:
            st.info("ℹ️ Run normalization in the Configuration tab first to generate visualizations.")
        else:
            result_df = st.session_state["norm_result_df"]
            norm_mode_used = st.session_state.get("norm_mode_used", "unknown")

            if norm_mode_used == "custom_light":
                st.info("💡 **Custom Light mode:** Visualizations are generated from structurally cleaned data. Domain-specific mappings (e.g., language consolidation, framework standardization) were NOT applied. Charts reflect raw category labels after whitespace/N/A cleanup only.")

            # Generate figures (cached in session state to avoid regeneration on rerun)
            if "norm_figures" not in st.session_state:
                with st.spinner("Generating visualizations..."):
                    st.session_state["norm_figures"] = generate_all_figures(result_df)

            figures = st.session_state.get("norm_figures", {})

            if not figures:
                st.info("ℹ️ No visualization data available. Ensure default normalization completed successfully.")
            else:
                st.subheader(f"📈 Normalization Visualizations ({len(figures)} charts)")
                st.caption("All charts show frequency counts with percentages. Generated from normalized data using the same logic as the validated pipeline notebook.")

                # Chart selector
                chart_names = list(figures.keys())
                selected_charts = st.multiselect(
                    "Select charts to display",
                    options=chart_names,
                    default=chart_names[:6],
                    key="norm_viz_selector"
                )

                # Display selected charts
                for chart_name in selected_charts:
                    if chart_name in figures:
                        st.markdown(f"#### {chart_name}")
                        st.pyplot(figures[chart_name])
                        st.divider()

                # Download all figures option
                st.divider()
                # Create ZIP file in memory for download + auto-save individual PNGs
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for name, fig in figures.items():
                        img_buffer = io.BytesIO()
                        fig.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
                        img_buffer.seek(0)
                        safe_name = f"{name.lower().replace(' ', '_').replace('/', '_')}.png"
                        zip_file.writestr(safe_name, img_buffer.read())
                        # Also auto-save individual PNG to disk
                        try:
                            fig.savefig(os.path.join(FIGURES_DIR, safe_name), dpi=150, bbox_inches="tight")
                        except Exception:
                            pass
                
                zip_buffer.seek(0)
                st.download_button(
                    label="💾 Download All Figures (ZIP)",
                    data=zip_buffer,
                    file_name="normalization_figures.zip",
                    mime="application/zip",
                    type="secondary",
                    key="norm_download_figs_zip_btn"
                )
                st.caption(f"💾 Individual figures also saved to `{FIGURES_DIR}`")