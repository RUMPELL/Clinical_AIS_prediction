from __future__ import annotations

import re

# Match multiple common date patterns (Korean + English) and replace with <DATE>
_DATE_RE = re.compile(
    r'(?<!\d)('
    r'20\d{2}[-/\.](0?[1-9]|1[0-2])[-/\.](0?[1-9]|[12]\d|3[01])|'  # 2024-05-22
    r'(0?[1-9]|1[0-2])[-/\.](0?[1-9]|[12]\d|3[01])|'                 # 05-22
    r'20\d{2}년\s*(0?[1-9]|1[0-2])월\s*(0?[1-9]|[12]\d|3[01])일|'  # 2024년 5월 22일
    r'(0?[1-9]|1[0-2])월\s*(0?[1-9]|[12]\d|3[01])일|'                # 5월 22일
    r'(0?[1-9]|[12]\d|3[01])\s*'
    r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
    r')(?!\d)',
    flags=re.I,
)

_SPACE_RE = re.compile(r"\s+")

# ChatML user block extractor
_USER_BLOCK_RE = re.compile(
    r"<\|im_start\|>user\s*(.*?)\s*<\|im_end\|>",
    flags=re.S,
)

def scrub_dates(text: str) -> str:
    return _DATE_RE.sub("<DATE>", str(text))

def normalize_whitespace(text: str) -> str:
    return _SPACE_RE.sub(" ", str(text)).strip()

def clean_text(text: str) -> str:
    text = scrub_dates(text)
    return normalize_whitespace(text)

def extract_chatml_user(prompt: str) -> str:
    m = _USER_BLOCK_RE.search(str(prompt))
    return m.group(1).strip() if m else str(prompt).strip()

def preprocess_prompt_to_text(prompt: str) -> str:
    # clean -> extract user -> clean again
    prompt = clean_text(prompt)
    report = extract_chatml_user(prompt)
    return clean_text(report)
