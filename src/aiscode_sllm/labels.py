from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

_CODE_RE_7 = re.compile(r"\b\d{7}\b")
_CODE_RE_6_7 = re.compile(r"\b\d{6,7}\b")

def load_code2idx(path: str | Path) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): int(v) for k, v in raw.items()}

def build_idx2code(code2idx: Dict[str, int]) -> Dict[int, str]:
    return {int(v): str(k) for k, v in code2idx.items()}

def normalize_display_code(code_3lvl: str) -> str:
    # "1/2/03" -> "1203"
    return str(code_3lvl).replace("/", "")

def pad_numeric_codes_in_text(text: str) -> str:
    # Pad any 1-6 digit numeric token to 7 digits (zfill) – use with care.
    return re.sub(r"\b(\d{1,6})\b", lambda m: m.group(1).zfill(7), str(text))

def parse_codes_from_completion(completion: str) -> List[str]:
    # Extract 6-7 digit codes from completion, pad 6->7, return as list of 7-digit strings
    codes = _CODE_RE_6_7.findall(str(completion))
    out = []
    for c in codes:
        c = c.strip()
        if len(c) == 6:
            c = c.zfill(7)
        if len(c) == 7:
            out.append(c)
    return out

def numeric7_to_3lvl(code7: str) -> str:
    """Convert a 7-digit AIS numeric code to a 3-level string code.

    Convention used in this repository:
      - level1: first digit
      - level2: second digit
      - level3: third+fourth digits (2-digit, zero-padded as-is)

    Example:
      "1406942" -> "1/4/06"
    """
    c = str(code7).strip()
    if len(c) != 7 or not c.isdigit():
        raise ValueError(f"Expected 7-digit numeric AIS code, got: {code7!r}")
    return f"{c[0]}/{c[1]}/{c[2:4]}"


def parse_3lvl_codes(completion_or_codes: str | Sequence[str]) -> List[str]:
    """Parse 3-level AIS codes from a string or a list.

    Accepts:
      - already-3lvl codes (e.g., "1/4/06, 2/5/10")
      - a completion string containing 6-7 digit AIS numeric codes
      - a list/tuple of either 3lvl strings or 7-digit numeric strings

    Returns:
      A list of normalized 3lvl code strings.
    """
    if isinstance(completion_or_codes, (list, tuple)):
        out: List[str] = []
        for x in completion_or_codes:
            s = str(x).strip()
            if not s:
                continue
            if "/" in s:
                out.append(s)
            elif len(s) == 7 and s.isdigit():
                out.append(numeric7_to_3lvl(s))
            else:
                # try extracting numeric codes from mixed strings
                out.extend([numeric7_to_3lvl(c) for c in parse_codes_from_completion(s)])
        return out

    s = str(completion_or_codes)
    if "/" in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return parts

    # fallback: numeric codes in string
    return [numeric7_to_3lvl(c) for c in parse_codes_from_completion(s)]
