#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from aiscode_sllm.io import read_jsonl, write_jsonl
from aiscode_sllm.text import preprocess_prompt_to_text
from aiscode_sllm.labels import pad_numeric_codes_in_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert ChatML JSONL (prompt/completion) into model-ready JSONL (text/completion).")
    p.add_argument("--src", required=True, help="Input JSONL containing 'prompt' and 'completion'.")
    p.add_argument("--dst", required=True, help="Output JSONL with 'text' and padded 'completion'.")
    p.add_argument("--keep-prompt", action="store_true", help="Keep original prompt field (default: drop).")
    p.add_argument("--pad-codes", action="store_true", help="Pad 1-6 digit numeric tokens in completion to 7 digits (zfill).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for obj in tqdm(list(read_jsonl(args.src)), desc="rewrite"):
        prompt = obj.get("prompt", "")
        obj["text"] = preprocess_prompt_to_text(prompt)
        if not args.keep_prompt and "prompt" in obj:
            obj.pop("prompt", None)
        if args.pad_codes and "completion" in obj:
            obj["completion"] = pad_numeric_codes_in_text(obj["completion"])
        rows.append(obj)
    write_jsonl(args.dst, rows)
    print(f"Done: {args.dst} (n={len(rows)})")


if __name__ == "__main__":
    main()
