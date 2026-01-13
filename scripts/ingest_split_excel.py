#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import List, Sequence

import pandas as pd
from tqdm import tqdm

from aiscode_sllm.prompt import build_chatml_prompt
from aiscode_sllm.split import multilabel_stratified_split
from aiscode_sllm.io import write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest 3-hospital Excel files and export stratified ChatML JSONL.")
    p.add_argument("--excel", nargs="+", required=True, help="Excel file paths (e.g., chu.xlsx cnu.xlsx gil.xlsx)")
    p.add_argument("--out-train", default="train_full_strat.jsonl")
    p.add_argument("--out-valid", default="ais_valid_strat.jsonl")
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--exclude-gold-xlsx", default=None, help="Optional: GOLD_LABEL.xlsx to exclude patient_ids.")
    p.add_argument("--exclude-ids-json", default=None, help="Optional: JSON file containing list of patient_ids to exclude.")
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def detect_university_tag(filename: str) -> str:
    m = re.search(r"(gil|cnu|chu)", os.path.basename(filename), flags=re.I)
    return m.group(1).lower() if m else "unk"


def load_gold_ids(gold_xlsx: str | None) -> set[str]:
    if not gold_xlsx:
        return set()
    gold_ids: set[str] = set()
    book = pd.ExcelFile(gold_xlsx)
    for sheet in book.sheet_names:
        df = book.parse(sheet)
        # expected columns in your notebooks
        df.columns = ["patient_id", "ct_text", "codes", "areas"]
        gold_ids.update(df["patient_id"].astype(str).str.strip())
    return gold_ids


def load_exclude_ids(path: str | None) -> set[str]:
    if not path:
        return set()
    with open(path, "r", encoding="utf-8") as f:
        ids = json.load(f)
    return {str(x) for x in ids}


def safe_list(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return [x]
    if isinstance(x, list):
        return x
    return [x]


def main() -> None:
    args = parse_args()
    gold_ids = load_gold_ids(args.exclude_gold_xlsx)
    exclude_ids = load_exclude_ids(args.exclude_ids_json)

    frames: List[pd.DataFrame] = []
    for f in args.excel:
        uni = detect_university_tag(f)
        df = pd.read_excel(f)
        df.columns = ["patient_id", "ct_text", "codes", "areas"]
        df["patient_id"] = df["patient_id"].apply(lambda x: f"{uni}_{x}")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # Exclusions
    before = len(df)
    if gold_ids:
        df = df[~df["patient_id"].isin(gold_ids)]
    if exclude_ids:
        df = df[~df["patient_id"].isin(exclude_ids)]
    after = len(df)
    print(f"Pool filtered: {before:,} -> {after:,}")

    # Normalize columns
    df["ct_text"] = df["ct_text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["codes"] = df["codes"].apply(safe_list)

    # Split
    train_idx, val_idx = multilabel_stratified_split(df["codes"].tolist(), val_ratio=args.val_ratio, random_state=args.random_state)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    print(f"Train size: {len(train_df):,} | Valid size: {len(val_df):,}")

    def row_to_json(row) -> dict:
        codes_str = ", ".join([str(c) for c in row["codes"]])
        prompt = build_chatml_prompt(row["ct_text"])
        completion = f"{codes_str}\n<|im_end|>"
        return {"prompt": prompt, "completion": completion, "patient_id": str(row["patient_id"])}

    write_jsonl(args.out_train, (row_to_json(r) for _, r in tqdm(train_df.iterrows(), total=len(train_df))))
    write_jsonl(args.out_valid, (row_to_json(r) for _, r in tqdm(val_df.iterrows(), total=len(val_df))))

    print(f"Wrote: {args.out_train}")
    print(f"Wrote: {args.out_valid}")


if __name__ == "__main__":
    main()
