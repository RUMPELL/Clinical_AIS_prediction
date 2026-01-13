#!/usr/bin/env python
"""External validation / inference on CBNU-style Excel.

This script converts the original external-validation notebook into a
GitHub-friendly CLI program.

CBNU Excel (as observed in the author's dataset) includes a ground-truth
AIS code list column:

  - de_list_외 : a Python-list string, e.g. "['6504182','6504182']"

The list elements are 7-digit numeric AIS codes. For evaluation we convert them
into 3-level codes (a/b/cd) and then to 4-digit prefixes (abcd) for set-based
metrics.

Clinical text columns vary by institution/export. You can specify the text
column via --text-col or let the script auto-detect common names.

If no text column exists, use --gt-only to export parsed GT and skip inference.

Example:
  python scripts/external_cbnu_infer.py --config configs/external_cbnu.yaml
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import torch
from tqdm import tqdm

from aiscode_sllm.config import parse_args_with_config
from aiscode_sllm.io import write_jsonl
from aiscode_sllm.labels import load_code2idx, numeric7_to_3lvl, parse_codes_from_completion
from aiscode_sllm.modeling import load_model_with_lora
from aiscode_sllm.prompt import build_chatml_prompt


def ensure_list(x: Any) -> List[str]:
    """Convert an Excel cell to a list of code strings.

    Supports:
      - python list object
      - python-list string: "['6504182','6504182']"
      - comma-separated string: "6504182,6504182"
    """

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]

    s = str(x).strip()
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, list):
                return [str(v).strip() for v in val if str(v).strip()]
        except Exception:
            pass

    # fallback
    return [p.strip() for p in s.split(",") if p.strip()]


def auto_detect_text_col(columns: List[str]) -> Optional[str]:
    """Try to find a plausible text column name in a CBNU export."""

    candidates = [
        "reading",
        "text",
        "report",
        "finding",
        "impression",
        "판독문",
        "판독",
        "소견",
        "판독내용",
        "결과",
        "진료내용",
    ]
    cols_set = {c.strip(): c for c in columns}
    for cand in candidates:
        if cand in cols_set:
            return cols_set[cand]
    # heuristic: choose first column that contains these substrings
    for c in columns:
        lc = str(c).lower()
        if any(k in lc for k in ["reading", "report", "impress", "finding"]):
            return c
        if any(k in str(c) for k in ["판독", "소견"]):
            return c
    return None


def to_prefix_set_from_numeric7(code_list: List[str]) -> Set[str]:
    """['6504182','5420102'] -> {'6504','5420'}"""
    out: Set[str] = set()
    for c in code_list:
        c = str(c).strip()
        # extract numeric tokens if the string is noisy
        toks = [c] if c.isdigit() else parse_codes_from_completion(c)
        for t in toks:
            if len(t) >= 4 and t[:4].isdigit():
                out.add(t[:4])
    return out


@torch.inference_mode()
def predict_prefixes(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    threshold: float,
    max_len: int,
    idx2label: Dict[int, str],
    device: torch.device,
    wrap_chatml: bool,
) -> Set[str]:
    prompt = build_chatml_prompt(text) if wrap_chatml else text
    enc = tokenizer(
        prompt,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    logits = out.logits.squeeze(0)
    probs = torch.sigmoid(logits).detach().cpu()
    pred_idx = (probs >= threshold).nonzero(as_tuple=True)[0].tolist()

    prefixes: Set[str] = set()
    for i in pred_idx:
        code3 = idx2label[int(i)]  # e.g., "1/4/06"
        try:
            a, b, cd = code3.split("/")
            prefixes.add(f"{a}{b}{cd}")
        except Exception:
            continue
    return prefixes


def compute_set_metrics(y_true: List[Set[str]], y_pred: List[Set[str]]) -> Dict[str, float]:
    """Set-based multilabel metrics over prefix sets."""
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    if n == 0:
        return {"subset_acc": 0.0, "jaccard": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    subset_acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    tp = sum(len(t & p) for t, p in zip(y_true, y_pred))
    fp = sum(len(p - t) for t, p in zip(y_true, y_pred))
    fn = sum(len(t - p) for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    jacc = sum((len(t & p) / len(t | p)) if (t | p) else 1.0 for t, p in zip(y_true, y_pred)) / n
    return {
        "subset_acc": float(subset_acc),
        "jaccard": float(jacc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="External validation / inference on CBNU-style Excel")
    ap.add_argument("--config", type=str, default=None, help="YAML config file (optional)")

    ap.add_argument("--excel", type=str, required=True, help="Path to external Excel file.")
    ap.add_argument(
        "--text-col",
        type=str,
        default=None,
        help="Text column name. If omitted, auto-detect will be attempted.",
    )
    ap.add_argument(
        "--code-col",
        type=str,
        default="de_list_외",
        help="GT AIS code list column (default: de_list_외).",
    )

    ap.add_argument("--gt-only", action="store_true", help="Skip inference and only export parsed GT")

    ap.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--model-dir", type=str, required=False, default=None, help="Tokenizer/config dir")
    ap.add_argument("--adapter-dir", type=str, required=False, default=None, help="LoRA adapter dir")
    ap.add_argument("--code2idx", type=str, default="code2idx_3lvl.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--wrap-chatml",
        action="store_true",
        default=True,
        help="Wrap input text with ChatML template before tokenization.",
    )

    ap.add_argument("--out-pred", type=str, default="outputs/external_cbnu_predictions.jsonl")
    ap.add_argument("--out-metrics", type=str, default="outputs/external_cbnu_metrics.json")
    return ap


def main() -> None:
    parser = build_parser()
    args, _ = parse_args_with_config(parser)

    df = pd.read_excel(args.excel)
    if args.code_col not in df.columns:
        raise KeyError(f"Missing code column: {args.code_col}. Available: {list(df.columns)}")

    text_col = args.text_col
    if not text_col:
        text_col = auto_detect_text_col(list(df.columns))

    if not text_col and not args.gt_only:
        raise KeyError(
            "No text column found. Provide --text-col or use --gt-only to export parsed GT without inference."
        )

    code2idx = load_code2idx(args.code2idx)

    loaded = None
    if not args.gt_only:
        if not args.model_dir or not args.adapter_dir:
            raise ValueError("--model-dir and --adapter-dir are required unless --gt-only is set")
        loaded = load_model_with_lora(
            base_model=args.base_model,
            model_dir=args.model_dir,
            adapter_dir=args.adapter_dir,
            code2idx=code2idx,
            device=args.device,
            torch_dtype=torch.bfloat16 if "cuda" in str(args.device) else torch.float32,
        )

    y_true: List[Set[str]] = []
    y_pred: List[Set[str]] = []
    rows_out: List[Dict[str, Any]] = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        gt_list = ensure_list(row[args.code_col])
        gt_list = list(dict.fromkeys(gt_list))  # de-duplicate, keep order
        gt_prefix = to_prefix_set_from_numeric7(gt_list)

        if args.gt_only:
            pred_prefix: Set[str] = set()
            text = ""
        else:
            text = str(row[text_col]) if row[text_col] is not None else ""
            pred_prefix = predict_prefixes(
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                text=text,
                threshold=float(args.threshold),
                max_len=int(args.max_len),
                idx2label=loaded.idx2label,
                device=loaded.device,
                wrap_chatml=bool(args.wrap_chatml),
            )
            y_true.append(gt_prefix)
            y_pred.append(pred_prefix)

        rows_out.append(
            {
                "text": text,
                "gt_codes": gt_list,
                "gt_prefixes": sorted(gt_prefix),
                "pred_prefixes": sorted(pred_prefix),
            }
        )

    Path(args.out_pred).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_pred, rows_out)

    metrics: Dict[str, float] = {}
    if not args.gt_only:
        metrics = compute_set_metrics(y_true, y_pred)
        Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        print("GT-only mode: wrote parsed ground truth to:", args.out_pred)


if __name__ == "__main__":
    main()
