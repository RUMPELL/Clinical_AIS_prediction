#!/usr/bin/env python
"""Evaluate a Qwen3-8B + LoRA multilabel classifier on JSONL.

This script is intended for *reproducible* evaluation runs (no notebooks required).

Input JSONL:
  - "text" (preferred) OR "prompt"
  - "completion" OR "code" (GT AIS codes)

GT code parsing supports both:
  - 3-level codes (e.g., "1/4/06")
  - 7-digit numeric codes (e.g., "1406942") which are converted to 3-level

Example:
  python scripts/evaluate.py --config configs/eval.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Set

import numpy as np
import torch
from tqdm import tqdm

from aiscode_sllm.config import parse_args_with_config
from aiscode_sllm.io import read_jsonl, write_jsonl
from aiscode_sllm.labels import load_code2idx, normalize_display_code, parse_3lvl_codes
from aiscode_sllm.metrics import compute_multilabel_metrics, compute_prefix_hits
from aiscode_sllm.modeling import load_qwen3_multilabel_with_mlp_head
from aiscode_sllm.prompt import build_chatml_prompt


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Qwen3-8B + LoRA AIS multilabel classifier")
    p.add_argument("--config", type=str, default=None, help="YAML config file (optional)")

    p.add_argument(
        "--data",
        required=True,
        help="Validation JSONL. Must contain 'text' or 'prompt' and 'completion' or 'code'.",
    )
    p.add_argument("--code2idx", required=True, help="Path to code2idx_3lvl.json")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--model-dir", required=True, help="Tokenizer/config directory (usually training output dir).")
    p.add_argument("--adapter-dir", required=True, help="LoRA adapter directory.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--topk-fallback", type=int, default=2)
    p.add_argument(
        "--wrap-chatml",
        action="store_true",
        default=True,
        help="Wrap input text with the project's ChatML template before tokenization.",
    )

    p.add_argument("--out-pred", default=None, help="Optional: write per-sample predictions JSONL.")
    p.add_argument("--out-metrics", default="metrics.json", help="Where to write a JSON metrics report.")
    return p


@torch.inference_mode()
def predict_one(
    text: str,
    loaded,
    threshold: float,
    max_len: int,
    topk_fallback: int,
    wrap_chatml: bool,
) -> List[int]:
    prompt = build_chatml_prompt(text) if wrap_chatml else text
    inputs = loaded.tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    ).to(loaded.device)
    logits = loaded.model(**inputs).logits[0]
    probs = torch.sigmoid(logits).detach().cpu()
    idxs = (probs >= threshold).nonzero(as_tuple=True)[0].tolist()
    if not idxs:
        idxs = probs.topk(topk_fallback).indices.tolist()
    idxs = sorted(idxs, key=lambda i: float(probs[i]), reverse=True)
    return idxs


def main() -> None:
    parser = build_parser()
    args, _ = parse_args_with_config(parser)

    code2idx = load_code2idx(args.code2idx)
    idx2code = {v: k for k, v in code2idx.items()}

    loaded = load_qwen3_multilabel_with_mlp_head(
        base_model=args.base_model,
        model_dir=args.model_dir,
        adapter_dir=args.adapter_dir,
        code2idx=code2idx,
        device=args.device,
        max_length=args.max_len,
    )

    y_true: List[Set[int]] = []
    y_pred: List[Set[int]] = []

    gt_codes_norm: List[List[str]] = []
    pred_codes_norm: List[List[str]] = []

    pred_rows = []

    for obj in tqdm(list(read_jsonl(args.data)), desc="eval"):
        text = obj.get("text")
        if text is None:
            text = obj.get("prompt", "")

        gt_codes = parse_3lvl_codes(obj.get("completion", obj.get("code", "")))
        gt_idx = {code2idx[c] for c in gt_codes if c in code2idx}

        pred_idx_list = predict_one(
            text=text,
            loaded=loaded,
            threshold=float(args.threshold),
            max_len=int(args.max_len),
            topk_fallback=int(args.topk_fallback),
            wrap_chatml=bool(args.wrap_chatml),
        )
        pr_idx = set(pred_idx_list)

        y_true.append(gt_idx)
        y_pred.append(pr_idx)

        gt_disp = [normalize_display_code(c) for c in gt_codes if isinstance(c, str)]
        pr_codes = [idx2code[i] for i in pred_idx_list]
        pr_disp = [normalize_display_code(c) for c in pr_codes]

        gt_codes_norm.append(gt_disp)
        pred_codes_norm.append(pr_disp)

        if args.out_pred:
            pred_rows.append(
                {
                    "text": text,
                    "gt_codes": gt_codes,
                    "pred_codes": pr_codes,
                    "pred_indices": pred_idx_list,
                }
            )

    metrics = compute_multilabel_metrics(y_true, y_pred, num_labels=len(code2idx)).as_dict()
    prefix = compute_prefix_hits(gt_codes_norm, pred_codes_norm, prefix_lens=(1, 2, 4))
    report = {
        "metrics": metrics,
        "prefix_metrics": prefix,
        "threshold": float(args.threshold),
        "num_samples": len(y_true),
        "wrap_chatml": bool(args.wrap_chatml),
    }

    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_metrics).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.out_pred:
        write_jsonl(args.out_pred, pred_rows)
        print(f"Wrote predictions to: {args.out_pred}")


if __name__ == "__main__":
    main()
