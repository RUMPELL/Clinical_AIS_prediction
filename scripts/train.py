#!/usr/bin/env python
"""Train Qwen3-8B for AIS multi-label classification (3-level labels).

This script mirrors the original training logic used in the project:

- Base model: Qwen/Qwen3-8B (sequence classification)
- Label space: `code2idx_3lvl.json` (3-level codes like "1/4/06")
- Head: replace `model.score` with an MLP head (hidden=512)
- PEFT: LoRA (r=32, alpha=128, dropout=0.1)
- Loss: BCEWithLogitsLoss with `pos_weight = (neg/pos).clamp(max=10)`
- Tokenization: padding="max_length", max_length=1024, pad_token=eos

The repository does not distribute clinical data.

Example:
  python scripts/train.py --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from aiscode_sllm.config import parse_args_with_config
from aiscode_sllm.labels import load_code2idx, numeric7_to_3lvl, parse_codes_from_completion


class WeightedBCETrainer(Trainer):
    """Trainer with BCEWithLogitsLoss(pos_weight=...)."""

    def __init__(self, pos_weight: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.args.device))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.criterion(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def make_onehot_3lvl(codes_str: str, code2idx: Dict[str, int], num_labels: int) -> List[float]:
    """Convert a completion string with numeric AIS codes -> multi-hot over 3-level labels."""
    vec = np.zeros(num_labels, dtype=np.float32)
    for token in parse_codes_from_completion(codes_str):
        code3 = numeric7_to_3lvl(token)  # 7-digit -> a/b/cd
        idx = code2idx.get(code3)
        if idx is not None:
            vec[int(idx)] = 1.0
    return vec.tolist()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Qwen3-8B + LoRA AIS multilabel classifier")

    # Config
    p.add_argument("--config", type=str, default=None, help="YAML config file (recommended)")

    # Data
    p.add_argument("--train-file", type=str, default="train_full_preproc.jsonl")
    p.add_argument("--val-file", type=str, default="ais_valid_preproc.jsonl")
    p.add_argument("--code2idx", type=str, default="code2idx_3lvl.json")

    # Model
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--mlp-hidden", type=int, default=512)

    # LoRA
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list",
    )

    # Training
    p.add_argument("--output-dir", type=str, default="qwen3_8b_ais_lora_onehot_4digit_mlphead_a128_epoch10")
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--grad-accum", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)

    # Scheduler/regularization
    p.add_argument("--lr-scheduler-type", type=str, default="cosine")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.01)

    # Logging/Eval/Save
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.3, help="Metric threshold used during eval")

    # Precision
    p.add_argument("--bf16", action="store_true", default=True)

    return p


def main() -> None:
    parser = build_parser()
    args, _ = parse_args_with_config(parser)

    set_seed(int(args.seed))

    code2idx = load_code2idx(args.code2idx)
    num_labels = len(code2idx)

    # Build label strings for HF config
    labels = [None] * num_labels
    for c, i in code2idx.items():
        labels[int(i)] = c

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    def tok_fn_cls(ex):
        text = ex.get("prompt", ex.get("text"))
        codes = ex.get("completion", ex.get("code", ""))
        tok = tokenizer(
            text,
            truncation=True,
            max_length=int(args.max_len),
            padding="max_length",
        )
        tok["labels"] = make_onehot_3lvl(str(codes), code2idx, num_labels)
        return tok

    # Load datasets
    train_raw = load_dataset("json", data_files=str(args.train_file), split="train")
    val_raw = load_dataset("json", data_files=str(args.val_file), split="train")

    train_ds = load_dataset("json", data_files=str(args.train_file), split="train").map(
        tok_fn_cls,
        remove_columns=train_raw.column_names,
        num_proc=4,
    )
    val_ds = load_dataset("json", data_files=str(args.val_file), split="train").map(
        tok_fn_cls,
        remove_columns=val_raw.column_names,
        num_proc=4,
    )

    # Build model config
    hf_cfg = AutoConfig.from_pretrained(
        args.base_model,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label={i: lab for i, lab in enumerate(labels)},
        label2id={lab: i for i, lab in enumerate(labels)},
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        config=hf_cfg,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Replace head with MLP (matches original)
    hidden_mid = int(args.mlp_hidden)
    base_model.score = nn.Sequential(
        nn.Linear(hf_cfg.hidden_size, hidden_mid, bias=False),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_mid, hf_cfg.num_labels, bias=False),
    )

    # LoRA
    target_modules = [m.strip() for m in str(args.lora_target_modules).split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        target_modules=target_modules,
        lora_dropout=float(args.lora_dropout),
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base_model, lora_cfg)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.print_trainable_parameters()

    # Compute pos_weight from training labels (neg/pos, clamp max=10)
    label_arr = np.stack(train_ds["labels"])  # [N, C]
    pos_cnt = label_arr.sum(0)
    neg_cnt = len(label_arr) - pos_cnt
    # avoid division by zero; if pos_cnt==0 -> weight becomes 1 (won't matter because that label never appears)
    safe_pos = np.where(pos_cnt == 0, 1.0, pos_cnt)
    pos_w = torch.tensor((neg_cnt / safe_pos), dtype=torch.float32).clamp(max=10)

    # Training arguments (mirrors original defaults)
    output_dir = str(args.output_dir)
    train_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=int(args.num_epochs),
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        learning_rate=float(args.lr),
        lr_scheduler_type=str(args.lr_scheduler_type),
        warmup_steps=int(args.warmup_steps),
        weight_decay=float(args.weight_decay),
        logging_steps=int(args.logging_steps),
        logging_dir=os.path.join(output_dir, "tb_logs"),
        eval_strategy="steps",
        eval_steps=int(args.eval_steps),
        save_strategy="steps",
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        bf16=bool(args.bf16),
        remove_unused_columns=False,
        report_to="tensorboard",
    )

    def compute_metrics(pred):
        logits, labels_true = pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= float(args.threshold)).astype(int)
        return {
            "f1": f1_score(labels_true, preds, average="macro", zero_division=0),
            "precision": precision_score(labels_true, preds, average="macro", zero_division=0),
            "recall": recall_score(labels_true, preds, average="macro", zero_division=0),
            "accuracy": accuracy_score(labels_true, preds),
        }

    data_collator = DataCollatorWithPadding(tokenizer, padding=False, pad_to_multiple_of=8)

    trainer = WeightedBCETrainer(
        pos_weight=pos_w,
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print(f"Num labels: {num_labels}")
    trainer.train()

    # Save adapter + tokenizer (matches original)
    adapter_dir = os.path.join(output_dir, "cls_lora_adapter")
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved adapter to: {adapter_dir}")
    print(f"Saved tokenizer/config to: {output_dir}")


if __name__ == "__main__":
    main()
