from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: any
    device: torch.device
    idx2label: Dict[int, str]


def load_qwen3_multilabel_with_mlp_head(
    base_model: str,
    model_dir: str | Path,
    adapter_dir: str | Path,
    code2idx: Dict[str, int],
    device: str = "cuda",
    max_length: int = 1024,
    torch_dtype: torch.dtype = torch.bfloat16,
    mlp_hidden: int = 512,
    dropout: float = 0.1,
) -> LoadedModel:
    """Load Qwen3-8B sequence classification model with custom MLP head + LoRA adapter."""
    num_labels = len(code2idx)
    labels: List[str] = [None] * num_labels
    for code, idx in code2idx.items():
        labels[int(idx)] = code
    idx2label = {i: lab for i, lab in enumerate(labels)}
    label2id = {lab: i for i, lab in idx2label.items()}

    config = AutoConfig.from_pretrained(
        base_model,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label=idx2label,
        label2id=label2id,
    )
    device_t = torch.device(device)
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch_dtype,
    ).to(device_t)

    # Replace classification head with an MLP (matches your notebooks)
    base.score = nn.Sequential(
        nn.Linear(config.hidden_size, mlp_hidden, bias=False),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_hidden, num_labels, bias=False),
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base,
        str(adapter_dir),
        torch_dtype=torch_dtype,
    ).to(device_t)
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer, device=device_t, idx2label=idx2label)
