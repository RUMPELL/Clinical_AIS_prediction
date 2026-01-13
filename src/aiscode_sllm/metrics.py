from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    jaccard_score,
    hamming_loss,
)

@dataclass
class MultiLabelMetrics:
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    subset_accuracy: float
    jaccard_samples: float
    hamming_loss: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "micro_precision": self.micro_precision,
            "micro_recall": self.micro_recall,
            "micro_f1": self.micro_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "subset_accuracy": self.subset_accuracy,
            "jaccard_samples": self.jaccard_samples,
            "hamming_loss": self.hamming_loss,
            "hamming_score": 1.0 - self.hamming_loss,
        }

def binarize_sets(
    y_true: Sequence[Set[int]],
    y_pred: Sequence[Set[int]],
    num_labels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    yt = np.zeros((len(y_true), num_labels), dtype=int)
    yp = np.zeros((len(y_pred), num_labels), dtype=int)
    for i,(t,p) in enumerate(zip(y_true, y_pred)):
        for j in t:
            if 0 <= j < num_labels:
                yt[i,j]=1
        for j in p:
            if 0 <= j < num_labels:
                yp[i,j]=1
    return yt, yp

def compute_multilabel_metrics(
    y_true: Sequence[Set[int]],
    y_pred: Sequence[Set[int]],
    num_labels: int,
) -> MultiLabelMetrics:
    yt, yp = binarize_sets(y_true, y_pred, num_labels)
    micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
        yt, yp, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        yt, yp, average="macro", zero_division=0
    )
    subset_acc = float(np.mean(np.all(yt == yp, axis=1)))
    jac = float(jaccard_score(yt, yp, average="samples", zero_division=0))
    hl = float(hamming_loss(yt, yp))
    return MultiLabelMetrics(
        micro_precision=float(micro_p),
        micro_recall=float(micro_r),
        micro_f1=float(micro_f),
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f),
        subset_accuracy=subset_acc,
        jaccard_samples=jac,
        hamming_loss=hl,
    )

def prefix_set(code: str, prefix_len: int) -> str:
    # codes are expected to be normalized already (e.g. "1203")
    return str(code)[:prefix_len]

def compute_prefix_hits(
    gt_codes: Sequence[Sequence[str]],
    pred_codes: Sequence[Sequence[str]],
    prefix_lens: Sequence[int] = (1,2,4),
) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for L in prefix_lens:
        hits = []
        exacts = []
        inclusions = []
        for gt, pr in zip(gt_codes, pred_codes):
            gt_set = {prefix_set(c, L) for c in gt}
            pr_set = {prefix_set(c, L) for c in pr}
            hits.append(len(gt_set & pr_set) > 0)
            exacts.append(gt_set == pr_set)
            inclusions.append(gt_set.issubset(pr_set))
        out[L] = {
            "sample_acc": float(np.mean(hits)),
            "exact_acc": float(np.mean(exacts)),
            "inclusion_acc": float(np.mean(inclusions)),
        }
    return out


def multilabel_metrics_from_logits(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    threshold: float = 0.5,
) -> MultiLabelMetrics:
    """Compute multi-label metrics from raw logits and 0/1 label matrix.

    logits: (N, C)
    labels: (N, C) float/bool
    """
    import torch

    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_pred = (probs >= threshold).astype(int)
    y_true = labels.detach().cpu().numpy().astype(int)

    # subset accuracy
    subset_acc = float((y_pred == y_true).all(axis=1).mean()) if len(y_true) else 0.0

    micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    jacc = jaccard_score(y_true, y_pred, average="samples", zero_division=0)
    hl = hamming_loss(y_true, y_pred)

    return MultiLabelMetrics(
        micro_precision=float(micro_p),
        micro_recall=float(micro_r),
        micro_f1=float(micro_f),
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f),
        subset_accuracy=float(subset_acc),
        jaccard_samples=float(jacc),
        hamming_loss=float(hl),
    )
