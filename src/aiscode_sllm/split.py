from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def multilabel_stratified_split(
    codes: Sequence[Sequence[str]],
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return train_idx, val_idx that preserve multilabel distribution."""
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(codes)
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=random_state
    )
    idx = np.arange(len(codes))
    train_idx, val_idx = next(splitter.split(idx, Y))
    return train_idx, val_idx
