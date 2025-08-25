from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Dict

def precision_at_k(recommended_ids: Iterable[int], relevant_ids: Iterable[int], k: int) -> float:
    """Compute Precision@K.
    recommended_ids: ordered list of recommended item ids (best first)
    relevant_ids: set/list of truly relevant items in ground truth
    """
    if k <= 0:
        return 0.0
    recs = list(recommended_ids)[:k]
    rel = set(relevant_ids)
    if len(recs) == 0:
        return 0.0
    hits = sum(1 for r in recs if r in rel)
    return hits / float(min(k, len(recs)))

def mean_center_rows(matrix: np.ndarray, mask_zero: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Mean-center each row. Returns (centered_matrix, row_means).
    If mask_zero, zeros (unrated) do not count toward the mean.
    """
    X = matrix.copy().astype(float)
    if mask_zero:
        row_sums = X.sum(axis=1)
        counts = (X != 0).sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            means = np.where(counts > 0, row_sums / counts, 0.0)
        X = X - means[:, None]
        X[X == -means[:, None]] = 0.0  # keep unrated as 0 after centering
        return X, means
    else:
        means = X.mean(axis=1, keepdims=False)
        return X - means[:, None], means

def top_n_from_scores(scores: np.ndarray, seen_mask: np.ndarray, n: int) -> np.ndarray:
    """Return indices of top-n scores where seen_mask==False."""
    scores = scores.copy()
    scores[seen_mask] = -np.inf
    n = min(n, (seen_mask == False).sum())
    if n <= 0:
        return np.array([], dtype=int)
    # argpartition then sort
    idx = np.argpartition(-scores, n-1)[:n]
    idx = idx[np.argsort(-scores[idx])]
    return idx
