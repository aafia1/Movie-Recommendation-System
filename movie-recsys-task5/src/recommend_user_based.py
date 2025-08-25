from __future__ import annotations
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .utils import mean_center_rows, top_n_from_scores

def user_based_scores(user_item: np.ndarray, target_user_idx: int, k_neighbors: int = 20) -> np.ndarray:
    """Predict scores for all items for a target user using user-based CF.
    Steps: mean-center rows (users), compute cosine similarities, use top-k neighbors to score.
    """
    Xc, means = mean_center_rows(user_item, mask_zero=True)
    sims = cosine_similarity(Xc, Xc)  # user-user
    np.fill_diagonal(sims, 0.0)
    # Top-k neighbors of target user
    neigh_idx = np.argsort(-sims[target_user_idx])[:k_neighbors]
    neigh_sims = sims[target_user_idx, neigh_idx]  # shape (k,)
    # Weighted sum of neighbors' centered ratings
    centered_preds = (neigh_sims[:, None] * Xc[neigh_idx]).sum(axis=0)
    denom = np.abs(neigh_sims).sum() + 1e-8
    centered_preds = centered_preds / denom
    # Add back target user's mean to get absolute rating predictions
    preds = centered_preds + means[target_user_idx]
    return preds

def recommend_user_based(user_item: np.ndarray, target_user_idx: int, top_n: int = 10, k_neighbors: int = 20) -> np.ndarray:
    preds = user_based_scores(user_item, target_user_idx, k_neighbors=k_neighbors)
    seen_mask = user_item[target_user_idx] > 0
    top_idx = top_n_from_scores(preds, seen_mask, top_n)
    return top_idx, preds[top_idx]
