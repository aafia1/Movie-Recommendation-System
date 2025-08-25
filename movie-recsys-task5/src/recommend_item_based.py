from __future__ import annotations
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .utils import mean_center_rows, top_n_from_scores

def item_based_scores(user_item: np.ndarray, target_user_idx: int, k_neighbors: int = 50) -> np.ndarray:
    """Predict scores for all items using item-based CF.
    Steps: mean-center columns (transpose trick), compute item-item similarity, weighted sum.
    """
    X = user_item
    # Center items by their means -> center rows of X.T
    Xc_items, item_means = mean_center_rows(X.T, mask_zero=True)  # shape (n_items, n_users)
    sims = cosine_similarity(Xc_items, Xc_items)                  # item-item
    np.fill_diagonal(sims, 0.0)
    # For each item, keep top-k neighbors to speed up
    n_items = X.shape[1]
    topk = min(k_neighbors, n_items-1) if n_items > 1 else 0
    if topk <= 0:
        return np.zeros(n_items)
    neigh_idx = np.argsort(-sims, axis=1)[:, :topk]               # (n_items, k)
    neigh_sim = np.take_along_axis(sims, neigh_idx, axis=1)       # (n_items, k)

    # Build predictions for the target user
    user_ratings = X[target_user_idx]                              # (n_items,)
    # Center the user's rated items using item means
    centered_user = np.where(user_ratings>0, user_ratings - item_means, 0.0)

    # Weighted sum of neighbors for each candidate item j
    num = (neigh_sim * centered_user[neigh_idx]).sum(axis=1)      # (n_items,)
    den = np.abs(neigh_sim).sum(axis=1) + 1e-8
    centered_preds = num / den
    preds = centered_preds + item_means                            # add back item means
    return preds

def recommend_item_based(user_item: np.ndarray, target_user_idx: int, top_n: int = 10, k_neighbors: int = 50) -> np.ndarray:
    preds = item_based_scores(user_item, target_user_idx, k_neighbors=k_neighbors)
    seen_mask = user_item[target_user_idx] > 0
    top_idx = top_n_from_scores(preds, seen_mask, top_n)
    return top_idx, preds[top_idx]
