from __future__ import annotations
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from .utils import top_n_from_scores

def svd_scores(user_item: np.ndarray, target_user_idx: int, n_components: int = 50, random_state: int = 42) -> np.ndarray:
    """Simple SVD baseline using TruncatedSVD on the user-item matrix.
    We fill missing with zeros (implicit). For a stronger baseline you can try different imputations.
    """
    X = user_item.astype(float)
    n_users, n_items = X.shape
    n_components = min(n_components, min(n_users-1, n_items-1)) if n_users>1 and n_items>1 else 1
    if n_components <= 0:
        return np.zeros(n_items)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    U = svd.fit_transform(X)           # (n_users, k)
    S = svd.singular_values_           # (k,)
    Vt = svd.components_               # (k, n_items)
    X_hat = np.dot(U, Vt)              # reconstructed ratings
    return X_hat[target_user_idx]

def recommend_svd(user_item: np.ndarray, target_user_idx: int, top_n: int = 10, n_components: int = 50) -> np.ndarray:
    preds = svd_scores(user_item, target_user_idx, n_components=n_components)
    seen_mask = user_item[target_user_idx] > 0
    top_idx = top_n_from_scores(preds, seen_mask, top_n)
    return top_idx, preds[top_idx]
