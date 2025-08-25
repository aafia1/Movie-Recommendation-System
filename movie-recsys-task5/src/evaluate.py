from __future__ import annotations
import argparse, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .data_utils import load_ratings, build_user_item_matrix, download_ml100k
from .recommend_user_based import recommend_user_based
from .recommend_item_based import recommend_item_based
from .recommend_svd import recommend_svd
from .utils import precision_at_k

def split_user_ratings(df: pd.DataFrame, test_size=0.2, min_ratings=5, seed=42):
    users = df['user_id'].value_counts()
    keep_users = users[users >= min_ratings].index
    df = df[df['user_id'].isin(keep_users)].copy()

    train_list, test_list = [], []
    rng = np.random.RandomState(seed)
    for uid, grp in df.groupby('user_id'):
        train_idx, test_idx = train_test_split(grp.index, test_size=test_size, random_state=rng)
        train_list.append(df.loc[train_idx])
        test_list.append(df.loc[test_idx])
    return pd.concat(train_list), pd.concat(test_list)

def evaluate_algo(algo: str, ratings_train: pd.DataFrame, ratings_test: pd.DataFrame, k: int = 10):
    # Build matrices on train only
    X, user_to_idx, item_to_idx = build_user_item_matrix(ratings_train)

    # Ground truth relevant items: ratings >= 4 in test
    relevant_by_user = (
        ratings_test[ratings_test['rating'] >= 4.0]
        .groupby('user_id')['item_id']
        .apply(set)
        .to_dict()
    )

    pks = []
    for uid in tqdm(relevant_by_user.keys(), desc=f"Evaluating {algo}"):
        if uid not in user_to_idx:
            continue  # user might be filtered out by min_ratings
        uidx = user_to_idx[uid]
        if algo == "user":
            rec_idx, _ = recommend_user_based(X, uidx, top_n=k)
        elif algo == "item":
            rec_idx, _ = recommend_item_based(X, uidx, top_n=k)
        else:
            rec_idx, _ = recommend_svd(X, uidx, top_n=k)

        inv_item = {v:k for k,v in item_to_idx.items()}
        rec_item_ids = [int(inv_item[i]) for i in rec_idx]
        pk = precision_at_k(rec_item_ids, relevant_by_user.get(uid, set()), k)
        pks.append(pk)

    return float(np.mean(pks)) if pks else 0.0

def main():
    ap = argparse.ArgumentParser(description="Evaluate recommenders with Precision@K")
    ap.add_argument("--k", type=int, default=10, help="K for Precision@K and #recs per user")
    ap.add_argument("--test_size", type=float, default=0.2, help="Per-user test split size")
    ap.add_argument("--min_ratings", type=int, default=5, help="Min ratings per user to keep")
    ap.add_argument("--download", action="store_true", help="Download dataset first")
    args = ap.parse_args()

    if args.download:
        download_ml100k()

    ratings = load_ratings()
    train_df, test_df = split_user_ratings(ratings, test_size=args.test_size, min_ratings=args.min_ratings)

    results = {}
    for algo in ["user", "item", "svd"]:
        score = evaluate_algo(algo, train_df, test_df, k=args.k)
        results[algo] = score

    print("Precision@{} results:".format(args.k))
    for algo, sc in results.items():
        print(f"- {algo:>4}: {sc:.4f}")

if __name__ == "__main__":
    main()
