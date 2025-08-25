from __future__ import annotations
import argparse, numpy as np, pandas as pd, os
from .data_utils import load_ratings, load_items, build_user_item_matrix, download_ml100k
from .recommend_user_based import recommend_user_based
from .recommend_item_based import recommend_item_based
from .recommend_svd import recommend_svd

def main():
    ap = argparse.ArgumentParser(description="Movie Recommender CLI")
    ap.add_argument("--algo", choices=["user","item","svd"], default="user", help="Algorithm")
    ap.add_argument("--user_id", type=int, required=True, help="MovieLens user id (e.g., 196)")
    ap.add_argument("--top_n", type=int, default=10, help="Number of recommendations to show")
    ap.add_argument("--download", action="store_true", help="If set, download the dataset first")
    args = ap.parse_args()

    if args.download:
        download_ml100k()

    ratings = load_ratings()
    items = load_items()
    X, user_to_idx, item_to_idx = build_user_item_matrix(ratings)

    if args.user_id not in user_to_idx:
        raise SystemExit(f"User id {args.user_id} not found in dataset. Try 196, 186, 22, 943, etc.")

    uidx = user_to_idx[args.user_id]

    if args.algo == "user":
        idx, scores = recommend_user_based(X, uidx, top_n=args.top_n)
    elif args.algo == "item":
        idx, scores = recommend_item_based(X, uidx, top_n=args.top_n)
    else:
        idx, scores = recommend_svd(X, uidx, top_n=args.top_n)

    # Map item indices back to movie ids and titles
    inv_item = {v:k for k,v in item_to_idx.items()}
    rec_movie_ids = [int(inv_item[i]) for i in idx]
    rec = items[items['item_id'].isin(rec_movie_ids)].copy()
    # Keep list order
    rec['order'] = rec['item_id'].apply(lambda x: rec_movie_ids.index(int(x)))
    rec = rec.sort_values('order').drop(columns=['order'])

    # Show results
    print(f"Top-{args.top_n} recommendations for user {args.user_id} using {args.algo}-based CF:")
    for i, row in rec.iterrows():
        print(f"- {row['title']} (item_id={row['item_id']})")

if __name__ == "__main__":
    main()
