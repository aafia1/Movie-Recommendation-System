from __future__ import annotations
import os, zipfile, io, urllib.request
import pandas as pd
import numpy as np
from typing import Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
ML100K_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"

def download_ml100k(dest_dir: str = DATA_DIR) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "ml-100k.zip")
    out_dir = os.path.join(dest_dir, "ml-100k")
    if os.path.exists(os.path.join(out_dir, "u.data")):
        return out_dir
    print("Downloading MovieLens 100K...")
    urllib.request.urlretrieve(ML100K_URL, zip_path)
    print("Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)
    os.remove(zip_path)
    print(f"Data ready at {out_dir}")
    return out_dir

def load_ratings(data_dir: str = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = os.path.join(DATA_DIR, "ml-100k")
    ratings_path = os.path.join(data_dir, "u.data")
    df = pd.read_csv(ratings_path, sep='\t', names=['user_id','item_id','rating','timestamp'], engine='python')
    return df

def load_items(data_dir: str = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = os.path.join(DATA_DIR, "ml-100k")
    # u.item is pipe-separated; item_id | title | release_date | video_release_date | IMDb URL | ...
    path = os.path.join(data_dir, "u.item")
    cols = ['item_id','title','release_date','video_release_date','imdb_url']
    df = pd.read_csv(path, sep='|', names=cols + [f'col{i}' for i in range(19)], encoding='latin-1')
    df = df[cols]
    return df

def build_user_item_matrix(ratings: pd.DataFrame) -> Tuple[np.ndarray, dict, dict]:
    users = np.sort(ratings['user_id'].unique())
    items = np.sort(ratings['item_id'].unique())
    user_to_idx = {u:i for i,u in enumerate(users)}
    item_to_idx = {m:i for i,m in enumerate(items)}
    mat = np.zeros((len(users), len(items)), dtype=np.float32)
    for _, row in ratings.iterrows():
        mat[user_to_idx[row.user_id], item_to_idx[row.item_id]] = row.rating
    return mat, user_to_idx, item_to_idx

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="Download MovieLens 100K to data/")
    args = ap.parse_args()
    if args.download:
        download_ml100k()
        print("Done.")
