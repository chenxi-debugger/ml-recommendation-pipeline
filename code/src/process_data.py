import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# ---- Paths: locate via __file__ so data/ is found no matter where we run from ----
BASE_DIR = Path(__file__).parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ---- Final feature column order (must stay fixed; shared by training & inference) ----
FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "user_activity",
    "video_popularity",
    "user_emb_0",
    "user_emb_1",
    "user_emb_2",
    "user_emb_3",
    "user_emb_4",
]
LABEL_COLUMN = "liked"
SVD_DIM = 5  # embedding dimension

def load_raw_data():
    """Step 1: Read the 3 raw CSV files.

    Use `usecols` to load only the necessary columns to reduce memory
    (project.md 4.2). interactions.csv has 500k rows; reading all columns
    would be slow and memory-heavy.
    """
    print("Loading raw CSV files ...")

    interactions = pd.read_csv(
        DATA_DIR / "interactions.csv",
        usecols=["user_id", "video_id", "timestamp", "liked", "watch_time_seconds"],
    )
    users = pd.read_csv(
        DATA_DIR / "users.csv",
        usecols=["user_id", "subscriber_count", "total_watch_time_minutes"],
    )
    videos = pd.read_csv(
        DATA_DIR / "videos.csv",
        usecols=["video_id", "view_count", "like_count"],
    )

    print(f"      interactions: {interactions.shape}")
    print(f"      users:        {users.shape}")
    print(f"      videos:       {videos.shape}")
    return interactions, users, videos

def join_tables(interactions, users, videos):
    """Step 2: Join the three tables (project.md 4.2).

    Main table is `interactions`; left-join `users` (on user_id) and
    `videos` (on video_id). Left join keeps every interaction row even
    if the user/video info is missing (those columns become NaN).
    """
    print("Joining tables ...")
    df = interactions.merge(users, on="user_id", how="left")
    df = df.merge(videos, on="video_id", how="left")
    return df

def add_time_features(df):
    """Step 3: Extract time features from timestamp (project.md 4.3).

    hour (0-23), day_of_week (Mon=0 ... Sun=6), is_weekend (Sat/Sun -> 1).
    """
    print("Building time features ...")
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df

def add_activity_popularity(df):
    """Step 4: User activity and video popularity (project.md 4.4 / 4.5).

    user_activity    = subscriber_count + total_watch_time_minutes
    video_popularity = view_count + like_count
    Missing values (from unmatched left-join rows) are filled with 0.
    """
    print("Building activity / popularity ...")
    df["subscriber_count"] = df["subscriber_count"].fillna(0)
    df["total_watch_time_minutes"] = df["total_watch_time_minutes"].fillna(0)
    df["view_count"] = df["view_count"].fillna(0)
    df["like_count"] = df["like_count"].fillna(0)

    df["user_activity"] = df["subscriber_count"] + df["total_watch_time_minutes"]
    df["video_popularity"] = df["view_count"] + df["like_count"]
    return df

def build_user_embeddings(df):
    """Step 5: Generate 5-dim user embeddings via TruncatedSVD (project.md 4.6).

    1. Encode user_id / video_id as integer row/col indices (category codes).
    2. Build a sparse user-video matrix M, shape = [n_users, n_videos],
       with 1 wherever an interaction exists.
    3. TruncatedSVD compresses M into a dense [n_users, 5] matrix:
       each user gets a 5-dim vector.
    4. Merge each user's vector back onto the main table by user_id.

    Returns:
        df            -- main table with user_emb_0..4 attached
        user_emb_map  -- {user_id: [5-dim vector]} for online inference
    """
    print("Building sparse matrix + TruncatedSVD ...")

    # --- Block 1: encode string ids into integer indices ---
    user_cat = df["user_id"].astype("category")
    video_cat = df["video_id"].astype("category")

    user_codes = user_cat.cat.codes.to_numpy()
    video_codes = video_cat.cat.codes.to_numpy()

    n_users = len(user_cat.cat.categories)
    n_videos = len(video_cat.cat.categories)
    print(f"      matrix size: {n_users} users x {n_videos} videos")

    # --- Block 2: build the sparse user-video matrix ---
    data = np.ones(len(df), dtype=np.float32)
    M = csr_matrix((data, (user_codes, video_codes)), shape=(n_users, n_videos))

    # --- Block 3: TruncatedSVD -> 5-dim user embeddings ---
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=42)
    user_embeddings = svd.fit_transform(M)

    # --- Block 4: attach embeddings back to df, and build lookup map ---
    emb_cols = [f"user_emb_{k}" for k in range(SVD_DIM)]
    emb_df = pd.DataFrame(user_embeddings, columns=emb_cols)
    emb_df["user_id"] = user_cat.cat.categories

    df = df.merge(emb_df, on="user_id", how="left")

    user_emb_map = {
        uid: vec.tolist()
        for uid, vec in zip(user_cat.cat.categories, user_embeddings)
    }
    return df, user_emb_map

def finalize_and_save(df, user_emb_map):
    """Step 6+7: Select final columns, convert label, save outputs.

    Writes:
      - data/processed_interactions.csv  (training feature table)
      - models/feature_store.pkl         (lookup dicts for online inference)
    """

    print("Finalizing feature table ...")

    # Label: bool -> 0/1
    df[LABEL_COLUMN] = df["liked"].astype(int)

    # Keep final 10 features + 1 label, fill any leftover NaN with 0
    output = df[FEATURE_COLUMNS + [LABEL_COLUMN]].copy()
    output = output.fillna(0)

    out_path = DATA_DIR / 'processed_interactions.csv'
    output.to_csv(out_path, index=False)
    print(f"      wrote {out_path}  shape={output.shape}")

    # ---- Build online feature store (used by api.py) ----
    print("Building feature_store.pkl ...")
    users = pd.read_csv(
        DATA_DIR / "users.csv",
        usecols=["user_id", "subscriber_count", "total_watch_time_minutes"],
    )
    users["user_activity"] = (
        users["subscriber_count"].fillna(0) + users["total_watch_time_minutes"].fillna(0)
    )
    user_activity_map = dict(zip(users["user_id"], users["user_activity"]))

    videos = pd.read_csv(
        DATA_DIR / "videos.csv",
        usecols=["video_id", "view_count", "like_count"],
    )
    videos["video_popularity"] = videos["view_count"].fillna(0) + videos["like_count"].fillna(0)
    video_popularity_map = dict(zip(videos["video_id"], videos["video_popularity"]))

    feature_store = {
        'user_activity': user_activity_map,
        'video_popularity': video_popularity_map,
        'user_emb': user_emb_map,
        'feature_columns': FEATURE_COLUMNS,
        'svd_dim': SVD_DIM,
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(feature_store, MODEL_DIR/ 'feature_store.pkl')
    print(f"      wrote {MODEL_DIR / 'feature_store.pkl'}")


def main():
    interactions, users, videos = load_raw_data()
    df = join_tables(interactions, users, videos)
    df = add_time_features(df)
    df = add_activity_popularity(df)
    df, user_emb_map = build_user_embeddings(df)
    finalize_and_save(df, user_emb_map)
    print("\n✅ Data processing complete")


if __name__ == "__main__":
    main()