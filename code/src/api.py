from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import pandera.pandas as pa
from fastapi import FastAPI, HTTPException

from .schema import InputSchema, PredictRequest, PredictResponse

app = FastAPI(title="Cognitive Shorts Recommendation API")

BASE_DIR = Path(__file__).parents[1]
MODEL_DIR = BASE_DIR / "models"

# Global cache: loaded once at startup, not on every request
best_model = None
model_metadata = None
feature_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, metadata, and feature store once at startup."""
    global best_model, model_metadata, feature_store
    try:
        best_model = joblib.load(MODEL_DIR / "best_model.pkl")
        model_metadata = joblib.load(MODEL_DIR / "model_metadata.pkl")
        feature_store = joblib.load(MODEL_DIR / "feature_store.pkl")
        print(f"Loaded model: {model_metadata['model_name']}")
    except Exception as e:
        print(f"Failed to load artifacts: {e}")
    yield
    # (Nothing to clean up on shutdown)


app = FastAPI(title="Cognitive Shorts Recommendation API", lifespan=lifespan)

@app.get("/")
def read_root():
    name = model_metadata["model_name"] if model_metadata else "None"
    return {"model_name": name}


@app.get("/health")
def health_check():
    ok = best_model is not None and feature_store is not None
    return {"status": "healthy" if ok else "degraded"}

def build_features(req: PredictRequest) -> pd.DataFrame:
    """Derive the 10 model features from the 4 request fields (project.md 7.2)."""
    # Time features
    hour = req.hour_of_day
    dow = datetime.now().weekday()
    is_weekend = 1 if dow >= 5 else 0

    # Lookup features (with cold-start fallback to 0, project.md 7.3)
    user_activity = feature_store["user_activity"].get(req.user_id, 0.0)
    video_popularity = feature_store["video_popularity"].get(req.video_id, 0.0)

    svd_dim = feature_store["svd_dim"]
    emb = feature_store["user_emb"].get(req.user_id, [0.0] * svd_dim)

    # Assemble one row
    row = {
        "hour": hour,
        "day_of_week": dow,
        "is_weekend": is_weekend,
        "user_activity": float(user_activity),
        "video_popularity": float(video_popularity),
    }
    for k in range(svd_dim):
        row[f"user_emb_{k}"] = float(emb[k])

    # Build DataFrame in the exact training column order
    df = pd.DataFrame([row])[feature_store["feature_columns"]]
    return df



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
