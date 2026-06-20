# Cognitive Shorts — Short-Video Like Prediction Pipeline

An end-to-end machine learning system that predicts whether a user will **like** a short video. The project covers the full ML lifecycle: data engineering, user embeddings, multi-model training with hyperparameter tuning, experiment tracking, a REST inference API, and an interactive web UI.

> **Repository:** `ml-recommendation-pipeline`
> **Stack:** Python · pandas · scikit-learn · LightGBM · TruncatedSVD · Pandera · Pydantic · FastAPI · Streamlit · Plotly · MLflow

---

## Overview

Given three raw tables — user–video interactions (500K rows), user profiles (25K), and video metadata (35K) — the system engineers 10 features per interaction and trains four classifiers to predict the binary `liked` label.

The dataset is **severely imbalanced** (only ~12% positive), which makes naive accuracy misleading. A large part of this project is about *correctly diagnosing and handling that imbalance*, and about recognizing when the real bottleneck is the data rather than the model.

Key capabilities:

- **Feature engineering** with time features, activity/popularity signals, and **TruncatedSVD user embeddings** derived from a sparse user–video interaction matrix.
- **Dual-layer validation**: Pandera validates the feature DataFrame; Pydantic validates the API request/response.
- **Four-model comparison** (Logistic Regression, Random Forest, Gradient Boosting, LightGBM) with **GridSearchCV** tuning and **MLflow** tracking.
- **Imbalance-aware selection**: models selected by **F1**, with Precision/Recall/F1/AUC all logged.
- **FastAPI inference service** that derives the 10 model features online from 4 request fields, with cold-start fallback.
- **Streamlit UI** with live prediction, a probability gauge, and an API health/uptime sidebar.

---

## Architecture

```
                 ┌──────────────┐
  raw CSVs  ───▶ │ process_data │ ──▶ processed_interactions.csv
 (3 tables)      │   + SVD      │ ──▶ feature_store.pkl
                 └──────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │   train.py   │  GridSearchCV · class_weight · MLflow
                 │ 4 models →F1 │ ──▶ best_model.pkl + model_metadata.pkl
                 └──────────────┘
                        │
                        ▼
                 ┌──────────────┐        ┌──────────────┐
   HTTP POST ──▶ │   api.py     │ ◀────▶ │   app.py     │ ◀── user
  {4 fields}     │ FastAPI      │  JSON  │  Streamlit   │
                 │ 4 → 10 feats │        │  gauge + UI  │
                 └──────────────┘        └──────────────┘
                 {prediction, probability, confidence}
```

**Online feature derivation (4 → 10).** The frontend sends only `user_id`, `video_id`, `watch_time_seconds`, and `hour_of_day`. The API derives the 10 model features at request time: `hour` is taken directly; `day_of_week` / `is_weekend` come from the server clock; `user_activity`, `video_popularity`, and the 5 SVD embedding dimensions are looked up from the precomputed feature store, with a zero fallback for unseen (cold-start) users.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & features | pandas, NumPy, SciPy (sparse), scikit-learn `TruncatedSVD` |
| Validation | Pandera (DataFrame schema), Pydantic (request/response) |
| Modeling | scikit-learn (LogReg, RF, GBDT), LightGBM, `GridSearchCV` |
| Tracking | MLflow (params, metrics, model artifacts) |
| Serving | FastAPI, Uvicorn |
| Frontend | Streamlit, Plotly |
| Persistence | joblib |

---

## Project Structure

```
cognitive_shorts/
├── code/
│   ├── src/
│   │   ├── process_data.py     # load, join, feature engineering, SVD embeddings
│   │   ├── schema.py           # Pandera + Pydantic schemas
│   │   ├── train.py            # GridSearch tuning, MLflow, model selection
│   │   ├── api.py              # FastAPI inference service
│   │   └── app.py              # Streamlit frontend
│   ├── data/                   # raw + processed CSVs (gitignored)
│   └── models/                 # best_model.pkl, metadata, feature_store (gitignored)
├── docs/
│   └── model_evaluation_analysis.md
└── README.md
```

---

## Quick Start

```bash
# 1. Environment
cd code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Build features (reads data/, writes processed CSV + feature_store.pkl)
python -m src.process_data

# 3. Train models (GridSearch + MLflow); saves best_model.pkl + metadata
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
python -m src.train

# 4. Start the API (terminal 1)
python -m src.api          # http://127.0.0.1:8000

# 5. Start the UI (terminal 2)
streamlit run src/app.py   # http://localhost:8501
```

Optional — inspect experiments:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Model Evaluation & Analysis

This is the core of the project. All experiments use the full 500K-row dataset, an 80/20 split, and `random_state=42`.

### Round 1 — Baseline: fixed params, no class weighting, accuracy only

| Model | Accuracy |
|---|---|
| Logistic Regression | 0.8803 |
| Random Forest | 0.8803 |
| Gradient Boosting | 0.8803 |
| LightGBM | 0.8803 |

All four models collapse to the **exact same accuracy (0.8803)** — the majority-class baseline. Each model "gives up" and predicts the majority class (no-like) almost every time, so Recall ≈ 0. **On imbalanced data, accuracy has no discriminative power**: four fundamentally different models look identical, and the "best" is decided by meaningless floating-point noise.

### Round 2 — `class_weight='balanced'` + GridSearchCV + multi-metric, selected by F1

| Model | F1 | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.1937** | 0.4966 | 0.4699 | 0.1184 | 0.5320 |
| Random Forest | 0.1866 | 0.4982 | 0.5655 | 0.1202 | 0.4163 |
| Gradient Boosting | 0.0002 | 0.4959 | 0.8803 | 0.2500 | 0.0001 |
| LightGBM | 0.1930 | 0.5005 | 0.5058 | 0.1199 | 0.4936 |

Adding `class_weight='balanced'` forces the models to stop ignoring the minority class — Recall jumps from ~0 to 0.40–0.53. Note **Gradient Boosting** (which does not support `class_weight` in scikit-learn) stays collapsed: accuracy 0.8803, Recall 0.0001.

**Metric choice changes the winner.** Selecting by accuracy would crown Gradient Boosting (0.8803) — a model that catches essentially zero likers. Selecting by F1 picks a model that actually works:

| Selection metric | Winner | Winner's Recall | Verdict |
|---|---|---|---|
| Accuracy | Gradient Boosting (0.8803) | 0.0001 | High score, useless model |
| **F1** | Logistic Regression (0.1937) | 0.5320 | Catches ~53% of real likers |

### Round 3 — Expanded hyperparameter grid (bottleneck verification)

Several Round-2 optima sat on grid boundaries (e.g. RF `max_depth=8` was the lower bound). Best practice is to expand the search to check whether a better optimum lies outside the grid. The boundary parameters were widened (e.g. RF `max_depth` → `[4,6,8,12,20]`, LightGBM `num_leaves` → `[15,31,63]`, etc.).

| Model | F1 | AUC | Accuracy | Recall | Best params |
|---|---|---|---|---|---|
| Logistic Regression | 0.1936 | 0.4966 | 0.4701 | 0.5314 | C=0.01 (lower bound) |
| Random Forest | 0.1866 | 0.4982 | 0.5655 | 0.4163 | max_depth=8, n_est=100 |
| Gradient Boosting | 0.0008 | 0.4967 | 0.8801 | 0.0004 | lr=0.1, depth=7, n_est=200 |
| **LightGBM** | **0.1941** | 0.4999 | 0.4830 | 0.5203 | lr=0.1, n_est=100, num_leaves=15 (lower bound) |

**Findings:**

1. **Performance barely moves.** Best F1 went from 0.1937 → 0.1941 (+0.0004); all AUCs stayed ≈ 0.50. Expanding the grid and tuning harder does almost nothing here — **the bottleneck is feature signal, not hyperparameters.**
2. **Optima drift toward the simplest configurations** (stronger regularization `C=0.01`, fewer leaves `num_leaves=15`). When the signal is weak, added model complexity buys nothing, so the search prefers simpler models.
3. **Gradient Boosting stays collapsed regardless of tuning** — its problem is the missing imbalance mechanism (`class_weight`), not the hyperparameters.
4. **The "winner" flips between Logistic Regression and LightGBM across rounds**, but by < 0.001 F1 — effectively a tie, since all models are capped by the same weak-signal ceiling.

### Takeaway

The honest conclusion is that **all models have AUC ≈ 0.50 — near-random ranking ability — because the provided features carry little signal about whether a specific user will like a specific video.** This is a property of the dataset, consistent with the project's goal of exercising the *engineering pipeline* rather than maximizing model performance. The deliverable value here is the *diagnosis*: identifying imbalance, choosing metrics that expose it, and proving the bottleneck is the data rather than the tuning.

The final saved model is **LightGBM (F1 = 0.1941)**.

---

## Design Notes

**`watch_time_seconds` is collected and validated but not used by the model.** The project spec asks the frontend to collect `watch_time_seconds` (single-session watch time, from the interactions table), but its training feature set does not include it. To stay faithful to the spec's 10-feature definition, the field is validated by Pydantic (`>= 0`) but not fed to the model. Note this is distinct from `total_watch_time_minutes` (a user's *historical cumulative* watch time, from the users table), which *does* enter the model via `user_activity`. Folding single-session watch time into the features is a natural extension (see Future Work), with a caveat about potential label leakage relative to "will like."

**Cold start.** Unseen `user_id` / `video_id` values fall back to zero activity/popularity and a zero embedding vector, so the service degrades gracefully instead of erroring.

**Why a separate `feature_store.pkl`.** User activity, video popularity, and SVD embeddings are computed offline once and cached, so the API can derive features with O(1) dictionary lookups at request time instead of recomputing.

---

## Future Work

- Add `sample_weight` to Gradient Boosting to approximate `class_weight` and remove its collapse.
- Engineer stronger features to break the AUC ≈ 0.50 ceiling (the real bottleneck).
- Add single-session `watch_time_seconds` as a feature (with leakage analysis).
- Study AUC vs F1 trade-offs and optionally switch the selection metric.
- Build out the Batch Prediction, Analytics Dashboard, and Model Info pages.
- Add a confusion-matrix view and threshold tuning to the UI.
