import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (                                         
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from lightgbm import LGBMClassifier

import mlflow
import mlflow.sklearn

from .schema import InputSchema

BASE_DIR = Path(__file__).parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = [
    "hour", "day_of_week", "is_weekend", "user_activity", "video_popularity",
    "user_emb_0", "user_emb_1", "user_emb_2", "user_emb_3", "user_emb_4",
]
LABEL_COLUMN = "liked"

def train_and_select_best_model():
    # ---- Load feature table ----
    df = pd.read_csv(DATA_DIR / "processed_interactions.csv")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    # ---- Validate features with the schema (the "gatekeeper") ----
    InputSchema.validate(X)

    # ---- Train/test split (project.md 5.3) ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # ---- Models + hyperparameter grids (project.md 5.4) ----
    model_configs = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            "params": {
                "C": [0.1, 1.0, 10.0],
            },
        },
        "Random_Forest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [8, 12, 20],
            },
        },
        "Gradient_Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            },
        },
        "LightGBM": {
            "model": LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, class_weight="balanced"),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "num_leaves": [31, 63],
            },
        },
    }
    
    # ---- Trackers for the best model (selected by F1) ----
    best_model_name = None
    best_f1 = 0.0
    best_model_object = None
    best_metrics = None

    mlflow.set_experiment("cognitive_shorts_recommendation")

    # ---- Tune each model with GridSearchCV, evaluate, track with MLflow ----
    for name, config in model_configs.items():
        with mlflow.start_run(run_name=f"tune_{name}"):
            print(f"Training {name} ...")

            grid_search = GridSearchCV(
                estimator=config["model"],
                param_grid=config["params"],
                cv=5,
                scoring="f1",
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)
            best_estimator = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Predictions on the test set
            y_pred = best_estimator.predict(X_test)
            y_proba = best_estimator.predict_proba(X_test)[:, 1]

            # Primary metric (for selection) + extra metrics (for the record)
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba)

            print(f"  Best params: {best_params}")
            print(f"  F1={f1:.4f} | AUC={auc:.4f} | Acc={acc:.4f} | "
                  f"Precision={precision:.4f} | Recall={recall:.4f}")

            # Log to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metrics({
                "f1": f1,
                "auc": auc,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
            })
            mlflow.sklearn.log_model(best_estimator, name)

            # Keep the best by F1
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_model_object = best_estimator
                best_metrics = {
                    "f1": f1, "auc": auc, "accuracy": acc,
                    "precision": precision, "recall": recall,
                }

    print(f"\nBest model: {best_model_name} (F1={best_f1:.4f})")

    # ---- Save the best model + metadata
    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(best_model_object, MODEL_DIR / "best_model.pkl")

    metadata = {
        "model_name": best_model_name,
        "selection_metric": "f1",
        "metrics": best_metrics,
        "feature_columns": FEATURE_COLUMNS,
        "label_meaning": "liked: 1 = like, 0 = no like",
    }
    joblib.dump(metadata, MODEL_DIR / "model_metadata.pkl")
    print(f"Model + metadata saved to {MODEL_DIR}")




if __name__ == "__main__":
    train_and_select_best_model()