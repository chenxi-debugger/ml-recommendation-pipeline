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

    # ---- Define the 4 models (project.md 5.4) ----
    model_configs = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        ),
        "Gradient_Boosting": GradientBoostingClassifier(random_state=42),
        "LightGBM": LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
    }
    
    # ---- Set MLflow experiment ----
    mlflow.set_experiment('cognitive_shorts_recommendation')

    best_model_name = None
    best_accuracy = 0.0
    best_model_object = None

    # ---- Train each model, track with MLflow, keep the best ----
    for name, model in model_configs.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name} ...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"  {name} accuracy: {acc:.4f}")

            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, name)

            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = name
                best_model_object = model

    print(f"\nBest model: {best_model_name} (accuracy={best_accuracy:.4f})")





if __name__ == "__main__":
    train_and_select_best_model()