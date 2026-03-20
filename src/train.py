"""
Train pipeline for Tree-Based Learning.
Trains RandomForest models for both datasets.
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

DATASETS = {
    "battery": {
        "file": "ev_battery_thermal_runaway.csv",
        "target": "thermal_runaway",
        "model": "rf_battery.pkl",
    },
    "wafer": {
        "file": "wafer_edge_yield.csv",
        "target": "yield_pass",
        "model": "rf_wafer.pkl",
    },
}


def train(name: str):
    """Train model pipeline for a single dataset and save to models/."""
    cfg = DATASETS[name]
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    df = pd.read_csv(DATA_DIR / cfg["file"])
    target_col = cfg["target"]
    X = df.select_dtypes(include="number").drop(columns=[target_col], errors="ignore")
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    model_path = MODEL_DIR / cfg["model"]
    joblib.dump(pipe, model_path)
    print(f"Model saved → {model_path}")
    return pipe


if __name__ == "__main__":
    for ds_name in DATASETS:
        train(ds_name)
