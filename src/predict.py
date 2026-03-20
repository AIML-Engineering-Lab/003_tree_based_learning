"""
Inference for Tree-Based Learning.
Load trained model and run predictions on new data.
"""
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def predict(data: pd.DataFrame, model_path: str = None) -> list:
    """Load model and predict on input DataFrame."""
    if model_path is None:
        model_path = str(MODEL_DIR / "rf_battery.pkl")

    pipe = joblib.load(model_path)
    X = data.select_dtypes(include="number")
    preds = pipe.predict(X)
    return preds.tolist()


if __name__ == "__main__":
    # Demo: EV Battery predictions
    df_b = pd.read_csv(ROOT / "data" / "ev_battery_thermal_runaway.csv")
    feats_b = df_b.drop(columns=["thermal_runaway"]).head(5)
    preds_b = predict(feats_b, str(MODEL_DIR / "rf_battery.pkl"))
    print(f"Battery predictions: {preds_b}")

    # Demo: Wafer predictions
    df_w = pd.read_csv(ROOT / "data" / "wafer_edge_yield.csv")
    feats_w = df_w.drop(columns=["yield_pass"]).head(5)
    preds_w = predict(feats_w, str(MODEL_DIR / "rf_wafer.pkl"))
    print(f"Wafer predictions:   {preds_w}")
