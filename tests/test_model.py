"""Tests for Tree-Based Learning models."""
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_battery_model_exists():
    assert (ROOT / "models" / "rf_battery.pkl").exists(), "Battery model not found. Run src/train.py first."


def test_wafer_model_exists():
    assert (ROOT / "models" / "rf_wafer.pkl").exists(), "Wafer model not found. Run src/train.py first."


def test_battery_prediction():
    from predict import predict
    model_path = ROOT / "models" / "rf_battery.pkl"
    if not model_path.exists():
        return
    df = pd.read_csv(ROOT / "data" / "ev_battery_thermal_runaway.csv")
    features = df.drop(columns=["thermal_runaway"]).head(3)
    preds = predict(features, str(model_path))
    assert len(preds) == 3


def test_wafer_prediction():
    from predict import predict
    model_path = ROOT / "models" / "rf_wafer.pkl"
    if not model_path.exists():
        return
    df = pd.read_csv(ROOT / "data" / "wafer_edge_yield.csv")
    features = df.drop(columns=["yield_pass"]).head(3)
    preds = predict(features, str(model_path))
    assert len(preds) == 3


if __name__ == "__main__":
    test_battery_model_exists()
    test_wafer_model_exists()
    test_battery_prediction()
    test_wafer_prediction()
    print("All tests passed.")
