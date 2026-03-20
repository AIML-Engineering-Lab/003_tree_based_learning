"""
FastAPI serving endpoint for Tree-Based Learning.
POST features -> model prediction.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Tree-Based Learning API", version="1.0.0")

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
_model = None


class PredictionInput(BaseModel):
    features: dict[str, float]


class PredictionResponse(BaseModel):
    prediction: float
    model: str = "RandomForest"


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_DIR / "rf_battery.pkl")
    return _model


@app.get("/health")
def health():
    return {"status": "healthy", "model": "RandomForest"}


@app.get("/info")
def info_endpoint():
    return {"project": "003_tree_based_learning", "description": "Tree-Based Learning", "task": "classification"}


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionInput):
    try:
        model = get_model()
        df = pd.DataFrame([input_data.features])
        pred = model.predict(df)[0]
        return PredictionResponse(prediction=float(pred))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
