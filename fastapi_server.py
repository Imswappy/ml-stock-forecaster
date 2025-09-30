# fastapi_server.py
import os
from typing import List

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Stock Forecasting Inference")

MODEL_PATH_ENV = "MODEL_PATH"  # you can set this env var or edit below

# Example: set default path if not provided via env
DEFAULT_MODEL_PATH = "AAPL_RandomForest.joblib"

class PredictRequest(BaseModel):
    # features must be in same order as saved 'features' list in joblib bundle
    features: List[float]

class PredictBatchRequest(BaseModel):
    instances: List[List[float]]


# Load model helper
def load_model(path=None):
    p = path or os.getenv(MODEL_PATH_ENV) or DEFAULT_MODEL_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(f"Model file not found: {p}")
    bundle = joblib.load(p)
    # Expecting bundle = {"model": model_obj, "scaler": scaler_obj, "features": feat_list}
    model = bundle.get("model")
    scaler = bundle.get("scaler")
    features = bundle.get("features")
    if model is None or scaler is None or features is None:
        raise ValueError("Model bundle missing required keys: 'model','scaler','features'")
    return model, scaler, features

# Load at startup (edit DEFAULT_MODEL_PATH or set MODEL_PATH env var)
try:
    MODEL, SCALER, FEATURE_LIST = load_model()
    print(f"Loaded model from default path. Features count: {len(FEATURE_LIST)}")
except Exception as e:
    # don't crash the server; return 503 for predict if not loaded
    MODEL = None
    SCALER = None
    FEATURE_LIST = None
    print("Model not loaded at startup:", e)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict")
def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")
    # Validate feature length
    if len(req.features) != len(FEATURE_LIST):
        raise HTTPException(status_code=400, detail=f"Expected {len(FEATURE_LIST)} features in order: {FEATURE_LIST}")
    X = np.array(req.features).reshape(1, -1)
    Xs = SCALER.transform(X)
    y_pred = MODEL.predict(Xs)
    return {"prediction": float(y_pred[0])}

@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")
    X = np.array(req.instances)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_LIST):
        raise HTTPException(status_code=400, detail=f"Each instance must have {len(FEATURE_LIST)} features")
    Xs = SCALER.transform(X)
    preds = MODEL.predict(Xs).tolist()
    return {"predictions": preds}

if __name__ == "__main__":
    # run with: python fastapi_server.py or uvicorn fastapi_server:app --reload
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8000, reload=True)
