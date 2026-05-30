import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

MODEL_PATH = "land_cost_predictor_pipeline.joblib"
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load(MODEL_PATH)
    yield
    model = None


app = FastAPI(title="Land Cost Predictor", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: dict  # e.g. {"area_sqft": 1200, "location": "urban", "distance_km": 5.2}


class PredictResponse(BaseModel):
    predicted_cost: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        import pandas as pd
        X = pd.DataFrame([request.features])
        prediction = model.predict(X)
        return PredictResponse(predicted_cost=float(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
