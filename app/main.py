from fastapi import FastAPI
from app.predict import predict, predict_and_explain
from app.schemas import PredictionRequest, Prediction, PredictionAndExplanation
from pathlib import Path

app = FastAPI(title="ML Spam Detector API")

@app.post("/predict", response_model=PredictionAndExplanation)
def predict(request: PredictionRequest):
    return predict_and_explain(request.message)