from fastapi import FastAPI
from app.model import load_model
from app.predict import predict_and_explain
from app.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="Spam Detector API")

model = load_model()

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    return predict_and_explain(model,request.message)