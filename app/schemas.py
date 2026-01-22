from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    message: str

class ExplanationItem(BaseModel):
    word: str
    direction: str
    percent_influence: float

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    explanation: List[ExplanationItem]