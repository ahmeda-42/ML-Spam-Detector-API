import joblib
from pathlib import Path
from model.train import MODEL_OUT

MODEL_PATH = Path(MODEL_OUT)

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)