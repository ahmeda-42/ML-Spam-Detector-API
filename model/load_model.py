import joblib
from pathlib import Path

MODEL_PATH = Path("artifacts/model.joblib")

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)