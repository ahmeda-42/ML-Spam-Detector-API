import joblib
from pathlib import Path

def load_model():
    model_path = Path("artifacts/model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)