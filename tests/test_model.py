import pytest
from model.load_model import MODEL_PATH, load_model

@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model artifact missing.")
def test_model_loads():
    model = load_model()
    assert model is not None

@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model artifact missing.")
def test_load_model_returns_pipeline():
    model = load_model()
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert "tfidf" in model.named_steps
    assert "classifier" in model.named_steps

@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model artifact missing.")
def test_model_predicts_binary_labels():
    model = load_model()
    preds = model.predict(["Win free money now", "Hey are you coming later?"])
    assert set(preds).issubset({0, 1})