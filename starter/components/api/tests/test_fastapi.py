# tests/test_api.py
"""
Three tests:
  • GET /
  • POST /predict  =>  0 (≤50K)
  • POST /predict  =>  1 (>50K)
We patch the module’s `encoder` and `model`, not the FastAPI instance.
"""
from pathlib import Path
import sys, importlib
import numpy as np
import pytest
from fastapi.testclient import TestClient

# --- Make sure the project root is on sys.path -----------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- Block heavy / local loading before we import app.py -------------------
import joblib
joblib.load = lambda *args, **kw: None   # always returns None

# Import the whole module – we get both the app instance and its globals
app_module = importlib.import_module("app")
client = TestClient(app_module.app)

# ---- Dummy helpers --------------------------------------------------------
class DummyEncoder:
    """Returns the right shape but no content – good enough for tests."""
    def transform(self, df):
        return np.zeros((len(df), 2))

class DummyModel:
    def __init__(self, out):
        self._out = out  # e.g. [0] or [1]

    def predict(self, X):
        # Ensure JSON‑friendly result (list of int)
        return list(self._out)


# ---- Common, valid payload ------------------------------------------------
VALID_PAYLOAD = {
    "age": 38,
    "fnlwgt": 284_582.0,
    "education": "HS-grad",
    "marital-status": "Divorced",
    "occupation": "Exec-managerial",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0.0,
    "capital-loss": 0.0,
    "hours-per-week": 40.0,
    "native-country": "United-States",
}

# ---- The actual tests -----------------------------------------------------
def test_read_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the Income Prediction API"}

def test_predict_low_income(monkeypatch):
    monkeypatch.setattr(app_module, "encoder", DummyEncoder(), raising=False)
    monkeypatch.setattr(app_module, "model", DummyModel([0]),   raising=False)

    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    assert r.json() == {"prediction": [0]}

def test_predict_high_income(monkeypatch):
    monkeypatch.setattr(app_module, "encoder", DummyEncoder(), raising=False)
    monkeypatch.setattr(app_module, "model", DummyModel([1]),   raising=False)

    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    assert r.json() == {"prediction": [1]}
