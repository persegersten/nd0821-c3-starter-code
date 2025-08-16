"""
Integration tests against a *running* FastAPI service
instead of an in-process TestClient.

What is covered
---------------
  • GET  /                 → 200 + welcome message
  • POST /predict  (low)   → 200 + {"prediction": [0]}
  • POST /predict  (high)  → 200 + {"prediction": [1]}

Configuration
-------------
Change the base URL in `API_BASE_URL`, or override it at runtime:

    cd integration_test/
    API_BASE_URL="https://udacity-c3-model-28ffc738d748.herokuapp.com/" pytest -q

NB: The service **must already be up** before you run these tests.
"""
from __future__ import annotations

import json
import os
import requests

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")


# -------- helper ------------------------------------------------------------
def _get(path: str = "/"):
    url = f"{API_BASE_URL}{path}"
    response = requests.get(url, timeout=5)
    print_response(response, "GET", url)
    return response


def _post(path: str, payload: dict):
    url = f"{API_BASE_URL}{path}"
    response = requests.post(url, json=payload, timeout=10)
    print_response(response, "POST", url)
    return response


def print_response(response: requests.Response, method: str, url: str):
    try:
        body_str = json.dumps(response.json(), ensure_ascii=False)
    except ValueError:
        body_str = response.text

    print("")
    print(f"Method:      {method}")
    print(f"URL:         {url}")
    print(f"Status code: {response.status_code}")
    print(f"Response:    {body_str}")


# -------- canonical payloads -----------------------------------------------
LOW_PAYLOAD: dict = {
    # numerical
    "age": 38,
    "fnlwgt": 284_582.0,
    "education-num": 9,
    "capital-gain": 0.0,
    "capital-loss": 0.0,
    "hours-per-week": 40.0,

    # categorical
    "workclass": "Private",
    "education": "HS-grad",
    "marital-status": "Divorced",
    "occupation": "Exec-managerial",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "native-country": "United-States",
}

HIGH_PAYLOAD: dict = {
    # numerical
    "age": 50,
    "fnlwgt": 120_000.0,
    "education-num": 16,
    "capital-gain": 99_999.0,
    "capital-loss": 0.0,
    "hours-per-week": 60.0,

    # categorical
    "workclass": "Private",
    "education": "Doctorate",
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "native-country": "United-States",
}


# -------- tests -------------------------------------------------------------
def test_read_root():
    r = _get("/")

    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the Income Prediction API"}


def test_predict_low_income():
    r = _post("/predict", LOW_PAYLOAD)
    assert r.status_code == 200
    assert r.json() == {"prediction": ["<=50K"]}


def test_predict_high_income():
    r = _post("/predict", HIGH_PAYLOAD)
    assert r.status_code == 200
    assert r.json() == {"prediction": [">50K"]}
