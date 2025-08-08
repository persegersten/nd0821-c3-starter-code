from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from typing import Any, Dict
import numpy as np
import os
from pathlib import Path

CATEGORICAL_COLS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

NUMERIC_COLS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

# Load trained objects
# Folder that holds the artefacts.
# Set MODEL_DIR=/some/absolute/or/relative/dir in your environment to override
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./model"))
model   = joblib.load(MODEL_DIR / "random_forest_model.joblib")
encoder = joblib.load(MODEL_DIR / "onehot_encoder.joblib")
# lb = joblib.load("../model/label_binarizer.joblib")

# Start app
# uvicorn components.api.app:app --reload --host 0.0.0.0 --port 8000
app = FastAPI()

class InferenceRequest(BaseModel):
    age: int
    fnlwgt: float
    education: str
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain:   float = Field(..., alias="capital-gain")
    capital_loss:   float = Field(..., alias="capital-loss")
    hours_per_week: float = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": 38,
                "fnlwgt": 284582.0,
                "education": "HS-grad",
                "marital-status": "Divorced",
                "occupation": "Exec-managerial",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0.0,
                "capital-loss": 0.0,
                "hours-per-week": 40.0,
                "native-country": "United-States"
            }
        }

@app.get("/")
def read_root() -> Dict[str, Any]:
    return {"message": "Welcome to the Income Prediction API"}

LABEL_MAP = {0: "<=50K", 1: ">50K"}

@app.post("/predict")
def predict_income(payload: dict):
    # --- DataFrame built from request -----------------------------------
    df = pd.DataFrame([payload])

    # --- split features exactly as during training ----------------------
    X_num = df[NUMERIC_COLS].reset_index(drop=True)

    # One-hot encode and build a DataFrame with column names
    X_cat_arr = encoder.transform(df[CATEGORICAL_COLS])
    X_cat = pd.DataFrame(
        X_cat_arr,
        columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
        index=X_num.index,
    )

    # --- concatenate into a single DataFrame ----------------------------
    X_final = pd.concat([X_num, X_cat], axis=1)

    # --- model prediction -----------------------------------------------
    pred = model.predict(X_final)

    pred_labels = [LABEL_MAP[int(y)] for y in pred]

    return {"prediction": pred_labels}

    return {"prediction": pred_labels}
