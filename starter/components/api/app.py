from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from typing import Any, Dict
import numpy as np

# Load trained objects (adjust paths as needed)
model = joblib.load("./model/random_forest_model.joblib")
encoder = joblib.load("./model/onehot_encoder.joblib")
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

@app.post("/predict")
def predict_income(payload: InferenceRequest) -> Dict[str, Any]:
    # Convert request to DataFrame with original column names
    data = payload.dict(by_alias=True)
    df = pd.DataFrame([data])

    # One-hot encode categorical features
    cat_feats = [f for f in df.columns if '-' in f]
    X_cat = encoder.transform(df[cat_feats])

    # Numeric features
    num_feats = [c for c in df.columns if c not in cat_feats]
    X_num = df[num_feats].to_numpy()

    # Combine and predict
    X_final = np.concatenate([X_num, X_cat], axis=1)
    pred = model.predict(X_final)
    #label = lb.inverse_transform(pred)[0]

    # return {"prediction": label}
    return {"prediction": pred}
