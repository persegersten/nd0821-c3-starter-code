from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from typing import List, Dict, Any


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def evaluate_slices(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    categorical_features: List[str]
) -> pd.DataFrame:
    """
    Evaluate model performance on slices of the data defined by each
    unique value of each categorical feature.

    Returns a DataFrame with columns:
      - feature: the categorical feature name
      - value: one of its unique categories
      - precision, recall, fbeta: metrics on that slice
      - n: number of samples in the slice
    """
    records: List[Dict[str, Any]] = []
    for feature in categorical_features:
        for val in X[feature].dropna().unique():
            mask = X[feature] == val
            X_slice = X.loc[mask]
            y_slice = y[mask]
            if len(y_slice) == 0:
                continue

            # turn DataFrame slice into model input (e.g., numpy array)
            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)

            records.append({
                "feature": feature,
                "value": val,
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta,
                "n": len(y_slice)
            })

    return pd.DataFrame.from_records(records)
