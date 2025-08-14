# test_model.py

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference


def test_train_model_returns_fitted_random_forest():
    """
    The model should be a RandomForestClassifier,
    fitted on the training data (has attribute n_features_in_).
    """
    X_train = np.array([[0, 0], [1, 1]])
    y_train = np.array([0, 1])

    model = train_model(X_train, y_train)

    # Check correct type and hyperparameters
    assert isinstance(model, RandomForestClassifier)

    # Check that the model has been fitted
    assert model.n_features_in_ == X_train.shape[1]


@pytest.mark.parametrize("y, preds, expected", [
    # Perfect prediction → all metrics 1.0
    (np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0]), (1.0, 1.0, 1.0)),
    # No positive predictions → precision=1 (zero_division), recall=0, fbeta=0
    (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 0]), (1.0, 0.0, 0.0)),
])
def test_compute_model_metrics_edge_and_happy_cases(y, preds, expected):
    """
    Test compute_model_metrics for both a perfect classifier
    and the zero-prediction edge case (using zero_division=1).
    """
    precision, recall, fbeta = compute_model_metrics(y, preds)
    exp_prec, exp_rec, exp_fbet = expected
    assert precision == exp_prec
    assert recall == exp_rec
    assert fbeta == exp_fbet


def test_inference_uses_model_predict():
    """
    The inference function should call model.predict exactly once
    and return a NumPy array of predictions.
    """

    class DummyModel:
        def __init__(self):
            self.called = False

        def predict(self, X):
            self.called = True
            # return a constant prediction array
            return np.array([42] * len(X))

    dummy = DummyModel()
    X = np.array([[1], [2], [3]])
    preds = inference(dummy, X)

    assert dummy.called, "inference() must call model.predict()"
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (3,)
    assert all(preds == 42)
