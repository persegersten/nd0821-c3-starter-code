# components/train_model/tests/test_evaluate_slices.py

import pandas as pd
import numpy as np
import ml.model as model
from ml.model import evaluate_slices

def test_evaluate_slices_with_stubbed_functions(monkeypatch):
    """
    Verify that evaluate_slices loops over each category/value,
    calls inference + compute_model_metrics, and records the correct n.
    """
    # Simple DataFrame with 2 categorical features
    X = pd.DataFrame({
        'feature1': ['v1', 'v2', 'v1'],
        'feature2': ['a',  'b',  'a']
    })
    y = np.array([1, 0, 1])

    # Stub inference to always return zeros
    def fake_inference(m, X_arr):
        return np.zeros(X_arr.shape[0], dtype=int)

    # Stub metrics to return fixed numbers
    def fake_compute_model_metrics(y_true, preds):
        return 0.1, 0.2, 0.3

    # Patch the functions on the ml.model module
    monkeypatch.setattr(model, 'inference', fake_inference)
    monkeypatch.setattr(model, 'compute_model_metrics', fake_compute_model_metrics)

    dummy_model = object()
    df = evaluate_slices(dummy_model, X, y, ['feature1', 'feature2'])

    # Expect 4 rows: 2 values in feature1 + 2 in feature2
    assert len(df) == 4

    # All metric columns should equal our stubbed values
    assert (df['precision'] == 0.1).all()
    assert (df['recall']    == 0.2).all()
    assert (df['fbeta']     == 0.3).all()

    # Check that 'n' matches the counts per category
    counts = dict(zip(df['value'], df['n']))
    assert counts['v1'] == 2
    assert counts['v2'] == 1
    assert counts['a']  == 2
    assert counts['b']  == 1

def test_evaluate_slices_with_perfect_model():
    """
    Using a model that always predicts the true label,
    metrics on each slice should be perfect (1.0), and n should match.
    """
    X = pd.DataFrame({'cat': ['x', 'y', 'x', 'y']})
    y = np.array([0, 1, 0, 1])

    class PerfectModel:
        def predict(self, X_df):
            # X_df is a DataFrame with one column, say "cat", containing 'x' or 'y'
            # Extract that column as a Series (1‑D)
            cats = X_df.iloc[:, 0]  # or X_df['cat'] if you know the name
            return np.array([0 if c == 'x' else 1 for c in cats])

    # No need to patch – inference() will call model.predict()
    df = evaluate_slices(PerfectModel(), X, y, ['cat'])

    # Two slices: 'x' and 'y'
    assert set(df['value']) == {'x', 'y'}

    for val in ['x', 'y']:
        rec = df[(df.feature=='cat') & (df.value==val)].iloc[0]
        assert rec.precision == 1.0
        assert rec.recall    == 1.0
        assert rec.fbeta     == 1.0
        # each value appears twice
        assert rec.n == 2
