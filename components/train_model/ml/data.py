import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : : pd.DataFrame
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """
    if categorical_features is None:
        categorical_features = []

    X_ = X.copy()

    # --------- Handle the label --------------------------------------------
    if label is not None:
        # Harmonise textual variants → strip '.' and whitespace
        y_series = (
            X_[label].astype(str)
            .str.replace(".", "", regex=False)
            .str.strip()
        )
        # Binary mapping: >50K  → 1   ,   <=50K → 0
        y = (y_series == ">50K").astype(int).values
        X_ = X_.drop(columns=[label])
    else:
        y = np.array([])

    # --------- Split out categorical / continuous --------------------------
    X_cat = X_[categorical_features].values
    X_cont = X_.drop(columns=categorical_features).values
    cont_features = [c for c in X_.columns if c not in categorical_features]

    # --------- Fit / transform encoder -------------------------------------
    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat_enc = encoder.fit_transform(X_cat)
    else:
        X_cat_enc = encoder.transform(X_cat)

    # --------- Concatenate --------------------------------------------------
    X_final = np.concatenate([X_cont, X_cat_enc], axis=1)

    # --------- Build datafram -----------------------------------------------
    # Build DataFrame with proper column names
    ohe_names = list(encoder.get_feature_names_out(categorical_features))
    all_cols = cont_features + ohe_names
    X_df = pd.DataFrame(X_final, columns=all_cols)

    # Safety check
    assert len(y) in (0, X_final.shape[0]), (
        f"X rows: {X_final.shape[0]}, y: {len(y)}"
    )

    return X_df, y, encoder, lb
