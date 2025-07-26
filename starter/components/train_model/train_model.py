# Script to train machine learning model.

import os
import argparse
import pandas as pd
import mlflow
import logging
from export_model import export_model
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
from ml.model import evaluate_slices

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Add code to load in the data.
if __name__ == "__main__":
    # Parse input argument for data file path
    parser = argparse.ArgumentParser(description="Train a model on census data")
    parser.add_argument("--data_path", type=str, default="starter/data/census_income.csv",
                        help="Path to the raw census data CSV file")
    parser.add_argument("--model_path", type=str, default="model",
                        help="Path to the folder of the trained model")
    args = parser.parse_args()

    # Start or attach to an MLflow run (if running via `mlflow run`)
    mlflow.set_experiment("Census_Income_Training")
    run_id = os.getenv("MLFLOW_RUN_ID")
    if not run_id:
        raise RuntimeError("MLFLOW_RUN_ID not set in environment. Ensure you're running via `mlflow run`.")

    # Seems not to be needed
    # mlflow.start_run(run_id=run_id)

    # Load the raw data
    data = pd.read_csv(args.data_path)
    # Rename label column to 'salary' if it's named 'income' in the dataset
    if "income" in data.columns and "salary" not in data.columns:
        data = data.rename(columns={"income": "salary"})

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Process the training data (one-hot encode categoricals)
    X_train, y_train, encoder, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    # Process the test data using the same encoder and label binarizer
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary",
        training=False, encoder=encoder
    )

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    # Train and save a model.
    model = train_model(X_train, y_train)

    # Evaluate the model on the test data
    preds = model.predict(X_test)
    precision, recall, f1 = compute_model_metrics(y_test, preds)
    logger.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    # Log metrics to MLflow
    client = MlflowClient()
    client.log_metric(run_id, "precision", precision)
    client.log_metric(run_id, "recall", recall)
    client.log_metric(run_id, "f1", f1)

    # (Optional) Log model parameters/hyperparameters
    client.log_param(run_id, "model_type", "RandomForestClassifier")
    client.log_param(run_id, "n_estimators", 100)
    client.log_param(run_id, "data_path", args.data_path)

    # Log the trained model as an MLflow model artifact
    # mlflow.sklearn.log_model(model, artifact_path="model")
    export_model(model, X_train, args.model_path)

    for category in cat_features:
        # 2) find all dummy columns that start with "category_"
        encoded_cols = [c for c in X_test.columns if c.startswith(f"{category}_")]
        logger.info(f"Analyse feature slices on categorical feature: {category}")

        #X_test_df = pd.DataFrame(X_test, columns=ohe_feature_names)
        slice_df = evaluate_slices(model, X_test, y_test, encoded_cols)
        logger.info(slice_df)

    mlflow.end_run()

