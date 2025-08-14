import os
import argparse
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from ucimlrepo import fetch_ucirepo


def download_and_save_data(dataset_id: int, target_dir: str, run_id: str) -> str:
    """
    Downloads a dataset from an external source and saves it locally,
    logging parameters and artifacts directly to the provided MLflow run.

    Args:
        dataset_id (int): UCI dataset ID (e.g., 20 for census_income).
        target_dir (str): Local directory to save the data.
        run_id (str): MLflow run ID to log into.

    Returns:
        str: Path to the saved data file.
    """
    client = MlflowClient()
    # Log parameters
    client.log_param(run_id, "dataset_id", dataset_id)
    client.log_param(run_id, "target_dir", target_dir)

    # Fetch data
    census_income = fetch_ucirepo(id=dataset_id)
    os.makedirs(target_dir, exist_ok=True)
    output_path = os.path.join(target_dir, "census_income.csv")

    X = census_income.data.features
    y = census_income.data.targets

    # metadata
    print(census_income.metadata)
    # variable information
    print(census_income.variables)

    # Combine features + target in same DataFrame
    df = pd.concat([X, y], axis=1)

    df.to_csv(output_path, index=False)

    # Log artifact
    client.log_artifact(run_id, output_path, artifact_path="data")
    return output_path


def main(dataset_id: int, target_dir: str):
    # Set or verify experiment name
    mlflow.set_experiment("Data_Ingestion_Experiment")

    # Grab the active run ID from the MLflow Projects environment
    run_id = os.getenv("MLFLOW_RUN_ID")
    if not run_id:
        raise RuntimeError("MLFLOW_RUN_ID not set in environment. Ensure you're running via `mlflow run`.")

    file_path = download_and_save_data(dataset_id, target_dir, run_id)
    print(f"Data saved to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download UCI Census Income dataset")
    parser.add_argument("--dataset_id", type=int, required=True, help="UCI dataset ID (e.g., 20)")
    parser.add_argument("--target_dir", type=str, required=True, help="Directory to save CSV")
    args = parser.parse_args()
    main(dataset_id=args.dataset_id, target_dir=args.target_dir)
