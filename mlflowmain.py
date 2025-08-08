import json

import mlflow
import argparse

def go(steps: str):

    # Steps to execute
    active_steps = steps.split(",")

    print(f"Active steps: {active_steps}")

    if "download_data" in active_steps:
        # Download file and load in W&B
        _ = mlflow.run(
            "components/download_data",
            "main",
            version='main',
            env_manager="conda",
        )

    if "train_model" in active_steps:
        _ = mlflow.run(
            "components/train_model",
            "main"
        )

    if "api" in active_steps:
        _ = mlflow.run(
            "components/api",
            "main"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default="all",
                        help="Comma‑separated list of pipeline steps to run")
    args = parser.parse_args()
    go(args.steps)
