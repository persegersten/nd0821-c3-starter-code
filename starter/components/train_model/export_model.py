import mlflow
import logging
import shutil
import os
import joblib
from mlflow.models import infer_signature

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def export_model_log(pipe, X_val):
    # Infer signature once
    val_pred = pipe.predict(X_val).reshape(-1, 1)
    signature = infer_signature(X_val, val_pred)

    mlflow.sklearn.log_model(
        pipe,
        artifact_path="model",
        signature=signature,
        input_example=X_val.iloc[:2],
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        registered_model_name="census_income_rf"  # valfritt
    )

def export_model(
    model,
    encoder,
    model_path: str
) -> None:
    # Ensure the directory exists
    os.makedirs(model_path, exist_ok=True)

    # 1) Save the model
    model_file = os.path.join(model_path, "random_forest_model.joblib")
    joblib.dump(model, model_file)

    # 2) Save the OneHotEncoder
    encoder_file = os.path.join(model_path, "onehot_encoder.joblib")
    joblib.dump(encoder, encoder_file)

    logger.info(f"Model saved to           {model_file}")
    logger.info(f"OneHotEncoder saved to   {encoder_file}")

def export_model2(pipe, X_val, model_dir_name):
    logger.info("Infer the signature of the model")
    val_pred = pipe.predict(X_val).reshape(-1, 1)  # <- gör kolumn-shape
    signature = infer_signature(X_val, val_pred)

    logger.info("Save model");

    export_path = model_dir_name
    # export_path = temp_dir # os.path.join(temp_dir, model_dir_name)

    # Delete the model directory if it already exists
    if os.path.exists(export_path):
        shutil.rmtree(export_path)

    # Saves the pipeline in the export_path directory using mlflow.sklearn.save_model
    # function. Provide the signature computed above ("signature") as well as a few
    # examples (input_example=X_val.iloc[:2]), and use the CLOUDPICKLE serialization
    # format (mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
    mlflow.sklearn.save_model(
        pipe,  # our pipeline
        export_path,  # Path to a directory for the produced package
        signature=signature,  # input and output schema
        input_example=X_val[:2], # the first few examples
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
    )
