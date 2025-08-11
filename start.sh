#!/bin/bash

# Check if GITHUB_TOKEN is set
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN environment variable is not set!"
    exit 1
fi

if [ -z "$PORT" ]; then
    echo "Error: PORT environment variable is not set!"
    exit 1
fi

# Create model directory if it doesn't exist
mkdir -p model

# Download and unzip model files if they don't exist
if [ ! -f "model/random_forest_model.joblib" ] || [ ! -f "model/onehot_encoder.joblib" ]; then
    echo "Downloading and extracting model files..."

    # Create a temporary directory for the download
    mkdir -p tmp_download

    # Download the artifact using GitHub token
    echo "Downloading model artifact..."
    curl -L -H "Authorization: token $GITHUB_TOKEN" \
         -o "tmp_download/model.zip" \
         "https://api.github.com/repos/persegersten/udacity-project3/actions/artifacts/latest/zip"

    # Check if download was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download model artifact!"
        rm -rf tmp_download
        exit 1
    fi

    # Unzip the files to the model directory
    echo "Extracting model files..."
    unzip -o "tmp_download/model.zip" -d model/

    # Check if unzip was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to extract model files!"
        rm -rf tmp_download
        exit 1
    fi

    # Clean up
    rm -rf tmp_download

    # Debug: List contents of model directory
    echo "Contents of model directory:"
    ls -la model/
fi

# Check if model files exist after download/extraction
if [ ! -f "model/random_forest_model.joblib" ] || [ ! -f "model/onehot_encoder.joblib" ]; then
    echo "Error: Model files are missing after download and extraction!"
    echo "Expected files:"
    echo "- model/random_forest_model.joblib"
    echo "- model/onehot_encoder.joblib"
    exit 1
fi

echo "Model files are ready!"

export MODEL_DIR="model"

# Start the application
exec gunicorn components.api.app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT