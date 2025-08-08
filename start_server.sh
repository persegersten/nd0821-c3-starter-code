#!/bin/bash

export MODEL_DIR="model"

uvicorn components.api.app:app \
        --reload \
        --host 0.0.0.0 \
        --port 8000
