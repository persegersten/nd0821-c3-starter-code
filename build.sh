.#!/bin/bash

# Run pipeline steps

 mlflow run . \
  -P steps=train_model,api