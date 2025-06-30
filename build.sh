#!/bin/bash
# build.sh - Render build script

set -o errexit  # exit on error

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Train the model if not present
if [ ! -f "logistic_model.pkl" ] || [ ! -f "count_vectorizer.pkl" ]; then
    echo "Training spam detection model..."
    python train_model.py
else
    echo "Model files found, skipping training"
fi

echo "Build completed successfully!"
