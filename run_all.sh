#!/bin/bash

# Activate virtual environment
source "/Users/shahdhruv/Desktop/Research/SNN/.venv/bin/activate"

# Create output directories if they don't exist
mkdir -p outputs/logs
mkdir -p outputs/plots
mkdir -p outputs/models

# Run all variants and save outputs to logs
echo "Running Variant A..."
python src/variantA.py --seed 42 --num-samples 1000 --num-features 20 > outputs/logs/variantA.log 2>&1

echo "Running Variant B..."
python src/variantB.py --seed 42 --dataset synthetic > outputs/logs/variantB.log 2>&1

echo "Running Variant C..."
python src/variantC.py --seed 42 --epochs 10 --device cpu > outputs/logs/variantC.log 2>&1

# Run tests
echo "Running tests..."
python -m pytest tests/

echo "All experiments completed. Check outputs/logs for detailed results."

# Run all tests
echo "Running tests..."
python -m pytest tests/

echo "All experiments completed. Check outputs/logs for detailed results."