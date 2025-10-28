# Dataset Information

This project uses both synthetic and real datasets for SNN vs ANN comparison experiments.

## Synthetic Datasets

### Variant A: Healthcare-like Data
- Type: Binary classification
- Features: Continuous values (normalized)
- Structure:
  - X shape: (num_samples, num_features)
  - y shape: (num_samples,)
  - Default: 1000 samples, 20 features
- Generation: Non-linear decision boundary using quadratic + linear terms

### Variant B: HAR-like Data
- Type: Multi-class classification
- Classes: 6 activities (by default)
- Features: Synthetic temporal patterns
- Structure:
  - X shape: (num_samples, num_features)
  - y shape: (num_samples,)
  - Default: 1000 samples, 30 features

### Variant C: Image-like Data
- Type: Multi-class classification
- Classes: 10 categories
- Structure:
  - X shape: (num_samples, 1, 28, 28) [MNIST-like]
  - y shape: (num_samples,)
  - Default: 1000 training samples

## UCI HAR Dataset (Optional)
For Variant B, the code can optionally use the UCI Human Activity Recognition dataset.
However, to ensure offline functionality, a synthetic alternative is always provided.

- Classes: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying
- Features: Accelerometer and Gyroscope readings
- Source: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

Note: The synthetic data generators use fixed random seeds (default=42) for reproducibility.