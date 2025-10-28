"""
Variant B: Mid-level NumPy experiment using a structured dataset
Multi-class classification with SNN timestepped simulation.
"""

import numpy as np
import argparse
from pathlib import Path
import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add parent directory to path for importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (set_seed, count_mac_ops, count_snn_active_macs,
                      safe_softmax, plot_confusion_matrix, calculate_metrics,
                      validate_shapes)

def generate_synthetic_har_data(num_samples: int = 1000, num_features: int = 30,
                              num_classes: int = 6, seed: int = 42) -> tuple:
    """Generate synthetic human activity recognition data."""
    np.random.seed(seed)
    
    # Generate feature matrix with temporal patterns
    X = np.zeros((num_samples, num_features))
    y = np.zeros(num_samples, dtype=int)
    
    # Create distinct patterns for each activity class
    patterns = np.random.randn(num_classes, num_features)
    
    for i in range(num_samples):
        class_idx = i % num_classes
        X[i] = patterns[class_idx] + np.random.randn(num_features) * 0.1
        y[i] = class_idx
        
    return X, y

class MultiClassANN:
    """Multi-class feedforward neural network."""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.w2 = np.random.randn(hidden_size, num_classes) * 0.1
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass returning class probabilities."""
        validate_shapes(x, self.w1)
        hidden = np.maximum(0, np.dot(x, self.w1))  # ReLU activation
        validate_shapes(hidden, self.w2)
        logits = np.dot(hidden, self.w2)
        return safe_softmax(logits)

class MultiClassSNN:
    """Multi-class spiking neural network."""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int,
                 threshold: float = 1.0, leak_rate: float = 0.1):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.w2 = np.random.randn(hidden_size, num_classes) * 0.1
        self.hidden_potentials = np.zeros(hidden_size)
        self.output_potentials = np.zeros(num_classes)
        self.threshold = threshold
        self.leak_rate = leak_rate
        
    def step(self, input_spikes: np.ndarray) -> tuple:
        """Single timestep simulation."""
        # Hidden layer
        validate_shapes(input_spikes.reshape(1, -1), self.w1)
        self.hidden_potentials = (self.hidden_potentials * (1 - self.leak_rate) +
                                np.dot(input_spikes, self.w1))
        hidden_spikes = (self.hidden_potentials >= self.threshold).astype(float)
        self.hidden_potentials[hidden_spikes > 0] = 0
        
        # Output layer
        validate_shapes(hidden_spikes.reshape(1, -1), self.w2)
        self.output_potentials = (self.output_potentials * (1 - self.leak_rate) +
                                np.dot(hidden_spikes, self.w2))
        output_spikes = (self.output_potentials >= self.threshold).astype(float)
        self.output_potentials[output_spikes > 0] = 0
        
        return hidden_spikes, output_spikes
    
    def forward(self, spike_train: np.ndarray) -> np.ndarray:
        """Forward pass accumulating spikes over time."""
        T = spike_train.shape[0]
        batch_size = spike_train.shape[1]
        num_classes = self.w2.shape[1]
        class_spikes = np.zeros((batch_size, num_classes))
        
        for b in range(batch_size):
            # Reset state for each sample
            self.hidden_potentials = np.zeros_like(self.hidden_potentials)
            self.output_potentials = np.zeros_like(self.output_potentials)
            
            for t in range(T):
                _, output_spikes = self.step(spike_train[t, b])
                class_spikes[b] += output_spikes
        
        return np.argmax(class_spikes, axis=1)

def main(args):
    # Set random seed
    set_seed(args.seed)
    
    # Generate or load data
    if args.dataset == 'synthetic':
        X, y = generate_synthetic_har_data(args.num_samples, args.num_features,
                                         args.num_classes, args.seed)
    else:
        # TODO: Add UCI HAR dataset loading
        print("Using synthetic data as fallback")
        X, y = generate_synthetic_har_data(args.num_samples, args.num_features,
                                         args.num_classes, args.seed)
    
    # Split and normalize data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate ANN
    ann = MultiClassANN(args.num_features, args.hidden_size, args.num_classes)
    ann_preds = np.argmax(ann.forward(X_test_scaled), axis=1)
    ann_accuracy = np.mean(ann_preds == y_test) * 100
    
    # Count ANN MACs
    ann_macs = count_mac_ops(args.num_features, args.hidden_size, args.num_classes)
    
    # Create and run SNN
    snn = MultiClassSNN(args.num_features, args.hidden_size, args.num_classes)
    spike_train = rate_encode(X_test_scaled, args.timesteps)
    snn_preds = snn.forward(spike_train)
    snn_accuracy = np.mean(snn_preds == y_test) * 100
    
    # Count SNN active MACs
    snn_macs = count_snn_active_macs(spike_train, snn.w1)
    
    # Calculate metrics
    ratio, savings = calculate_metrics(ann_macs, snn_macs)
    
    # Plot confusion matrices
    os.makedirs('outputs/plots', exist_ok=True)
    plot_confusion_matrix(y_test, ann_preds, 'ANN Confusion Matrix',
                         'outputs/plots/ann_confusion.png')
    plot_confusion_matrix(y_test, snn_preds, 'SNN Confusion Matrix',
                         'outputs/plots/snn_confusion.png')
    
    # Save results if requested
    if args.save_results:
        results = pd.DataFrame({
            'Model': ['ANN', 'SNN'],
            'Accuracy': [ann_accuracy, snn_accuracy],
            'MACs': [ann_macs, snn_macs]
        })
        results.to_csv('outputs/logs/variant_b_results.csv', index=False)
    
    # Print results
    print("\nResults for Variant B:")
    print(f"ANN Accuracy: {ann_accuracy:.2f}%")
    print(f"SNN Accuracy: {snn_accuracy:.2f}%")
    print(f"ANN MACs: {ann_macs}")
    print(f"SNN Active MACs: {snn_macs}")
    print(f"MAC Reduction Ratio (ANN/SNN): {ratio:.2f}")
    print(f"Power Savings: {savings:.2f}%")

def rate_encode(data: np.ndarray, timesteps: int = 100) -> np.ndarray:
    """Convert continuous values to spike trains."""
    normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    spike_probs = normalized.reshape(1, -1, data.shape[-1])
    spike_probs = np.repeat(spike_probs, timesteps, axis=0)
    return (np.random.rand(*spike_probs.shape) < spike_probs).astype(float)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, choices=['synthetic', 'uci-har'],
                       default='synthetic')
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-features", type=int, default=30)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--save-results", action='store_true')
    args = parser.parse_args()
    
    main(args)