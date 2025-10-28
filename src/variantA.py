"""
Variant A: Lightweight NumPy toy comparison
Compares ANN vs SNN using a synthetic healthcare-like dataset.
"""

import numpy as np
import argparse
from pathlib import Path
import sys
import os

# Add parent directory to path for importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (set_seed, count_mac_ops, count_snn_active_macs,
                      safe_sigmoid, calculate_metrics, validate_shapes)

class LIFNeuron:
    """Leaky Integrate and Fire neuron implementation."""
    
    def __init__(self, threshold: float = 1.0, leak_rate: float = 0.1):
        self.threshold = threshold
        self.leak_rate = leak_rate
        self.membrane_potential = 0
        
    def step(self, input_current: float) -> int:
        """Simulate one timestep and return spike (0 or 1)."""
        self.membrane_potential = (self.membrane_potential * (1 - self.leak_rate) + 
                                 input_current)
        
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = 0
            return 1
        return 0

def generate_synthetic_data(num_samples: int, num_features: int, 
                          seed: int = 42) -> tuple:
    """Generate synthetic healthcare-like data."""
    np.random.seed(seed)
    
    # Generate feature matrix
    X = np.random.randn(num_samples, num_features)
    
    # Create non-linear decision boundary
    w = np.random.randn(num_features)
    y = np.zeros(num_samples)
    
    # Non-linear rule: combine quadratic terms
    quadratic_term = np.sum(X[:, :num_features//2]**2, axis=1)
    linear_term = np.dot(X[:, num_features//2:], w[num_features//2:])
    
    y = (quadratic_term + linear_term > np.median(linear_term))
    return X, y.astype(int)

def rate_encode(data: np.ndarray, timesteps: int = 100) -> np.ndarray:
    """Convert continuous values to spike trains."""
    normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    spike_probs = normalized.reshape(1, -1, data.shape[-1])
    spike_probs = np.repeat(spike_probs, timesteps, axis=0)
    return (np.random.rand(*spike_probs.shape) < spike_probs).astype(float)

class SimpleANN:
    """Simple feedforward neural network."""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.w2 = np.random.randn(hidden_size, 1) * 0.1
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass returning predictions."""
        validate_shapes(x, self.w1)
        hidden = safe_sigmoid(np.dot(x, self.w1))
        validate_shapes(hidden, self.w2)
        output = safe_sigmoid(np.dot(hidden, self.w2))
        return (output > 0.5).flatten()

class SimpleSNN:
    """Simple spiking neural network."""
    
    def __init__(self, input_size: int, hidden_size: int,
                 threshold: float = 1.0, leak_rate: float = 0.1):
        self.hidden_neurons = [LIFNeuron(threshold, leak_rate) 
                             for _ in range(hidden_size)]
        self.output_neuron = LIFNeuron(threshold, leak_rate)
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.w2 = np.random.randn(hidden_size, 1) * 0.1
        self.spike_counter = np.zeros(hidden_size)
        
    def forward(self, spike_train: np.ndarray) -> np.ndarray:
        """Forward pass with spike train input."""
        T = spike_train.shape[0]
        batch_size = spike_train.shape[1]
        predictions = np.zeros(batch_size)
        
        for b in range(batch_size):
            output_spikes = 0
            hidden_states = np.zeros(len(self.hidden_neurons))
            
            for t in range(T):
                # Input layer to hidden layer
                input_spikes = spike_train[t, b]
                validate_shapes(input_spikes.reshape(1, -1), self.w1)
                hidden_current = np.dot(input_spikes, self.w1)
                
                # Update hidden neurons
                for i, neuron in enumerate(self.hidden_neurons):
                    spike = neuron.step(hidden_current[i])
                    hidden_states[i] = spike
                    self.spike_counter[i] += spike
                
                # Hidden layer to output
                validate_shapes(hidden_states.reshape(1, -1), self.w2)
                output_current = np.dot(hidden_states, self.w2)
                output_spikes += self.output_neuron.step(output_current[0])
            
            predictions[b] = output_spikes > T/4  # Threshold for classification
            
        return predictions

def main(args):
    # Set random seed
    set_seed(args.seed)
    
    # Generate synthetic data
    X, y = generate_synthetic_data(args.num_samples, args.num_features, args.seed)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train ANN
    ann = SimpleANN(args.num_features, args.hidden_size)
    ann_preds = ann.forward(X_test)
    ann_accuracy = np.mean(ann_preds == y_test) * 100
    
    # Count ANN MACs
    ann_macs = count_mac_ops(args.num_features, args.hidden_size, 1)
    
    # Create and run SNN
    snn = SimpleSNN(args.num_features, args.hidden_size)
    spike_train = rate_encode(X_test, args.timesteps)
    snn_preds = snn.forward(spike_train)
    snn_accuracy = np.mean(snn_preds == y_test) * 100
    
    # Count SNN active MACs
    snn_macs = count_snn_active_macs(spike_train, snn.w1)
    
    # Calculate metrics
    ratio, savings = calculate_metrics(ann_macs, snn_macs)
    
    # Print results
    print("\nResults for Variant A:")
    print(f"ANN Accuracy: {ann_accuracy:.2f}%")
    print(f"SNN Accuracy: {snn_accuracy:.2f}%")
    print(f"ANN MACs: {ann_macs}")
    print(f"SNN Active MACs: {snn_macs}")
    print(f"MAC Reduction Ratio (ANN/SNN): {ratio:.2f}")
    print(f"Power Savings: {savings:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-features", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=50)
    parser.add_argument("--timesteps", type=int, default=100)
    args = parser.parse_args()
    
    main(args)