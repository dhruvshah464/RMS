"""
Unit tests for Variant A implementation.
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path for importing src modules
sys.path.append(str(Path(__file__).parent.parent))
from src.variantA import (generate_synthetic_data, SimpleANN, SimpleSNN,
                         rate_encode)

def test_data_generation():
    """Test synthetic data generation."""
    num_samples, num_features = 100, 10
    X, y = generate_synthetic_data(num_samples, num_features)
    
    assert X.shape == (num_samples, num_features)
    assert y.shape == (num_samples,)
    assert np.all((y == 0) | (y == 1))

def test_ann_forward():
    """Test ANN forward pass."""
    input_size, hidden_size = 10, 5
    batch_size = 3
    
    ann = SimpleANN(input_size, hidden_size)
    X = np.random.randn(batch_size, input_size)
    
    predictions = ann.forward(X)
    assert predictions.shape == (batch_size,)
    assert np.all((predictions == 0) | (predictions == 1))

def test_rate_encoding():
    """Test spike train encoding."""
    data = np.random.randn(5, 10)
    timesteps = 100
    spike_train = rate_encode(data, timesteps)
    
    assert spike_train.shape == (timesteps, 5, 10)
    assert np.all((spike_train == 0) | (spike_train == 1))

def test_snn_forward():
    """Test SNN forward pass."""
    input_size, hidden_size = 10, 5
    batch_size = 3
    timesteps = 50
    
    snn = SimpleSNN(input_size, hidden_size)
    spike_train = np.random.choice([0, 1], 
                                 size=(timesteps, batch_size, input_size))
    
    predictions = snn.forward(spike_train)
    assert predictions.shape == (batch_size,)
    assert np.all((predictions == 0) | (predictions == 1))

def test_end_to_end():
    """Test entire pipeline."""
    num_samples, num_features = 50, 8
    hidden_size = 10
    timesteps = 30
    
    # Generate data
    X, y = generate_synthetic_data(num_samples, num_features)
    
    # Test ANN
    ann = SimpleANN(num_features, hidden_size)
    ann_preds = ann.forward(X)
    ann_acc = np.mean(ann_preds == y) * 100
    
    assert 0 <= ann_acc <= 100
    
    # Test SNN
    snn = SimpleSNN(num_features, hidden_size)
    spike_train = rate_encode(X, timesteps)
    snn_preds = snn.forward(spike_train)
    snn_acc = np.mean(snn_preds == y) * 100
    
    assert 0 <= snn_acc <= 100