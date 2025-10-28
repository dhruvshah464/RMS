"""
Unit tests for Variant B implementation.
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path for importing src modules
sys.path.append(str(Path(__file__).parent.parent))
from src.variantB import (generate_synthetic_har_data, MultiClassANN, 
                         MultiClassSNN, rate_encode)

def test_har_data_generation():
    """Test synthetic HAR data generation."""
    num_samples, num_features = 120, 30
    num_classes = 6
    
    X, y = generate_synthetic_har_data(num_samples, num_features, num_classes)
    
    assert X.shape == (num_samples, num_features)
    assert y.shape == (num_samples,)
    assert len(np.unique(y)) == num_classes
    assert np.min(y) == 0
    assert np.max(y) == num_classes - 1

def test_multiclass_ann_forward():
    """Test multi-class ANN forward pass."""
    input_size, hidden_size, num_classes = 30, 20, 6
    batch_size = 5
    
    ann = MultiClassANN(input_size, hidden_size, num_classes)
    X = np.random.randn(batch_size, input_size)
    
    probs = ann.forward(X)
    assert probs.shape == (batch_size, num_classes)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    assert np.all(probs >= 0) and np.all(probs <= 1)

def test_rate_encoding():
    """Test spike train encoding."""
    data = np.random.randn(5, 10)
    timesteps = 100
    spike_train = rate_encode(data, timesteps)
    
    assert spike_train.shape == (timesteps, 5, 10)
    assert np.all((spike_train == 0) | (spike_train == 1))

def test_multiclass_snn_forward():
    """Test multi-class SNN forward pass."""
    input_size, hidden_size, num_classes = 30, 20, 6
    batch_size = 5
    timesteps = 50
    
    snn = MultiClassSNN(input_size, hidden_size, num_classes)
    spike_train = np.random.choice([0, 1], 
                                 size=(timesteps, batch_size, input_size))
    
    predictions = snn.forward(spike_train)
    assert predictions.shape == (batch_size,)
    assert np.all(predictions >= 0) and np.all(predictions < num_classes)

def test_end_to_end():
    """Test entire pipeline with small dataset."""
    num_samples, num_features = 60, 20
    num_classes = 4
    hidden_size = 15
    timesteps = 30
    
    # Generate data
    X, y = generate_synthetic_har_data(num_samples, num_features, num_classes)
    
    # Test ANN
    ann = MultiClassANN(num_features, hidden_size, num_classes)
    ann_probs = ann.forward(X)
    ann_preds = np.argmax(ann_probs, axis=1)
    ann_acc = np.mean(ann_preds == y) * 100
    
    assert 0 <= ann_acc <= 100
    
    # Test SNN
    snn = MultiClassSNN(num_features, hidden_size, num_classes)
    spike_train = rate_encode(X, timesteps)
    snn_preds = snn.forward(spike_train)
    snn_acc = np.mean(snn_preds == y) * 100
    
    assert 0 <= snn_acc <= 100