"""
Unit tests for Variant C implementation.
"""

import pytest
import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path for importing src modules
sys.path.append(str(Path(__file__).parent.parent))
from src.variantC import (SNNCNN, generate_synthetic_data, train, validate)

def test_data_generation():
    """Test synthetic image data generation."""
    num_samples = 50
    input_size = 28
    num_classes = 10
    
    images, labels = generate_synthetic_data(num_samples, input_size, num_classes)
    
    assert images.shape == (num_samples, 1, input_size, input_size)
    assert labels.shape == (num_samples,)
    assert torch.min(labels) >= 0
    assert torch.max(labels) < num_classes

def test_snn_model():
    """Test SNN model architecture."""
    num_inputs = 784
    num_hidden = 256
    num_outputs = 10
    batch_size = 4
    timesteps = 5
    
    model = SNNCNN(num_inputs, num_hidden, num_outputs)
    x = torch.randn(timesteps, batch_size, 1, 28, 28)
    
    output = model(x)
    assert output.shape == (timesteps, batch_size, num_outputs)
    assert not torch.isnan(output).any()

def test_afr_computation():
    """Test Average Firing Rate computation."""
    num_inputs = 784
    num_hidden = 256
    num_outputs = 10
    batch_size = 4
    timesteps = 5
    
    model = SNNCNN(num_inputs, num_hidden, num_outputs)
    x = torch.randn(timesteps, batch_size, 1, 28, 28)
    
    spk_rec = model(x)
    afr = model.compute_afr(spk_rec)
    
    assert isinstance(afr, float)
    assert 0 <= afr <= 100

@pytest.mark.skipif(not torch.cuda.is_available(),
                   reason="CUDA not available")
def test_gpu_support():
    """Test model can run on GPU if available."""
    device = torch.device('cuda')
    
    model = SNNCNN(784, 256, 10).to(device)
    x = torch.randn(5, 2, 1, 28, 28, device=device)
    
    output = model(x)
    assert output.device.type == 'cuda'

def test_training_step():
    """Test single training step."""
    batch_size = 4
    num_steps = 5
    
    # Create small synthetic dataset
    X, y = generate_synthetic_data(batch_size)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=batch_size
    )
    
    model = SNNCNN(784, 256, 10)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Run one training step
    stats = train(model, train_loader, optimizer,
                 torch.device('cpu'), num_steps)
    
    assert 'loss' in stats
    assert 'accuracy' in stats
    assert 0 <= stats['accuracy'] <= 100

def test_validation_step():
    """Test validation step."""
    batch_size = 4
    num_steps = 5
    
    # Create small synthetic dataset
    X, y = generate_synthetic_data(batch_size)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=batch_size
    )
    
    model = SNNCNN(784, 256, 10)
    
    # Run validation
    stats = validate(model, val_loader,
                    torch.device('cpu'), num_steps)
    
    assert 'accuracy' in stats
    assert 'afr' in stats
    assert 0 <= stats['accuracy'] <= 100
    assert 0 <= stats['afr'] <= 100