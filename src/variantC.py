"""
Variant C: PyTorch + snnTorch implementation
Trainable LIF-based SNN model with AFR computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as sf
import numpy as np
import argparse
from pathlib import Path
import sys
import os
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple

# Add parent directory to path for importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import set_seed, plot_training_curves

class SNNCNN(nn.Module):
    """
    Spiking CNN using LIF neurons.
    Architecture: Conv2d -> LIF -> MaxPool -> Conv2d -> LIF -> MaxPool -> FC -> LIF
    """
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int,
                 beta: float = 0.95):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        # Calculate size after convolutions and pooling
        self.num_flat_features = 64 * 4 * 4
        
        self.fc1 = nn.Linear(self.num_flat_features, num_outputs)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        self.num_neurons = (
            12 * 24 * 24 +  # After conv1
            64 * 8 * 8 +    # After conv2
            num_outputs     # Output layer
        )
    
    def forward(self, x):
        """Forward pass with membrane potential reset between batches."""
        # Initialize hidden states and outputs
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        spk1_rec = []
        spk2_rec = []
        out_rec = []
        
        for step in range(x.size(0)):  # Time steps
            cur1 = F.max_pool2d(self.conv1(x[step]), 2)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = F.max_pool2d(self.conv2(spk1), 2)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur3 = self.fc1(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)
            
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            out_rec.append(spk3)
        
        return torch.stack(out_rec, dim=0)
    
    def compute_afr(self, spk_rec):
        """Compute Average Firing Rate as percentage of total possible spikes."""
        total_spikes = torch.sum(spk_rec)
        total_neurons = self.num_neurons
        total_timesteps = spk_rec.size(0)
        total_samples = spk_rec.size(1)
        
        max_possible_spikes = total_neurons * total_timesteps * total_samples
        afr = 100.0 * total_spikes / max_possible_spikes
        return afr.item()

def generate_synthetic_data(num_samples: int = 1000,
                          input_size: int = 28,
                          num_classes: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic image-like data for CNN."""
    # Generate random images
    images = torch.randn(num_samples, 1, input_size, input_size)
    
    # Generate labels based on simple patterns
    labels = torch.zeros(num_samples, dtype=torch.long)
    for i in range(num_samples):
        # Simple rule: strongest activated region determines class
        regions = F.avg_pool2d(images[i], kernel_size=input_size//2)
        labels[i] = torch.argmax(regions.view(-1)) % num_classes
    
    return images, labels

def train(model: nn.Module,
          train_loader: DataLoader,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          num_steps: int = 25) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc = 0
    total_samples = 0

    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)

        # Expand input for time steps
        data = data.unsqueeze(0).repeat(num_steps, 1, 1, 1, 1)
        spk_rec = model(data)
        
        # Loss and accuracy
        loss = F.cross_entropy(spk_rec.mean(0), targets)  # Average over time steps
        acc = (spk_rec.mean(0).argmax(1) == targets).float().mean() * 100

        # Gradient computation and weight updates
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_size = len(targets)
        total_loss += loss.item() * batch_size
        total_acc += acc.item() * batch_size
        total_samples += batch_size
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_acc / total_samples
    }

def validate(model: nn.Module,
            val_loader: DataLoader,
            device: torch.device,
            num_steps: int = 25) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_acc = 0
    total_samples = 0
    spike_recordings = []
    
    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            data = data.unsqueeze(0).repeat(num_steps, 1, 1, 1, 1)
            spk_rec = model(data)
            
            acc = sf.accuracy_rate(spk_rec, targets) * 100
            spike_recordings.append(spk_rec)
            
            batch_size = len(targets)
            total_acc += acc.item() * batch_size
            total_samples += batch_size
    
    # Compute AFR using all spike recordings
    afr = model.compute_afr(torch.cat(spike_recordings, dim=1))
    
    return {
        'accuracy': total_acc / total_samples,
        'afr': afr
    }

def main(args):
    # Set device and random seed
    device = torch.device(args.device)
    set_seed(args.seed)
    
    # Generate dataset
    X_train, y_train = generate_synthetic_data(args.num_samples)
    X_test, y_test = generate_synthetic_data(args.num_samples // 5)
    
    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                            batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_test, y_test),
                          batch_size=args.batch_size)
    
    # Initialize model
    model = SNNCNN(784, args.hidden_size, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training history
    history = {'loss': [], 'accuracy': [], 'val_accuracy': [], 'val_afr': []}
    
    # Training loop
    for epoch in range(args.epochs):
        train_stats = train(model, train_loader, optimizer, device, args.num_steps)
        val_stats = validate(model, val_loader, device, args.num_steps)
        
        history['loss'].append(train_stats['loss'])
        history['accuracy'].append(train_stats['accuracy'])
        history['val_accuracy'].append(val_stats['accuracy'])
        history['val_afr'].append(val_stats['afr'])
        
        print(f"Epoch {epoch + 1}/{args.epochs}:")
        print(f"  Train Loss: {train_stats['loss']:.4f}")
        print(f"  Train Accuracy: {train_stats['accuracy']:.2f}%")
        print(f"  Val Accuracy: {val_stats['accuracy']:.2f}%")
        print(f"  Average Firing Rate: {val_stats['afr']:.2f}%")
    
    # Save training curves
    os.makedirs('outputs/plots', exist_ok=True)
    plot_training_curves(history, 'outputs/plots/training_curves.png')
    
    # Save model checkpoint
    os.makedirs('outputs/models', exist_ok=True)
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, 'outputs/models/snn_model.pth')
    
    # Final evaluation
    final_stats = validate(model, val_loader, device, args.num_steps)
    print("\nFinal Results for Variant C:")
    print(f"Test Accuracy: {final_stats['accuracy']:.2f}%")
    print(f"Average Firing Rate: {final_stats['afr']:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default='cpu')
    args = parser.parse_args()
    
    main(args)