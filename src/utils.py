"""
Common utilities for SNN experiments.
Includes MAC counting, seed setting, and plotting functions.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import os
from typing import Dict, Tuple, Union, Optional

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def count_mac_ops(input_size: int, hidden_size: int, output_size: int) -> int:
    """Count total multiply-accumulate operations for a simple feedforward network."""
    return (input_size * hidden_size) + (hidden_size * output_size)

def count_snn_active_macs(spikes: np.ndarray, weights: np.ndarray) -> int:
    """
    Count active MACs in SNN simulation (only when spikes occur).
    
    Args:
        spikes: Binary spike array of shape (timesteps, batch_size, n_neurons)
        weights: Weight matrix
    
    Returns:
        Total number of active MACs
    """
    active_macs = 0
    for t in range(spikes.shape[0]):
        active_inputs = np.sum(spikes[t] > 0)
        if active_inputs > 0:
            active_macs += active_inputs * weights.shape[1]
    return active_macs

def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    x_clipped = np.clip(x, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-x_clipped))

def safe_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax function."""
    x_max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str, save_path: str) -> None:
    """Plot and save confusion matrix."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_curves(history: Dict[str, list], save_path: str) -> None:
    """Plot training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['loss'], label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history['accuracy'], label='Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(ann_macs: int, snn_macs: int) -> Tuple[float, float]:
    """Calculate MAC reduction ratio and power savings."""
    ratio = ann_macs / max(snn_macs, 1)  # Avoid division by zero
    savings = (1 - 1/ratio) * 100 if ratio > 1 else 0
    return ratio, savings

def estimate_energy(macs: int, scale_factor: float = 1.0) -> float:
    """Convert MACs to energy units (arbitrary units)."""
    return macs * scale_factor

def validate_shapes(*arrays: np.ndarray) -> None:
    """Validate shapes for matrix operations."""
    if len(arrays) < 2:
        return
        
    for i in range(len(arrays) - 1):
        if arrays[i].shape[-1] != arrays[i + 1].shape[0]:
            raise ValueError(f"Shape mismatch: {arrays[i].shape} vs {arrays[i + 1].shape}")