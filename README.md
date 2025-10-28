# SNN vs ANN Comparison Project

This project implements and compares Spiking Neural Networks (SNNs) with conventional Artificial Neural Networks (ANNs) on various classification tasks. Three variants are provided, ranging from simple NumPy implementations to PyTorch-based SNNs.

## Project Structure
```
snn-project/
  README.md
  requirements.txt
  run_all.sh
  Dockerfile
  data/
    README.md
  src/
    utils.py            # Common helpers
    variantA.py        # NumPy toy comparison
    variantB.py        # Multi-class structured data
    variantC.py        # PyTorch + snnTorch
  notebooks/
    demo_variantC_colab.ipynb
  outputs/
    logs/
    plots/
    models/
  tests/
    test_variantA.py
    test_variantB.py
    test_variantC.py
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd snn-project
```

2. Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

## Running the Experiments

### Run all variants:
```bash
bash run_all.sh
```

### Run individual variants:
```bash
# Variant A: NumPy toy comparison
python src/variantA.py --seed 42 --num-samples 1000 --num-features 20

# Variant B: Multi-class structured
python src/variantB.py --seed 42 --dataset synthetic

# Variant C: PyTorch + snnTorch
python src/variantC.py --seed 42 --epochs 10 --device cpu  # use 'cuda' for GPU
```

## MAC Counting Methodology

The project uses the following approach for counting multiply-accumulate operations (MACs):

1. ANN MACs:
   - Layer MACs = input_features × output_features
   - Total = sum of MACs across layers
   - Counted once per forward pass

2. SNN MACs:
   - Only counted when spikes occur
   - Active MAC = MAC operation performed when input or hidden neuron spikes
   - Accumulated across timesteps
   - Total active MACs typically lower due to sparsity

## Energy Proxy Model

Energy estimation uses a simple linear model:
- energy_units = macs * scale_factor
- Default scale_factor = 1.0 (arbitrary units)
- Relative comparison between ANN/SNN is key metric

## Key Features

1. Deterministic Behavior:
   - Fixed random seeds
   - Reproducible results
   - Stable numerical operations

2. MAC Accounting:
   - Accurate spike-based counting
   - Layer-wise tracking
   - Clear percentage savings

3. Performance Metrics:
   - Classification accuracy
   - MAC reduction ratio
   - Average Firing Rate (AFR)
   - Runtime comparison

## Docker Support

Build and run with Docker:
```bash
docker build -t snn-project .
docker run -it snn-project
```

GPU support (optional):
```bash
docker run --gpus all -it snn-project
```

## Testing

Run all tests:
```bash
python -m pytest tests/
```

## Limitations and Disclaimers

1. Synthetic Data:
   - Uses simplified data generators
   - Real-world performance may vary

2. Training:
   - Basic training procedures implemented
   - More sophisticated methods possible

3. Energy Estimation:
   - Simple MAC-based proxy
   - Hardware-specific factors not considered

4. SNN Implementation:
   - Basic LIF neuron model
   - More complex dynamics possible

## License

MIT License - see LICENSE file

## Updates from Sample Codes

Key improvements over the original samples:
1. Corrected MAC counting logic for actual spike operations
2. Implemented deterministic seeding across all random operations
3. Added stable sigmoid/softmax with proper numerical clipping
4. Fixed matrix shape validation and explicit error messages
5. Improved SNN thresholds and membrane potential handling