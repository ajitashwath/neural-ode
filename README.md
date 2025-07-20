# Neural Ordinary Differential Equations (Neural ODEs)
A JAX-based implementation of Neural Ordinary Differential Equations using Equinox and Diffrax for continuous-depth neural networks.

![Spiral Visualization](https://github.com/ajitashwath/neural-ode/blob/main/res/spiral.png?raw=true)

## Overview
Neural ODEs represent a continuous analog of residual networks, where instead of discrete layers, we have a continuous transformation parameterized by an ordinary differential equation. This implementation demonstrates how to train Neural ODEs for binary classification tasks on synthetic 2D datasets.

## Features
- **JAX-based implementation** for high-performance computing with automatic differentiation
- **Equinox integration** for clean, functional neural network definitions
- **Diffrax solver** for efficient ODE integration with adaptive step sizes
- **Synthetic datasets** including spiral and concentric circles
- **Real-time visualization** of learned dynamics and trajectories
- **Docker support** for easy deployment and reproducibility

## Mathematical Background

A Neural ODE defines the hidden state evolution as:

```
dz/dt = f(z(t), t; θ)
```

Where:
- `z(t)` is the hidden state at time `t`
- `f` is a neural network parameterized by `θ`
- The output is obtained by integrating from `t=0` to `t=1`

The gradient computation uses the adjoint sensitivity method, making it memory-efficient compared to standard backpropagation through explicit Euler steps.

## Installation

### Using pip (Recommended)

```bash
pip install .
```

This will install all required dependencies including JAX, Equinox, Diffrax, Optax, and Matplotlib.

### Using Docker

```bash
# Build the Docker image
docker build -t neural-ode .

# Run the container
docker run -it --rm neural-ode
```

## Project Structure

```
neural-ode/
├── notebooks/              
├── res/
├── Dockerfile              # Docker configuration
├── pyproject.toml         # Project configuration and dependencies
├── main.py                # Main training script
├── model.py               # Neural ODE model definitions
├── data_loader.py         # Synthetic dataset generators
├── training.py            # Training loop and visualization
└── README.md           
```

## Usage

### Basic Training
Run the main training script with default parameters:

```bash
python main.py
```

### Available Datasets

The implementation includes two synthetic datasets:

1. **Spiral Dataset**: Two interleaving spirals representing different classes
2. **Circle Dataset**: Two concentric circles with different radii

### Model Architecture

- **ODE Function**: 2-layer MLP with 64 hidden units and tanh activation
- **Solver**: Dopri5 (Dormand-Prince) adaptive step-size solver
- **Loss**: Sigmoid binary cross-entropy
- **Optimizer**: Adam with learning rate 3e-3

### Customization

You can modify the hyperparameters in `main.py`:

```python
learning_rate = 3e-3      # Learning rate for Adam optimizer
num_steps = 2000          # Number of training steps
dataset_name = "spiral"   # "spiral" or "circle"
data_size = 1000          # Number of samples per class
batch_size = 256          # Batch size for training
seed = 42                 # Random seed for reproducibility
```

## Visualization

The training process automatically generates two types of visualizations:

1. **Training Loss Curve**: Shows convergence over training steps (log scale)
2. **Learned Dynamics**: Displays vector field, trajectories, and data points

The dynamics visualization includes:
- Vector field showing the learned ODE dynamics
- Sample trajectories from initial data points
- Original data points colored by class

## Dependencies

- **JAX**: High-performance numerical computing
- **Equinox**: Functional neural networks in JAX
- **Diffrax**: Differential equation solvers in JAX
- **Optax**: Gradient-based optimization
- **Matplotlib**: Plotting and visualization

## Performance Notes

- Uses JAX's JIT compilation for fast execution
- Memory-efficient adjoint method for backpropagation
- Adaptive step-size ODE solver for numerical stability
- Vectorized operations for batch processing

## References

1. Chen, Ricky T. Q., et al. "Neural ordinary differential equations." Advances in neural information processing systems 31 (2018).
2. [Diffrax Documentation](https://docs.kidger.site/diffrax/)
3. [Equinox Documentation](https://docs.kidger.site/equinox/)
