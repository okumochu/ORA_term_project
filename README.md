# Flexible Job Shop Scheduling Problem (FJSP) with DANIEL

This project implements the **DANIEL (Dual Attention Network)** approach for solving the Flexible Job Shop Scheduling Problem using deep reinforcement learning. DANIEL uses specialized attention mechanisms to capture both operation-level and machine-level dependencies in FJSP scheduling decisions.

## Overview

The implementation is based on the paper "Flexible job shop scheduling via dual attention network-based reinforcement learning" (Wang et al., 2023), which introduces a dual attention network architecture combining:
- **Operation Attention Blocks (OAB)**: Capture precedence relationships and temporal dependencies between operations
- **Machine Attention Blocks (MAB)**: Model machine workload and compatibility information using node and edge features

The model is trained using Proximal Policy Optimization (PPO) to learn effective scheduling policies.

## Project Structure

```
term_project/
├── config.py                 # Centralized configuration for experiments
├── exp_daniel_main.py        # Main training entry point for DANIEL
├── requirements.txt          # Python dependencies
├── src/
│   ├── DANIEL/              # DANIEL (Dual Attention Network) implementation
│   │   ├── DANIEL_env.py    # FJSP Environment with state representation
│   │   ├── DANIEL_network.py # Dual Attention Network architecture
│   │   │                     # - Operation/Machine Attention Blocks (OAB/MAB)
│   │   │                     # - Actor-Critic networks
│   │   ├── DANIEL_trainer.py # PPO training logic
│   │   └── __init__.py
│   ├── utils/               # Utility functions
│   │   ├── config_util.py   # Configuration management
│   │   ├── test_util.py     # Testing and evaluation utilities
│   │   └── common_utils.py
│   └── data_generator.py    # FJSP instance generation
├── benchmarks_data/          # Benchmark datasets
│   ├── BenchData/           # Standard FJSP benchmarks
│   ├── test_data/           # Test instances
│   └── or_solution/         # OR-Tools optimal solutions
└── result/                  # Training results (excluded from git)
```

## Architecture

### DANIEL Network Components

1. **Feature Extraction**
   - Operation features: 10-dimensional vectors (job info, timing, completion status)
   - Machine features: 8-dimensional vectors (workload, utilization, availability)

2. **Dual Attention Layers** (configurable, default: 2 layers)
   - **Operation Attention Block (OAB)**:
     - Multi-head attention over operation predecessors and successors
     - Captures precedence constraints and temporal dependencies
     - Configurable heads per layer (default: 4 heads)
   
   - **Machine Attention Block (MAB)**:
     - Multi-head attention with both node and edge features
     - Models machine compatibility and processing time information
     - Configurable heads per layer (default: 4 heads)

3. **Actor-Critic Networks**
   - Actor: Selects operation-machine pairs from valid candidates
   - Critic: Estimates state values for PPO training
   - 3-layer MLPs with 32 hidden units

### Training Configuration

The model uses PPO (Proximal Policy Optimization) with:
- Parallel environments: 20
- Learning rate: 3e-4
- PPO epochs per update: 4
- GAE (λ=0.98) for advantage estimation
- Gradient clipping: 0.5
- Entropy coefficient: 0.01

## Requirements

- Python 3.10+
- PyTorch 2.0.0+ (with CUDA support recommended)
- NumPy
- WandB (for experiment tracking)
- See `requirements.txt` for full dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd term_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For CUDA support, install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/)

## Configuration

The `config.py` file contains all experiment parameters organized into sections:

### Problem Configuration (`problem_config`)
```python
{
    'job_num': 10,                    # Number of jobs
    'machine_num': 5,                 # Number of machines
    'operation_per_job_min': 5,       # Min operations per job
    'operation_per_job_max': 5,       # Max operations per job
    'machine_per_operation_min': 1,   # Min machines per operation
    'machine_per_operation_max': 5,   # Max machines per operation
    'process_time_min': 1,            # Min processing time
    'process_time_max': 99,           # Max processing time
}
```

### DANIEL Network Parameters (`method_params['daniel']`)
- `num_layers`: Number of dual attention layers (default: 2)
- `layer_fea_output_dim`: Output dimensions for each layer (default: [64, 64])
- `num_heads_OAB`: Attention heads for operation blocks (default: [4, 4])
- `num_heads_MAB`: Attention heads for machine blocks (default: [4, 4])
- `hidden_dim_actor/critic`: MLP hidden dimensions (default: 32)

### Training Parameters
- `num_envs`: Parallel environments (default: 20)
- `max_updates`: Maximum training updates (default: 1000)
- `test_episode`: Episodes between testing (default: 1000)
- `lr`, `gamma`, `k_epochs`, `eps_clip`, etc.

## Usage

### Training

Run the main training script:
```bash
python exp_daniel_main.py
```

Optional arguments:
```bash
python exp_daniel_main.py \
    --experiment_name my_experiment \
    --wandb_run_name custom_run_name
```

### Using Custom Configurations

You can modify `config.py` directly or pass a JSON config file:
```bash
python exp_daniel_main.py --config_path path/to/config.json
```

### Monitoring Training

Training progress is logged to WandB (Weights & Biases):
- Project name: `FJSP_oversmoothing`
- Metrics: makespan, entropy, policy loss, value loss, etc.
- Results saved to: `result/{experiment_name}/{timestamp}/`

## Data Generation

The project uses a unified data generation system (`CLDataGenerator` in `data_generator.py`):

- **Fixed parameters**: Job/machine counts and parameter ranges stay constant throughout training
- **Random instances**: Each batch generates new random FJSP instances within the parameter ranges
- **Uniform sampling**: All parameters (operation counts, machine compatibility, processing times) use uniform distribution
- **Batch consistency**: All environments in a parallel batch have the same number of operations to ensure PPO compatibility

## Features

- **DANIEL Architecture**: Dual attention network specifically designed for FJSP
- **Multi-head Attention**: Separate attention mechanisms for operations and machines
- **PPO Training**: Stable policy optimization with parallel environments
- **Flexible Configuration**: Centralized config system for easy experimentation
- **WandB Integration**: Comprehensive experiment tracking and visualization
- **Benchmark Support**: Includes standard FJSP benchmark datasets for evaluation
- **GPU Acceleration**: CUDA support for faster training

## Research Focus

This is a term project for "Operations Research Applications and Implementation", focusing on:
- Analyzing the effectiveness of attention-based architectures for FJSP
- Investigating deep reinforcement learning approaches versus traditional optimization
- Comparing with graph neural network methods (e.g., HGT)

## References

1. **Wang, R., Wang, G., Sun, J., Deng, F., & Chen, J.** (2023). Flexible job shop scheduling via dual attention network-based reinforcement learning. *IEEE Transactions on Neural Networks and Learning Systems*, 35(3), 3091-3102.
   - Primary reference for DANIEL architecture

2. Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. *arXiv preprint arXiv:1903.02428*.

## License

See LICENSE file for details.

## Team Members

[Add your team members here]
