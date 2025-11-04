# Flexible Job Shop Scheduling Problem (FJSP) with Graph-Based Reinforcement Learning

This project implements a Graph-Based Reinforcement Learning approach using Heterogeneous Graph Transformer (HGT) for solving the Flexible Job Shop Scheduling Problem.

## Project Structure

```
term_project/
├── config.py              # Centralized configuration for experiments
├── exp_graph_rl.py        # Main training entry point
├── requirements.txt       # Python dependencies
├── src/
│   ├── HGT/              # HGT (Heterogeneous Graph Transformer) implementation
│   │   ├── hgt_env.py    # Environment implementation
│   │   ├── hgt_network.py # HGT network architecture
│   │   └── hgt_trainer.py # Training logic
│   ├── utils/            # Utility functions
│   │   ├── config_util.py
│   │   ├── data_util.py
│   │   └── ...
│   └── data_generator.py # Data generation utilities
├── benchmarks_data/       # Benchmark datasets
└── result/               # Training results (excluded from git)
```

## Requirements

- Python 3.10+
- PyTorch 2.0.0+ (with CUDA support recommended)
- PyTorch Geometric 2.0.0+
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

Edit `config.py` to customize experiment parameters:
- Problem configuration (job number, machine number, etc.)
- Network architecture (HGT layers, hidden dimensions, etc.)
- Training hyperparameters (learning rate, batch size, etc.)

## Usage

Run training:
```bash
python exp_graph_rl.py
```

## Features

- **HGT-based Architecture**: Uses Heterogeneous Graph Transformer for graph-based scheduling
- **Flexible Configuration**: Centralized config system for easy experimentation
- **Wandb Integration**: Experiment tracking and visualization
- **Benchmark Support**: Includes benchmark datasets for evaluation

## License

See LICENSE file for details.

## Team Members

[Add your team members here]

## About

This is a term project for "Operations Research Applications and Implementation", focusing on analyzing the limitations of Flexible Job Shop Problem (FJSP) using GNN, particularly over-smoothing issues.

## References

1. Wang, R., Wang, G., Sun, J., Deng, F., & Chen, J. (2023). Flexible job shop scheduling via dual attention network-based reinforcement learning. *IEEE Transactions on Neural Networks and Learning Systems*, 35(3), 3091-3102.

2. Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020, April). Heterogeneous graph transformer. In *Proceedings of the web conference 2020* (pp. 2704-2710).

3. Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. *arXiv preprint arXiv:1903.02428*.
