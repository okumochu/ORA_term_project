"""
DANIEL: Dual Attention Network for Flexible Job Shop Scheduling

This module contains the implementation of DANIEL, a deep reinforcement learning
approach for solving Flexible Job Shop Scheduling Problems (FJSP).

Components:
- DANIEL_network.py: The neural network architecture
- DANIEL_env.py: The FJSP environment implementation
- DANIEL_trainer.py: Training and evaluation utilities
"""

from .DANIEL_network import DANIEL, DualAttentionNetwork
from .DANIEL_env import FJSPEnvForVariousOpNums, EnvState
from .DANIEL_trainer import DANIELTrainer, PPO, Memory

# Import config utilities
try:
    from config import create_daniel_config, get_default_daniel_config
    __all__ = [
        'DANIEL',
        'DualAttentionNetwork', 
        'FJSPEnvForVariousOpNums',
        'EnvState',
        'DANIELTrainer',
        'PPO',
        'Memory',
        'create_daniel_config',
        'get_default_daniel_config'
    ]
except ImportError:
    __all__ = [
        'DANIEL',
        'DualAttentionNetwork', 
        'FJSPEnvForVariousOpNums',
        'EnvState',
        'DANIELTrainer',
        'PPO',
        'Memory'
    ]
