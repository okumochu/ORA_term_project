"""
Configuration utility functions for HGT training.

This module provides functions to create and manage configuration objects
for HGT training experiments. It uses a centralized ExperimentConfig singleton
to provide consistent configuration across the project.
"""

import os
import time
import argparse
from typing import Dict, Any, List

# Import the data container only, logic lives here
from config import ExperimentConfig


# Singleton config holder to avoid scattered globals
# This ensures all configuration comes from a single source of truth
_CONFIG = ExperimentConfig()


def setup_wandb_env(result_dir: str, project_name: str) -> None:
    """
    Setup Weights & Biases (wandb) environment variables.
    
    Args:
        result_dir: Directory where wandb logs will be stored
        project_name: Name of the wandb project
    """
    os.environ["WANDB_DIR"] = result_dir
    os.environ["WANDB_PROJECT"] = project_name
    if os.environ.get("WANDB_MODE") is None:
        os.environ["WANDB_MODE"] = "online"


def create_experiment_result_dir(experiment_name: str) -> str:
    """
    Create a timestamped result directory for an experiment.
    
    Directory structure: result/{experiment_name}/{timestamp}/
    where timestamp is in format YYYYMMDD_HH (hour precision).
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Path to the created timestamped directory
    """
    base_result_dir = "result"
    experiment_dir = os.path.join(base_result_dir, experiment_name)
    timestamp = time.strftime("%Y%m%d_%H")
    timestamp_dir = os.path.join(experiment_dir, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)
    return timestamp_dir


def create_model_filename(prefix: str = "model") -> str:
    """
    Create a timestamped model filename.
    
    Args:
        prefix: Prefix for the filename (default: "model")
        
    Returns:
        Filename in format: {prefix}_YYYYMMDD_HHMM.pth
    """
    timestamp = time.strftime('%Y%m%d_%H%M')
    return f"{prefix}_{timestamp}.pth"


def get_config(method: str, experiment_name: str = None, **overrides) -> Dict[str, Any]:
    """
    Get configuration dictionary for a specific method.
    
    Merges common parameters with method-specific parameters and applies any overrides.
    Also sets up problem_config for data generation and test configuration.
    
    Args:
        method: Method name (e.g., 'hgt')
        experiment_name: Name of the experiment (default: f"exp_{method}")
        **overrides: Additional configuration parameters to override defaults
        
    Returns:
        Configuration dictionary with all parameters merged
    """
    if experiment_name is None:
        experiment_name = f"exp_{method}"

    result_dir = create_experiment_result_dir(experiment_name)

    # Merge common params with method-specific params
    config: Dict[str, Any] = {**_CONFIG.common_params, **_CONFIG.method_params.get(method, {})}
    config.update(overrides)

    # Add problem_config for data generation (same throughout training session)
    config['problem_config'] = _CONFIG.problem_config.copy()
    
    # Add fixed_parameter from problem_config (kept for backward compatibility)
    # Note: 'op_num' is always True within each batch (auto-enforced by data_generator)
    config['fixed_parameter'] = config['problem_config'].get('fixed_parameter', {
        'pt': False,
        'compatibility': False
    })
    
    # Add test configuration
    config['test_num'] = _CONFIG.test_num

    config['result_dir'] = result_dir
    setup_wandb_env(result_dir, config['wandb_project'])
    return config


def get_daniel_config(experiment_name: str = "exp_daniel", **overrides) -> Dict[str, Any]:
    """
    Get configuration dictionary specifically for DANIEL method.
    
    Args:
        experiment_name: Name of the experiment (default: "exp_daniel")
        **overrides: Additional configuration parameters to override defaults
        
    Returns:
        Configuration dictionary for DANIEL
    """
    config = get_config('daniel', experiment_name, **overrides)
    return config


def create_daniel_config(experiment_name: str = "exp_daniel", create_result_dir: bool = True, **overrides) -> argparse.Namespace:
    """
    Create DANIEL configuration as argparse.Namespace object.
    
    This function converts the configuration dictionary into a Namespace object
    for easier attribute access. It also extracts and sets problem_config parameters
    as direct attributes for convenience.
    
    Args:
        experiment_name: Name of the experiment (default: "exp_daniel")
        create_result_dir: Whether to create result directory (default: True)
        **overrides: Additional configuration parameters to override defaults
        
    Returns:
        argparse.Namespace object with all configuration parameters as attributes
    """
    config_dict = get_daniel_config(experiment_name, **overrides)
    daniel_config = argparse.Namespace()
    for key, value in config_dict.items():
        setattr(daniel_config, key, value)
    
    # Extract parameters from problem_config for convenient access
    problem_config = config_dict.get('problem_config', _CONFIG.problem_config)
    n_j = problem_config['job_num']
    n_m = problem_config['machine_num']
    
    # Set problem dimensions as direct attributes
    daniel_config.n_j = n_j
    daniel_config.n_m = n_m
    
    # Set data generation parameters from problem_config
    daniel_config.low = problem_config['process_time_min']
    daniel_config.high = problem_config['process_time_max']
    daniel_config.operation_per_job_min = problem_config['operation_per_job_min']
    daniel_config.operation_per_job_max = problem_config['operation_per_job_max']
    daniel_config.machine_per_operation_min = problem_config['machine_per_operation_min']
    daniel_config.machine_per_operation_max = problem_config['machine_per_operation_max']
    
    # Preserve common_params dict for trainers
    daniel_config.common_params = _CONFIG.common_params.copy()
    
    # Set model and test metadata
    daniel_config.model_suffix = config_dict.get('model_suffix', '')
    daniel_config.train_size = f"{n_j}x{n_m}"
    daniel_config.test_data = [f"{n_j}x{n_m}"]
    daniel_config.test_model = [f"{n_j}x{n_m}"]
    daniel_config.test_method = []
    
    return daniel_config


def save_daniel_config(config: argparse.Namespace, filepath: str) -> None:
    """
    Save DANIEL configuration to a JSON file.
    
    Args:
        config: Configuration object (argparse.Namespace)
        filepath: Path to save the configuration JSON file
    """
    import json
    
    # Convert Namespace to dict
    config_dict = vars(config).copy()
    
    # Remove non-serializable items if any
    non_serializable = []
    for key, value in config_dict.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            non_serializable.append(key)
    
    for key in non_serializable:
        config_dict.pop(key)
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"[Config] Saved DANIEL configuration to {filepath}")


def load_daniel_config(filepath: str) -> argparse.Namespace:
    """
    Load DANIEL configuration from a JSON file.
    
    Args:
        filepath: Path to the configuration JSON file
        
    Returns:
        Configuration object (argparse.Namespace)
    """
    import json
    
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    # Convert dict to Namespace
    config = argparse.Namespace()
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    print(f"[Config] Loaded DANIEL configuration from {filepath}")
    return config