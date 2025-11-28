#!/usr/bin/env python3
"""
DANIEL Training Entry Point

This script provides the main entry point for training
the DANIEL model for Flexible Job Shop Scheduling Problems.

Usage:
    python exp_daniel_main.py
"""

import os
import sys
import argparse
from typing import Optional

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_util import save_daniel_config, load_daniel_config, create_daniel_config
from src.DANIEL.DANIEL_trainer import DANIELTrainer


def main(config_path: Optional[str] = None,
         output_dir: str = './results',
         experiment_name: str = 'exp_daniel',
         wandb_run_name: Optional[str] = None,
         **config_overrides):
    """
    Main function for DANIEL training.
    
    Args:
        config_path: Path to configuration file (JSON)
        output_dir: Output directory for results
        **config_overrides: Additional configuration parameters
    """
    
    # Load configuration if provided, otherwise create a default one
    if config_path is not None:
        config = load_daniel_config(config_path)
    else:
        # Build a default config using the centralized config system
        overrides = dict(config_overrides)
        if wandb_run_name is not None:
            overrides['wandb_run_name'] = wandb_run_name
        config = create_daniel_config(experiment_name=experiment_name, create_result_dir=True, **overrides)
    if wandb_run_name is not None and getattr(config, 'wandb_run_name', None) != wandb_run_name:
        setattr(config, 'wandb_run_name', wandb_run_name)
    
    # Use the result directory from config (includes experiment name and timestamp)
    # The config system creates: result/{experiment_name}/{timestamp}/
    print(f"Models will be saved to: {config.result_dir}")
    
    print("Starting DANIEL training...")
    print(f"Configuration: {config.n_j} jobs, {config.n_m} machines, {config.max_updates} updates")
    
    trainer = DANIELTrainer(config)
    trainer.train()
    
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DANIEL Training for FJSP')
    parser.add_argument('--config_path', type=str, default=None, help='Path to JSON config (optional)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save outputs')
    parser.add_argument('--experiment_name', type=str, default='exp_daniel', help='Experiment name for results/wandb')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Run name for wandb')
    args, unknown = parser.parse_known_args()
    main(config_path=args.config_path, output_dir=args.output_dir, experiment_name=args.experiment_name, wandb_run_name=args.wandb_run_name)
