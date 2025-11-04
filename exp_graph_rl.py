#!/usr/bin/env python3
"""
HGT Training Entry Point

This script provides the main entry point for training the HGT-based model
for Flexible Job Shop Scheduling Problems, mirroring the DANIEL entry point.

Usage:
    python exp_graph_rl.py
"""

import os
import sys
import argparse
from typing import Optional

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_util import create_hgt_config
from src.HGT.hgt_trainer import HGTTrainer


def main(config_path: Optional[str] = None,
         output_dir: str = './results',
         experiment_name: str = 'exp_hgt',
         wandb_run_name: Optional[str] = None,
         **config_overrides):
    """
    Main function for HGT training.

    Args:
        config_path: (unused for HGT) path to a configuration file
        output_dir: Output directory for results
        **config_overrides: Additional configuration parameters
    """

    # Build configuration using centralized config system
    overrides = dict(config_overrides)
    if wandb_run_name is not None:
        overrides['wandb_run_name'] = wandb_run_name
    config = create_hgt_config(experiment_name, create_result_dir=True, **overrides)

    # Use the result directory from config (includes experiment name and timestamp)
    # The config system creates: result/{experiment_name}/{timestamp}/
    print(f"Models will be saved to: {config.result_dir}")

    print("Starting HGT training...")
    print(f"Configuration: {config.n_j} jobs, {config.n_m} machines, {config.max_updates} updates")

    trainer = HGTTrainer(config)
    trainer.train()

    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HGT Training for FJSP')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save outputs')
    parser.add_argument('--experiment_name', type=str, default='exp_hgt', help='Experiment name for results/wandb')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Run name for wandb')
    args, unknown = parser.parse_known_args()
    main(output_dir=args.output_dir, experiment_name=args.experiment_name, wandb_run_name=args.wandb_run_name)
