"""
Data utility functions for FJSP instance handling.

This module provides functions for:
- Converting between matrix and text formats for FJSP instances
- Loading FJSP instances from files
- Generating and saving FJSP instances to files

The text format follows the standard FJSP file format (.fjs files).
The matrix format is used internally for efficient processing.
"""

import numpy as np
import re
import random
import time
import os
import sys
from typing import List, Dict, Any, Tuple


def matrix_to_text(job_length, op_pt, op_per_mch):
    """
    Convert matrix form of FJSP data into standard text format (.fjs file format).
    
    Args:
        job_length: Array with number of operations in each job [J]
        op_pt: Processing time matrix [N, M]
               - op_pt[i,j] > 0: Processing time of operation i on machine j
               - op_pt[i,j] == 0: Machine j cannot process operation i
        op_per_mch: Average number of compatible machines per operation (for header)
    
    Returns:
        List of strings representing the instance in .fjs text format
        First line: "{job_num}\t{machine_num}\t{op_per_mch}"
        Subsequent lines: One per job with operation details
    """
    n_j = job_length.shape[0]
    n_op, n_m = op_pt.shape
    text = [f'{n_j}\t{n_m}\t{op_per_mch}']

    op_idx = 0
    for j in range(n_j):
        line = f'{job_length[j]}'
        for _ in range(job_length[j]):
            use_mch = np.where(op_pt[op_idx] != 0)[0]
            line = line + ' ' + str(use_mch.shape[0])
            for k in use_mch:
                line = line + ' ' + str(k + 1) + ' ' + str(op_pt[op_idx][k])
            op_idx += 1

        text.append(line)

    return text


def text_to_matrix(text):
    """
    Convert standard text format (.fjs file) into matrix form.
    
    Parses the .fjs file format and converts it to numpy arrays for efficient processing.
    
    Args:
        text: List of strings representing the instance in .fjs format
             First line should be: "{job_num}\t{machine_num}\t{op_per_mch}"
             Subsequent lines: One per job with operation details
    
    Returns:
        job_length: Array with number of operations per job [J]
        op_pt: Processing time matrix [N, M]
               - op_pt[i,j] > 0: Processing time of operation i on machine j
               - op_pt[i,j] == 0: Machine j cannot process operation i
    """
    n_j = int(re.findall(r'\d+\.?\d*', text[0])[0])
    n_m = int(re.findall(r'\d+\.?\d*', text[0])[1])

    job_length = np.zeros(n_j, dtype='int32')
    op_pt = []

    for i in range(n_j):
        content = np.array([int(float(s)) for s in re.findall(r'\d+\.?\d*', text[i + 1])])
        job_length[i] = content[0]

        idx = 1
        for j in range(content[0]):
            op_pt_row = np.zeros(n_m, dtype='int32')
            mch_num = content[idx]
            next_idx = idx + 2 * mch_num + 1
            for k in range(mch_num):
                mch_idx = content[idx + 2 * k + 1]
                pt = content[idx + 2 * k + 2]
                op_pt_row[mch_idx - 1] = pt

            idx = next_idx
            op_pt.append(op_pt_row)

    op_pt = np.array(op_pt)

    return job_length, op_pt


def load_single_file(filepath):
    """
    Load a single FJSP instance from a .fjs file.
    
    Args:
        filepath: Path to the .fjs file (absolute or relative)
    
    Returns:
        Tuple of (job_length, op_pt) for the instance
        Returns (None, None) if file doesn't exist or cannot be read
    """
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} does not exist")
        return None, None
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    job_length, op_pt = text_to_matrix(lines)
    return job_length, op_pt


def load_data_from_files(directory):
    """
    Load all FJSP instance files from a directory.
    
    Scans the directory recursively and loads all .fjs files, sorting them
    by filename to ensure consistent ordering.
    
    Args:
        directory: Path to directory containing .fjs files
    
    Returns:
        Tuple of (dataset_job_length, dataset_op_pt)
        - dataset_job_length: List of job_length arrays, one per file
        - dataset_op_pt: List of op_pt matrices, one per file
    """
    if not os.path.exists(directory):
        return [], []

    dataset_job_length = []
    dataset_op_pt = []
    for root, dirs, files in os.walk(directory):
        # sort files by index
        files.sort(key=lambda s: int(re.findall(r"\d+", s)[0]))
        files.sort(key=lambda s: int(re.findall(r"\d+", s)[-1]))
        for f in files:
            # print(f)
            g = open(os.path.join(root, f), 'r').readlines()
            job_length, op_pt = text_to_matrix(g)
            dataset_job_length.append(job_length)
            dataset_op_pt.append(op_pt)
    return dataset_job_length, dataset_op_pt


def pack_data_from_config(data_source, test_data):
    """
    Load multiple datasets from a data source directory.
    
    This function loads data from legacy data sources (SD1/SD2/BenchData).
    Note: This function may not be actively used in current training pipeline.
    
    Args:
        data_source: Source directory name (e.g., 'SD1', 'SD2', 'BenchData')
        test_data: List of dataset names to load
    
    Returns:
        List of tuples: (data, name) where data is (job_length, op_pt) and name is the dataset name
    """
    data_list = []
    for data_name in test_data:
        data_path = f'./data/{data_source}/{data_name}'
        data_list.append((load_data_from_files(data_path), data_name))
    return data_list


def generate_data_to_files(directory, config, seed=None):
    """
    Generate FJSP instances and save them to files using problem_config.
    
    Uses the same data generation logic as CLDataGenerator but saves instances
    to .fjs files instead of returning them in memory. Useful for generating
    test datasets that can be reused across experiments.
    
    Args:
        directory: Directory path where .fjs files will be saved
        config: Configuration object with problem_config dict
               Must contain problem_config with job_num, machine_num, and parameter ranges
        seed: Optional random seed for reproducible generation (default: None = random)
    
    Note:
        If directory already exists and cover_data_flag is False, generation is skipped.
        Set config.cover_data_flag = True to regenerate existing data.
    """
    # Get problem_config from config
    problem_config = getattr(config, 'problem_config', None)
    if problem_config is None:
        raise ValueError("problem_config not found in config. Please set config.problem_config.")
    
    n_j = problem_config['job_num']
    n_m = problem_config['machine_num']
    batch_size = getattr(config, 'data_size', 100)
    
    # Get parameters from problem_config
    op_per_job_min = problem_config['operation_per_job_min']
    op_per_job_max = problem_config['operation_per_job_max']
    mch_per_op_min = problem_config['machine_per_operation_min']
    mch_per_op_max = problem_config['machine_per_operation_max']
    pt_min = problem_config['process_time_min']
    pt_max = problem_config['process_time_max']
    
    # Set seed only if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Create filename pattern (no nested directory)
    filename = '{}x{}'.format(n_j, n_m)

    print("-" * 25 + "Data Generation Setting" + "-" * 25)
    print(f"seed : {seed if seed is not None else 'None (random)'}")
    print(f"data size : {batch_size}")
    print(f"directory : {directory}")
    print(f"filename pattern : {filename}")
    print(f"jobs : {n_j}")
    print(f"machines : {n_m}")
    print(f"operations per job : [{op_per_job_min}, {op_per_job_max}]")
    print(f"machines per operation : [{mch_per_op_min}, {mch_per_op_max}]")
    print(f"processing time : [{pt_min}, {pt_max}]")
    print("-" * 50)

    # Use directory directly (no nested subdirectory)
    path = directory

    if (not os.path.exists(path)) or getattr(config, 'cover_data_flag', False):
        if not os.path.exists(path):
            os.makedirs(path)

        for idx in range(batch_size):
            # Generate instance using CL-style generation (inline)
            # Sample job lengths
            if op_per_job_min == op_per_job_max:
                job_length = np.full(n_j, op_per_job_min, dtype=int)
            else:
                job_length = np.random.randint(op_per_job_min, op_per_job_max + 1, size=n_j)
            
            total_ops = np.sum(job_length)
            
            # Sample compatibility (machines per operation)
            if mch_per_op_min == mch_per_op_max:
                compatibility = np.full(total_ops, mch_per_op_min, dtype=int)
            else:
                compatibility = np.random.randint(mch_per_op_min, mch_per_op_max + 1, size=total_ops)
            
            # Sample processing times
            op_pt = np.zeros((total_ops, n_m), dtype=np.float64)
            
            for op_idx in range(total_ops):
                mch_num = int(compatibility[op_idx])
                
                # Randomly select which machines can process this operation
                compatible_machines = np.random.choice(n_m, mch_num, replace=False)
                
                # Sample processing times for compatible machines
                if pt_min == pt_max:
                    op_pt[op_idx, compatible_machines] = pt_min
                else:
                    op_pt[op_idx, compatible_machines] = np.random.randint(pt_min, pt_max + 1, size=mch_num)
            
            # Calculate average compatibility for the instance
            op_per_mch = np.mean(compatibility)
            
            # Convert to text format
            lines_doc = matrix_to_text(job_length, op_pt, op_per_mch)

            # Save to file
            file_path = os.path.join(path, f'{filename}_{str(idx + 1).zfill(3)}.fjs')
            with open(file_path, 'w') as doc:
                for line in lines_doc:
                    print(line, file=doc)
            
            if (idx + 1) % 10 == 0:
                print(f"Generated {idx + 1}/{batch_size} instances...")
        
        print(f"Successfully generated {batch_size} instances in {path}")
    else:
        print("the data already exists...")