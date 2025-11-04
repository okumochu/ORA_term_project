"""
Test utility functions for discovering and managing test files.

This module provides functions to discover and organize test files from
benchmarks_data directory. Test files are organized by directory structure:
- BenchData/: Contains benchmark instances with known optimal solutions
- test_data/: Contains generated test instances with solution files

Test files are used during training for periodic evaluation (does not affect
model selection or training process).
"""
import os
from typing import List


def discover_test_files(test_num: int = -1) -> List[List[str]]:
    """
    Discover test files from BenchData/ and test_data/ directories.
    Always includes all BenchData files, limits test_data files based on test_num.
    
    Args:
        test_num: Number of files to use from test_data/ (-1 for all files, 0 to skip test_data)
    
    Returns:
        List of file groups, where each group is a list of file paths from the same subdirectory
    """
    groups = []
    
    # BenchData structure: multiple subdirs containing .fjs
    # Always include all BenchData files (as requested)
    bd_root = os.path.join('benchmarks_data', 'BenchData')
    if os.path.isdir(bd_root):
        for d in sorted(os.listdir(bd_root)):
            full_d = os.path.join(bd_root, d)
            if os.path.isdir(full_d):
                files = [os.path.join('BenchData', d, f) for f in sorted(os.listdir(full_d)) if f.endswith('.fjs')]
                if files:
                    groups.append(files)
    
    # test_data structure - limit based on test_num
    if test_num != 0:  # Skip test_data if test_num is 0
        td_root = os.path.join('benchmarks_data', 'test_data')
        if os.path.isdir(td_root):
            for d in sorted(os.listdir(td_root)):
                full_d = os.path.join(td_root, d)
                if os.path.isdir(full_d):
                    files = [os.path.join('test_data', d, f) for f in sorted(os.listdir(full_d)) if f.endswith('.fjs')]
                    if files:
                        # Apply test_num limit if not -1 (which means all files)
                        if test_num != -1:
                            files = files[:test_num]
                        groups.append(files)
            # top-level .fjs under test_data
            top = [os.path.join('test_data', f) for f in sorted(os.listdir(td_root)) if f.endswith('.fjs')]
            if top:
                # Apply test_num limit if not -1 (which means all files)
                if test_num != -1:
                    top = top[:test_num]
                groups.append(top)
    
    return groups


def get_test_files_from_config(config) -> List[List[str]]:
    """
    Get test files using configuration parameters.
    
    Args:
        config: Configuration object with test_num attribute
    
    Returns:
        List of file groups for testing
    """
    test_num = getattr(config, 'test_num', -1)
    
    return discover_test_files(test_num)
