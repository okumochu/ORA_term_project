"""
OR (Optimal/Reference) Solution Store for FJSP benchmarks.

This module provides functionality to load and lookup optimal or reference
makespan values for benchmark instances. These values are used to compute
gap percentages during testing.

The store supports multiple formats:
- .npy files: Contains makespan values as numpy arrays
- Text files: Contains makespan values as integers
- CSV files: Mapping files for BenchData instances
"""

import os
import re
import numpy as np
from typing import Optional, Dict


def _read_first_int_from_file(path: str) -> Optional[int]:
    """
    Try to extract an integer OR value from a text-like file.
    Strategy: scan lines, pick the first positive integer token.
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                nums = re.findall(r"\d+", line)
                for tok in nums:
                    try:
                        val = int(tok)
                        if val > 0:
                            return val
                    except Exception:
                        continue
    except Exception:
        return None
    return None


def _read_or_value_from_npy(path: str) -> Optional[int]:
    """
    Try to extract an integer OR value from a .npy file.
    Strategy: load numpy array and take the first element (makespan value).
    """
    try:
        data = np.load(path)
        if isinstance(data, np.ndarray) and len(data) > 0:
            # OR solution files contain [makespan, solve_time]
            # We want the makespan (first element)
            val = int(data[0])
            if val > 0:
                return val
    except Exception:
        return None
    return None


def _load_csv_mapping(csv_path: str) -> Dict[str, int]:
    """
    Load CSV mapping for BenchData solutions.
    Expected format: benchname,dataname,filename,lb,ub
    """
    mapping: Dict[str, int] = {}
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            # Skip header line
            for line in lines[1:]:
                parts = [p.strip() for p in line.strip().split(',') if p.strip()]
                if len(parts) >= 5:  # benchname,dataname,filename,lb,ub
                    filename = parts[2]  # filename column
                    try:
                        ub_value = int(parts[4])  # ub (upper bound) column
                        # Use filename as key (without extension)
                        inst = os.path.splitext(filename)[0]
                        mapping[inst.lower()] = ub_value
                    except (ValueError, IndexError):
                        continue
    except Exception:
        pass
    return mapping


class ORSolutionStore:
    """
    Store for loading and looking up OR (optimal/reference) makespan values.
    
    Loads OR solutions from `benchmarks_data/or_solution` directory and provides
    lookup functionality to find optimal makespan for test instances. The store
    is robust to filename prefixes (e.g., SD1_, SD2_, CL_) and matches instances
    by their canonical name (e.g., MK01, LA06).
    
    Supports multiple file formats:
    - .npy files: NumPy arrays with makespan values
    - Text files: Plain text files with integer makespan values
    - CSV files: Mapping files for BenchData instances
    """

    def __init__(self, base_dir: str = 'benchmarks_data/or_solution'):
        self.base_dir = base_dir
        self._cache: Dict[str, int] = {}
        self._loaded = False

    def _index_directory(self):
        if self._loaded:
            return

        # CSV mapping support (optional)
        csv_path = os.path.join(self.base_dir, 'or_solutions.csv')
        if os.path.isfile(csv_path):
            self._cache.update(_load_csv_mapping(csv_path))
        
        # BenchData CSV mapping support
        benchdata_csv_path = os.path.join('benchmarks_data', 'BenchData', 'BenchDataSolution.csv')
        if os.path.isfile(benchdata_csv_path):
            self._cache.update(_load_csv_mapping(benchdata_csv_path))

        # Walk all files under or_solution and index by stem token (e.g., MK01)
        for root, _, files in os.walk(self.base_dir):
            for fname in files:
                # Skip CSV already handled
                if fname.lower().endswith('.csv'):
                    continue
                # Try to derive an instance key from the filename
                stem = os.path.splitext(fname)[0]
                # Extract canonical token like MK01 or LA12 from stem
                token = self._extract_token(stem)
                if not token:
                    # Fallback: use complete stem
                    token = stem
                token = token.lower()

                full_path = os.path.join(root, fname)
                
                # Try to read OR value from file
                val = None
                if fname.lower().endswith('.npy'):
                    val = _read_or_value_from_npy(full_path)
                else:
                    val = _read_first_int_from_file(full_path)
                
                if val is not None:
                    # Prefer explicit CSV mapping if present; otherwise fill
                    if token not in self._cache:
                        self._cache[token] = val

        self._loaded = True

    @staticmethod
    def _extract_token(stem: str) -> Optional[str]:
        """
        Try to extract instance token patterns such as MK01, LA11, etc.
        Returns the matched token or None.
        """
        m = re.search(r"\b(MK\d{1,2})\b", stem, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.search(r"\b(LA\d{1,2})\b", stem, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return None

    def lookup_or_value(self, test_file_path: str) -> Optional[int]:
        """
        Given a test file path, attempt to find its OR value.
        Strategy: 
        1. For BenchData files: use CSV upper bounds (ub) only
        2. For test_data files: use npy files only
        """
        self._index_directory()
        
        if 'BenchData' in test_file_path:
            # BenchData: Use CSV ub only, no fallback
            fname = os.path.basename(test_file_path)
            stem = os.path.splitext(fname)[0]
            return self._cache.get(stem.lower())
        
        elif 'test_data' in test_file_path:
            # test_data: Use npy files only
            path_parts = test_file_path.split(os.sep)
            if len(path_parts) >= 3 and path_parts[0] == 'test_data':
                dir_name = path_parts[1]
                file_name = path_parts[2]
                
                if '_' in dir_name:
                    prefix, size = dir_name.split('_', 1)
                    full_dir = os.path.join('benchmarks_data', 'test_data', dir_name)
                    if os.path.isdir(full_dir):
                        all_files = sorted([f for f in os.listdir(full_dir) if f.endswith('.fjs')])
                        if file_name in all_files:
                            file_index = all_files.index(file_name)
                            or_file = os.path.join(self.base_dir, prefix, f'solution_{size}.npy')
                            if os.path.exists(or_file):
                                try:
                                    data = np.load(or_file)
                                    if isinstance(data, np.ndarray) and file_index < len(data):
                                        return int(data[file_index][0])
                                except Exception:
                                    pass
        
        return None

    @staticmethod
    def compute_gap_percent(makespan: float, or_value: float) -> float:
        """gap% = (makespan - OR) / OR * 100"""
        if or_value is None or or_value <= 0:
            return float('nan')
        return (float(makespan) - float(or_value)) / float(or_value) * 100.0

 