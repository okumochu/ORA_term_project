#!/usr/bin/env python3
"""
Solve all FJSSP benchmark files from benchmarks_data/FJSSP_data using OR-Tools.

This script:
1. Reads all .txt files from benchmarks_data/FJSSP_data (180 files)
2. Solves each instance using OR-Tools CP-SAT solver with 30-minute time limit
3. Saves results in three formats:
   - Individual .npy files: OR_sol/benchmark_results/solutions/<filename>.npy
   - Detailed CSV: OR_sol/benchmark_results/detailed_results.csv
   - Summary JSON: OR_sol/benchmark_results/makespan_summary.json (for gap calculation)
"""

import os
import sys
import json
import time
import collections
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.data_util import load_single_file


def _matrix_to_cp_format(
    job_length: np.ndarray,
    op_pt: np.ndarray,
) -> Tuple[List[List[List[Tuple[int, int]]]], int]:
    """
    Convert matrix format to CP-SAT format.
    
    Args:
        job_length: Array with number of operations per job [J]
        op_pt: Processing time matrix [N, M]
    
    Returns:
        jobs: List of jobs, each job is a list of operations, 
              each operation is a list of (processing_time, machine_id) tuples
        num_machines: Number of machines
    """
    num_ops, num_machines = op_pt.shape
    num_jobs = job_length.shape[0]
    jobs: List[List[List[Tuple[int, int]]]] = []
    op_idx = 0
    for j in range(num_jobs):
        job_msg: List[List[Tuple[int, int]]] = []
        for _ in range(job_length[j]):
            able_mchs = np.where(op_pt[op_idx] != 0)[0]
            op_msg = [(int(op_pt[op_idx, k]), int(k)) for k in able_mchs]
            job_msg.append(op_msg)
            op_idx += 1
        jobs.append(job_msg)
    return jobs, num_machines


def _solve_with_cp_sat(
    job_length: np.ndarray,
    op_pt: np.ndarray,
    time_limit: int,
) -> Optional[Tuple[int, float]]:
    """
    Solve FJSP instance using OR-Tools CP-SAT solver.
    
    Args:
        job_length: Array with number of operations per job [J]
        op_pt: Processing time matrix [N, M]
        time_limit: Time limit in seconds
    
    Returns:
        (makespan, solve_time) if solved, None otherwise
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise RuntimeError(
            "OR-Tools is required. Install with: pip install ortools"
        ) from exc

    jobs, num_machines = _matrix_to_cp_format(job_length, op_pt)
    num_jobs = len(jobs)
    all_jobs = range(num_jobs)
    all_machines = range(num_machines)

    model = cp_model.CpModel()
    horizon = int(np.sum(op_pt))

    intervals_per_resources = collections.defaultdict(list)
    starts = {}
    job_ends = []

    for job_id in all_jobs:
        job = jobs[job_id]
        num_tasks = len(job)
        previous_end = None
        for task_id in range(num_tasks):
            task = job[task_id]
            if not task:
                continue
            min_duration = min(alt[0] for alt in task)
            max_duration = max(alt[0] for alt in task)

            suffix_name = f"_j{job_id}_t{task_id}"
            start = model.NewIntVar(0, horizon, "start" + suffix_name)
            duration = model.NewIntVar(min_duration, max_duration, "duration" + suffix_name)
            end = model.NewIntVar(0, horizon, "end" + suffix_name)
            interval = model.NewIntervalVar(start, duration, end, "interval" + suffix_name)
            starts[(job_id, task_id)] = start

            if previous_end is not None:
                model.Add(start >= previous_end)
            previous_end = end

            if len(task) > 1:
                l_presences = []
                for alt_id, (alt_duration, machine_id) in enumerate(task):
                    alt_suffix = f"{suffix_name}_a{alt_id}"
                    presence = model.NewBoolVar("presence" + alt_suffix)
                    alt_start = model.NewIntVar(0, horizon, "start" + alt_suffix)
                    alt_end = model.NewIntVar(0, horizon, "end" + alt_suffix)
                    alt_interval = model.NewOptionalIntervalVar(
                        alt_start, alt_duration, alt_end, presence, "interval" + alt_suffix
                    )
                    model.Add(start == alt_start).OnlyEnforceIf(presence)
                    model.Add(duration == alt_duration).OnlyEnforceIf(presence)
                    model.Add(end == alt_end).OnlyEnforceIf(presence)
                    intervals_per_resources[machine_id].append(alt_interval)
                    l_presences.append(presence)
                model.AddExactlyOne(l_presences)
            else:
                machine_id = task[0][1]
                intervals_per_resources[machine_id].append(interval)

        if previous_end is not None:
            job_ends.append(previous_end)

    for machine_id in all_machines:
        intervals = intervals_per_resources[machine_id]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    makespan = model.NewIntVar(0, horizon, "makespan")
    if job_ends:
        model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    
    start_time = time.time()
    status = solver.Solve(model)
    solve_time = time.time() - start_time

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return int(solver.ObjectiveValue()), float(solve_time)


def get_benchmark_files(benchmark_dir: str) -> List[str]:
    """
    Get all benchmark files from directory, sorted by name.
    
    Args:
        benchmark_dir: Path to benchmarks_data/FJSSP_data
    
    Returns:
        List of absolute file paths
    """
    if not os.path.exists(benchmark_dir):
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")
    
    files = []
    for f in os.listdir(benchmark_dir):
        if f.endswith('.txt'):
            files.append(os.path.join(benchmark_dir, f))
    
    files.sort()
    return files


def solve_benchmarks(
    benchmark_dir: str,
    output_dir: str,
    time_limit: int = 1800,  # 30 minutes
    resume: bool = True,
):
    """
    Solve all benchmark instances and save results.
    
    Args:
        benchmark_dir: Path to benchmarks_data/FJSSP_data
        output_dir: Path to output directory (OR_sol/benchmark_results)
        time_limit: Time limit per instance in seconds (default: 1800 = 30 min)
        resume: If True, skip already-solved instances
    """
    # Create output directories
    solutions_dir = os.path.join(output_dir, "solutions")
    os.makedirs(solutions_dir, exist_ok=True)
    
    # Get all benchmark files
    benchmark_files = get_benchmark_files(benchmark_dir)
    print(f"Found {len(benchmark_files)} benchmark files")
    
    # Track results
    results = []
    makespan_dict = {}
    
    # Process each file
    for file_path in tqdm(benchmark_files, desc="Solving benchmarks", unit="file"):
        filename = os.path.basename(file_path)
        stem = os.path.splitext(filename)[0]
        
        # Check if already solved
        npy_path = os.path.join(solutions_dir, f"{stem}.npy")
        if resume and os.path.exists(npy_path):
            # Load existing solution
            try:
                data = np.load(npy_path)
                makespan, solve_time = int(data[0]), float(data[1])
                tqdm.write(f"  Skipped {filename} (already solved, makespan={makespan})")
                
                # Still need to load file info for CSV
                job_length, op_pt = load_single_file(file_path)
                if job_length is not None and op_pt is not None:
                    results.append({
                        'filename': stem,
                        'makespan': makespan,
                        'solve_time': solve_time,
                        'num_jobs': int(job_length.shape[0]),
                        'num_machines': int(op_pt.shape[1]),
                        'num_operations': int(op_pt.shape[0]),
                    })
                    makespan_dict[stem] = makespan
                continue
            except Exception as e:
                tqdm.write(f"  Warning: Could not load {npy_path}, re-solving: {e}")
        
        # Load instance
        job_length, op_pt = load_single_file(file_path)
        if job_length is None or op_pt is None:
            tqdm.write(f"  Error: Could not load {filename}, skipping")
            continue
        
        # Solve instance
        tqdm.write(f"  Solving {filename}...")
        result = _solve_with_cp_sat(job_length, op_pt, time_limit=time_limit)
        
        if result is None:
            tqdm.write(f"  Warning: No solution found for {filename}")
            continue
        
        makespan, solve_time = result
        tqdm.write(f"  Solved {filename}: makespan={makespan}, time={solve_time:.2f}s")
        
        # Save individual .npy file
        np.save(npy_path, np.array([makespan, solve_time], dtype=np.float64))
        
        # Record result
        results.append({
            'filename': stem,
            'makespan': makespan,
            'solve_time': solve_time,
            'num_jobs': int(job_length.shape[0]),
            'num_machines': int(op_pt.shape[1]),
            'num_operations': int(op_pt.shape[0]),
        })
        makespan_dict[stem] = makespan
    
    # Save detailed CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "detailed_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved detailed results to: {csv_path}")
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Total instances solved: {len(results)}")
        print(f"Average makespan: {df['makespan'].mean():.2f}")
        print(f"Average solve time: {df['solve_time'].mean():.2f}s")
        print(f"Max solve time: {df['solve_time'].max():.2f}s")
        print(f"Min solve time: {df['solve_time'].min():.2f}s")
        print("=" * 60)
    
    # Save makespan summary JSON
    if makespan_dict:
        json_path = os.path.join(output_dir, "makespan_summary.json")
        with open(json_path, 'w') as f:
            json.dump(makespan_dict, f, indent=2)
        print(f"Saved makespan summary to: {json_path}")
        print(f"Total entries in JSON: {len(makespan_dict)}")
    
    return results, makespan_dict


def main():
    """Main entry point."""
    # Define paths
    project_root = Path(__file__).parent.parent
    benchmark_dir = project_root / "benchmarks_data" / "FJSSP_data"
    output_dir = project_root / "OR_sol" / "benchmark_results"
    
    # Solve all benchmarks
    print("=" * 60)
    print("FJSSP Benchmark Solver")
    print("=" * 60)
    print(f"Benchmark directory: {benchmark_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Time limit per instance: 1800s (30 minutes)")
    print("=" * 60)
    print()
    
    results, makespan_dict = solve_benchmarks(
        str(benchmark_dir),
        str(output_dir),
        time_limit=1800,
        resume=True,
    )
    
    print("\n✓ Benchmark solving complete!")


if __name__ == "__main__":
    main()
