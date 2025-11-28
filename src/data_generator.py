"""
Data Generator for FJSP (Flexible Job Shop Scheduling Problem)

This module provides a unified data generation system using problem_config
with min/max parameters. Uses uniform distribution for all sampling.

The generator uses the same problem_config throughout the entire training session,
generating new random instances for each batch while maintaining the same problem
structure (job count, machine count, and parameter ranges).
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class CLDataGenerator:
    """
    Data Generator for FJSP instances.
    
    This generator uses a fixed problem_config dict with min/max parameters
    and generates instances using uniform distribution sampling. The same
    problem_config is used throughout the entire training session, ensuring
    consistent problem structure while providing diverse training instances.
    
    Generation Process:
    1. Sample number of operations per job (uniform within [min, max])
    2. Sample number of compatible machines per operation (uniform within [min, max])
    3. Randomly assign compatible machines for each operation
    4. Sample processing times for operation-machine pairs (uniform within [min, max])
    """
    
    def __init__(self, config):
        """
        Initialize the data generator.
        
        Args:
            config: Configuration object with problem_config dict
                   Must contain problem_config with the following keys:
                   - job_num: Number of jobs
                   - machine_num: Number of machines
                   - operation_per_job_min: Minimum operations per job
                   - operation_per_job_max: Maximum operations per job
                   - machine_per_operation_min: Minimum machines per operation
                   - machine_per_operation_max: Maximum machines per operation
                   - process_time_min: Minimum processing time
                   - process_time_max: Maximum processing time
        
        Raises:
            ValueError: If problem_config is not found in config
        """
        self.config = config
        
        # Get problem_config from config (same throughout training session)
        self.problem_config = getattr(config, 'problem_config', None)
        if self.problem_config is None:
            raise ValueError("[DataGenerator] problem_config not found in config. Please set config.problem_config.")
        
        print(f"[DataGenerator] Initialized with problem_config:")
        print(f"  Jobs: {self.problem_config['job_num']}, Machines: {self.problem_config['machine_num']}")
        print(f"  Operations per job: [{self.problem_config['operation_per_job_min']}, {self.problem_config['operation_per_job_max']}]")
        print(f"  Machines per operation: [{self.problem_config['machine_per_operation_min']}, {self.problem_config['machine_per_operation_max']}]")
        print(f"  Process time: [{self.problem_config['process_time_min']}, {self.problem_config['process_time_max']}]")
    
    def sample_env(self, num_envs: int, fixed_parameter: Optional[Dict[str, bool]] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate a batch of FJSP instances with random sampling.
        
        IMPORTANT: All environments within a batch have the same number of operations
        (to prevent PPO update errors in parallel rollout). However, different batches
        can have different operation numbers.
        
        Uses the same problem_config throughout training, but generates new
        random instances for each batch to provide diversity in training.
        
        Args:
            num_envs: Number of parallel environments to generate
            fixed_parameter: Optional dict for fixing parameters across environments
                           (kept for backward compatibility)
        
        Returns:
            dataset_job_length: List of job length arrays, one per environment
                              Each array has shape [job_num] with operation counts per job
                              All environments in batch have same total operation count
            dataset_op_pt: List of operation processing time matrices, one per environment
                         Each matrix has shape [total_ops, machine_num]
                         where op_pt[i,j] is processing time of operation i on machine j
                         (0 if machine j cannot process operation i)
        """
        if fixed_parameter is None:
            fixed_parameter = {}
        
        # ALWAYS fix operation numbers within a batch for parallel rollout
        # This prevents dimension mismatch errors in PPO updates
        fixed_parameter['op_num'] = True
        
        return self._generate_batch(num_envs, self.problem_config, fixed_parameter)
    
    def _generate_batch(self, num_envs: int, problem_params: Dict[str, Any], 
                       fixed_parameter: Dict[str, bool]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Core batch generation logic.
        
        Generates num_envs instances using uniform distribution sampling. All environments
        within a batch have the same number of operations (fixed via fixed_parameter['op_num']).
        This ensures parallel rollout works correctly with PPO.
        
        Args:
            num_envs: Number of parallel environments to generate
            problem_params: Problem parameters dict (problem_config) - same for all batches
            fixed_parameter: Dict for fixing parameters within batch
                           - 'op_num': Always True (enforced in sample_env)
                           - 'compatibility': Optional
                           - 'pt': Optional
        
        Returns:
            dataset_job_length: List of job length arrays, one per environment
                              All environments have same total operation count within batch
            dataset_op_pt: List of operation processing time matrices, one per environment
        """
        job_num = problem_params['job_num']
        machine_num = problem_params['machine_num']
        op_per_job_min = problem_params['operation_per_job_min']
        op_per_job_max = problem_params['operation_per_job_max']
        mch_per_op_min = problem_params['machine_per_operation_min']
        mch_per_op_max = problem_params['machine_per_operation_max']
        pt_min = problem_params['process_time_min']
        pt_max = problem_params['process_time_max']
        
        # Pre-sample fixed parameters if needed
        if fixed_parameter.get('op_num', False):
            # All envs have same operation structure
            fixed_job_length = self._sample_job_length(job_num, op_per_job_min, op_per_job_max)
        else:
            fixed_job_length = None
        
        if fixed_parameter.get('compatibility', False):
            # All envs have same compatibility structure
            # Generate for maximum possible operations to avoid index errors
            max_total_ops = np.sum(fixed_job_length) if fixed_job_length is not None else \
                           job_num * op_per_job_max
            fixed_compatibility = self._sample_compatibility(max_total_ops, mch_per_op_min, mch_per_op_max)
        else:
            fixed_compatibility = None
        
        if fixed_parameter.get('pt', False):
            # All envs have same processing times (structure must be fixed too)
            if fixed_job_length is None:
                fixed_job_length = self._sample_job_length(job_num, op_per_job_min, op_per_job_max)
            if fixed_compatibility is None:
                total_ops = np.sum(fixed_job_length)
                fixed_compatibility = self._sample_compatibility(total_ops, mch_per_op_min, mch_per_op_max)
            fixed_pt = self._sample_processing_times(fixed_job_length, fixed_compatibility, 
                                                     machine_num, pt_min, pt_max)
        else:
            fixed_pt = None
        
        # Generate instances for all environments
        dataset_job_length = []
        dataset_op_pt = []
        
        for env_idx in range(num_envs):
            if fixed_job_length is not None:
                job_length = fixed_job_length.copy()
            else:
                job_length = self._sample_job_length(job_num, op_per_job_min, op_per_job_max)
            
            total_ops = np.sum(job_length)
            
            if fixed_compatibility is not None:
                compatibility = fixed_compatibility[:total_ops].copy()
            else:
                compatibility = self._sample_compatibility(total_ops, mch_per_op_min, mch_per_op_max)
            
            if fixed_pt is not None:
                op_pt = fixed_pt[:total_ops].copy()
            else:
                op_pt = self._sample_processing_times(job_length, compatibility, 
                                                      machine_num, pt_min, pt_max)
            
            dataset_job_length.append(job_length)
            dataset_op_pt.append(op_pt)
        
        return dataset_job_length, dataset_op_pt
    
    def _sample_job_length(self, job_num: int, op_min: int, op_max: int) -> np.ndarray:
        """
        Sample the number of operations for each job using uniform distribution.
        
        For each job, randomly samples the number of operations from the range
        [op_min, op_max] (inclusive). If op_min == op_max, all jobs have the same
        number of operations (deterministic).
        
        Args:
            job_num: Number of jobs in the instance
            op_min: Minimum operations per job (inclusive)
            op_max: Maximum operations per job (inclusive)
        
        Returns:
            Array of job lengths [job_num], where each element is the number
            of operations for that job
        """
        if op_min == op_max:
            # Deterministic: all jobs have same number of operations
            return np.full(job_num, op_min, dtype=int)
        else:
            # Random: sample uniformly for each job
            return np.random.randint(op_min, op_max + 1, size=job_num)
    
    def _sample_compatibility(self, total_ops: int, mch_min: int, mch_max: int) -> np.ndarray:
        """
        Sample the number of compatible machines for each operation using uniform distribution.
        
        For each operation, randomly samples the number of machines that can process
        it from the range [mch_min, mch_max] (inclusive). If mch_min == mch_max,
        all operations have the same compatibility (deterministic).
        
        Args:
            total_ops: Total number of operations across all jobs
            mch_min: Minimum machines per operation (inclusive)
            mch_max: Maximum machines per operation (inclusive)
        
        Returns:
            Array of compatible machine counts [total_ops], where each element
            is the number of machines that can process that operation
        """
        if mch_min == mch_max:
            # Deterministic: all operations have same compatibility
            return np.full(total_ops, mch_min, dtype=int)
        else:
            # Random: sample uniformly for each operation
            return np.random.randint(mch_min, mch_max + 1, size=total_ops)
    
    def _sample_processing_times(self, job_length: np.ndarray, compatibility: np.ndarray,
                                 machine_num: int, pt_min: int, pt_max: int) -> np.ndarray:
        """
        Sample processing times for operation-machine pairs using uniform distribution.
        
        For each operation:
        1. Randomly selects compatible machines based on compatibility array
        2. Samples processing times for compatible machines uniformly from [pt_min, pt_max]
        3. Sets processing time to 0 for incompatible machines
        
        Args:
            job_length: Number of operations per job [job_num]
            compatibility: Number of compatible machines per operation [total_ops]
            machine_num: Total number of machines
            pt_min: Minimum processing time (inclusive)
            pt_max: Maximum processing time (inclusive)
        
        Returns:
            Processing time matrix [total_ops, machine_num]
            - op_pt[i, j] > 0: Processing time of operation i on machine j
            - op_pt[i, j] == 0: Machine j cannot process operation i
        """
        total_ops = np.sum(job_length)
        op_pt = np.zeros((total_ops, machine_num), dtype=np.float64)
        
        for op_idx in range(total_ops):
            mch_num = int(compatibility[op_idx])
            
            # Randomly select which machines can process this operation
            compatible_machines = np.random.choice(machine_num, mch_num, replace=False)
            
            # Sample processing times for compatible machines uniformly
            if pt_min == pt_max:
                # Deterministic processing time
                op_pt[op_idx, compatible_machines] = pt_min
            else:
                # Random processing times uniformly sampled
                op_pt[op_idx, compatible_machines] = np.random.randint(pt_min, pt_max + 1, size=mch_num)
        
        return op_pt


def create_cl_generator(config) -> CLDataGenerator:
    """
    Factory function to create a data generator instance.
    
    Args:
        config: Configuration object with problem_config dict
    
    Returns:
        CLDataGenerator instance initialized with the provided config
        
    Example:
        >>> generator = create_cl_generator(config)
        >>> job_lengths, op_pts = generator.sample_env(num_envs=20)
    """
    return CLDataGenerator(config)

