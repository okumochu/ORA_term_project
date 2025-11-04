"""
HGT Environment for Flexible Job Shop Scheduling Problem.

This module implements the FJSP environment for HGT training, providing:
- State representation as heterogeneous graphs (operations and machines)
- Action space: job-machine pairs for scheduling decisions
- Reward computation: based on makespan improvement (lower bound or direct)
- Parallel processing: supports multiple environments simultaneously

The environment converts FJSP instances into heterogeneous graph structures
where operations and machines are nodes, and edges represent precedence
constraints and machine-operation compatibility.
"""

import numpy as np
import numpy.ma as ma
import copy
import sys
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn.norm import LayerNorm as PygLayerNorm


class HGTEnvState:
    """Environment state for HGT environment following DANIEL structure"""
    
    def __init__(self):
        self.fea_j_tensor = None  # Operation features [sz_b, N, fea_dim]
        self.fea_m_tensor = None  # Machine features [sz_b, M, fea_dim]
        self.dynamic_pair_mask_tensor = None  # Dynamic pair mask [sz_b, J, M]
        self.candidate_tensor = None  # Candidate operations [sz_b, J]
        self.fea_pairs_tensor = None  # Pair features [sz_b, J, M, 8]
        self.graph_tensor = None  # HeteroData graph for HGT processing
    
    def update(self, fea_j, fea_m, dynamic_pair_mask, candidate, fea_pairs, graph, device):
        """Update state tensors"""
        self.fea_j_tensor = torch.from_numpy(fea_j).float().to(device)
        self.fea_m_tensor = torch.from_numpy(fea_m).float().to(device)
        self.dynamic_pair_mask_tensor = torch.from_numpy(dynamic_pair_mask).float().to(device)
        self.candidate_tensor = torch.from_numpy(candidate).long().to(device)
        self.fea_pairs_tensor = torch.from_numpy(fea_pairs).float().to(device)
        self.graph_tensor = graph


class FJSPEnvForHGTVariousOpNums:
    """
    FJSP Environment for HGT with parallel processing following DANIEL structure.
    Handles multiple environments simultaneously with different operation numbers.
    """
    
    def __init__(self, n_j, n_m, device):
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.device = device
        
        # Feature dimensions (following DANIEL structure)
        self.op_fea_dim = 10  # Same as DANIEL
        self.mch_fea_dim = 8  # Same as DANIEL
        
        # Graph structure caching for efficiency
        self.cached_graph_structure = None  # Will store edge indices
        self.graph_structure_initialized = False
        
        # Initialize state
        self.old_state = HGTEnvState()
        
        # Will be set when data is loaded
        self.number_of_envs = 0
        self.max_number_of_ops = 0
        self.env_number_of_ops = None
        self.job_length = None
        self.op_pt = None
        self.true_op_pt = None
        self.process_relation = None
        self.compatible_op = None
        self.compatible_mch = None
        self.job_first_op_id = None
        self.job_last_op_id = None
        self.candidate = None
        
        # Parallel processing variables
        self.done_flag = None
        self.current_makespan = None
        self.step_count = 0
        
    def set_static_properties(self):
        """Define static properties for parallel processing"""
        self.env_idxs = np.arange(self.number_of_envs)
        self.env_job_idx = self.env_idxs.repeat(self.number_of_jobs).reshape(self.number_of_envs, self.number_of_jobs)
        
        # [E, N] - mask for dummy nodes
        self.mask_dummy_node = np.full(shape=[self.number_of_envs, self.max_number_of_ops],
                                       fill_value=False, dtype=bool)
        
        cols = np.arange(self.max_number_of_ops)
        self.mask_dummy_node[cols >= self.env_number_of_ops[:, None]] = True
        
        a = self.mask_dummy_node[:, :, np.newaxis]
        self.dummy_mask_fea_j = np.tile(a, (1, 1, self.op_fea_dim))
        
        self.flag_exist_dummy_node = ~(self.env_number_of_ops == self.max_number_of_ops).all()
    
    def set_initial_data(self, job_length_list, op_pt_list):
        """Set initial data from data_util.py format like DANIEL"""
        self.number_of_envs = len(job_length_list)
        self.job_length = np.array(job_length_list)
        self.number_of_jobs = job_length_list[0].shape[0]
        
        # Update number of machines from actual data
        self.number_of_machines = op_pt_list[0].shape[1]
        
        # Handle various operation numbers like DANIEL
        self.env_number_of_ops = np.array([op_pt_list[k].shape[0] for k in range(self.number_of_envs)])
        self.max_number_of_ops = np.max(self.env_number_of_ops)
        
        self.set_static_properties()
        
        self.virtual_job_length = np.copy(self.job_length)
        self.virtual_job_length[:, -1] += self.max_number_of_ops - self.env_number_of_ops
        
        # Pad operations to handle different numbers of operations
        self.op_pt = np.array([np.pad(op_pt_list[k],
                                      ((0, self.max_number_of_ops - self.env_number_of_ops[k]),
                                       (0, 0)),
                                      'constant', constant_values=0)
                               for k in range(self.number_of_envs)]).astype(np.float64)
        
        # Normalize processing times like DANIEL
        self.pt_lower_bound = np.min(self.op_pt)
        self.pt_upper_bound = np.max(self.op_pt)
        self.true_op_pt = np.copy(self.op_pt)
        self.op_pt = (self.op_pt - self.pt_lower_bound) / (self.pt_upper_bound - self.pt_lower_bound + 1e-8)
        
        # Process relation (which operations can be processed by which machines)
        self.process_relation = (self.op_pt != 0)
        self.reverse_process_relation = ~self.process_relation
        self.compatible_op = np.sum(self.process_relation, 2)  # [E, N]
        self.compatible_mch = np.sum(self.process_relation, 1)  # [E, M]
        
        # Job structure
        head_op_id = np.zeros((self.number_of_envs, 1))
        self.job_first_op_id = np.concatenate([head_op_id, np.cumsum(self.job_length, axis=1)[:, :-1]], axis=1).astype('int')
        self.job_last_op_id = self.job_first_op_id + self.job_length - 1
        self.job_last_op_id[:, -1] = self.env_number_of_ops - 1
        
        self.initial_vars()
        
        self.op_pt = ma.array(self.op_pt, mask=self.reverse_process_relation)
        self.op_mean_pt = np.mean(self.op_pt, axis=2).data
        
        # Use filled() to handle masked values (padding operations) safely
        self.op_min_pt = np.min(self.op_pt, axis=-1).filled(0).astype(np.float64)
        self.op_max_pt = np.max(self.op_pt, axis=-1).filled(0).astype(np.float64)
        self.pt_span = self.op_max_pt - self.op_min_pt
        
        self.mch_min_pt = np.max(self.op_pt, axis=1).data
        
        self.op_ct_lb = copy.deepcopy(self.op_min_pt)
        
        for k in range(self.number_of_envs):
            for i in range(self.number_of_jobs):
                self.op_ct_lb[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1] = np.cumsum(
                    self.op_ct_lb[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1])
        
        self.op_match_job_left_op_nums = np.array([np.repeat(self.job_length[k],
                                                             repeats=self.virtual_job_length[k])
                                                   for k in range(self.number_of_envs)])
        self.job_remain_work = []
        for k in range(self.number_of_envs):
            self.job_remain_work.append(
                [np.sum(self.op_mean_pt[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1])
                 for i in range(self.number_of_jobs)])
        
        self.op_match_job_remain_work = np.array([np.repeat(self.job_remain_work[k], repeats=self.virtual_job_length[k])
                                                  for k in range(self.number_of_envs)])
        
        self.construct_op_features()
        
        # Shape reward
        self.init_quality = np.max(self.op_ct_lb, axis=1)
        self.max_endTime = self.init_quality
        
        # Old
        self.mch_current_available_op_nums = np.copy(self.compatible_mch)
        self.candidate_pt = np.array([self.op_pt[k][self.candidate[k]] for k in range(self.number_of_envs)])
        
        self.dynamic_pair_mask = (self.candidate_pt == 0)
        self.candidate_process_relation = np.copy(self.dynamic_pair_mask)
        self.mch_current_available_jc_nums = np.sum(~self.candidate_process_relation, axis=1)
        
        self.mch_mean_pt = np.mean(self.op_pt, axis=1).filled(0)
        
        
        # Construct machine features [E, M, 8]
        self.construct_mch_features()
        
        self.construct_pair_features()
        
        # Initialize graph structure (edges) once - will be reused every step
        self._initialize_graph_structure()
        
        self.old_state.update(self.fea_j, self.fea_m,
                              self.dynamic_pair_mask, self.candidate,
                              self.fea_pairs, self._build_batch_hetero_graphs(), self.device)
        
        # Old record
        self.old_op_ct_lb = np.copy(self.op_ct_lb)
        self.old_op_match_job_left_op_nums = np.copy(self.op_match_job_left_op_nums)
        self.old_op_match_job_remain_work = np.copy(self.op_match_job_remain_work)
        self.old_init_quality = np.copy(self.init_quality)
        self.old_candidate_pt = np.copy(self.candidate_pt)
        self.old_candidate_process_relation = np.copy(self.candidate_process_relation)
        self.old_mch_current_available_op_nums = np.copy(self.mch_current_available_op_nums)
        self.old_mch_current_available_jc_nums = np.copy(self.mch_current_available_jc_nums)
        
        # State
        self.state = copy.deepcopy(self.old_state)
        return self.state
    
    def initial_vars(self):
        """Initialize variables for parallel processing"""
        self.step_count = 0
        self.done_flag = np.full(shape=(self.number_of_envs,), fill_value=0, dtype=bool)
        self.current_makespan = np.full(self.number_of_envs, float("-inf"))
        
        self.mch_queue = np.full(shape=[self.number_of_envs, self.number_of_machines,
                                        self.max_number_of_ops + 1], fill_value=-99, dtype=int)
        self.mch_queue_len = np.zeros((self.number_of_envs, self.number_of_machines), dtype=int)
        self.mch_queue_last_op_id = np.zeros((self.number_of_envs, self.number_of_machines), dtype=int)
        self.op_ct = np.zeros((self.number_of_envs, self.max_number_of_ops))
        
        self.mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_remain_work = np.zeros((self.number_of_envs, self.number_of_machines))
        
        self.mch_waiting_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_working_flag = np.zeros((self.number_of_envs, self.number_of_machines))
        
        self.next_schedule_time = np.zeros(self.number_of_envs)
        self.candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs))
        
        self.true_op_ct = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.true_candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs))
        self.true_mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))
        
        self.candidate = np.copy(self.job_first_op_id)
        
        self.mask = np.full(shape=(self.number_of_envs, self.number_of_jobs), fill_value=0, dtype=bool)
        
        self.op_scheduled_flag = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.op_waiting_time = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.op_remain_work = np.zeros((self.number_of_envs, self.max_number_of_ops))
        
        self.op_available_mch_nums = np.copy(self.compatible_op) / self.number_of_machines
        self.pair_free_time = np.zeros((self.number_of_envs, self.number_of_jobs,
                                        self.number_of_machines))
        self.remain_process_relation = np.copy(self.process_relation)
    
    
    def construct_op_features(self):
        """Construct operation features following DANIEL structure"""
        self.fea_j = np.stack((self.op_scheduled_flag,
                               self.op_ct_lb,
                               self.op_min_pt,
                               self.pt_span,
                               self.op_mean_pt,
                               self.op_waiting_time,
                               self.op_remain_work,
                               self.op_match_job_left_op_nums,
                               self.op_match_job_remain_work,
                               self.op_available_mch_nums), axis=2)
        
        self.norm_operation_features()
    
    def norm_operation_features(self):
        """Normalize operation features with proper dummy node exclusion"""
        if self.flag_exist_dummy_node:
            # Zero out dummy nodes first
            self.fea_j[self.dummy_mask_fea_j] = 0
            
            # Calculate statistics excluding dummy nodes
            num_actual_ops = self.env_number_of_ops[:, np.newaxis]  # [E, 1]
            
            # Sum features only over actual operations
            sum_fea_j = np.sum(self.fea_j, axis=1, keepdims=True)  # [E, 1, fea_dim]
            mean_fea_j = sum_fea_j / np.maximum(num_actual_ops[:, :, np.newaxis], 1e-8)
            
            # Calculate std excluding dummy nodes
            actual_op_mask = (~self.mask_dummy_node)[:, :, np.newaxis]  # [E, N, 1]
            diff_sq = np.where(actual_op_mask, (self.fea_j - mean_fea_j) ** 2, 0)
            var_fea_j = np.sum(diff_sq, axis=1, keepdims=True) / np.maximum(num_actual_ops[:, :, np.newaxis], 1e-8)
            std_fea_j = np.sqrt(var_fea_j)
        else:
            # No dummy nodes - standard normalization
            mean_fea_j = np.mean(self.fea_j, axis=1, keepdims=True)
            std_fea_j = np.std(self.fea_j, axis=1, keepdims=True)
        
        # Normalize all features
        self.fea_j = (self.fea_j - mean_fea_j) / (std_fea_j + 1e-8)
    
    def construct_mch_features(self):
        """Construct machine features following DANIEL structure"""
        self.fea_m = np.stack((self.mch_current_available_jc_nums,
                               self.mch_current_available_op_nums,
                               self.mch_min_pt,
                               self.mch_mean_pt,
                               self.mch_waiting_time,
                               self.mch_remain_work,
                               self.mch_free_time,
                               self.mch_working_flag), axis=2)
        
        self.norm_machine_features()
    
    def norm_machine_features(self):
        """Normalize machine features using clean standard normalization"""
        mean_fea_m = np.mean(self.fea_m, axis=1, keepdims=True)
        std_fea_m = np.std(self.fea_m, axis=1, keepdims=True)
        self.fea_m = (self.fea_m - mean_fea_m) / (std_fea_m + 1e-8)
    
    def construct_pair_features(self):
        """Construct pair features following DANIEL structure"""
        remain_op_pt = ma.array(self.op_pt, mask=~self.remain_process_relation)
        
        chosen_op_max_pt = np.expand_dims(self.op_max_pt[self.env_job_idx, self.candidate], axis=-1)
        
        max_remain_op_pt = np.max(np.max(remain_op_pt, axis=1, keepdims=True), axis=2, keepdims=True) \
            .filled(0 + 1e-8)
        
        mch_max_remain_op_pt = np.max(remain_op_pt, axis=1, keepdims=True). \
            filled(0 + 1e-8)
        
        pair_max_pt = np.max(np.max(self.candidate_pt, axis=1, keepdims=True),
                             axis=2, keepdims=True) + 1e-8
        
        mch_max_candidate_pt = np.max(self.candidate_pt, axis=1, keepdims=True) + 1e-8
        
        pair_wait_time = self.op_waiting_time[self.env_job_idx, self.candidate][:, :,
                         np.newaxis] + self.mch_waiting_time[:, np.newaxis, :]
        
        chosen_job_remain_work = np.expand_dims(self.op_match_job_remain_work
                                                [self.env_job_idx, self.candidate],
                                                axis=-1) + 1e-8
        
        self.fea_pairs = np.stack((self.candidate_pt,
                                   self.candidate_pt / chosen_op_max_pt,
                                   self.candidate_pt / mch_max_candidate_pt,
                                   self.candidate_pt / max_remain_op_pt,
                                   self.candidate_pt / mch_max_remain_op_pt,
                                   self.candidate_pt / pair_max_pt,
                                   self.candidate_pt / chosen_job_remain_work,
                                   pair_wait_time), axis=-1)
    
    def _build_batch_hetero_graphs(self):
        """Build batch of heterogeneous graphs efficiently using cached structure
        
        IMPORTANT: This method correctly excludes dummy/padded operations from the graph.
        Only actual operations [:num_actual_ops] are included, ensuring that:
        1. HGT message passing only operates on real nodes
        2. Global pooling (in hgt_network.py) only aggregates over actual operations
        3. No dummy vectors contaminate the learned representations
        """
        batch_graphs = []
        
        for env_idx in range(self.number_of_envs):
            # Get CACHED graph structure
            if not self.graph_structure_initialized:
                raise RuntimeError("Graph structure not initialized. Call _initialize_graph_structure() first.")
            
            cached = self.cached_graph_structure[env_idx]
            num_actual_ops = cached['num_actual_ops']
            
            # Only update node features (the part that changes each step)
            # CRITICAL: [:num_actual_ops] slice excludes dummy/padded operations
            op_x = torch.from_numpy(self.fea_j[env_idx][:num_actual_ops]).float().to(self.device)
            machine_x = torch.from_numpy(self.fea_m[env_idx]).float().to(self.device)
            
            # Create HeteroData with CACHED edges (including bidirectional op-machine edges)
            data = HeteroData()
            data['op'].x = op_x
            data['machine'].x = machine_x
            data['op', 'precedes', 'op'].edge_index = cached['precedence_edges']
            data['machine', 'can_process', 'op'].edge_index = cached['op_machine_edges']
            data['op', 'compatible_with', 'machine'].edge_index = cached['reverse_op_machine_edges']
            
            batch_graphs.append(data)
        
        return batch_graphs
    
    def _get_precedence_edges(self, env_idx):
        """Get precedence edges between operations for specific environment"""
        src_ops = []
        dst_ops = []
        
        for job_id in range(self.number_of_jobs):
            first_op = self.job_first_op_id[env_idx, job_id]
            last_op = self.job_last_op_id[env_idx, job_id]
            
            # Add precedence edges within job
            for i in range(first_op, last_op):
                src_ops.append(i)
                dst_ops.append(i + 1)
        
        return [src_ops, dst_ops]
    
    def _get_operation_machine_edges(self, env_idx):
        """Get edges between machines and operations they can process for specific environment"""
        machine_ids = []
        op_ids = []
        
        for op_id in range(self.env_number_of_ops[env_idx]):
            for m_id in range(self.number_of_machines):
                if self.op_pt[env_idx, op_id, m_id] > 0:  # Machine can process this operation
                    machine_ids.append(m_id)
                    op_ids.append(op_id)
        
        return [machine_ids, op_ids]
    
    def _get_reverse_operation_machine_edges(self, env_idx):
        """Get reverse edges (operation -> machine compatibility) for bidirectional information flow"""
        op_ids = []
        machine_ids = []
        
        for op_id in range(self.env_number_of_ops[env_idx]):
            for m_id in range(self.number_of_machines):
                if self.op_pt[env_idx, op_id, m_id] > 0:  # Operation compatible with this machine
                    op_ids.append(op_id)
                    machine_ids.append(m_id)
        
        return [op_ids, machine_ids]
    
    def _initialize_graph_structure(self):
        """
        Initialize and cache static graph structure (edges).
        Called once per episode - edges never change, only node features update.
        """
        self.cached_graph_structure = []
        
        for env_idx in range(self.number_of_envs):
            # Precedence edges (STATIC - based on job structure)
            precedence_edges = self._get_precedence_edges(env_idx)
            if len(precedence_edges[0]) > 0:
                precedence_edge_index = torch.tensor(precedence_edges, dtype=torch.long, device=self.device)
            else:
                precedence_edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            
            # Machine -> Operation edges (STATIC - based on compatibility)
            op_machine_edges = self._get_operation_machine_edges(env_idx)
            if len(op_machine_edges[0]) > 0:
                op_machine_edge_index = torch.tensor(op_machine_edges, dtype=torch.long, device=self.device)
            else:
                op_machine_edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            
            # Operation -> Machine reverse edges (STATIC - for bidirectional flow)
            reverse_op_machine_edges = self._get_reverse_operation_machine_edges(env_idx)
            if len(reverse_op_machine_edges[0]) > 0:
                reverse_op_machine_edge_index = torch.tensor(reverse_op_machine_edges, dtype=torch.long, device=self.device)
            else:
                reverse_op_machine_edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            
            # Store edge structure for this environment
            self.cached_graph_structure.append({
                'precedence_edges': precedence_edge_index,
                'op_machine_edges': op_machine_edge_index,
                'reverse_op_machine_edges': reverse_op_machine_edge_index,
                'num_actual_ops': self.env_number_of_ops[env_idx]
            })
        
        self.graph_structure_initialized = True
    
    def calculate_reward(self):
        """
        Calculate reward using lower bound (lb) reward type.
        
        Returns:
            reward: numpy array of shape (number_of_envs,)
        
        Reward calculation:
        - reward = previous_max_endTime - current_max_endTime
        - With gamma=1, cumulative reward = initial_lower_bound - makespan
        - Reward is given at every step (dense rewards)
        """
        current_lb = np.max(self.op_ct_lb, axis=1)
        # Dense rewards: every step
        reward = self.max_endTime - current_lb
        self.max_endTime = current_lb
        return reward
    
    def reset(self):
        """Reset environment to initial state"""
        self.initial_vars()
        self.op_ct_lb = np.copy(self.old_op_ct_lb)
        self.op_match_job_left_op_nums = np.copy(self.old_op_match_job_left_op_nums)
        self.op_match_job_remain_work = np.copy(self.old_op_match_job_remain_work)
        self.init_quality = np.copy(self.old_init_quality)
        self.max_endTime = self.init_quality
        self.candidate_pt = np.copy(self.old_candidate_pt)
        self.candidate_process_relation = np.copy(self.old_candidate_process_relation)
        self.mch_current_available_op_nums = np.copy(self.old_mch_current_available_op_nums)
        self.mch_current_available_jc_nums = np.copy(self.old_mch_current_available_jc_nums)
        
        # State
        self.state = copy.deepcopy(self.old_state)
        return self.state
    
    def step(self, actions):
        """Execute one step in the environment following DANIEL structure"""
        self.incomplete_env_idx = np.where(self.done_flag == 0)[0]
        self.number_of_incomplete_envs = int(self.number_of_envs - np.sum(self.done_flag))
        # Filter actions to only incomplete environments
        actions_incomplete = actions[self.incomplete_env_idx]
        chosen_job = actions_incomplete // self.number_of_machines
        chosen_mch = actions_incomplete % self.number_of_machines
        chosen_op = self.candidate[self.incomplete_env_idx, chosen_job]
        
        if (self.reverse_process_relation[self.incomplete_env_idx, chosen_op, chosen_mch]).any():
            print(f'HGT_Env.py Error from choosing action: Op {chosen_op} can\'t be processed by Mch {chosen_mch}')
            sys.exit()
        
        self.step_count += 1
        
        # Update candidate
        candidate_add_flag = (chosen_op != self.job_last_op_id[self.incomplete_env_idx, chosen_job])
        self.candidate[self.incomplete_env_idx, chosen_job] += candidate_add_flag
        self.mask[self.incomplete_env_idx, chosen_job] = (1 - candidate_add_flag)
        
        self.mch_queue[
            self.incomplete_env_idx, chosen_mch, self.mch_queue_len[self.incomplete_env_idx, chosen_mch]] = chosen_op
        
        self.mch_queue_len[self.incomplete_env_idx, chosen_mch] += 1
        
        # [E]
        chosen_op_st = np.maximum(self.candidate_free_time[self.incomplete_env_idx, chosen_job],
                                  self.mch_free_time[self.incomplete_env_idx, chosen_mch])
        
        self.op_ct[self.incomplete_env_idx, chosen_op] = chosen_op_st + self.op_pt[
            self.incomplete_env_idx, chosen_op, chosen_mch]
        self.candidate_free_time[self.incomplete_env_idx, chosen_job] = self.op_ct[self.incomplete_env_idx, chosen_op]
        self.mch_free_time[self.incomplete_env_idx, chosen_mch] = self.op_ct[self.incomplete_env_idx, chosen_op]
        
        true_chosen_op_st = np.maximum(self.true_candidate_free_time[self.incomplete_env_idx, chosen_job],
                                       self.true_mch_free_time[self.incomplete_env_idx, chosen_mch])
        self.true_op_ct[self.incomplete_env_idx, chosen_op] = true_chosen_op_st + self.true_op_pt[
            self.incomplete_env_idx, chosen_op, chosen_mch]
        self.true_candidate_free_time[self.incomplete_env_idx, chosen_job] = self.true_op_ct[
            self.incomplete_env_idx, chosen_op]
        self.true_mch_free_time[self.incomplete_env_idx, chosen_mch] = self.true_op_ct[
            self.incomplete_env_idx, chosen_op]
        
        self.current_makespan[self.incomplete_env_idx] = np.maximum(self.current_makespan[self.incomplete_env_idx],
                                                                    self.true_op_ct[
                                                                        self.incomplete_env_idx, chosen_op])
        
        for k, j in enumerate(self.incomplete_env_idx):
            if candidate_add_flag[k]:
                self.candidate_pt[j, chosen_job[k]] = self.op_pt[j, chosen_op[k] + 1]
                self.candidate_process_relation[j, chosen_job[k]] = self.reverse_process_relation[j, chosen_op[k] + 1]
            else:
                self.candidate_process_relation[j, chosen_job[k]] = 1
        
        candidateFT_for_compare = np.expand_dims(self.candidate_free_time, axis=2)
        mchFT_for_compare = np.expand_dims(self.mch_free_time, axis=1)
        self.pair_free_time = np.maximum(candidateFT_for_compare, mchFT_for_compare)
        
        pair_free_time = self.pair_free_time[self.incomplete_env_idx]
        schedule_matrix = ma.array(pair_free_time, mask=self.candidate_process_relation[self.incomplete_env_idx])
        
        self.next_schedule_time[self.incomplete_env_idx] = np.min(
            schedule_matrix.reshape(self.number_of_incomplete_envs, -1), axis=1).data
        
        self.remain_process_relation[self.incomplete_env_idx, chosen_op] = 0
        self.op_scheduled_flag[self.incomplete_env_idx, chosen_op] = 1
        
        self.mch_queue_last_op_id[self.incomplete_env_idx, chosen_mch] = chosen_op
        
        diff = self.op_ct[self.incomplete_env_idx, chosen_op] - self.op_ct_lb[self.incomplete_env_idx, chosen_op]
        for k, j in enumerate(self.incomplete_env_idx):
            self.op_ct_lb[j][chosen_op[k]:self.job_last_op_id[j, chosen_job[k]] + 1] += diff[k]
            self.op_match_job_left_op_nums[j][
            self.job_first_op_id[j, chosen_job[k]]:self.job_last_op_id[j, chosen_job[k]] + 1] -= 1
            self.op_match_job_remain_work[j][
            self.job_first_op_id[j, chosen_job[k]]:self.job_last_op_id[j, chosen_job[k]] + 1] -= \
                self.op_mean_pt[j, chosen_op[k]]
        
        self.op_waiting_time = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.op_waiting_time[self.env_job_idx, self.candidate] = \
            (1 - self.mask) * np.maximum(np.expand_dims(self.next_schedule_time, axis=1)
                                         - self.candidate_free_time, 0) + self.mask * self.op_waiting_time[
                self.env_job_idx, self.candidate]
        
        self.op_remain_work = np.maximum(self.op_ct -
                                         np.expand_dims(self.next_schedule_time, axis=1), 0)
        
        self.construct_op_features()
        
        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)
        
        self.dynamic_pair_mask[self.incomplete_env_idx] = np.logical_or(self.dynamic_pair_mask[self.incomplete_env_idx],
                                                                        np.array([pair_free_time[k] > self.next_schedule_time[j]
                                                                                 for k, j in enumerate(self.incomplete_env_idx)]))
        
        
        self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
        self.mch_current_available_op_nums[self.incomplete_env_idx] -= self.process_relation[
            self.incomplete_env_idx, chosen_op]
        
        mch_free_duration = np.expand_dims(self.next_schedule_time[self.
                                           incomplete_env_idx], axis=1) - self.mch_free_time[self.incomplete_env_idx]
        mch_free_flag = mch_free_duration < 0
        self.mch_working_flag[self.incomplete_env_idx] = mch_free_flag + 0
        self.mch_waiting_time[self.incomplete_env_idx] = (1 - mch_free_flag) * mch_free_duration
        
        self.mch_remain_work[self.incomplete_env_idx] = np.maximum(-mch_free_duration, 0)
        
        self.construct_mch_features()
        
        self.construct_pair_features()
        
        # Calculate reward using the reward calculation method
        reward = self.calculate_reward()
        
        self.state.update(self.fea_j, self.fea_m,
                          self.dynamic_pair_mask, self.candidate,
                          self.fea_pairs, self._build_batch_hetero_graphs(), self.device)
        self.done_flag = self.done()
        
        return self.state, np.array(reward), self.done_flag
    
    def done(self):
        """Check if all environments are done"""
        return self.step_count >= self.env_number_of_ops
    
    def get_makespan(self):
        """Get current makespan for all environments"""
        return self.current_makespan
