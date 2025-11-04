"""
HGT Trainer for Flexible Job Shop Scheduling Problem.

This module implements the training pipeline for Heterogeneous Graph Transformer (HGT)
based reinforcement learning for FJSP. Uses PPO (Proximal Policy Optimization) algorithm
with HGT-based feature extraction.

Training Process:
1. Uses fixed problem_config throughout training session (same problem structure)
2. Generates new random instances for each batch using CLDataGenerator
3. Trains using PPO with parallel environments
4. Periodically tests on benchmark files (doesn't affect training)
5. Saves model at end of training session
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys
from copy import deepcopy
from typing import List
from tqdm import tqdm
import wandb

from .hgt_network import HGT
from .hgt_env import FJSPEnvForHGTVariousOpNums, HGTEnvState
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.data_util import load_data_from_files
from src.data_generator import create_cl_generator
from src.utils.common_utils import sample_action, eval_actions, greedy_select_action
from src.utils.or_solution import ORSolutionStore
from src.utils.test_util import get_test_files_from_config



class Memory:
    """
    Memory buffer for PPO training.
    
    Stores trajectories collected during environment interaction for batch updates.
    Includes all state variables needed for HGT processing (graph structures, features)
    and action/reward sequences for PPO computation.
    """
    
    def __init__(self, gamma, gae_lambda):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Input variables for HGT (following DANIEL structure)
        self.fea_j_seq = []  # [N, tensor[sz_b, N, op_feature_dim]]
        self.fea_m_seq = []  # [N, tensor[sz_b, M, machine_feature_dim]]
        self.dynamic_pair_mask_seq = []  # [N, tensor[sz_b, J, M]]
        self.candidate_seq = []  # [N, tensor[sz_b, J]]
        self.fea_pairs_seq = []  # [N, tensor[sz_b, J, M, 8]]
        self.graph_seq = []  # [N, HeteroData] - Additional for HGT
        
        # Other variables
        self.action_seq = []  # action index with shape [N, tensor[sz_b]]
        self.reward_seq = []  # reward value with shape [N, tensor[sz_b]]
        self.val_seq = []  # state value with shape [N, tensor[sz_b]]
        self.done_seq = []  # done flag with shape [N, tensor[sz_b]]
        self.log_probs = []  # log(p_{\theta_old}(a_t|s_t)) with shape [N, tensor[sz_b]]
    
    def clear_memory(self):
        self.clear_state()
        del self.action_seq[:]
        del self.reward_seq[:]
        del self.val_seq[:]
        del self.done_seq[:]
        del self.log_probs[:]
    
    def clear_state(self):
        del self.fea_j_seq[:]
        del self.fea_m_seq[:]
        del self.dynamic_pair_mask_seq[:]
        del self.candidate_seq[:]
        del self.fea_pairs_seq[:]
        del self.graph_seq[:]
    
    def push(self, state: HGTEnvState):
        """Push a state into the memory"""
        self.fea_j_seq.append(state.fea_j_tensor)
        self.fea_m_seq.append(state.fea_m_tensor)
        self.dynamic_pair_mask_seq.append(state.dynamic_pair_mask_tensor)
        self.candidate_seq.append(state.candidate_tensor)
        self.fea_pairs_seq.append(state.fea_pairs_tensor)
        self.graph_seq.append(state.graph_tensor)
    
    def transpose_data(self):
        """Transpose the first and second dimension of collected variables"""
        t_Fea_j_seq = torch.stack(self.fea_j_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_Fea_m_seq = torch.stack(self.fea_m_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_dynamicMask_seq = torch.stack(self.dynamic_pair_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_candidate_seq = torch.stack(self.candidate_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_pairMessage_seq = torch.stack(self.fea_pairs_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_action_seq = torch.stack(self.action_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_reward_seq = torch.stack(self.reward_seq, dim=0).transpose(0, 1).flatten(0, 1)
        self.t_old_val_seq = torch.stack(self.val_seq, dim=0).transpose(0, 1)
        t_val_seq = self.t_old_val_seq.flatten(0, 1)
        t_done_seq = torch.stack(self.done_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_logprobs_seq = torch.stack(self.log_probs, dim=0).transpose(0, 1).flatten(0, 1)
        
        # Flatten graph sequence to align EXACTLY with tensor flattening order (env-major, then time)
        # Original tensors are stacked as [T, B, ...] -> transpose to [B, T, ...] -> flatten(0,1)
        # So the order is: for each env b in 0..B-1, for each t in 0..T-1
        t_graph_seq = []
        if len(self.graph_seq) > 0:
            num_envs = len(self.graph_seq[0])  # Number of environments (B)
            num_time = len(self.graph_seq)      # Number of time steps (T)
            for env_idx in range(num_envs):
                for time_step in range(num_time):
                    t_graph_seq.append(self.graph_seq[time_step][env_idx])
        
        return (t_Fea_j_seq, t_Fea_m_seq, t_dynamicMask_seq,
                t_candidate_seq, t_pairMessage_seq, t_graph_seq,
                t_action_seq, t_reward_seq, t_val_seq, t_done_seq, t_logprobs_seq)
    
    def get_gae_advantages(self):
        """Compute the generalized advantage estimates"""
        reward_arr = torch.stack(self.reward_seq, dim=0)
        t_old_val_seq = torch.stack(self.val_seq, dim=0).transpose(0, 1)
        values = t_old_val_seq.transpose(0, 1)
        len_trajectory, len_envs = reward_arr.shape
        
        advantage = torch.zeros(len_envs, device=values.device)
        advantage_seq = []
        for i in reversed(range(len_trajectory)):
            if i == len_trajectory - 1:
                delta_t = reward_arr[i] - values[i].squeeze()
            else:
                delta_t = reward_arr[i] + self.gamma * values[i + 1].squeeze() - values[i].squeeze()
            advantage = delta_t + self.gamma * self.gae_lambda * advantage
            advantage_seq.insert(0, advantage)
        
        t_advantage_seq = torch.stack(advantage_seq, dim=0).transpose(0, 1).to(torch.float32)
        v_target_seq = (t_advantage_seq + t_old_val_seq.squeeze(-1)).flatten(0, 1)
        
        # Normalization
        t_advantage_seq = (t_advantage_seq - t_advantage_seq.mean(dim=1, keepdim=True)) / \
                          (t_advantage_seq.std(dim=1, keepdim=True) + 1e-8)
        
        return t_advantage_seq.flatten(0, 1), v_target_seq


class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm implementation.
    
    Implements PPO with clipped surrogate objective for stable policy updates.
    Uses GAE (Generalized Advantage Estimation) for advantage computation.
    Supports gradient clipping for HGT stability.
    """
    
    def __init__(self, config):
        self.lr = config.lr
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.eps_clip = config.eps_clip
        self.k_epochs = config.k_epochs
        self.tau = config.tau
        
        self.ploss_coef = config.ploss_coef
        self.vloss_coef = config.vloss_coef
        self.entloss_coef = config.entloss_coef
        self.minibatch_size = config.minibatch_size
        self.max_grad_norm = getattr(config, 'max_grad_norm', 2.0)  # HGT gradient clipping
        
        self.policy = HGT(config)
        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.V_loss_2 = nn.MSELoss()
        self.device = torch.device(config.device)
        self.wandb_enabled = getattr(config, 'wandb_enabled', True)
    
    def update(self, memory, update_step=0, raw_rewards=None, raw_makespan=None):
        """
        Update policy using PPO algorithm.
        
        Args:
            memory: Memory buffer containing collected trajectories
            update_step: Current training update step (for logging)
            raw_rewards: Raw reward values for logging
            raw_makespan: Raw makespan values for logging
        """
        t_data = memory.transpose_data()
        t_advantage_seq, v_target_seq = memory.get_gae_advantages()
        
        full_batch_size = len(t_data[-1])
        num_batch = np.ceil(full_batch_size / self.minibatch_size)
        
        loss_epochs = 0
        v_loss_epochs = 0
        p_loss_epochs = 0
        
        # Accumulators for logging metrics across all minibatches
        total_v_loss = 0
        total_p_loss = 0
        total_total_loss = 0
        total_ent_loss = 0
        total_grad_norm = 0
        total_advantages_mean = 0
        total_advantages_std = 0
        total_ratios_mean = 0
        total_ratios_std = 0
        num_minibatch_updates = 0
        
        for epoch in range(self.k_epochs):
            for i in range(int(num_batch)):
                if i + 1 < num_batch:
                    start_idx = i * self.minibatch_size
                    end_idx = (i + 1) * self.minibatch_size
                else:
                    start_idx = i * self.minibatch_size
                    end_idx = full_batch_size
                
                # Get batch data
                fea_j = t_data[0][start_idx:end_idx]
                fea_m = t_data[1][start_idx:end_idx]
                dynamic_pair_mask = t_data[2][start_idx:end_idx]
                candidate = t_data[3][start_idx:end_idx]
                fea_pairs = t_data[4][start_idx:end_idx]
                graph_seq = t_data[5][start_idx:end_idx]
                
                # Forward pass
                pis, vals = self.policy(
                    fea_j=fea_j,
                    candidate=candidate,
                    fea_m=fea_m,
                    dynamic_pair_mask=dynamic_pair_mask,
                    fea_pairs=fea_pairs,
                    graph=graph_seq
                )
                
                action_batch = t_data[6][start_idx:end_idx]
                logprobs, ent_loss = eval_actions(pis, action_batch)
                ratios = torch.exp(logprobs - t_data[10][start_idx:end_idx].detach())
                
                advantages = t_advantage_seq[start_idx:end_idx]
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                
                v_loss = self.V_loss_2(vals.squeeze(1).squeeze(-1), v_target_seq[start_idx:end_idx])
                p_loss = -torch.min(surr1, surr2)
                ent_loss = -ent_loss.clone()
                loss = self.vloss_coef * v_loss + self.ploss_coef * p_loss + self.entloss_coef * ent_loss
                
                self.optimizer.zero_grad()
                loss_epochs += loss.mean().detach()
                v_loss_epochs += v_loss.mean().detach()
                p_loss_epochs += p_loss.mean().detach()
                loss.mean().backward()
                
                # Calculate gradient norms for logging (before clipping)
                grad_norm = 0
                for param in self.policy.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                grad_norm = grad_norm ** (1. / 2)
                
                # Apply gradient clipping for HGT stability
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Accumulate metrics for averaged logging
                total_v_loss += v_loss.mean().item()
                total_p_loss += p_loss.mean().item()
                total_total_loss += loss.mean().item()
                total_ent_loss += ent_loss.mean().item()
                total_grad_norm += grad_norm
                total_advantages_mean += advantages.mean().item()
                total_advantages_std += advantages.std().item()
                total_ratios_mean += ratios.mean().item()
                total_ratios_std += ratios.std().item()
                num_minibatch_updates += 1
        
        # Log averaged metrics once per update (not per minibatch)
        if hasattr(self, 'wandb_enabled') and self.wandb_enabled and num_minibatch_updates > 0:
            wandb.log({
                'update_step': update_step,
                'value_loss': total_v_loss / num_minibatch_updates,
                'policy_loss': total_p_loss / num_minibatch_updates,
                'total_loss': total_total_loss / num_minibatch_updates,
                'entropy_loss': total_ent_loss / num_minibatch_updates,
                'gradient_norm': total_grad_norm / num_minibatch_updates,
                'raw_reward': np.mean(raw_rewards) if raw_rewards is not None else 0,
                'raw_makespan': np.mean(raw_makespan) if raw_makespan is not None else 0,
                'raw_objective': np.mean(raw_makespan) if raw_makespan is not None else 0,  # Same as makespan for FJSP
                'advantages_mean': total_advantages_mean / num_minibatch_updates,
                'advantages_std': total_advantages_std / num_minibatch_updates,
                'ratios_mean': total_ratios_mean / num_minibatch_updates,
                'ratios_std': total_ratios_std / num_minibatch_updates,
            })
        
        # Soft update
        for policy_old_params, policy_params in zip(self.policy_old.parameters(), self.policy.parameters()):
            policy_old_params.data.copy_(self.tau * policy_old_params.data + (1 - self.tau) * policy_params.data)
        
        return (loss_epochs.item() / self.k_epochs, 
                v_loss_epochs.item() / self.k_epochs, 
                p_loss_epochs.item() / self.k_epochs)


class HGTTrainer:
    """HGT Trainer following DANIEL structure"""
    
    def __init__(self, config):
        self.n_j = config.n_j
        self.n_m = config.n_m
        self.config = config
        self.max_updates = config.max_updates
        self.reset_env_episode = config.reset_env_episode  # Number of episodes to use same batch
        self.num_envs = config.num_envs
        
        # Initialize data generator
        self.cl_generator = create_cl_generator(config)
        print("[HGT] Using unified data generator")
        
        # Use pre-defined result directory from config if provided, otherwise use a default
        self.result_dir = getattr(config, 'result_dir', f'./result')
        
        self.data_name = f'{self.n_j}x{self.n_m}'
        # Get model_name directly from config (should be in common_params)
        self.model_name = getattr(config, "model_name", self.data_name)
        
        # Initialize environment
        self.env = self._create_environment()
        
        # Initialize PPO
        self.ppo = PPO(config)
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)
        
        # Initialize wandb
        self.wandb_project = getattr(config, 'wandb_project', 'exp_test_small_instance')
        self.wandb_enabled = getattr(config, 'wandb_enabled', True)

        if self.wandb_enabled:
            run_name = getattr(config, 'wandb_run_name', None)
            if not run_name or not isinstance(run_name, str):
                run_name = f"HGT_{self.data_name}_{self.model_name}"
            wandb.init(
                project=self.wandb_project,
                name=run_name,
                config=vars(config)
            )

        # Testing: OR solution store for computing gap percentages
        self.or_store = ORSolutionStore()

        # Timing accumulators for performance tracking
        self._train_time_acc = 0.0
        self._test_time_acc = 0.0
    
    
    def _create_environment(self):
        """Create environment for training"""
        # Create parallel environment
        env = FJSPEnvForHGTVariousOpNums(self.n_j, self.n_m, device=self.config.device)
        
        return env
    
    def sample_training_instances(self):
        """
        Sample training instances using the data generator.
        
        Returns:
            new training instances (dataset_job_length, dataset_op_pt)
        """
        # Get fixed_parameter from config (if available)
        fixed_parameter = getattr(self.config, 'fixed_parameter', None)
        dataset_job_length, dataset_op_pt = self.cl_generator.sample_env(self.num_envs, fixed_parameter=fixed_parameter)
        return dataset_job_length, dataset_op_pt
    
    def train(self):
        """
        Train the HGT model using PPO algorithm.
        
        Training process:
        1. Sample training instances using data generator with fixed problem_config
        2. Run episodes in parallel environments
        3. Update policy using PPO
        4. Periodically test on benchmark files (doesn't affect training)
        5. Save model at end of training session
        """
        
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"model name: {self.model_name}")
        print(f"using unified data generator with fixed problem_config")
        print(f"\n--- Test Configuration ---")
        test_num = getattr(self.config, 'test_num', -1)
        if test_num == 0:
            print(f"test_data files: 0 (skipped)")
        elif test_num == -1:
            print(f"test_data files: all available")
        else:
            print(f"test_data files: {test_num} per directory")
        print(f"BenchData files: all available (always included)")
        print(f"\n--- Environment Reset Strategy ---")
        print(f"Parallel environments per batch: {self.num_envs}")
        print(f"Same batch will be reused for: {self.reset_env_episode} episodes")
        print(f"New batch sampled every: {self.reset_env_episode} episodes (at updates 0, {self.reset_env_episode}, {self.reset_env_episode*2}, ...)")
        print(f"Total training episodes: {self.max_updates}")
        print(f"Total batches to be sampled: ~{self.max_updates // self.reset_env_episode + 1}")
        print("\n")
        
        self.train_st = time.time()
        
        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'):
            ep_st = time.time()
            
            # Resample training data every reset_env_episode episodes
            if i_update % self.reset_env_episode == 0:
                # Sample new training instances
                dataset_job_length, dataset_op_pt = self.sample_training_instances()
                state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)
                tqdm.write(f"[Update {i_update}] Sampled new batch of {self.num_envs} environments")
            else:
                # Reset to initial state but keep the same problem instances
                state = self.env.reset()
            
            ep_rewards = -deepcopy(self.env.init_quality)
            
            while True:
                # Store state
                self.memory.push(state)
                
                with torch.no_grad():
                    pi_envs, vals_envs = self.ppo.policy_old(
                        fea_j=state.fea_j_tensor,  # [sz_b, N, op_feature_dim]
                        candidate=state.candidate_tensor,  # [sz_b, J]
                        fea_m=state.fea_m_tensor,  # [sz_b, M, machine_feature_dim]
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
                        fea_pairs=state.fea_pairs_tensor,  # [sz_b, J, M, 8]
                        graph=state.graph_tensor  # List of HeteroData for batched processing
                    )
                
                # Sample action
                action_envs, action_logprob_envs = sample_action(pi_envs)
                
                # State transition
                state, reward, done = self.env.step(actions=action_envs.cpu().numpy())
                ep_rewards += reward
                reward = torch.from_numpy(reward).to(self.ppo.device)
                
                # Collect transition
                self.memory.done_seq.append(torch.from_numpy(done).to(self.ppo.device))
                self.memory.reward_seq.append(reward)
                self.memory.action_seq.append(action_envs)
                self.memory.log_probs.append(action_logprob_envs)
                self.memory.val_seq.append(vals_envs.squeeze(1))
                
                if done.all():
                    break
            
            # Run PPO update
            loss, v_loss, p_loss = self.ppo.update(self.memory, update_step=i_update, 
                                                  raw_rewards=ep_rewards, raw_makespan=self.env.current_makespan)
            
            self.memory.clear_memory()
            
            mean_rewards_all_env = np.mean(ep_rewards)
            mean_makespan_all_env = np.mean(self.env.current_makespan)
            
            # Test model (separate from validation, doesn't affect model saving)
            test_episode = getattr(self.config, 'test_episode', None)
            if test_episode and (i_update + 1) % test_episode == 0:
                test_files = self._discover_test_files()
                if test_files:
                    test_st = time.time()
                    self.test_on_file_list()
                    self._test_time_acc += (time.time() - test_st)
                else:
                    tqdm.write('Test skipped (no test files provided)')
            
            ep_et = time.time()
            
            # Print training info
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    i_update + 1, mean_rewards_all_env, mean_makespan_all_env, loss, ep_et - ep_st))
        
        self.train_et = time.time()
        
        # Save model at end of training session
        self.save_model()
        tqdm.write(f'Model saved at end of training: {self.model_name}.pth')
        
        # Log training-only wall time (excluding test time)
        import numpy as _np
        effective_train_time = (self.train_et - self.train_st) - self._test_time_acc
        if self.wandb_enabled:
            wandb.log({'train/training_wall_clock_s': effective_train_time})

        # Finish wandb logging
        if self.wandb_enabled:
            wandb.finish()
    
    def _evaluate_on_file_group(self, file_list: List[str], metric_prefix: str, compute_gap: bool = True):
        """
        Evaluate policy on a group of test files from the same directory.
        
        Args:
            file_list: List of file paths (all from same directory)
            metric_prefix: Prefix for wandb metrics ('test' only)
            compute_gap: If True, compute gap vs OR value (for test data)
        
        Returns:
            dict with {file_path: makespan_or_gap} results
        """
        from src.utils.data_util import load_single_file
        
        if not file_list:
            return {}
        
        results = {}
        
        # Extract directory name from first file
        first_file = file_list[0]
        dir_name = os.path.basename(os.path.dirname(first_file))
        
        for file_path in tqdm(file_list, desc=f"Processing {dir_name}", leave=False, disable=len(file_list) <= 1):
            # Construct full path for test data
            # test -> benchmarks_data/<rel_path> (supports BenchData/ and test_data/)
            full_path = os.path.join('benchmarks_data', file_path)
            
            # Load the single file
            job_length, op_pt = load_single_file(full_path)
            
            if job_length is None or op_pt is None:
                continue
            
            # Create a temporary environment for this single instance
            temp_env = FJSPEnvForHGTVariousOpNums(self.n_j, self.n_m, device=self.config.device)
            temp_env.set_initial_data([job_length], [op_pt])
            
            state = temp_env.reset()
            
            while True:
                with torch.no_grad():
                    batch_idx = torch.from_numpy(~temp_env.done_flag)
                    filtered_graphs = [state.graph_tensor[i] for i in range(len(state.graph_tensor)) if batch_idx[i]]
                    pi, _ = self.ppo.policy(
                        fea_j=state.fea_j_tensor[batch_idx],
                        candidate=state.candidate_tensor[batch_idx],
                        fea_m=state.fea_m_tensor[batch_idx],
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor[batch_idx],
                        fea_pairs=state.fea_pairs_tensor[batch_idx],
                        graph=filtered_graphs
                    )
                
                action = greedy_select_action(pi)
                state, _, done = temp_env.step(action.cpu().numpy())
                
                if done.all():
                    break
            
            makespan = float(temp_env.current_makespan[0])
            
            if compute_gap:
                # Compute gap vs OR (for test data)
                # Use relative path from benchmarks_data for OR lookup
                relative_path = file_path
                or_val = self.or_store.lookup_or_value(relative_path)
                result_value = float('nan') if or_val is None else self.or_store.compute_gap_percent(makespan, or_val)
            else:
                # Return makespan directly
                result_value = makespan
            
            results[file_path] = result_value

            # Log per-file metrics for test data
            if self.wandb_enabled:
                filename = os.path.splitext(os.path.basename(file_path))[0]
                metric_name = f"{metric_prefix}_file/{dir_name}/{filename}"
                wandb.log({metric_name: result_value}, commit=False)
        
        # Log group mean for test data
        if results and self.wandb_enabled:
            vals = [v for v in results.values() if isinstance(v, (int, float))]
            if vals:
                group_mean = np.mean(vals)
                metric_suffix = "gap" if compute_gap else "makespan"
                wandb.log({f"{metric_prefix}/{dir_name}_{metric_suffix}": group_mean}, commit=False)
        
        return results
    
    def test_on_file_list(self):
        """
        Test policy on grouped test files using greedy strategy.
        
        Loads files grouped by directory, evaluates them, and logs results to wandb.
        Uses gap percentages vs OR solutions for test data.
        """
        test_files = self._discover_test_files()
        if not test_files:
            return None

        self.ppo.policy.eval()
        all_results = []
        
        # Process each group of files (from same directory) with progress bar
        for file_group in tqdm(test_files, desc="Testing", leave=False):
            group_results = self._evaluate_on_file_group(file_group, 'test', compute_gap=True)
            all_results.extend(list(group_results.values()))

        self.ppo.policy.train()
        return None

    def _discover_test_files(self):
        """Discover all benchmark test files using utility function."""
        return get_test_files_from_config(self.config)

    
    
    
    def save_model(self):
        """Save the model"""
        save_obj = {
            'state_dict': self.ppo.policy.state_dict(),
            'config': vars(self.config) if hasattr(self.config, '__dict__') else self.config
        }
        torch.save(save_obj, f"{self.result_dir}/{self.model_name}.pth")
    
    def load_model(self, model_path=None):
        """Load the trained model"""
        if model_path is None:
            model_path = f"{self.result_dir}/{self.model_name}.pth"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        loaded = torch.load(model_path, map_location=self.ppo.device if hasattr(self, 'ppo') else 'cuda')
        if isinstance(loaded, dict) and 'state_dict' in loaded:
            self.ppo.policy.load_state_dict(loaded['state_dict'])
            self.saved_config = loaded.get('config', None)
        else:
            self.ppo.policy.load_state_dict(loaded)

    # Removed CL generation helpers; data must be prepared before training


def main(config):
    """Main function"""
    trainer = HGTTrainer(config)
    trainer.train()


if __name__ == '__main__':
    pass
