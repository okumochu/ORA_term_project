import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import random
import time
import subprocess
import os
import sys
from copy import deepcopy
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
import wandb

from .DANIEL_network import DANIEL
from .DANIEL_env import FJSPEnvForVariousOpNums
from src.utils.data_util import load_data_from_files
from src.data_generator import create_cl_generator  # Use unified data generator
from src.utils.common_utils import sample_action, eval_actions, greedy_select_action
from src.utils.or_solution import ORSolutionStore
from src.utils.test_util import get_test_files_from_config



class Memory:
    def __init__(self, gamma, gae_lambda):
        """
            the memory used for collect trajectories for PPO training
        :param gamma: discount factor
        :param gae_lambda: GAE parameter for PPO algorithm
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # input variables of DANIEL
        self.fea_j_seq = []  # [N, tensor[sz_b, N, 8]]
        self.op_mask_seq = []  # [N, tensor[sz_b, N, 3]]
        self.fea_m_seq = []  # [N, tensor[sz_b, M, 6]]
        self.mch_mask_seq = []  # [N, tensor[sz_b, M, M]]
        self.dynamic_pair_mask_seq = []  # [N, tensor[sz_b, J, M]]
        self.comp_idx_seq = []  # [N, tensor[sz_b, M, M, J]]
        self.candidate_seq = []  # [N, tensor[sz_b, J]]
        self.fea_pairs_seq = []  # [N, tensor[sz_b, J]]

        # other variables
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
        del self.op_mask_seq[:]
        del self.fea_m_seq[:]
        del self.mch_mask_seq[:]
        del self.dynamic_pair_mask_seq[:]
        del self.comp_idx_seq[:]
        del self.candidate_seq[:]
        del self.fea_pairs_seq[:]

    def push(self, state):
        """
            push a state into the memory
        :param state: the MDP state
        :return:
        """
        self.fea_j_seq.append(state.fea_j_tensor)
        self.op_mask_seq.append(state.op_mask_tensor)
        self.fea_m_seq.append(state.fea_m_tensor)
        self.mch_mask_seq.append(state.mch_mask_tensor)
        self.dynamic_pair_mask_seq.append(state.dynamic_pair_mask_tensor)
        self.comp_idx_seq.append(state.comp_idx_tensor)
        self.candidate_seq.append(state.candidate_tensor)
        self.fea_pairs_seq.append(state.fea_pairs_tensor)

    def transpose_data(self):
        """
            transpose the first and second dimension of collected variables
        """
        # 14
        t_Fea_j_seq = torch.stack(self.fea_j_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_op_mask_seq = torch.stack(self.op_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_Fea_m_seq = torch.stack(self.fea_m_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_mch_mask_seq = torch.stack(self.mch_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_dynamicMask_seq = torch.stack(self.dynamic_pair_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_Compete_m_seq = torch.stack(self.comp_idx_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_candidate_seq = torch.stack(self.candidate_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_pairMessage_seq = torch.stack(self.fea_pairs_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_action_seq = torch.stack(self.action_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_reward_seq = torch.stack(self.reward_seq, dim=0).transpose(0, 1).flatten(0, 1)
        self.t_old_val_seq = torch.stack(self.val_seq, dim=0).transpose(0, 1)
        t_val_seq = self.t_old_val_seq.flatten(0, 1)
        t_done_seq = torch.stack(self.done_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_logprobs_seq = torch.stack(self.log_probs, dim=0).transpose(0, 1).flatten(0, 1)

        return t_Fea_j_seq, t_op_mask_seq, t_Fea_m_seq, t_mch_mask_seq, t_dynamicMask_seq, \
               t_Compete_m_seq, t_candidate_seq, t_pairMessage_seq, \
               t_action_seq, t_reward_seq, t_val_seq, t_done_seq, t_logprobs_seq

    def get_gae_advantages(self):
        """
            Compute the generalized advantage estimates
        :return: advantage sequences, state value sequence
        """

        reward_arr = torch.stack(self.reward_seq, dim=0)
        values = self.t_old_val_seq.transpose(0, 1)
        len_trajectory, len_envs = reward_arr.shape

        advantage = torch.zeros(len_envs, device=values.device)
        advantage_seq = []
        for i in reversed(range(len_trajectory)):

            if i == len_trajectory - 1:
                delta_t = reward_arr[i] - values[i]
            else:
                delta_t = reward_arr[i] + self.gamma * values[i + 1] - values[i]
            advantage = delta_t + self.gamma * self.gae_lambda * advantage
            advantage_seq.insert(0, advantage)

        # [sz_b, N]
        t_advantage_seq = torch.stack(advantage_seq, dim=0).transpose(0, 1).to(torch.float32)

        # [sz_b, N]
        v_target_seq = (t_advantage_seq + self.t_old_val_seq).flatten(0, 1)

        # normalization
        t_advantage_seq = (t_advantage_seq - t_advantage_seq.mean(dim=1, keepdim=True)) \
                          / (t_advantage_seq.std(dim=1, keepdim=True) + 1e-8)

        return t_advantage_seq.flatten(0, 1), v_target_seq


class PPO:
    def __init__(self, config):
        """
            The implementation of PPO algorithm
        :param config: a package of parameters
        """
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

        self.policy = DANIEL(config)
        self.policy_old = deepcopy(self.policy)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.V_loss_2 = nn.MSELoss()
        self.device = torch.device(config.device)
        self.wandb_enabled = getattr(config, 'wandb_enabled', True)

    def update(self, memory, update_step=0, raw_rewards=None, raw_makespan=None, 
              validate_fn=None, validation_callback_fn=None):
        '''
        :param memory: data used for PPO training
        :param update_step: current update step for logging
        :param raw_rewards: raw rewards from episode
        :param raw_makespan: raw makespan from episode
        :param validate_fn: Optional validation function to call after each epoch
        :param validation_callback_fn: Optional callback with validation results (epoch, results)
        :return: total_loss and critic_loss
        '''

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

            # Split into multiple batches of updates due to memory limitations
            for i in range(int(num_batch)):
                if i + 1 < num_batch:
                    start_idx = i * self.minibatch_size
                    end_idx = (i + 1) * self.minibatch_size
                else:
                    # the last batch
                    start_idx = i * self.minibatch_size
                    end_idx = full_batch_size

                pis, vals = self.policy(fea_j=t_data[0][start_idx:end_idx],
                                        op_mask=t_data[1][start_idx:end_idx],
                                        candidate=t_data[6][start_idx:end_idx],
                                        fea_m=t_data[2][start_idx:end_idx],
                                        mch_mask=t_data[3][start_idx:end_idx],
                                        comp_idx=t_data[5][start_idx:end_idx],
                                        dynamic_pair_mask=t_data[4][start_idx:end_idx],
                                        fea_pairs=t_data[7][start_idx:end_idx])

                action_batch = t_data[8][start_idx: end_idx]
                logprobs, ent_loss = eval_actions(pis, action_batch)
                ratios = torch.exp(logprobs - t_data[12][start_idx: end_idx].detach())

                advantages = t_advantage_seq[start_idx: end_idx]
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                v_loss = self.V_loss_2(vals.squeeze(1), v_target_seq[start_idx: end_idx])
                p_loss = - torch.min(surr1, surr2)
                ent_loss = - ent_loss.clone()
                loss = self.vloss_coef * v_loss + self.ploss_coef * p_loss + self.entloss_coef * ent_loss

                self.optimizer.zero_grad()
                loss_epochs += loss.mean().detach()
                v_loss_epochs += v_loss.mean().detach()
                p_loss_epochs += p_loss.mean().detach()
                loss.mean().backward()
                
                # Calculate gradient norms for logging
                grad_norm = 0
                for param in self.policy.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                grad_norm = grad_norm ** (1. / 2)
                
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
            
            # Validate after each epoch (for TSCL)
            # For NAIVE: validate every epoch
            # For WINDOW: validate only on the last epoch
            if validate_fn is not None and validation_callback_fn is not None:
                # Check if we should validate this epoch
                should_validate_epoch = validation_callback_fn(epoch, None, self.k_epochs, check_only=True)
                if should_validate_epoch:
                    validation_results = validate_fn()
                    validation_callback_fn(epoch, validation_results, self.k_epochs, check_only=False)
        
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
        
        # soft update
        for policy_old_params, policy_params in zip(self.policy_old.parameters(), self.policy.parameters()):
            policy_old_params.data.copy_(self.tau * policy_old_params.data + (1 - self.tau) * policy_params.data)

        return loss_epochs.item() / self.k_epochs, v_loss_epochs.item() / self.k_epochs, p_loss_epochs.item() / self.k_epochs


class DANIELTrainer:
    def __init__(self, config):

        self.n_j = config.n_j
        self.n_m = config.n_m
        self.config = config
        self.max_updates = config.max_updates
        self.reset_env_episode = config.reset_env_episode  # Number of episodes to use same batch
        self.num_envs = config.num_envs

        # Initialize data generator
        self.cl_generator = create_cl_generator(config)
        print("[DANIEL] Using unified data generator")

        # Use pre-defined result directory from config if provided
        self.result_dir = getattr(config, 'result_dir', f'./result')

        if torch.cuda.is_available() and config.device == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        self.data_name = f'{self.n_j}x{self.n_m}'
        # Get model_name directly from config (should be in common_params)
        self.model_name = getattr(config, "model_name", self.data_name)

        self.device = torch.device(config.device)
        # Always use lower-bound reward with dense step=1; rely on env defaults
        self.env = FJSPEnvForVariousOpNums(self.n_j, self.n_m, device=self.device)
        
        # Load validation files list (if provided)
        self.validation_files = getattr(config, 'validation_files', [])

        self.ppo = PPO(config)
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)
        
        # Initialize wandb
        self.wandb_project = getattr(config, 'wandb_project', 'exp_test_small_instance')
        self.wandb_enabled = getattr(config, 'wandb_enabled', True)
   
        if self.wandb_enabled:
            run_name = getattr(config, 'wandb_run_name', None)
            if not run_name or not isinstance(run_name, str):
                run_name = f"DANIEL_{self.data_name}_{self.model_name}"
            wandb.init(
                project=self.wandb_project,
                name=run_name,
                config=vars(config)
            )

        # Testing: OR solution store
        self.or_store = ORSolutionStore()

        # Timing accumulators
        self._train_time_acc = 0.0
        self._val_time_acc = 0.0
        self._test_time_acc = 0.0
        self._last_train_tick = None

        # Validation AUC data
        self._val_overall_history = []
        self._test_group_histories = {}

        # Removed on-the-fly CL generation; validation data should be generated beforehand

    def train(self):
        """
            train the model following the config
        """
        # Removed custom file-based logging; rely on stdout and W&B
        self.record = float('inf')

        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"model name: {self.model_name}")
        print(f"using unified data generator: True")
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
                dataset_job_length, dataset_op_pt = self.sample_training_instances()
                self._assert_uniform_operation_counts(dataset_job_length)
                state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)
                tqdm.write(f"[Update {i_update}] Sampled new batch of {self.num_envs} environments")
            else:
                # Reset to initial state but keep the same problem instances
                state = self.env.reset()

            ep_rewards = - deepcopy(self.env.init_quality)

            while True:

                # state store
                self.memory.push(state)
                with torch.no_grad():

                    pi_envs, vals_envs = self.ppo.policy_old(fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
                                                             op_mask=state.op_mask_tensor,  # [sz_b, N, N]
                                                             candidate=state.candidate_tensor,  # [sz_b, J]
                                                             fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
                                                             mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
                                                             comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
                                                             dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
                                                             fea_pairs=state.fea_pairs_tensor)  # [sz_b, J, M]

                # sample the action
                action_envs, action_logprob_envs = sample_action(pi_envs)

                # state transition
                state, reward, done = self.env.step(actions=action_envs.cpu().numpy())
                ep_rewards += reward
                reward = torch.from_numpy(reward).to(self.ppo.device)

                # collect the transition
                self.memory.done_seq.append(torch.from_numpy(done).to(self.ppo.device))
                self.memory.reward_seq.append(reward)
                self.memory.action_seq.append(action_envs)
                self.memory.log_probs.append(action_logprob_envs)
                self.memory.val_seq.append(vals_envs.squeeze(1))

                if done.all():
                    break

            # Standard PPO update (no validation during k_epochs)
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
                    group_means = self.test_on_file_list()
                    if group_means:
                        for dir_name, group_mean in group_means.items():
                            if dir_name not in self._test_group_histories:
                                self._test_group_histories[dir_name] = []
                            self._test_group_histories[dir_name].append(group_mean)
                    self._test_time_acc += (time.time() - test_st)
                else:
                    tqdm.write('Test skipped (no test files provided)')

            ep_et = time.time()
            
            # print the reward, makespan, loss and training time of the current episode
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    i_update + 1, mean_rewards_all_env, mean_makespan_all_env, loss, ep_et - ep_st))

        self.train_et = time.time()

        # Log training-only wall time and per-test-directory AUCs (exclude validation and testing time)
        import numpy as _np
        effective_train_time = (self.train_et - self.train_st) - (self._val_time_acc + self._test_time_acc)
        if self.wandb_enabled:
            wandb.log({'train/training_wall_clock_s': effective_train_time})
            for dir_name, history in self._test_group_histories.items():
                if len(history) >= 2:
                    x = _np.arange(len(history))
                    dir_auc = float(_np.trapz(history, x))
                else:
                    dir_auc = float('nan')
                wandb.log({f'train/{dir_name}_AUC': dir_auc}, commit=False)

        # Save model once at the end of training
        self.save_model()

        # Finish wandb logging
        if self.wandb_enabled:
            wandb.finish()

    def sample_training_instances(self):
        """
        Sample training instances using the data generator.
        
        Returns:
            Tuple of (dataset_job_length, dataset_op_pt)
        """
        return self.cl_generator.sample_env(self.num_envs)

    @staticmethod
    def _assert_uniform_operation_counts(job_lengths: List[np.ndarray]) -> None:
        totals = {int(np.sum(length)) for length in job_lengths}
        if len(totals) > 1:
            raise ValueError(
                f"Parallel rollout batch contains mixed operation counts: {sorted(totals)}. "
                "Regenerate training data so every environment has the same number of operations."
            )
    
    def _validate_tscl_all_tasks(self, current_timestep: int) -> Dict[int, float]:
        """
        Validate on all TSCL validation sets and return average makespan for each.
        
        Args:
            current_timestep: Current training timestep
        
        Returns:
            Dictionary mapping task_idx to average makespan
        """
        self.ppo.policy.eval()
        
        task_makespans = {}
        
        env_config = {'n_j': self.n_j, 'n_m': self.n_m, 'device': self.device}

        # Progress over all TSCL validation tasks
        for task_idx in tqdm(range(self.tscl_learner.n_tasks), desc=f"TSCL validating tasks (t={current_timestep})", leave=False):
            val_job_lengths, val_op_pts = self.tscl_learner.get_validation_instances(task_idx)
            
            makespans = batched_inference(
                policy=self.ppo.policy,
                batch_job_lengths=val_job_lengths,
                batch_op_pts=val_op_pts,
                env_class=FJSPEnvForVariousOpNums,
                env_config=env_config,
                is_hgt=False
            )
            
            # Compute average makespan for this task
            avg_makespan = compute_validation_score(makespans)
            task_makespans[task_idx] = avg_makespan
            
            # Prepare names for potential logging outside (no per-file or makespan logs here)
            task_name = self.tscl_learner.get_complexity_level_name(task_idx)
            flex_level, pt_level = parse_complexity_name(task_name)
        
        self.ppo.policy.train()
        
        return task_makespans

    def _evaluate_on_file_group(self, file_list: List[str], metric_prefix: str, compute_gap: bool = True):
        """
        Evaluate policy on a group of files from the same directory.
        
        Args:
            file_list: List of file paths (all from same directory)
            metric_prefix: Prefix for wandb metrics ('validation' or 'test')
            compute_gap: If True, compute gap vs OR value; if False, return makespan directly
        
        Returns:
            dict with {file_path: makespan_or_gap} results
        """
        from src.utils.data_util import load_single_file
        
        if not file_list:
            return {}

        # Extract directory name from first file
        first_file = file_list[0]
        dir_name = os.path.basename(os.path.dirname(first_file))
        
        results = {}
        
        # Evaluate each file individually (files may have different problem sizes)
        for file_path in file_list:
            if metric_prefix == 'validation':
                full_path = os.path.join('benchmarks_data', 'data_train_vali', file_path)
            else:
                full_path = os.path.join('benchmarks_data', file_path)
            
            job_length, op_pt = load_single_file(full_path)
            
            if job_length is None or op_pt is None:
                continue
            
            # Get problem dimensions from the actual file
            n_j = job_length.shape[0]
            n_m = op_pt.shape[1]
            
            # Evaluate single file (batch_size=1)
            env_config = {'n_j': n_j, 'n_m': n_m, 'device': self.device}
            makespans = batched_inference(
                policy=self.ppo.policy,
                batch_job_lengths=[job_length],
                batch_op_pts=[op_pt],
                env_class=FJSPEnvForVariousOpNums,
                env_config=env_config,
                is_hgt=False
            )
            
            makespan = makespans[0]
            if compute_gap:
                relative_path = file_path
                or_val = self.or_store.lookup_or_value(relative_path)
                result_value = float('nan') if or_val is None else self.or_store.compute_gap_percent(makespan, or_val)
            else:
                result_value = makespan
            
            results[file_path] = result_value

            if self.wandb_enabled and metric_prefix != 'validation':
                filename = os.path.splitext(os.path.basename(file_path))[0]
                metric_name = f"{metric_prefix}_file/{dir_name}/{filename}"
                wandb.log({metric_name: result_value}, commit=False)
        
        if results and self.wandb_enabled and metric_prefix != 'validation':
            vals = [v for v in results.values() if isinstance(v, (int, float))]
            if vals:
                group_mean = np.mean(vals)
                metric_suffix = "gap" if compute_gap else "makespan"
                wandb.log({f"{metric_prefix}/{dir_name}_{metric_suffix}": group_mean}, commit=False)
        
        return results
    
    def validate_on_file_list(self):
        """
        Validate the policy on grouped validation files using greedy strategy.
        
        Loads files grouped by directory, evaluates them, and logs results to wandb.
        Returns overall mean performance for model saving decisions.
        
        :return: overall mean makespan across all validation files
        """
        if not self.validation_files:
            return float('inf')
        
        self.ppo.policy.eval()
        all_results = []
        subdir_results = {}  # Store results by subdirectory
        
        # Process each group of files (from same directory) with progress bar
        for file_group in tqdm(self.validation_files, desc="Validating", leave=False):
            group_results = self._evaluate_on_file_group(file_group, 'validation', compute_gap=False)
            all_results.extend(list(group_results.values()))
            
            # Store results by subdirectory for detailed logging
            if file_group:
                subdir_name = os.path.basename(os.path.dirname(file_group[0]))
                subdir_results[subdir_name] = list(group_results.values())
        
        self.ppo.policy.train()
        
        # Calculate overall mean makespan (no wandb logging)
        if all_results:
            overall_mean = np.mean(all_results)
            return overall_mean
        else:
            return float('inf')
    
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
        dir_means = {}
        
        # Process each group of files (from same directory) with progress bar
        for file_group in tqdm(test_files, desc="Testing", leave=False):
            group_results = self._evaluate_on_file_group(file_group, 'test', compute_gap=True)
            if file_group:
                dir_name = os.path.basename(os.path.dirname(file_group[0]))
                vals = [v for v in group_results.values() if isinstance(v, (int, float))]
                if vals:
                    dir_means[dir_name] = float(np.mean(vals))

        self.ppo.policy.train()
        return dir_means

    def _discover_test_files(self):
        """Discover all benchmark test files using utility function."""
        return get_test_files_from_config(self.config)

    

    def save_model(self):
        """
            save the model
        """
        save_obj = {
            'state_dict': self.ppo.policy.state_dict(),
            'config': vars(self.config) if hasattr(self.config, '__dict__') else self.config
        }
        torch.save(save_obj, f"{self.result_dir}/{self.model_name}.pth")

    def load_model(self, model_path=None):
        """
            load the trained model
        """
        if model_path is None:
            model_path = f"{self.result_dir}/{self.model_name}.pth"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        loaded = torch.load(model_path, map_location=self.device)
        if isinstance(loaded, dict) and 'state_dict' in loaded:
            self.ppo.policy.load_state_dict(loaded['state_dict'])
            # keep saved config if present
            self.saved_config = loaded.get('config', None)
        else:
            # backward compatibility with pure state_dict files
            self.ppo.policy.load_state_dict(loaded)

    # Removed CL generation helpers; data must be prepared before training


def PPO_initialize(config):
    ppo = PPO(config)
    return ppo


def main(config):
    trainer = DANIELTrainer(config)
    trainer.train()


if __name__ == '__main__':
    # This would need to be adapted based on your config system
    pass
