"""
Common utility functions for HGT training.

This module provides utility functions for:
- Action sampling and evaluation (for PPO training)
- Environment interaction (heuristic action selection)
- Feature processing (nonzero averaging for graph features)
"""

import json
import random
from torch.distributions.categorical import Categorical
import sys
import numpy as np
import torch
import copy
from typing import List, Tuple, Dict, Any, Optional


def sample_action(p):
    """
    Sample an action from a categorical probability distribution.
    
    Used during training to sample actions from the policy network output.
    
    Args:
        p: Probability distribution tensor [batch_size, num_actions]
           Each row should sum to 1.0 (softmax output)
    
    Returns:
        Tuple of (action, log_prob):
        - action: Sampled action indices [batch_size]
        - log_prob: Log probability of the sampled actions [batch_size]
    """
    dist = Categorical(p)
    s = dist.sample()  # index
    return s, dist.log_prob(s)


def eval_actions(p, actions):
    """
    Evaluate log probabilities and entropy for given actions.
    
    Computes the log probability of the given actions under the policy distribution
    and the entropy of the policy (for regularization).
    
    Args:
        p: Policy distribution [batch_size, num_actions] (softmax probabilities)
        actions: Action indices [batch_size] (long tensor)
    
    Returns:
        Tuple of (log_probs, entropy):
        - log_probs: Log probability of actions [batch_size]
        - entropy: Mean entropy of the policy distribution (scalar)
    """
    softmax_dist = Categorical(p.squeeze())
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


def greedy_select_action(p):
    """
    Select actions greedily (argmax) from policy distribution.
    
    Used during evaluation/testing to select the best action rather than sampling.
    
    Args:
        p: Policy distribution [batch_size, num_actions] (softmax probabilities)
    
    Returns:
        Action indices [batch_size] with maximum probability
    """
    _, index = torch.max(p, dim=1)
    return index


def min_element_index(array):
    """
    :param array: an array with numbers
    :return: Index set corresponding to the minimum element of the array
    """
    min_element = np.min(array)
    candidate = np.where(array == min_element)
    return candidate


def max_element_index(array):
    """
    :param array: an array with numbers
    :return: Index set corresponding to the maximum element of the array
    """
    max_element = np.max(array)
    candidate = np.where(array == max_element)
    return candidate


def available_mch_list_for_job(chosen_job, env):
    """
    :param chosen_job: the selected job
    :param env: the production environment
    :return: the machines which can immediately process the chosen job
    """
    mch_state = ~env.candidate_process_relation[0, chosen_job]
    available_mch_list = np.where(mch_state == True)[0]
    mch_free_time = env.mch_free_time[0][available_mch_list]
    job_free_time = env.candidate_free_time[0][chosen_job]
    # case1 eg:
    # JF: 50
    # MchF: 55 60 65 70
    if (job_free_time < mch_free_time).all():
        chosen_mch_list = available_mch_list[min_element_index(mch_free_time)]
    # case2 eg:
    # JF: 50
    # MchF: 35 40 55 60
    else:
        chosen_mch_list = available_mch_list[np.where(mch_free_time <= job_free_time)]

    return chosen_mch_list


def heuristic_select_action(method, env):
    """
    :param method: the name of heuristic method
    :param env: the environment
    :return: the action selected by the heuristic method

    here are heuristic methods selected for comparison:

    FIFO: First in first out
    MOR(or MOPNR): Most operations remaining
    SPT: Shortest processing time
    MWKR: Most work remaining
    """
    chosen_job = -1
    chosen_mch = -1

    job_state = (env.mask[0] == 0)

    process_job_state = (env.candidate_free_time[0] <= env.next_schedule_time[0])
    job_state = process_job_state & job_state

    available_jobs = np.where(job_state == True)[0]
    available_ops = env.candidate[0][available_jobs]

    if method == 'FIFO':
        # selecting the earliest ready candidate operation
        candidate_free_time = env.candidate_free_time[0][available_jobs]
        chosen_job_list = available_jobs[min_element_index(candidate_free_time)]
        chosen_job = np.random.choice(chosen_job_list)

        # select the earliest ready machine
        mch_state = ~env.candidate_process_relation[0, chosen_job]
        available_mchs = np.where(mch_state == True)[0]
        mch_free_time = env.mch_free_time[0][available_mchs]
        chosen_mch_list = available_mchs[min_element_index(mch_free_time)]
        chosen_mch = np.random.choice(chosen_mch_list)

    elif method == 'MOR':
        remain_ops = env.op_match_job_left_op_nums[0][available_ops]
        chosen_job_list = available_jobs[max_element_index(remain_ops)]
        chosen_job = np.random.choice(chosen_job_list)

        # select a machine which can immediately process the chosen job
        chosen_mch_list = available_mch_list_for_job(chosen_job, env)
        chosen_mch = np.random.choice(chosen_mch_list)

    elif method == 'SPT':

        temp_pt = copy.deepcopy(env.candidate_pt[0])
        temp_pt[env.dynamic_pair_mask[0]] = float("inf")
        pt_list = temp_pt.reshape(-1)

        action_list = np.where(pt_list == np.min(pt_list))[0]

        action = np.random.choice(action_list)
        return action

    elif method == 'MWKR':
        job_remain_work_list = env.op_match_job_remain_work[0][available_ops]

        chosen_job = available_jobs[np.random.choice(max_element_index(job_remain_work_list)[0])]

        # select a machine which can immediately process the chosen job
        chosen_mch_list = available_mch_list_for_job(chosen_job, env)
        chosen_mch = np.random.choice(chosen_mch_list)

    else:
        print(f'Error From rule select: undefined method {method}')
        sys.exit()

    if chosen_job == -1 or chosen_mch == -1:
        print(f'Error From choosing action: choose job {chosen_job}, mch {chosen_mch}')
        sys.exit()

    action = chosen_job * env.number_of_machines + chosen_mch
    return action


"""
    common utils
"""



def nonzero_averaging(x):
    """
    Compute mean of feature vectors, excluding zero vectors (dummy nodes).
    
    In graph-based models, dummy/padding nodes are represented by zero vectors.
    This function computes the mean over only the actual (nonzero) nodes.
    
    Args:
        x: Feature vectors [sz_b, node_num, d]
           Zero vectors represent dummy/padding nodes
    
    Returns:
        Mean feature vector [sz_b, d], averaged over nonzero nodes only
    """
    b = x.sum(dim=-2)
    y = torch.count_nonzero(x, dim=-1)
    z = (y != 0).sum(dim=-1, keepdim=True)
    p = 1 / z
    p[z == 0] = 0
    return torch.mul(p, b)
