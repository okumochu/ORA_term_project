"""
Heterogeneous Graph Transformer (HGT) Network for FJSP.

This module implements the HGT-based neural network architecture for solving
Flexible Job Shop Scheduling Problems using reinforcement learning.

Architecture:
- HGTFeatureExtractor: Uses HGT layers to process heterogeneous graph structure
  (operations and machines with precedence and compatibility edges)
- HGT: Complete policy network with actor-critic heads for action selection
  and value estimation

The network processes FJSP instances as heterogeneous graphs where:
- Operations are nodes with precedence edges (job constraints)
- Machines are nodes with compatibility edges to operations
- HGT layers enable message passing between different node types
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import HGTConv, Linear, MLP, global_mean_pool
from torch_geometric.nn.norm import LayerNorm as PygLayerNorm
from torch_geometric.data import Batch
from src.utils.common_utils import nonzero_averaging


class HGTFeatureExtractor(nn.Module):
    """
    HGT-based feature extractor following DANIEL structure.
    Replaces DANIEL's DualAttentionNetwork with proper HGT layers.
    """
    
    def __init__(self, op_feature_dim: int, machine_feature_dim: int, 
                 hidden_dim: int, num_hgt_layers: int, num_heads: int, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_hgt_layers = num_hgt_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Output dimension of HGT layers (multiply by num_head)
        hgt_out_dim = hidden_dim * num_heads
        
        # Input embedding layers
        self.op_embedding = Linear(op_feature_dim, hgt_out_dim, bias=True)
        self.machine_embedding = Linear(machine_feature_dim, hgt_out_dim, bias=True)
        
        # Define metadata for HGT with bidirectional edges for machine-op compatibility
        self.metadata = (['op', 'machine'], [('op', 'precedes', 'op'), 
                                            ('machine', 'can_process', 'op'),
                                            ('op', 'compatible_with', 'machine')])
        
        # HGT layers
        self.hgt_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residual_dropout = nn.Dropout(dropout)
        
        for i in range(num_hgt_layers):
            # Create HGT layer
            hgt_layer = HGTConv(
                in_channels=hgt_out_dim,
                out_channels=hgt_out_dim,  # Keep same dimension for residual connection
                metadata=self.metadata,
                heads=num_heads
            )
            self.hgt_layers.append(hgt_layer)
            
            # Single layer normalization per layer (not per node type)
            self.layer_norms.append(PygLayerNorm(hgt_out_dim))
        
        # Layer normalization for input embeddings
        self.op_norm = PygLayerNorm(hgt_out_dim)
        self.machine_norm = PygLayerNorm(hgt_out_dim)
    
    def forward(self, graph):
        """
        Forward pass with efficient PyG batching.
        
        Args:
            graph: List of HeteroData graphs for HGT processing
            
        Returns:
            fea_j: Updated operation features [sz_b, N, hidden_dim]
            fea_m: Updated machine features [sz_b, M, hidden_dim]
            fea_j_global: Global operation features [sz_b, hidden_dim]
            fea_m_global: Global machine features [sz_b, hidden_dim]
        """
        # === EFFICIENT BATCHING: Process all graphs at once ===
        # Store original node counts for unbatching later
        num_ops_per_graph = [g['op'].x.size(0) for g in graph]
        num_machines_per_graph = [g['machine'].x.size(0) for g in graph]
        
        # Batch all graphs into a single disconnected graph
        batched_graph = Batch.from_data_list(graph)
        
        # Get embeddings (now batched)
        op_x = self.op_embedding(batched_graph['op'].x)
        machine_x = self.machine_embedding(batched_graph['machine'].x)
        
        # Normalize embeddings
        op_x = self.op_norm(op_x)
        machine_x = self.machine_norm(machine_x)
        
        # Prepare for HGT processing
        x_dict = {'op': op_x, 'machine': machine_x}
        
        # Apply HGT layers (now on batched graph - MUCH FASTER!)
        for layer_idx, hgt_layer in enumerate(self.hgt_layers):
            residual_dict = {k: v for k, v in x_dict.items()}
            
            # HGT forward pass on batched graph
            out_dict = hgt_layer(x_dict, batched_graph.edge_index_dict)
            
            # Apply residual connection and layer norm
            x_dict = {}
            for node_type in ['op', 'machine']:
                hgt_output = out_dict.get(node_type, torch.zeros_like(residual_dict[node_type]))
                combined = hgt_output + residual_dict[node_type]
                
                # Apply dropout and layer normalization
                x_dict[node_type] = self.layer_norms[layer_idx](
                    self.residual_dropout(combined)
                )
        
        # Get final embeddings (still batched)
        final_op_embeddings = x_dict['op']
        final_machine_embeddings = x_dict['machine']
        
        # Global pooling per graph (using batch indices from PyG)
        # Note: This correctly excludes dummy/padded nodes because graphs only contain
        # actual operations before batching (see _build_batch_hetero_graphs in hgt_env.py)
        op_batch_idx = batched_graph['op'].batch
        machine_batch_idx = batched_graph['machine'].batch
        
        # Shape: [num_graphs, hidden_dim * heads]
        fea_j_global = global_mean_pool(final_op_embeddings, op_batch_idx)
        fea_m_global = global_mean_pool(final_machine_embeddings, machine_batch_idx)
        
        # Unbatch node features to match DANIEL's expected format
        # Split concatenated features back to individual graphs
        op_splits = torch.split(final_op_embeddings, num_ops_per_graph)
        machine_splits = torch.split(final_machine_embeddings, num_machines_per_graph)
        
        # Pad to max size for batched tensor format [sz_b, max_N, hidden_dim]
        max_ops = max(num_ops_per_graph)
        max_machines = max(num_machines_per_graph)
        
        batch_size = len(graph)
        device = final_op_embeddings.device
        hidden_dim = final_op_embeddings.size(-1)
        
        # Create padded tensors
        fea_j = torch.zeros(batch_size, max_ops, hidden_dim, device=device)
        fea_m = torch.zeros(batch_size, max_machines, hidden_dim, device=device)
        
        for i, (op_feat, mch_feat) in enumerate(zip(op_splits, machine_splits)):
            fea_j[i, :op_feat.size(0), :] = op_feat
            fea_m[i, :mch_feat.size(0), :] = mch_feat
        
        # Ensure global features have correct shape [sz_b, 1, hidden_dim]
        fea_j_global = fea_j_global.unsqueeze(1)
        fea_m_global = fea_m_global.unsqueeze(1)
        
        return fea_j, fea_m, fea_j_global, fea_m_global


class HGT(nn.Module):
    """
    HGT-based network following DANIEL structure.
    Replaces DANIEL's dual attention with HGT for feature extraction.
    """
    
    def __init__(self, config):
        super(HGT, self).__init__()
        device = torch.device(config.device)
        
        # Pair features input dim (same as DANIEL)
        self.pair_input_dim = 8
        
        # Feature dimensions
        self.op_feature_dim = config.op_feature_dim if hasattr(config, 'op_feature_dim') else 10
        self.machine_feature_dim = config.machine_feature_dim if hasattr(config, 'machine_feature_dim') else 8
        self.hidden_dim = config.hidden_dim
        self.num_hgt_layers = config.num_hgt_layers
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        
        # Output dimension - HGT outputs hidden_dim * heads
        self.embedding_output_dim = self.hidden_dim * self.num_heads
        
        # Feature extractor (HGT instead of DualAttentionNetwork)
        self.feature_extractor = HGTFeatureExtractor(
            op_feature_dim=self.op_feature_dim,
            machine_feature_dim=self.machine_feature_dim,
            hidden_dim=self.hidden_dim,
            num_hgt_layers=self.num_hgt_layers,
            num_heads=self.num_heads,
            dropout=self.dropout
        ).to(device)
        
        # Actor and Critic networks replaced with PyG MLP
        self.actor = MLP(
            in_channels=4 * self.embedding_output_dim + self.pair_input_dim,
            hidden_channels=config.hidden_dim_actor,
            out_channels=1,
            num_layers=config.num_mlp_layers_actor,
            act=torch.tanh,
            dropout=0.0,
            norm=None,
            bias=True,
        ).to(device)
        
        self.critic = MLP(
            in_channels=2 * self.embedding_output_dim,
            hidden_channels=config.hidden_dim_critic,
            out_channels=1,
            num_layers=config.num_mlp_layers_critic,
            act=torch.tanh,
            dropout=0.0,
            norm=None,
            bias=True,
        ).to(device)
    
    def forward(self, fea_j, candidate, fea_m, 
                dynamic_pair_mask, fea_pairs, graph):
        """
        Forward pass following DANIEL structure.
        
        Args:
            fea_j: Operation features [sz_b, N, op_feature_dim]
            candidate: Candidate operations [sz_b, J]
            fea_m: Machine features [sz_b, M, machine_feature_dim]
            dynamic_pair_mask: Dynamic pair mask [sz_b, J, M]
            fea_pairs: Pair features [sz_b, J, M, 8]
            graph: HeteroData graph for HGT processing
            
        Returns:
            pi: Policy [sz_b, J*M]
            v: Value [sz_b, 1]
        """
        # Extract features using HGT
        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_extractor(graph)
        
        # Extract dimensions from existing tensors
        sz_b = fea_j.size(0)  # batch size
        M = fea_m.size(1)     # number of machines
        J = candidate.size(1) # number of candidate operations
        d = fea_j.size(-1)    # feature dimension
        
        # Collect input for decision-making network (same as DANIEL)
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64).to(fea_j.device)
        
        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)
        
        Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)
        
        Fea_Gj_input = fea_j_global.expand_as(Fea_j_JC_serialized)
        Fea_Gm_input = fea_m_global.expand_as(Fea_j_JC_serialized)
        
        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim).to(fea_j.device)
        
        # Candidate features [sz_b, J*M, 4*output_dim + 8]
        candidate_feature = torch.cat((
            Fea_j_JC_serialized, Fea_m_serialized, Fea_Gj_input,
            Fea_Gm_input, fea_pairs
        ), dim=-1)
        
        # Get policy
        candidate_scores = self.actor(candidate_feature)
        candidate_scores = candidate_scores.squeeze(-1)
        
        # Mask incompatible op-mch pairs
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1).bool().to(fea_j.device)] = float('-inf')
        
        # Handle case where all actions are masked (avoid NaN from softmax)
        # Check if any row has all -inf values
        all_masked = torch.all(torch.isinf(candidate_scores) & (candidate_scores < 0), dim=1)
        if all_masked.any():
            # For rows with all -inf, set them to zeros to get uniform distribution
            candidate_scores[all_masked] = 0.0
        
        pi = F.softmax(candidate_scores, dim=1)
        
        # Get value
        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)
        
        return pi, v
