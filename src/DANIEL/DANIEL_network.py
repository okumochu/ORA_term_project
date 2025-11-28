import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from src.utils.common_utils import nonzero_averaging


class SingleOpAttnBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob):
        """
            The implementation of Operation Message Attention Block
        :param input_dim: the dimension of input feature vectors
        :param output_dim: the dimension of output feature vectors
        :param dropout_prob: the parameter p for nn.Dropout()

        """
        super(SingleOpAttnBlock, self).__init__()
        self.in_features = input_dim
        self.out_features = output_dim
        self.alpha = 0.2

        self.W = nn.Parameter(torch.empty(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, h, op_mask):
        """
        :param h: operation feature vectors with shape [sz_b, N, input_dim]
        :param op_mask: used for masking nonexistent predecessors/successor
                        with shape [sz_b, N, 3]
        :return: output feature vectors with shape [sz_b, N, output_dim]
        """
        Wh = torch.matmul(h, self.W)
        sz_b, N, _ = Wh.size()

        Wh_concat = torch.stack([Wh.roll(1, dims=1), Wh, Wh.roll(-1, dims=1)], dim=-2)

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        Wh2_concat = torch.stack([Wh2.roll(1, dims=1), Wh2, Wh2.roll(-1, dims=1)], dim=-1)

        # broadcast add: [sz_b, N, 1, 1] + [sz_b, N, 1, 3]
        e = Wh1.unsqueeze(-1) + Wh2_concat
        e = self.leaky_relu(e)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(op_mask.unsqueeze(-2) > 0, zero_vec, e)

        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        h_new = torch.matmul(attention, Wh_concat).squeeze(-2)

        return h_new


class MultiHeadOpAttnBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob, num_heads, activation, concat=True):
        """
            The implementation of Operation Message Attention Block with multi-head attention
        :param input_dim: the dimension of input feature vectors
        :param output_dim: the dimension of each head's output
        :param dropout_prob: the parameter p for nn.Dropout()
        :param num_heads: the number of attention heads
        :param activation: the activation function used before output
        :param concat: the aggregation operator, true/false means concat/averaging
        """
        super(MultiHeadOpAttnBlock, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.num_heads = num_heads
        self.concat = concat
        self.activation = activation
        self.attentions = [
            SingleOpAttnBlock(input_dim, output_dim, dropout_prob) for
            _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, h, op_mask):
        """
        :param h: operation feature vectors with shape [sz_b, N, input_dim]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :return: output feature vectors with shape
                [sz_b, N, num_heads * output_dim] (if concat == true)
                or [sz_b, N, output_dim]
        """
        h = self.dropout(h)

        # shape: [ [sz_b, N, output_dim], ... [sz_b, N, output_dim]]
        h_heads = [att(h, op_mask) for att in self.attentions]

        if self.concat:
            h = torch.cat(h_heads, dim=-1)
        else:
            # h.shape : [sz_b, N, output_dim, num_heads]
            h = torch.stack(h_heads, dim=-1)
            # h.shape : [sz_b, N, output_dim]
            h = h.mean(dim=-1)

        return h if self.activation is None else self.activation(h)


class SingleMchAttnBlock(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, output_dim, dropout_prob):
        """
            The implementation of Machine Message Attention Block
        :param node_input_dim: the dimension of input node feature vectors
        :param edge_input_dim: the dimension of input edge feature vectors
        :param output_dim: the dimension of output feature vectors
        :param dropout_prob: the parameter p for nn.Dropout()
        """
        super(SingleMchAttnBlock, self).__init__()
        self.node_in_features = node_input_dim
        self.edge_in_features = edge_input_dim
        self.out_features = output_dim
        self.alpha = 0.2
        self.W = nn.Parameter(torch.empty(size=(node_input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.W_edge = nn.Parameter(torch.empty(size=(edge_input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W_edge.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(3 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, h, mch_mask, comp_val):
        """
        :param h: operation feature vectors with shape [sz_b, M, node_input_dim]
        :param mch_mask:  used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_val: a tensor with shape [sz_b, M, M, edge_in_features]
                    comp_val[i, k, q] corresponds to $c_{kq}$ in the paper,
                    which serves as a measure of the intensity of competition
                    between machine $M_k$ and $M_q$
        :return: output feature vectors with shape [sz_b, N, output_dim]
        """
        # Wh.shape: [sz_b, M, out_features]
        # W_edge.shape: [sz_b, M, M, out_features]

        Wh = torch.matmul(h, self.W)
        W_edge = torch.matmul(comp_val, self.W_edge)

        # compute attention matrix

        e = self.get_attention_coef(Wh, W_edge)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(mch_mask > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def get_attention_coef(self, Wh, W_edge):
        """
            compute attention coefficients using node and edge features
        :param Wh: transformed node features
        :param W_edge: transformed edge features
        :return:
        """

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # [sz_b, M, 1]
        Wh2 = torch.matmul(Wh, self.a[self.out_features:2 * self.out_features, :])  # [sz_b, M, 1]
        edge_feas = torch.matmul(W_edge, self.a[2 * self.out_features:, :])  # [sz_b, M, M, 1]

        # broadcast add
        e = Wh1 + Wh2.transpose(-1, -2) + edge_feas.squeeze(-1)

        return self.leaky_relu(e)


class MultiHeadMchAttnBlock(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, output_dim, dropout_prob, num_heads, activation, concat=True):
        """
            The implementation of Machine Message Attention Block with multi-head attention
        :param node_input_dim: the dimension of input node feature vectors
        :param edge_input_dim: the dimension of input edge feature vectors
        :param output_dim: the dimension of each head's output
        :param dropout_prob: the parameter p for nn.Dropout()
        :param num_heads: the number of attention heads
        :param activation: the activation function used before output
        :param concat: the aggregation operator, true/false means concat/averaging
        """
        super(MultiHeadMchAttnBlock, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.concat = concat
        self.activation = activation
        self.num_heads = num_heads

        self.attentions = [SingleMchAttnBlock
                           (node_input_dim, edge_input_dim, output_dim, dropout_prob) for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, h, mch_mask, comp_val):
        """
        :param h: operation feature vectors with shape [sz_b, M, node_input_dim]
        :param mch_mask:  used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_val: a tensor with shape [sz_b, M, M, edge_in_features]
                    comp_val[i, k, q] (any i) corresponds to $c_{kq}$ in the paper,
                    which serves as a measure of the intensity of competition
                    between machine $M_k$ and $M_q$
        :return: output feature vectors with shape
                [sz_b, M, num_heads * output_dim] (if concat == true)
                or [sz_b, M, output_dim]
        """
        h = self.dropout(h)

        h_heads = [att(h, mch_mask, comp_val) for att in self.attentions]

        if self.concat:
            # h.shape : [sz_b, M, output_dim*num_heads]
            h = torch.cat(h_heads, dim=-1)
        else:
            # h.shape : [sz_b, M, output_dim, num_heads]
            h = torch.stack(h_heads, dim=-1)
            h = h.mean(dim=-1)

        return h if self.activation is None else self.activation(h)


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
            the implementation of multi layer perceptrons (refer to L2D)
        :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                            If num_layers=1, this reduces to linear model.
        :param input_dim: dimensionality of input features
        :param hidden_dim: dimensionality of hidden units at ALL layers
        :param output_dim:  number of classes for prediction
        """

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class Actor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
            the implementation of Actor network (refer to L2D)
        :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                            If num_layers=1, this reduces to linear model.
        :param input_dim: dimensionality of input features
        :param hidden_dim: dimensionality of hidden units at ALL layers
        :param output_dim:  number of classes for prediction
        """
        super(Actor, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        self.activative = torch.tanh

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.activative((self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class Critic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
            the implementation of Critic network (refer to L2D)
        :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                            If num_layers=1, this reduces to linear model.
        :param input_dim: dimensionality of input features
        :param hidden_dim: dimensionality of hidden units at ALL layers
        :param output_dim:  number of classes for prediction
        """
        super(Critic, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        self.activative = torch.tanh

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.activative((self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class DualAttentionNetwork(nn.Module):
    def __init__(self, config):
        """
            The implementation of dual attention network (DAN)
        :param config: a package of parameters
        """
        super(DualAttentionNetwork, self).__init__()

        self.fea_j_input_dim = config.fea_j_input_dim
        self.fea_m_input_dim = config.fea_m_input_dim
        self.output_dim_per_layer = config.layer_fea_output_dim
        self.num_heads_OAB = config.num_heads_OAB
        self.num_heads_MAB = config.num_heads_MAB
        self.last_layer_activate = nn.ELU()

        self.num_dan_layers = len(self.num_heads_OAB)
        assert len(config.num_heads_MAB) == self.num_dan_layers
        assert len(self.output_dim_per_layer) == self.num_dan_layers
        self.alpha = 0.2
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout_prob = config.dropout_prob

        num_heads_OAB_per_layer = [1] + self.num_heads_OAB
        num_heads_MAB_per_layer = [1] + self.num_heads_MAB

        # mid_dim = [self.embedding_output_dim] * (self.num_dan_layers - 1)
        mid_dim = self.output_dim_per_layer[:-1]

        j_input_dim_per_layer = [self.fea_j_input_dim] + mid_dim

        m_input_dim_per_layer = [self.fea_m_input_dim] + mid_dim

        self.op_attention_blocks = torch.nn.ModuleList()
        self.mch_attention_blocks = torch.nn.ModuleList()

        for i in range(self.num_dan_layers):
            self.op_attention_blocks.append(
                MultiHeadOpAttnBlock(
                    input_dim=num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_OAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate,
                    dropout_prob=self.dropout_prob
                )
            )

        for i in range(self.num_dan_layers):
            self.mch_attention_blocks.append(
                MultiHeadMchAttnBlock(
                    node_input_dim=num_heads_MAB_per_layer[i] * m_input_dim_per_layer[i],
                    edge_input_dim=num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_MAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate,
                    dropout_prob=self.dropout_prob
                )
            )

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx):
        """
        :param candidate: the index of candidates  [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :return:
            fea_j.shape = [sz_b, N, output_dim]
            fea_m.shape = [sz_b, M, output_dim]
            fea_j_global.shape = [sz_b, output_dim]
            fea_m_global.shape = [sz_b, output_dim]
        """
        sz_b, M, _, J = comp_idx.size()

        comp_idx_for_mul = comp_idx.reshape(sz_b, -1, J)

        for layer in range(self.num_dan_layers):
            candidate_idx = candidate.unsqueeze(-1). \
                repeat(1, 1, fea_j.shape[-1]).type(torch.int64)

            # fea_j_jc: candidate features with shape [sz_b, N, J]
            fea_j_jc = torch.gather(fea_j, 1, candidate_idx).type(torch.float32)
            comp_val_layer = torch.matmul(comp_idx_for_mul,
                                     fea_j_jc).reshape(sz_b, M, M, -1)
            fea_j = self.op_attention_blocks[layer](fea_j, op_mask)
            fea_m = self.mch_attention_blocks[layer](fea_m, mch_mask, comp_val_layer)

        fea_j_global = nonzero_averaging(fea_j)
        fea_m_global = nonzero_averaging(fea_m)

        return fea_j, fea_m, fea_j_global, fea_m_global


class DANIEL(nn.Module):
    def __init__(self, config):
        """
            The implementation of the proposed learning framework for fjsp
        :param config: a package of parameters
        """
        super(DANIEL, self).__init__()
        device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        self.feature_exact = DualAttentionNetwork(config).to(device)
        self.actor = Actor(config.num_mlp_layers_actor, 4 * self.embedding_output_dim + self.pair_input_dim,
                           config.hidden_dim_actor, 1).to(device)
        self.critic = Critic(config.num_mlp_layers_critic, 2 * self.embedding_output_dim, config.hidden_dim_critic,
                             1).to(device)

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs):
        """
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :return:
            pi: scheduling policy with shape [sz_b, J*M]
            v: the value of state with shape [sz_b, 1]
        """

        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(fea_j, op_mask, candidate, fea_m, mch_mask,
                                                                      comp_idx)
        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # collect the input of decision-making network
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)

        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)

        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
        # candidate_feature.shape = [sz_b, J*M, 4*output_dim + 8]
        candidate_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, Fea_Gj_input,
                                       Fea_Gm_input, fea_pairs), dim=-1)

        candidate_scores = self.actor(candidate_feature)
        candidate_scores = candidate_scores.squeeze(-1)

        # masking incompatible op-mch pairs
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1).bool()] = float('-inf')
        
        # Handle case where all actions are masked (avoid NaN from softmax)
        # Check if any row has all -inf values
        all_masked = torch.all(torch.isinf(candidate_scores) & (candidate_scores < 0), dim=1)
        if all_masked.any():
            # For rows with all -inf, set them to zeros to get uniform distribution
            candidate_scores[all_masked] = 0.0
        
        pi = F.softmax(candidate_scores, dim=1)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)
        return pi, v
