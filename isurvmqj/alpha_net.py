import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class AlphaNet(nn.Module):
    """
    AlphaNet is a neural network module designed to compute attention weights 
    between a set of 'keys' and a 'query' based on learned embeddings.

    It uses a shared feedforward network to embed both keys and queries, 
    followed by linear projections for computing attention scores via scaled dot-product attention.

    Parameters:
        input_dim (int): Dimensionality of the input features.
        embed_dim (int): Dimensionality of the learned embeddings.
        dropout_rate (float): Dropout rate used in the embedding network.
        hidden_dim (int): Number of hidden units in the embedding network (default: 64).
    """

    def __init__(self, input_dim, embed_dim, dropout_rate, hidden_dim=64):
        super(AlphaNet, self).__init__()

        self.alpha_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout_rate)
        )

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, keys, query, attn_mask=None):
        keys_embed = self.alpha_net(keys)  
        query_embed = self.alpha_net(query)  

        keys_proj = self.W_k(keys_embed)  
        query_proj = self.W_q(query_embed)  

        weights = torch.matmul(query_proj, keys_proj.T) / (keys_proj.shape[-1] ** 0.5)

        if attn_mask is not None:
            weights = weights.masked_fill_(attn_mask == 0, float('-inf'))

        weights = F.softmax(weights, dim=1)

        return weights