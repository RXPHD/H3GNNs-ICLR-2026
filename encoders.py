import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

import torch_geometric.transforms as T

from premodels import GCN, GAT, APPNP_AE, SGC, MLP, WGCN
from torch_geometric.nn.conv.gcn_conv import gcn_norm


from torch_sparse import SparseTensor


def create_sparse_identity(n, dtype=torch.float32):
    indices = torch.arange(n, dtype=torch.long)
    edge_index = torch.stack([indices, indices], dim=0) 
    values = torch.ones(n, dtype=dtype)
    identity_sparse = SparseTensor(row=edge_index[0], col=edge_index[1], value=values)
    return identity_sparse



class HomoEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_hidden, nlayers, dropout, hidden_heads, out_heads, encoder_type, num_edges, sparse):
        super(HomoEncoder, self).__init__()
        self.dropout = dropout
        self.sparse = sparse

        self.encoder_type = encoder_type.lower()
        if encoder_type == 'gcn':
            self.encoder = GCN(in_dim, num_hidden, out_dim, nlayers, self.dropout)
        elif encoder_type == 'wgcn':
            self.encoder = WGCN(in_dim, num_hidden, out_dim, nlayers, num_edges, self.dropout)
        elif encoder_type == 'gat':
            self.encoder = GAT(in_dim, num_hidden, hidden_heads, out_dim, out_heads, nlayers, self.dropout)
        elif encoder_type == 'appnp':
            self.encoder = APPNP_AE(in_dim, out_dim, num_hidden, self.dropout, nlayers, alpha = 0.1)
        elif encoder_type == 'sgc':
            self.encoder = SGC(in_dim, out_dim, nlayers)
        elif encoder_type == 'mlp':
            self.encoder = MLP(in_dim, num_hidden, out_dim, nlayers, self.dropout)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

    def forward(self, x, adj_t):

        return self.encoder(x, adj_t)



class HeteroEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_hidden, nlayers, dropout, hidden_heads, out_heads, beta, encoder_type, num_edges, sparse):
        super(HeteroEncoder, self).__init__()
        self.dropout = dropout
        self.sparse = sparse
        self.encoder_type = encoder_type.lower()
        self.beta = beta

        if encoder_type == 'gcn':
            self.encoder = GCN(in_dim, num_hidden, out_dim, nlayers, self.dropout)
        elif encoder_type == 'wgcn':
            self.encoder = WGCN(in_dim, num_hidden, out_dim, nlayers, num_edges, self.dropout)
        elif encoder_type == 'gat':
            self.encoder = GAT(in_dim, num_hidden, hidden_heads, out_dim, out_heads, nlayers, self.dropout)
        elif encoder_type == 'appnp':
            self.encoder = APPNP_AE(in_dim, out_dim, num_hidden, self.dropout, nlayers, alpha = 0.1)
        elif encoder_type == 'sgc':
            self.encoder = SGC(in_dim, out_dim, nlayers)
        elif encoder_type == 'mlp':
            self.encoder = MLP(in_dim, num_hidden, out_dim, nlayers, self.dropout)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

    def forward(self, x, adj_t):

        if self.encoder_type == 'sgc':
            n = adj_t.size(0)
            row, col, value = adj_t.coo()

            identity = SparseTensor.eye(n, device=adj_t.device)

            scaled_value = value * (-self.beta)
            scaled_adj = SparseTensor(row=row, col=col, value=scaled_value, sparse_sizes=(n, n)).to(adj_t.device)

            adj_t = identity.add(scaled_adj)

            return self.encoder(x, adj_t)

        else:
          return self.encoder(x, adj_t)
      
      
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

      
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class SelfAttentionFusion(nn.Module):
    def __init__(self, d_model, n_heads, dropout, droppath=0.0, fusion="cls"):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

        self.dropout = nn.Dropout(dropout)
        
        self.droppath = DropPath(droppath) if droppath > 0. else nn.Identity()
        
        self.fusion = fusion

    def forward(self, node_feat, gnn_feat, gnn_feat2, mlp_feat):
        
        N = node_feat.size(0)

        seq = torch.stack([node_feat, gnn_feat, gnn_feat2, mlp_feat], dim=1)  # [N, 4, d_model]

        Q = self.W_q(seq).view(N, 4, self.n_heads, self.d_k).transpose(1, 2)  # [N, n_heads, 4, d_k]
        K = self.W_k(seq).view(N, 4, self.n_heads, self.d_k).transpose(1, 2)  # [N, n_heads, 4, d_k]
        V = self.W_v(seq).view(N, 4, self.n_heads, self.d_k).transpose(1, 2)  # [N, n_heads, 4, d_k]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # [N, n_heads, 4, 4]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # [N, n_heads, 4, d_k]

        attn_output = attn_output.transpose(1, 2).contiguous().view(N, 4, self.d_model)  # [N, 4, d_model]
        attn_output = self.W_o(attn_output)  # [N, 4, d_model]

        attn_output = self.layer_norm1(seq + self.droppath(attn_output))  # [N, 4, d_model]

        # FFN
        ffn_output = self.ffn(attn_output)  # [N, 4, d_model]

        output = self.layer_norm2(attn_output + ffn_output)  # [N, 4, d_model]

        if self.fusion == "cls":
           fused_feat = output[:, 0]  # [N, d_model]
        else:
            fused_feat = output.max(dim=1)[0]


        return fused_feat, attn_weights
    


class Encoder(nn.Module):
    def __init__(self, in_dim, num_edges, out_dim, num_hidden_homo, nlayers_homo, num_hidden_homo2, nlayers_homo2, num_hidden_hetero, nlayers_hetero, dropout, 
                 dropout_fusion, branch_heads, hidden_heads, out_heads, beta, homo_encoder_type, hetero_encoder_type, share, sparse, norm, no_root, droppath=0.0):
        super().__init__()

        self.gnn_branch = HomoEncoder(in_dim, out_dim, num_hidden_homo, nlayers_homo, dropout, hidden_heads, out_heads, homo_encoder_type, num_edges, sparse)
        self.gnn_branch2 = HomoEncoder(in_dim, out_dim, num_hidden_homo2, nlayers_homo2, dropout, hidden_heads, out_heads, homo_encoder_type, num_edges, sparse)
        self.mlp_branch = HeteroEncoder(in_dim, out_dim, num_hidden_hetero, nlayers_hetero, dropout, hidden_heads, out_heads, beta, hetero_encoder_type, num_edges, sparse)

        self.fusion_attn = SelfAttentionFusion(out_dim, branch_heads, dropout_fusion, droppath)
        self.projector = nn.Linear(in_dim, out_dim)

        self.share = share
        self.norm = norm
        self.no_root = no_root

    def forward(self, x_masked, adj_t, device):

        h_gnn = self.gnn_branch(x_masked, adj_t)  # [N, in_channels]
        h_gnn2 = self.gnn_branch2(x_masked, adj_t)

        if self.share:
            n = adj_t.size(0)  # adj_t is SparseTensor
            identity_sparse = create_sparse_identity(n, dtype=torch.float32).to(device)
            h_mlp = self.gnn_branch2(x_masked, identity_sparse)
        else:
            h_mlp = self.mlp_branch(x_masked, adj_t)              # [N, in_channels]

        x_proj = self.projector(x_masked)

        h_fused, attn_weights = self.fusion_attn(x_proj, h_gnn, h_gnn2, h_mlp)

        return h_fused, attn_weights
