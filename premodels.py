import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import APPNP
from torch_geometric.nn import SGConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True, add_self_loops= None, normalize= False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True, add_self_loops= None, normalize= False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True, add_self_loops= None, normalize= False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels: int,
                 hidden_heads: int, out_channels: int, out_heads: int,
                 num_layers: int, dropout: float = 0.0):
        super(GAT, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_heads = hidden_heads
        self.out_channels = out_channels
        self.out_heads = out_heads
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            conv = GATConv(in_channels, out_channels, out_heads,
                          concat=False, dropout=dropout, add_self_loops=False)
            self.convs.append(conv)
        else:
            for i in range(num_layers - 1):
                in_dim = in_channels if i == 0 else hidden_channels * hidden_heads
                conv = GATConv(in_dim, hidden_channels, hidden_heads, concat=True,
                            dropout=dropout, add_self_loops=False)
                self.convs.append(conv)

            conv = GATConv(hidden_channels * hidden_heads, out_channels, out_heads,
                        concat=False, dropout=dropout, add_self_loops=False)
            self.convs.append(conv)

    def reset_parameters(self):
        super().reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
    
class APPNP_AE(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_hidden, dropout, k, alpha = 0.1):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim, num_hidden,)
        self.lin2 = torch.nn.Linear(num_hidden, out_dim)
        self.prop1 = APPNP(k, alpha)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, return_hidden=True):
        hidden_list = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        hidden_list.append(x)
        
        return x
        
class SGC(torch.nn.Module):
    def __init__(self, in_dim, out_dim, k):
        super().__init__()
        self.conv1 = SGConv(
            in_channels=in_dim,
            out_channels=out_dim,
            K=k,
            cached=True,
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim, num_layers, dropout):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_dim))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers):
            self.lins.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.lins.append(torch.nn.Linear(hidden_dim, out_dim))
        self.dropout = dropout

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
    
    def forward(self, x, edge_index):
        x = self.lins[0](x).relu_()
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, lin in enumerate(self.lins[1:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class WGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_edges,
                 dropout):
        super(WGCN, self).__init__()
        self.edge_weight = torch.nn.Parameter(torch.ones(num_edges))
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, add_self_loops= None, normalize= False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False, add_self_loops= None, normalize= False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False, add_self_loops= None, normalize= False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, self.edge_weight.sigmoid())
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, self.edge_weight.sigmoid())
        return x
    
    
class LogisticReg(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)