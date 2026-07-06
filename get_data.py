import torch
import torch.nn.functional as F
import os

import torch_geometric.transforms as T

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected, degree
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import HeterophilousGraphDataset
from data_preprocess.heterodata_loader import Hetero


from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from os import path as path


def load_dataset(dataset_name, device):

    if dataset_name in (
        "squirrel-filtered",
        "chameleon-filtered",
        ):
        dataset = Hetero(dataset_name, device)
        graph = dataset.data

    num_features = dataset.num_features
    num_classes = dataset.num_classes
    return graph, num_features, num_classes


def load_data(dataset_name):

    path = os.path.join(os.path.dirname(os.path.realpath("__file__")), '.', 'data', dataset_name)

    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, dataset_name)
    elif dataset_name in ['actor']:
        dataset = Actor(path)
    elif dataset_name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(path, dataset_name)
    elif dataset_name in ['roman_empire']:
        dataset = HeterophilousGraphDataset(path, dataset_name)

    data = dataset[0]

    data.edge_index = remove_self_loops(data.edge_index)[0]


    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

    if len(train_mask.shape) < 2:
        train_mask = train_mask.unsqueeze(1)
        val_mask = val_mask.unsqueeze(1)
        test_mask = test_mask.unsqueeze(1)

    return data, dataset.num_features, dataset.num_classes

def load_arxiv():

    torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    graph = dataset[0]
    num_nodes = graph.x.shape[0]
    graph.edge_index = to_undirected(graph.edge_index)
    # graph.edge_index = remove_self_loops(graph.edge_index)[0]
    # graph.edge_index = add_self_loops(graph.edge_index)[0]
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    if not torch.is_tensor(train_idx):
        train_idx = torch.as_tensor(train_idx)
        val_idx = torch.as_tensor(val_idx)
        test_idx = torch.as_tensor(test_idx)
    train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
    val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
    test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
    graph.train_mask, graph.val_mask, graph.test_mask = train_mask, val_mask, test_mask
    graph.y = graph.y.view(-1)
    # graph.x = scale_feats(graph.x)
    
    return graph, dataset.num_features, dataset.num_classes
