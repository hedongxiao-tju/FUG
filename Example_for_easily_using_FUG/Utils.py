import torch as pt
import torch.nn as nn
import torch_geometric as pyg
from Dataset_Load import load_dataset
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import warnings
import torch
import torch_geometric


def dimensional_sample_random(sample_size, x, edge_index, if_rand=False):
    with pt.no_grad():
        if if_rand!=True:
            d_sample_matrix = x[:sample_size, :]
        else:
            d_sample_matrix = x[pt.randperm(x.shape[0]),:][:sample_size, :]
        return d_sample_matrix


def DAD_edge_index(edge_index, size):
    a = torch.sparse_coo_tensor(edge_index, torch.ones_like(edge_index)[0], size).to_dense().to(edge_index.device)
    a = a + torch.eye(n=a.size()[0]).to(edge_index.device)
    d = a.sum(dim=0)
    d_2 = torch.diag(d.pow(-0.5))
    a = d_2 @ a @ d_2
    return a


def get_embedding(x, edge_index, model, num_hop, if_rand=False, feature_samples=None):
    with pt.no_grad():
        model.eval()
        model.update_sample(x, edge_index, if_rand)
        if feature_samples!= None:
            model.d_sample_matrix = feature_samples
        z = model.embed(x, edge_index)
        if num_hop != 0:
            a = DAD_edge_index(edge_index, (z.size()[0], z.size()[0]))
            for i in range(num_hop):
                z = a @ z
    return z

    