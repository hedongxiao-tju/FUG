import torch as pt
import torch.nn as nn
import torch_geometric as pyg
from Dataset_Load import load_dataset
import GCL
from GCL.eval import get_split, LREvaluator
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import warnings
import torch
import torch_geometric
import time
import argparse

class DimensionNN_V2(nn.Module):
    def __init__(self, n_in, n_h, n_out, activator):
        # n_in = sample_size
        super(DimensionNN_V2, self).__init__()
        self.act = activator()
        self.lin_in = nn.Linear(n_in, n_h)
        self.lin_h1 = nn.Linear(n_h, n_h)
        self.lin_out = nn.Linear(n_h, n_out)
        self.sample = []

    def encode(self, x):
        z = self.act(self.lin_in(x))
        z = self.act(self.lin_h1(z))
        #return self.act(self.lin_out(z))
        return self.lin_out(z)

    def forward(self, x: pt.Tensor):
        '''
        x size: (sample_size, feature) 
        '''
        self.sample = x
        self.out = F.normalize(self.encode(x.T)) #(feature, sig_embed)
        return self.out
         

    def dimensional_loss(self):
        return self.out.mean(dim=0).pow(2).mean()

class GCN_encoder(nn.Module):
    def __init__(self, n_in, n_h, activator):
        super(GCN_encoder, self).__init__()
        self.gcn_in = GCNConv(n_in, n_h)
        self.gcn_out = GCNConv(n_h, n_h)
        self.act = activator()

    def encode(self, x, edge_index):
        out = self.act(self.gcn_in(x, edge_index))
        out = self.gcn_out(out, edge_index)
        return out

    def proj(self, z):
        return self.lin_2(self.act(self.lin_1(z)))
        
    def forward(self, x, edge_index):
        out = self.encode(x,edge_index)
        return out

    def embed(self, x, edge_index):
        self.eval()
        return self.encode(x, edge_index)


class FUG(nn.Module):
    def __init__(self, D_NN, G_NN, S_mtd, sample_size):
        super(FUG, self).__init__()
        self.dnn = D_NN
        self.gnn = G_NN
        self.smtd = S_mtd
        self.sample_size = sample_size
        self.d_sample_matrix = []

    def update_sample(self, x, edge_index, if_rand=False):
        with torch.no_grad():
            self.d_sample_matrix = self.smtd(self.sample_size, x, edge_index, if_rand)

    def forward(self, x, edge_index):
        dimension_sig = self.dnn(self.d_sample_matrix)
        x = self.feature_sig_propagate(x, dimension_sig)
        return self.gnn(x, edge_index)

    def embed(self, x, edge_index):
        with torch.no_grad():
            self.eval()
            dimension_sig = self.dnn(self.d_sample_matrix)
            x = self.feature_sig_propagate(x, dimension_sig)
            return self.gnn.embed(x, edge_index)

    def ssl_loss_fn_infoNCE(self, z):
        z = F.normalize(z, dim=1)
        return z.mean(dim=0).pow(2).mean()

    def ssl_loss_fn_pos(self, z, edge_index):
        return (z[edge_index[0]]-z[edge_index[1]]).pow(2).mean()

    def dim_loss_fn(self):
        return self.dnn.dimensional_loss()

    def feature_sig_propagate(self, x, dimension_sig):
        return F.normalize(x @ dimension_sig)