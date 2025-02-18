
# We directly copied and modified the GCN model 
# from other recent works with good impact.

# At the same time, in order not to cause unnecessary misunderstandings 
# during the blind review process and to respect the work of others, 
# we append URLs that link to their code.

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.utils import expand_as_pair


# Modified from https://github.com/THUDM/GraphMAE/blob/main/graphmae/models/gcn.py
class GCN(nn.Module):
    def __init__(self, in_dim, num_hidden, out_dim, num_layers, dropout, residual, norm, encoding=False):
        super(GCN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        gcn_layers = [GraphConv(in_dim, num_hidden, residual=residual, norm=norm, activation=nn.PReLU)]
        gcn_layers.extend([GraphConv(num_hidden, num_hidden, residual=residual, norm=norm, activation=nn.PReLU) for _ in range(1, num_layers - 1)])
        last_activation = nn.PReLU if encoding else None
        last_residual = residual if encoding else False
        last_norm = norm if encoding else None
        gcn_layers.append(GraphConv(num_hidden, out_dim, residual=last_residual, norm=last_norm, activation=last_activation))
        self.gcn_layers = nn.ModuleList(gcn_layers)

    def forward(self, g, inputs):
        h = inputs
        for layer in self.gcn_layers:
            h = layer(g, F.dropout(h, p=self.dropout, training=self.training))
        return h


# Modified from https://github.com/THUDM/GraphMAE/blob/main/graphmae/models/gcn.py
class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, norm=None, activation=None):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.res_fc = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = norm(out_dim)
        self._activation = activation
        self.fc.reset_parameters()

    def _apply_norm(self, graph, feat, degree_type):
        norm = torch.pow(graph.degrees(degree_type).float().clamp(min=1), -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        return feat * torch.reshape(norm, shp)

    def forward(self, graph, feat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = self._apply_norm(graph, feat_src, 'out_degrees')
            graph.update_all(fn.copy_src('h', 'm'), fn.sum(msg='m', out='h'))
            rst = self.fc(graph.dstdata['h'])
            rst = self._apply_norm(graph, rst, 'in_degrees')
            return self._activation(self.norm(rst + self.res_fc(feat_dst)))

