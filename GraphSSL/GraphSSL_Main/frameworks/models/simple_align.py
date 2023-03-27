import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import block_diag
import dgl.function as fn
from dgl.utils import expand_as_pair


class Agg(nn.Module):
    def __init__(self):
        super(Agg, self).__init__()

    def forward(self, node):
        neighbour = node.mailbox['m']
        c = self.aggre(neighbour)
        return {"c": c}

    def aggre(self, neighbour):
        raise NotImplementedError


class Maxasign(Agg):
    def __init__(self, in_feats, out_feats):
        super(Maxasign, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=True)
        nn.init.xavier_uniform_(self.linear.weight,
                                gain=nn.init.calculate_gain('relu'))

    def aggre(self, n):
        n = self.linear(n)
        n_new = torch.max(n, dim=1)[0]
        return n_new


class Linearlayer(nn.Module):
    def __init__(self, in_feats, out_feats, dropout, bias=True):
        super(Linearlayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_feats * 2, out_feats, bias)

        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))


    def forward(self, node):
        d = self.linear(torch.cat((node.data['h'], node.data['c']), 1))
        d = F.normalize(d, p=2, dim=1)

        return {"h": d}


class GC(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 ):
        super().__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim
        self.fc = nn.Linear(in_dim, out_dim)
        self.fc.reset_parameters()
        

    def forward(self, graph, feat):
        with graph.local_scope():
            aggregate_fn = fn.copy_src('h', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)

            degs = graph.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat_src = feat_src * norm

            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            
            rst = self.fc(rst)

            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

            return rst


class GCLayer(nn.Module):
    def __init__(self, in_feats, out_feats, dropout):
        super(GCLayer, self).__init__()

        self.gc = GC(in_feats, in_feats)
        self.Linearlayer = Linearlayer(in_feats, out_feats, dropout)
        self.ma = Maxasign(in_feats, in_feats)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, g, h):
        h = self.dropout(h)
        g.ndata['h'] = h

        h = self.gc(g, h)
        g.update_all(fn.copy_src(src='h', out='m'), self.ma, self.Linearlayer)
        h = g.ndata.pop('h')

        return h


class Simplealign(nn.Module):

    def __init__(self, input_dim, align_dim, dropout):
        super(Simplealign, self).__init__()

        self.assign_gc = GCLayer(
            input_dim,
            align_dim,
            dropout)

    def forward(self, g, h):

        align = torch.split(F.softmax(self.assign_gc(g, h), dim=1), g.batch_num_nodes().tolist())
        align = torch.block_diag(*align)
        adj = g.adjacency_matrix(transpose=True, ctx="cuda")
        adj_new = torch.mm(torch.t(align), torch.sparse.mm(adj, align))

        return adj_new, h


class Reversealign(nn.Module):

    def __init__(self, input_dim, align_dim, dropout):
        super(Reversealign, self).__init__()

        self.assign_gc = GCLayer(
            input_dim,
            align_dim,
            dropout)

    def forward(self, g, h, fea):

        align = torch.split(F.softmax(self.assign_gc(g, h), dim=1), g.batch_num_nodes().tolist())
        align = torch.block_diag(*align)

        return torch.mm(align, fea)
