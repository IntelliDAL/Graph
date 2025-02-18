from functools import partial
import torch
import torch.nn as nn
import dgl
import numpy as np
from .gcn import GCN
from .stralign import AlignS as dpp
from .stralign import UpS as rpp
from .bone import DualTopologyLearning as DTL
from .bone import biEnLearning, motifD, sce_loss
from torch_cluster import random_walk

def define_tel(args):
    return graphtel(
        in_dim=args.num_features,
        num_hidden=args.num_hidden,
        num_layers=args.num_layers,
        feat_drop=args.in_drop,
        residual=args.residual,
        drop_edge_rate=args.drop_edge_rate,
        alpha_l=args.alpha_l,
        concat_hidden=args.concat_hidden,
        fea_len=args.fea_len,
        use5=args.use5,
        use6=args.use6)

class graphtel(nn.Module):
    def __init__(self, in_dim, num_hidden, num_layers, feat_drop, residual, drop_edge_rate, alpha_l, concat_hidden, fea_len, use5, use6):
        super(graphtel, self).__init__()
        self.perturbedge = drop_edge_rate
        self.nodewise = GCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=num_hidden, 
            num_layers=num_layers, 
            dropout=feat_drop,
            residual=residual, 
            norm=nn.BatchNorm1d,
            encoding=True
        )
        self.topo_dec = GCN(
            in_dim=num_hidden, 
            num_hidden=num_hidden, 
            out_dim=in_dim, 
            num_layers=1, 
            dropout=feat_drop,
            residual=residual, 
            norm=nn.BatchNorm1d,
            encoding=False
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(num_hidden * num_layers, num_hidden, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(num_hidden, num_hidden, bias=False)
        self.aligned_size = 32
        self.alignment = dpp(fea_len, self.aligned_size, 32, False, 0.33)
        self.upsample = rpp(num_hidden, self.aligned_size, False, 0.2)
        self.dtl = DTL(self.aligned_size, num_hidden, use5=use5, use6=use6)
        self.biencode = biEnLearning(num_hidden)
        self.motif_discovery = motifD(use5=use5, use6=use6)
        self.criterion = partial(sce_loss, alpha=alpha_l)

    def edge_perturb(self, graph):
        if self.perturbedge <= 0:
            return graph, None
        edge_mask = torch.bernoulli(1 - torch.FloatTensor(np.ones(graph.num_edges()) * self.perturbedge)).nonzero().squeeze(1)
        src = graph.edges()[0]
        dst = graph.edges()[1]
        nsrc = src[edge_mask]
        ndst = dst[edge_mask]
        ng = dgl.graph((nsrc, ndst), num_nodes=graph.num_nodes())
        ng = ng.add_self_loop()

        num_edges = src.size(0)
        edge_mask = torch.ones(num_edges, dtype=torch.bool)
        startn = (torch.normal(1 - torch.FloatTensor(np.ones(graph.num_nodes()))).nonzero().squeeze(1).dtype(torch.bool)) and ((graph.in_degrees() + graph.out_degrees()) > 1)
        rowptr = torch.zeros(graph.num_nodes() + 1, dtype=torch.bool)
        torch.cumsum((graph.in_degrees() + graph.out_degrees()), 0, out=rowptr[1:])
        road = []
        for tryR in range(0, 20):
            _, e_id = random_walk(rowptr, dst, startn, int(graph.num_edges() * self.perturbedge / 2), 1.0, 1.0)
            e_id = e_id[e_id != -1].view(-1)
            road.append(e_id)
        cho_r = []
        for i in range(0, len(road)):
            c = True
            for j in range(i, len(road)):
                if set(road[i]) & set(road[j]):
                    c = False
            if c:
                cho_r.append(road[i])
            if len(cho_r) == 2:
                break
        edge_mask[road[0]] = False
        edge_mask[road[1]] = False
        nsrc = src[edge_mask]
        ng2 = dgl.graph((nsrc, dst), num_nodes=graph.num_nodes())
        ng2 = ng2.add_self_loop()
        return ng, ng2

    def tel_forward(self, g, x):
        g_maskded = g
        use_x = x
        aligned_adj, AlignRegLoss = self.alignment(g_maskded, use_x)
        aligned_adj_temp = aligned_adj.unsqueeze(0)
        aligned_adj_batch = aligned_adj_temp[:, 0:self.aligned_size, 0:self.aligned_size]
        for i in range(self.aligned_size, aligned_adj.shape[0], self.aligned_size):
            aligned_adj_batch = torch.cat((aligned_adj_batch, aligned_adj_temp[:, i:i + self.aligned_size, i:i + self.aligned_size]), dim=0)
        local_adj_batch = self.motif_discovery.Mds(aligned_adj_batch)
        topo_fea = self.dtl(aligned_adj_batch, local_adj_batch).permute(0, 2, 1, 3)
        topo_fea = self.upsample(g_maskded, use_x, torch.reshape(topo_fea, (topo_fea.shape[0] * topo_fea.shape[1], topo_fea.shape[2])))
        return topo_fea, AlignRegLoss

    def enc_dec(self, g, x, topo_fea):
        nodeemb, graphemb = self.biencode(topo_fea, self.nodewise(g, x))
        rec_topo = self.topo_dec(nodeemb)
        return nodeemb, graphemb, torch.mm(rec_topo, rec_topo.permute(1, 0))

    def forward(self, g, x):
        gr, gw = self.edge_perturb(g)        # if not repeat
        topo_fear, AlignRegLossr = self.tel_forward(gr, x.clone())   
        topo_feaw, AlignRegLossw = self.tel_forward(gw, x.clone())   
        nodeembr, graphembr, rec_topor = self.enc_dec(gr, x.clone(), topo_fear)
        nodeembw, graphembw, rec_topow = self.enc_dec(gw, x.clone(), topo_feaw) 
        init_topo = g.adjacency_matrix()
        rec_loss = self.criterion(init_topo, rec_topor) + self.criterion(init_topo, rec_topow)  
        return rec_loss + 0.3*AlignRegLossr + 0.3*AlignRegLossw, (nodeembr, graphembr), (nodeembw, graphembw)

    def learn_rep(self, g, x):
        topo_fea, _ = self.tel_forward(g, x)
        nodeemb, graphemb, _ = self.enc_dec(g, x, topo_fea)
        return nodeemb, graphemb

