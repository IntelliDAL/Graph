import torch
import torch.nn as nn
import torch.nn.functional as F
from dotmotif import GrandIsoExecutor as dotmf
from dotmotif import Motif as dotmft
from moDic import motifDict, motifADict
import numpy as np
import networkx as nx

class filblock(nn.Module):
    def __init__(self, in_channel, out_channel, input_shape):
        super().__init__()
        self.in_channel = in_channel
        self.d = input_shape[0]
        self.conv1xd = nn.Conv2d(in_channel, out_channel, (self.d, 1))
        self.convdx1 = nn.Conv2d(in_channel, out_channel, (1, self.d))
        self.nodes = input_shape[0]

    def forward(self, A):
        A = A.view(-1, self.in_channel, self.nodes, self.nodes)
        return torch.cat([self.conv1xd(A)] * self.d, 2) + torch.cat([self.convdx1(A)] * self.d, 3)    
    
class MotifC(nn.Module):
    def __init__(self, num_hidden, use5=False, use6=False):
        super().__init__()
        self.use5 = use5
        self.use6 = use6
        self.t1 = nn.Sequential(nn.Conv2d(1, num_hidden, (3, 3)), nn.LeakyReLU(0.33))
        self.t2 = nn.Sequential(nn.Conv2d(1, num_hidden, (4, 4)), nn.LeakyReLU(0.33))
        self.t3 = nn.Sequential(nn.Conv2d(1, num_hidden, (5, 5)), nn.LeakyReLU(0.33)) if self.use5 else None
        self.t4 = nn.Sequential(nn.Conv2d(1, num_hidden, (6, 6)), nn.LeakyReLU(0.33)) if self.use6 else None

    def forward(self, lx):
        emb1 = torch.sum(self.t1(item) for item in lx[0]) / len(lx[0])
        emb2 = torch.sum(self.t2(item) for item in lx[1]) / len(lx[1])
        emb3 = torch.sum(self.t3(item) for item in lx[2]) / len(lx[2]) if self.use5 or len(lx[2]) != 0 else torch.zeros_like(emb2)
        emb4 = torch.sum(self.t4(item) for item in lx[3]) / len(lx[3]) if self.use6 or len(lx[3]) != 0 else torch.zeros_like(emb2)
        return torch.cat((emb1, emb2, emb3, emb4), dim=1)

class DualTopologyLearning(nn.Module):
    def __init__(self, pool_size, num_hidden, use5=False, use6=False):
        super().__init__()
        self.nodes = pool_size
        self.fT = nn.Sequential(
            filblock(1, 8, (self.nodes, self.nodes)),
            nn.LeakyReLU(0.33),
            filblock(8, 64, (self.nodes, self.nodes)),
            nn.LeakyReLU(0.33)
        )
        self.fN = nn.Sequential(nn.Conv2d(64, num_hidden, (1, self.nodes)), nn.LeakyReLU(0.33))
        self.fL = nn.Sequential(nn.Conv2d(4, 1, (1, self.nodes)), nn.LeakyReLU(0.33))
        self.motif_c = MotifC(num_hidden, use5=use5, use6=use6)

    def forward(self, gx, lx):
        return self.fN(self.fT(gx)) + self.fL(self.motif_c(lx))


class biEnLearning(nn.Module):
    def __init__(self, ind):
        super().__init__()
        self.genc = torch.nn.Conv2d(ind*2, ind, (self.d,1))

    def forward(self, tfea, nfea):
        neb = torch.cat((tfea, nfea), dim=1)
        geb = F.dropout(F.leaky_relu(self.genc(neb), negative_slope=0.33), p=0.5)
        geb = geb.view(geb.size(0), -1)
        return neb, geb

class motifD():
    def __init__(self, use5=False, use6=False):
        self.use5, self.use6, self.dotm = use5, use6, dotmf

    def nx_graph_load(self, adj_matrices):
        return [nx.DiGraph([(i, j, {'weight': adj_matrix[i, j].item()}) 
                            for i in range(adj_matrix.size(0)) for j in range(adj_matrix.size(1)) 
                            if adj_matrix[i, j] > 0]) 
                            for adj_matrix in adj_matrices]

    def uniq_(self, results):
        return np.unique([np.sort(sum(oneR.items(), [])) for oneR in results], axis=0)

    def combiMfFinder(self, DicA, DicB, a1, a2, tc):
        return self.uniq_([self.return_a(la, motifADict[a1][ka], lb, motifADict[a2][kb]) 
                           for ka, la in DicA.items() for kb, lb in DicB.items()
                           if ka != kb and len(s := set(la) | set(lb)) == tc and set(la) & set(lb)])

    def return_a(self, n1, adj1, n2, adj2):
        nodes, m_adj = sorted(set(n1) | set(n2)), torch.zeros((len(s := set(n1) | set(n2)), len(s)))
        f_m = lambda ns, adj: [[m_adj.__setitem__((ni := nodes.index(n), nodes.index(ns[j])), adj[i][j]) for j in range(len(ns))] for i, n in enumerate(ns)]
        f_m(n1, adj1), f_m(n2, adj2)
        return m_adj

    def Mds(self, aligned_adj_batch):
        aab = self.nx_graph_load(aligned_adj_batch)
        mf, proc = [], lambda o, idx: [self.uniq_(self.dotm(aab[i]).find(dotmft(o))) for i in range(len(aab))]
        mf.extend([sum([[motifADict[d][k] for _ in proc(motifDict[d][k], k)] for k in range(len(motifDict[d]))], []) for d in [3, 4]])
        # Since there are too many motif modes for 5-vertex (21 types) or 6-vertex (112 types),
        # Besides, the 5-vertex and 6-vertex motifs found in real graphs are relatively less than 2, 3 and 4-vertex motifs with high overlap.
        # So we do not choose to directly give all the motif formulas manually in moDic.py .
        # We here treat a 5/6-vertex motif as a combination of the 3/4-vertex motifs.
        # By combining the adj of the 3/4-vertex motifs at node-level, 5/6-vertex motif adj matrix is obtained.
        mf_t = [{k: proc(motifDict[d][k], k) for k in range(len(motifDict[d]))} for d in [3, 4]]
        if self.use5:
            mf.extend([[self.combiMfFinder(mf_t[0], mf_t[0], 3, 3, 5)], []])
            mf.extend([[self.combiMfFinder(mf_t[0], mf_t[1], 3, 4, 5)], []])
            mf.extend([[self.combiMfFinder(mf_t[1], mf_t[1], 4, 4, 5)], []])
        if self.use6:
            mf.extend([[self.combiMfFinder(mf_t[0], mf_t[0], 3, 3, 6)], []])
            mf.extend([[self.combiMfFinder(mf_t[0], mf_t[1], 3, 4, 6)], []])
            mf.extend([[self.combiMfFinder(mf_t[1], mf_t[1], 4, 4, 6)], []])

        return [torch.FloatTensor(item) for item in mf]

def sce_loss(x, y, alpha=3):
    return (1 - (F.normalize(x, p=2, dim=-1) * F.normalize(y, p=2, dim=-1)).sum(dim=-1)).pow_(alpha).mean()
