import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn


# we directly copied and modified part of the DiffPool model class, including GraphSageLayer("...") and EntropyLoss().
# At the same time, in order not to cause unnecessary misunderstandings 
# during the blind review process and to respect the work of others,
# we append URLs that link to their code.


# Modified from https://github.com/RexYing/diffpool/blob/master/graphsage.py
class AlignS(nn.Module):

    def __init__(self, ):
        super(AlignS, self).__init__()

        self.asigngc = GraphSageLayer("...")
        self.reg_loss = EntropyLoss()

    def AlignReg(self, assign_tensor):
        return self.reg_loss(assign_tensor) + self.l2_loss(assign_tensor)

    def forward(self, g, h):
        assign_tensor = torch.split(F.softmax(self.asigngc(g, h), dim=1), g.batch_num_nodes().tolist())
        assign_tensor = torch.block_diag(*assign_tensor)

        adj = g.adjacency_matrix(transpose=True, ctx="cuda")
        adj_new = torch.sparse.mm(adj, assign_tensor)
        adj_new = torch.mm(torch.t(assign_tensor), adj_new)

        return adj_new, self.AlignReg(assign_tensor)


class UpS(nn.Module):

    def __init__(self, ):
        super(UpS, self).__init__()

        self.pool_gc = GraphSageLayer("...")

    def forward(self, g, h, fea):

        assign_tensor = self.pool_gc(g, h)
        assign_tensor = F.softmax(assign_tensor, dim=1)
        assign_tensor = torch.split(assign_tensor, g.batch_num_nodes().tolist())
        assign_tensor = torch.block_diag(*assign_tensor)

        return torch.mm(assign_tensor, fea)

