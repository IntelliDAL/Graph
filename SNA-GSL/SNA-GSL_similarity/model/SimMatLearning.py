import torch

from model.ChannelAlignmentModule import ChannelAlignment
from model.SimCNNModule import SimCNN
from model.SimMatPooling import SimMatPooling


class SimMatLearning(torch.nn.Module):
    def __init__(self, args):
        super(SimMatLearning, self).__init__()
        self.args = args

        if args.channel_align:
            self.channel_alignment = ChannelAlignment(args).to(args.device)

        if args.sim_mat_learning_ablation:
            self.sim_mat_pooling = SimMatPooling(args).to(args.device)
        else:
            self.sim_CNN = SimCNN(args).to(args.device)

    def forward(self, mat, mask_ij):
        if self.args.channel_align:
            mat = self.channel_alignment(mat, mask_ij)

        if self.args.sim_mat_learning_ablation:
            score = self.sim_mat_pooling(mat)
        else:
            score = self.sim_CNN(mat, mask_ij)

        return score
