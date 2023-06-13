import torch

from ChannelAlignmentModule import ChannelAlignment
from SimCNNModule import SimCNN


class SimMatLearning(torch.nn.Module):
    def __init__(self, args):
        super(SimMatLearning, self).__init__()
        self.args = args
        self.channel_alignment = ChannelAlignment(args).to(args.device)

        self.sim_CNN = SimCNN(args).to(args.device)

    def forward(self, mat, mask_ij):
        aligned_mat = self.channel_alignment(mat, mask_ij)
        # aligned_mat=mat
        # if self.args.sim_mat_learning_ablation:
        #     score = self.sim_mat_pooling(aligned_mat)
        # else:
        score = self.sim_CNN(aligned_mat, mask_ij)

        return score
