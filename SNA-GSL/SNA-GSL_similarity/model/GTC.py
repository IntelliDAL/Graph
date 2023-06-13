import torch

from model.EmbeddingLearning import GCNTransformerEncoder
from model.EmbeddingInteraction import CrossTransformer
from model.SimMatLearning import SimMatLearning


class GTCNet(torch.nn.Module):
    def __init__(self, args):
        super(GTCNet, self).__init__()
        self.args = args
        self.embedding_interaction = CrossTransformer(args).to(args.device)
        self.sim_mat_learning = SimMatLearning(args).to(args.device)

    def forward(self, sub_embeddings_0,sub_embeddings_1):

        # 128x10x32

        # mask_ij = torch.einsum('ij,ik->ijk', mask_0, mask_1)
        mask_ij = None
        # sim_mat = self.embedding_interaction(sub_embeddings_0, mask_0, sub_embeddings_1, mask_1, mask_ij)
        sim_mat = torch.cat([sub_embeddings_0, sub_embeddings_1], dim=1)
        score = self.sim_mat_learning(sim_mat, mask_ij)
        return score

