import torch
from torch import nn
from utils import write_log_file, create_dir_if_not_exists
from torch.nn import Sequential, Linear, ReLU
from model.EmbeddingLearning import  GCNTransformerEncoder
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool
from model.EmbeddingInteraction import CrossTransformer


def to_sparse_batch(batch_x, mask, device):
    B, N = batch_x.shape[0], batch_x.shape[1]
    x = batch_x.reshape(B * N, -1)
    mask = mask.reshape(-1)
    x = x[mask][:]
    # x = np.compress(mask, x, axis=0)
    return x


def padding(x, shape_len):
    # 左右上下
    pad = nn.ZeroPad2d(padding=(0, shape_len, 0, 0))

    return pad(x)


######################################################################
class Transformer_Coss_Encoder(nn.Module):
    '''
    '''

    def __init__(self, args):
        super(Transformer_Coss_Encoder, self).__init__()
        self.args = args
        self.encoder = TransformerEncoder(self.args)
        fc_embeddings = self.args.fc_embeddings
        in_features=self.args.n_max_nodes*self.args.n_heads
        if self.args.dataset == 'IMDBMulti':
            in_features=self.args.pooling_res*self.args.n_heads
        self.projectionHead = Sequential(Linear(in_features, fc_embeddings), nn.ReLU(),
                                         nn.Dropout(p=self.args.dropout),
                                         Linear(fc_embeddings, fc_embeddings), nn.ReLU(),
                                         nn.Dropout(p=self.args.dropout),
                                         Linear(fc_embeddings, fc_embeddings))
        self.layernorm = nn.LayerNorm(in_features)

    def forward(self, G1,G2):
        x1, x2, = self.encoder(G1,G2)
        x1 = torch.sum(x1.reshape(x1.shape[0],x1.shape[2],-1).squeeze(), dim=1)
        x2 = torch.sum(x2.reshape(x2.shape[0],x2.shape[2],-1).squeeze(), dim=1)
        x1=self.layernorm(x1)
        x2=self.layernorm(x2)
        y1 = self.projectionHead(x1)
        y2 = self.projectionHead(x2)
        return y1,y2

    def get_embeddings(self, G1,G2):
        x1, x2 = self.encoder.forward(G1,G2)
        return x1, x2

    def loss_cal(self, x, x_aug):
        T = self.args.T
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.args = args
        if self.args.share_qk:
            q = torch.nn.Linear(args.embedding_size, args.embedding_size * args.n_heads, bias=args.msa_bias)
            k = torch.nn.Linear(args.embedding_size, args.embedding_size * args.n_heads, bias=args.msa_bias)
        else:
            q = k = None
        self.embedding_learning = GCNTransformerEncoder(args, q, k).to(args.device)
        self.embedding_interaction = CrossTransformer(args, q, k).to(args.device)

    def forward(self, G1,G2):
        x_0 = G1['x']
        adj_0 = G1['adj']
        mask_0 = G1['mask']
        dist_0 = G1['dist']
        x_1 = G2['x']
        adj_1 = G2['adj']
        mask_1 = G2['mask']
        dist_1 = G2['dist']
        mask_ij = None
        embeddings_0 = self.embedding_learning(x_0, adj_0, mask_0, dist_0)
        embeddings_1 = self.embedding_learning(x_1, adj_1, mask_1, dist_1)
        sim_mat1, sim_mat2 = self.embedding_interaction(embeddings_0, mask_0, embeddings_1, mask_1, mask_ij)

        return sim_mat1, sim_mat2
