import torch
import torch.nn as nn


class CrossTransformer(nn.Module):
    def __init__(self, args, q=None, k=None):
        super(CrossTransformer, self).__init__()
        self.args = args

        self.cross_attention = CrossAttention(args)

        if self.args.n_max_nodes > self.args.pooling_res:
            self.d = args.pooling_res
            self.pooling = torch.nn.AdaptiveAvgPool2d((args.pooling_res, args.pooling_res))

    def forward(self, embeddings_i, mask_i, embeddings_j, mask_j, mask_ij=None):
        y1, y2 = self.cross_attention(embeddings_i, mask_i, embeddings_j, mask_j, mask_ij)
        if y1.shape[-1] > self.args.pooling_res:
            y = self.pooling(torch.cat([y1, y2], dim=1))
            # y=y.reshape(2,y.shape[0],self.args.GCA_n_heads,y.shape[-1],-1)
            # return y[0],y[1]
            return y[:, :self.args.GCA_n_heads, :, :], y[:, self.args.GCA_n_heads:, :, :]
        return y1, y2


class CrossAttention(nn.Module):
    def __init__(self, args):
        super(CrossAttention, self).__init__()
        self.args = args
        self.n_heads = n_heads = args.GCA_n_heads
        self.embedding_size = embedding_size = args.embedding_size
        # if self.args.readout=='cat':
        #     self.embedding_size = embedding_size = args.embedding_size*3
        self.scale = embedding_size ** -0.5

        self.linear_q = nn.Linear(embedding_size, n_heads * embedding_size, bias=args.msa_bias)
        self.linear_k = nn.Linear(embedding_size, n_heads * embedding_size, bias=args.msa_bias)

    def forward(self, embeddings_i, mask_i, embeddings_j, mask_j, mask_ij=None):
        # batch_size = embeddings_i.shape[0]
        node_size_i = embeddings_i.shape[0]
        node_size_j = embeddings_j.shape[0]

        q_i = self.linear_q(embeddings_i).view(node_size_i, -1, self.n_heads, self.embedding_size).transpose(-2, -3)
        k_i = self.linear_k(embeddings_i).view(node_size_i, -1, self.n_heads, self.embedding_size).transpose(-2,-3).transpose(-1, -2)
        q_j = self.linear_q(embeddings_j).view(node_size_j, -1, self.n_heads, self.embedding_size).transpose(-2, -3)
        k_j = self.linear_k(embeddings_j).view(node_size_j, -1, self.n_heads, self.embedding_size).transpose(-2,-3).transpose(-1, -2)

        # q_i = torch.einsum('bhne,bn->bhne', q_i, mask_i)
        # k_i = torch.einsum('bhen,bn->bhen', k_i, mask_i)
        # q_j = torch.einsum('bhne,bn->bhne', q_j, mask_j)
        # k_j = torch.einsum('bhen,bn->bhen', k_j, mask_j)

        a_i = torch.matmul(q_i, k_j)
        a_i *= self.scale

        a_j = torch.matmul(q_j, k_i).transpose(-1, -2)
        a_j *= self.scale

        # a_i = a_i.transpose(0, 1).masked_fill(mask_ij == 0, -1e9).transpose(0, 1)
        a_i = torch.softmax(a_i, dim=3)
        # a_i = torch.einsum('bhij,bij->bhij', a_i, mask_ij)

        # a_j = a_j.transpose(0, 1).masked_fill(mask_ij == 0, -1e9).transpose(0, 1)
        a_j = torch.softmax(a_j, dim=2)
        # a_j = torch.einsum('bhij,bij->bhij', a_j, mask_ij)

        # a = torch.cat([a_i, a_j], dim=1)
        # print(a.size())
        return a_i, a_j
