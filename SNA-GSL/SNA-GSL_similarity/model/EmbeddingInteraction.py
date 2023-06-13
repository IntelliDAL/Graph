import torch
import torch.nn as nn


class CrossTransformer(nn.Module):
    def __init__(self, args, q=None, k=None):
        super(CrossTransformer, self).__init__()
        self.args = args

        self.cross_attention = CrossAttention(args, q, k).to(args.device)

        if args.dataset == 'IMDBMulti':
            self.d = args.pooling_res
            self.pooling = torch.nn.AdaptiveAvgPool2d((args.pooling_res, args.pooling_res)).to(self.args.device)

    def forward(self, embeddings_i, mask_i, embeddings_j, mask_j, mask_ij=None):

        y1, y2 = self.cross_attention(embeddings_i, mask_i, embeddings_j, mask_j, mask_ij)

        if self.args.dataset == 'IMDBMulti':
            y = self.pooling(torch.cat([y1, y2], dim=1))
            # y=y.reshape(2,y.shape[0],self.args.GCA_n_heads,y.shape[-1],-1)
            # return y[0],y[1]
            return y[:, :self.args.n_heads, :, :], y[:, self.args.n_heads:, :, :]
        return y1, y2


class CrossAttention(nn.Module):
    def __init__(self, args, q=None, k=None):
        super(CrossAttention, self).__init__()
        self.args = args
        self.n_heads = n_heads = args.n_heads
        self.embedding_size = embedding_size = args.embedding_size
        self.scale = embedding_size ** -0.5

        self.linear_q = q if q else nn.Linear(embedding_size, n_heads * embedding_size, bias=args.msa_bias)
        self.linear_k = k if k else nn.Linear(embedding_size, n_heads * embedding_size, bias=args.msa_bias)

    def forward(self, embeddings_i, mask_i, embeddings_j, mask_j, mask_ij=None):
        batch_size = embeddings_i.size(0)

        q_i = self.linear_q(embeddings_i).view(batch_size, -1, self.n_heads, self.embedding_size).transpose(-2, -3)
        k_i = self.linear_k(embeddings_i).view(batch_size, -1, self.n_heads, self.embedding_size).transpose(-2, -3).transpose(-1, -2)
        q_j = self.linear_q(embeddings_j).view(batch_size, -1, self.n_heads, self.embedding_size).transpose(-2, -3)
        k_j = self.linear_k(embeddings_j).view(batch_size, -1, self.n_heads, self.embedding_size).transpose(-2, -3).transpose(-1, -2)

        # q_i = torch.einsum('bhne,bn->bhne', q_i, mask_i)
        # k_i = torch.einsum('bhen,bn->bhen', k_i, mask_i)
        # q_j = torch.einsum('bhne,bn->bhne', q_j, mask_j)
        # k_j = torch.einsum('bhen,bn->bhen', k_j, mask_j)

        a_i = torch.matmul(q_i, k_j)
        a_i *= self.scale

        a_j = torch.matmul(q_j, k_i).transpose(-1, -2)
        a_j *= self.scale

        # a_i = a_i.transpose(0, 1).masked_fill(mask_ij == 0, -1e9).transpose(0, 1)
        # a_i = torch.softmax(a_i, dim=3)
        # a_i = torch.einsum('bhij,bij->bhij', a_i, mask_ij)
        #
        # a_j = a_j.transpose(0, 1).masked_fill(mask_ij == 0, -1e9).transpose(0, 1)
        # a_j = torch.softmax(a_j, dim=2)
        # a_j = torch.einsum('bhij,bij->bhij', a_j, mask_ij)
        return a_i, a_j
