import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, args):
        super(FeedForwardNetwork, self).__init__()
        self.args = args

        hidden_size = args.n_max_nodes ** 2
        ffn_size = args.channel_ffn_size
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.args = args
        args.n_max_nodes = args.n_max_nodes
        self.num_heads = num_heads = args.n_channel_transformer_heads
        embedding_size = args.n_max_nodes ** 2

        self.att_size = embedding_size
        self.scale = embedding_size ** -0.5

        self.linear_q = nn.Linear(embedding_size, num_heads * embedding_size, bias=args.msa_bias)
        self.linear_k = nn.Linear(embedding_size, num_heads * embedding_size, bias=args.msa_bias)
        self.linear_v = nn.Linear(embedding_size, num_heads * embedding_size, bias=args.msa_bias)
        self.att_dropout = nn.Dropout(args.dropout)
        self.output_layer = nn.Linear(num_heads * embedding_size, embedding_size)

        torch.nn.init.xavier_uniform_(self.linear_q.weight)
        torch.nn.init.xavier_uniform_(self.linear_k.weight)
        torch.nn.init.xavier_uniform_(self.linear_v.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x, flatten_mask):
        d_k = self.att_size
        d_v = self.att_size
        batch_size = x.size(0)

        q = self.linear_q(x).view(batch_size, -1, self.num_heads, d_k).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, d_k).transpose(1, 2).transpose(-1, -2)
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, d_v).transpose(1, 2)
        # if flatten_mask is not None:
        #     q = torch.einsum('bhnl,bl->bhnl', q, flatten_mask)
        #     k = torch.einsum('bhln,bl->bhln', k, flatten_mask)
        #     v = torch.einsum('bhnl,bl->bhnl', v, flatten_mask)

        q = q * self.scale
        a = torch.matmul(q, k)

        a = torch.softmax(a, dim=3)
        a = self.att_dropout(a)
        y = a.matmul(v).transpose(-2, -3).contiguous().view(batch_size, -1, self.num_heads * d_v)
        y = self.output_layer(y)
        # if flatten_mask is not None:
        #     y = torch.einsum('bnl,bl->bnl', y, flatten_mask)

        return y


class ChannelAlignment(nn.Module):
    def __init__(self, args):
        super(ChannelAlignment, self).__init__()
        self.args = args
        if self.args.n_max_nodes > args.pooling_res:
            args.n_max_nodes = args.pooling_res
        # if args.n_max_nodes >20:
        #     args.n_max_nodes = args.pooling_res
        self.self_attention_norm = nn.LayerNorm(args.n_max_nodes ** 2)
        self.self_attention = MultiHeadAttention(args)
        self.self_attention_dropout = nn.Dropout(args.dropout)
        self.ffn_norm = nn.LayerNorm(args.n_max_nodes ** 2)
        self.ffn = FeedForwardNetwork(args)
        self.ffn_dropout = nn.Dropout(args.dropout)

    def forward(self, x, mask_ij):
        # (B, Heads, n, n) -> (B, Heads, L)
        B = x.size(0)
        H = x.size(1)
        x = x.view(B, H, -1)
        # flatten_mask = mask_ij.view(B, -1) if self.args.dataset != 'IMDBMulti' else None
        flatten_mask = None
        # print(x.size())
        y = self.self_attention_norm(x)
        y = self.self_attention(y, flatten_mask)
        y = self.self_attention_dropout(y)
        x = x + y
        y = self.ffn_norm(x)
        y = self.ffn(y)
        # if flatten_mask is not None:
        #     y = torch.einsum('bhl,bl->bhl', y, flatten_mask)
        y = self.ffn_dropout(y)
        x = x + y
        x = x.view(B, H, self.args.n_max_nodes, self.args.n_max_nodes)
        return x


def padding(x, shape_len):
    # 左右上下
    pad = nn.ZeroPad2d(padding=(0, shape_len, 0, 0))

    return pad(x)
