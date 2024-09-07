# In[]
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as scio
import random
from sklearn import metrics
from thop import profile
from thop import clever_format
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(123)

#### some values
N_subjects = 871

####

class DenseGCNConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(DenseGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = nn.Parameter(torch.Tensor(self.in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.bias.data = nn.init.constant_(self.bias.data, 0.0)
            # init.xavier_uniform_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

class GCN(nn.Module):
    def __init__(self, in_dim=48, out_dim=48, neg_penalty=0.2):
        super(GCN, self).__init__()
        self.in_dim = in_dim  # 输入的维度
        self.out_dim = out_dim  # 输出的维度
        self.neg_penalty = neg_penalty  # 负值
        self.kernel = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        nn.init.kaiming_normal_(self.kernel)
        # init.uniform_(weight, -stdv, stdv)
        # nn.init.zeros_(layer.bias)
        self.c = 0.85
        self.losses = []

    def forward(self, x, adj):
        # GCN-node
        feature_dim = int(adj.shape[-1])
        eye = torch.eye(feature_dim).cuda()  # 生成对角矩阵 feature_dim * feature_dim
        if x is None:  # 如果没有初始特征
            AXW = torch.tensordot(adj, self.kernel, [[-1], [0]])  # batch_size * num_node * feature_dim
        else:
            XW = torch.tensordot(x, self.kernel, [[-1], [0]])  # batch *  num_node * feature_dim
            AXW = torch.matmul(adj, XW)  # batch *  num_node * feature_dim
        # I_cAXW = eye+self.c*AXW
        I_cAXW = self.c * AXW
        # y_relu = torch.nn.functional.relu(I_cAXW)
        # temp = torch.mean(input=y_relu, dim=-2, keepdim=True) + 1e-6
        # col_mean = temp.repeat([1, feature_dim, 1])
        # y_norm = torch.divide(y_relu, col_mean)  # 正则化后的值
        # output = torch.nn.functional.softplus(y_norm)
        # print(output)
        # output = y_relu
        # 做个尝试
        if self.neg_penalty != 0:
            neg_loss = torch.multiply(torch.tensor(self.neg_penalty),
                                      torch.sum(torch.nn.functional.relu(1e-6 - self.kernel)))
            self.losses.append(neg_loss)
        print(I_cAXW)
        return I_cAXW

class Aggregator(nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()

    def forward(self, node):
        neighbour = node.mailbox['m']
        c = self.aggre(neighbour)
        return {"c": c}

    def aggre(self, neighbour):
        raise NotImplementedError


class MeanAggregator(Aggregator):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def aggre(self, neighbour):
        mean_neighbour = torch.mean(neighbour, dim=1)
        return mean_neighbour


class MaxPoolAggregator(Aggregator):
    def __init__(self, in_feats, out_feats, activation, bias):
        super(MaxPoolAggregator, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.activation = activation
        nn.init.xavier_uniform_(self.linear.weight,
                                gain=nn.init.calculate_gain('relu'))

    def aggre(self, neighbour):
        neighbour = self.linear(neighbour)
        if self.activation:
            neighbour = self.activation(neighbour)
        maxpool_neighbour = torch.max(neighbour, dim=1)[0]
        return maxpool_neighbour


class Bundler(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout, bias=True):
        super(Bundler, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_feats * 2, out_feats, bias)
        self.activation = activation

        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

    def concat(self, h, aggre_result):
        bundle = torch.cat((h, aggre_result), 1)
        bundle = self.linear(bundle)
        return bundle

    def forward(self, node):
        h = node.data['h']
        c = node.data['c']
        bundle = self.concat(h, c)
        bundle = F.normalize(bundle, p=2, dim=1)
        if self.activation:
            bundle = self.activation(bundle)
        return {"h": bundle}

class GraphSageLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type, bn=False, bias=True):
        super(GraphSageLayer, self).__init__()
        self.use_bn = bn
        self.bundler = Bundler(in_feats, out_feats, activation, dropout,
                               bias=bias)
        self.dropout = nn.Dropout(p=dropout)

        if aggregator_type == "maxpool":
            self.aggregator = MaxPoolAggregator(in_feats, in_feats,
                                                activation, bias)
        else:
            self.aggregator = MeanAggregator()

    def forward(self, g, h):
        h = self.dropout(h)
        g.ndata['h'] = h
        if self.use_bn and not hasattr(self, 'bn'):
            device = h.device
            self.bn = nn.BatchNorm1d(h.size()[1]).to(device)
        g.update_all(fn.copy_src(src='h', out='m'), self.aggregator,
                     self.bundler)
        if self.use_bn:
            h = self.bn(h)
        h = g.ndata.pop('h')

        return h

class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=8,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        # self.kernel_size = kernel_size
        self.kernel_size1, self.kernel_size2 = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)
        

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, self.kernel_size1, self.kernel_size2), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x, att):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        # x batch 1 200 200   # att batch 1
        # print(att.shape)
        softmax_attention = self.attention(x)
        # print(softmax_attention.shape) # batch K # weight alpha
        att = torch.unsqueeze(att, 1)
        att = att.repeat(1, self.K)
        # print(att.shape)
        softmax_attention = att * softmax_attention 
        # print(softmax_attention)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积 
        # print(x.shape) # 1 batch nodes nodes
        weight = self.weight.view(self.K, -1)
        # print(weight.shape) # K 

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size1, self.kernel_size2)
        # print(aggregate_weight.shape) # Batch*K channel nodes 1
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

class E2E(nn.Module):

    def __init__(self, in_channel, out_channel, input_shape, **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.d = input_shape[0]
        self.k = 4
        # self.conv1xd = nn.Conv2d(in_channel, out_channel, (self.d, 1))
        # self.convdx1 = nn.Conv2d(in_channel, out_channel, (1, self.d))
        self.conv1xd = Dynamic_conv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=(self.d, 1), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.convdx1 = Dynamic_conv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=(1, self.d), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        
        self.nodes = 64

    def forward(self, A, att):
        
#         print(A.shape)
        A = A.view(-1, self.in_channel, self.nodes, self.nodes)

        a = self.conv1xd(A, att)
        b = self.convdx1(A, att)

        # print(a.shape)
        concat1 = torch.cat([a]*self.d, 2)
        concat2 = torch.cat([b]*self.d, 3)
        
        # A = torch.mean(concat1+concat2, 1)
        # print('e2e', (concat1+concat2).shape)
        return concat1+concat2

class DeE2E(nn.Module):

    def __init__(self, in_channel, out_channel, input_shape, **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.d = input_shape[0]
        self.conv1xd = nn.ConvTranspose2d(in_channel, out_channel, (self.d, 1))
        self.convdx1 = nn.ConvTranspose2d(in_channel, out_channel, (1, self.d))
        self.nodes = 64

    def forward(self, A):
        
        A = A.view(-1, self.in_channel, self.nodes, self.nodes)

        A1 = torch.sum(A, dim=2).reshape(A.shape[0], A.shape[1], 1, A.shape[3])
        # print('A1,',A1.shape)
        A2 = torch.sum(A, dim=3).reshape(A.shape[0], A.shape[1], A.shape[2], 1)

        a = self.conv1xd(A1)
        b = self.convdx1(A2)

        # print('a ', a.shape)
        # print('b ', b.shape)

        # concat1 = torch.cat([a]*self.d, 2)
        # concat2 = torch.cat([b]*self.d, 3)
        
        # A = torch.mean(concat1+concat2, 1)
        # print('e2e', (concat1+concat2).shape)
        return (a+b)/2.0

class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super(FeedForwardNetwork, self).__init__()

        hidden_size = 48
        ffn_size = 64
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

        nn.init.kaiming_normal_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)

        nn.init.kaiming_normal_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)


    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class CrossTransformer(nn.Module):
    def __init__(self, q=None, k=None):
        super(CrossTransformer, self).__init__()

        self.cross_attention = CrossAttention()

        # if args.dataset == 'IMDBMulti':
        #     pooling_kernel = args.n_max_nodes - args.pooling_res + 1
        #     self.d = args.pooling_res
        #     self.pooling = torch.nn.AdaptiveAvgPool2d((args.pooling_res,args.pooling_res))

        # if self.args.channel_activate:
        #     self.channel_transformer = ChannelTransformer(args)

        # self.conv = BrainNetCNN(args)

    def forward(self, embeddings_i):

        s = self.cross_attention(embeddings_i)

        # y = torch.cat([y_i, y_j], dim=1)

        # if self.args.dataset == 'IMDBMulti':
        #     y = self.pooling(y)

        # if self.args.channel_activate:
        #     y = self.channel_transformer(y)

        # s = self.conv(y)

        return s


class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()
        self.n_heads = 4 # 几个头 # 2-50 675 1-60 659 3-50 681 4-60 649 5-675 6-664 7-654 8-0.673
        self.embedding_size = 48
        self.scale = self.embedding_size ** -0.5

        self.linear_q = nn.Linear(self.embedding_size, self.n_heads * self.embedding_size)
        self.linear_k = nn.Linear(self.embedding_size, self.n_heads * self.embedding_size)

        nn.init.kaiming_normal_(self.linear_q.weight)
        nn.init.zeros_(self.linear_q.bias)

        nn.init.kaiming_normal_(self.linear_k.weight)
        nn.init.zeros_(self.linear_k.bias)

        self.W = nn.Parameter(torch.Tensor(200, 200))
        nn.init.xavier_uniform_(self.W)



    def forward(self, embeddings_i):

        q_i = self.linear_q(embeddings_i).view(embeddings_i.size(0), self.n_heads, embeddings_i.size(1), -1)
        k_i = self.linear_k(embeddings_i).view(embeddings_i.size(0), self.n_heads, embeddings_i.size(1), -1).transpose(-1, -2)

        # q_j = self.linear_q(embeddings_j).view(embeddings_i.size(0), self.args.n_heads, embeddings_i.size(1), -1)
        # k_j = self.linear_k(embeddings_j).view(embeddings_i.size(0), self.args.n_heads, embeddings_i.size(1), -1).transpose(-1, -2)

        x_i = torch.matmul(q_i, k_i)
        x_i_T = x_i.transpose(-1, -2)
        x_i = x_i * x_i_T
        # x_i = x_i * self.Wx
        # x_i *= self.scale

        # x_j = torch.matmul(q_j, k_i)
        # x_j *= self.scale

        return x_i

class MultiHeadAttention(nn.Module):
    def __init__(self, dropout=0.2):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = 8 # 0.667-2 0.656-4 680-8
        embedding_size = 48

        self.att_size = embedding_size
        self.scale = embedding_size ** -0.5

        self.linear_q = nn.Linear(embedding_size, self.num_heads * embedding_size)
        self.linear_k = nn.Linear(embedding_size, self.num_heads * embedding_size)

        nn.init.kaiming_normal_(self.linear_q.weight)
        nn.init.zeros_(self.linear_q.bias)

        nn.init.kaiming_normal_(self.linear_k.weight)
        nn.init.zeros_(self.linear_k.bias)

        self.linear_v = nn.Linear(embedding_size, self.num_heads * embedding_size, bias=False)
        self.att_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(self.num_heads * embedding_size, embedding_size)

        torch.nn.init.xavier_uniform_(self.linear_v.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

        self.loss_rec = nn.MSELoss()


    def forward(self, x, x_corr, dist=None):
        orig_q_size = x.size()
        # print('x', x.shape) # batch*nodes 7 48

        d_k = self.att_size
        d_v = self.att_size
        batch_size = x.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        # q = self.linear_q(x).view(batch_size, -1, self.num_heads, d_k).transpose(-2, -3)
        # # print('q', q.shape) #32 8 200 48
        # k = self.linear_k(x).view(batch_size, -1, self.num_heads, d_k).transpose(-2, -3).transpose(-1, -2)
        # v = self.linear_v(x).view(batch_size, -1, self.num_heads, d_v).transpose(-2, -3)

        q = self.linear_q(x).view(x.size(0), self.num_heads, x.size(1), -1)
        k = self.linear_k(x).view(x.size(0), self.num_heads, x.size(1), -1).transpose(-1, -2)
        v = self.linear_v(x).view(x.size(0), self.num_heads, x.size(1), -1)


        q = q * self.scale
        a = torch.matmul(q, k)
        # print('a', a.shape) # batch*nodes heads 7 7 

        a_heads = a.size(1)
        a = a.reshape(a.size(0)//64, -1, a.size(2), a.size(3)) # batch nodes*heads 7 7 
        # print(a.shape)
        x_corr = torch.unsqueeze(x_corr, 1)
        x_corr = x_corr.repeat(1, a.size(1), 1, 1)
        # a = a * x_corr
        loss_corr = self.loss_rec(a, x_corr)
        a = a.reshape(a.size(0)*64, a_heads, a.size(2), a.size(3))
        # a = a.reshape(a.size(0)*a.size(1))

        # if self.args.dist_decay != 0:
        #     assert 1 >= self.args.dist_decay >= 0
        #     vanish_iter = self.args.iter_val_start * self.args.dist_decay
        #     dist_decay = max(0, (vanish_iter - self.args.temp['cur_iter'])) / vanish_iter
        # else:
        #     dist_decay = 1
        # if dist is not None:
        #     dist *= dist_decay
        #     a += torch.stack([dist] * self.args.n_heads, dim=1).to(self.args.device)

        a = torch.softmax(a, dim=3)
        a = self.att_dropout(a)
        y = a.matmul(v).contiguous().view(batch_size, -1, self.num_heads * d_v)

        y = self.output_layer(y)

        assert y.size() == orig_q_size
        return y, loss_corr


nums_train = np.ones(200) # 制作mask模板
# nums_train[:150] = 0 # 根据设置的nodes number 决定多少是mask 即mask比例 # 写错了 应该是[:nodes]
Mask_train = nums_train.reshape(nums_train.shape[0], 1) * nums_train # 200 200
for i in range(150):
    Mask_train[i][:150] = 0
# np.repeat(Mask_train, X_train.shape[0], 0)
Mask_train_tensor = torch.from_numpy(Mask_train).float().to(device)
# Mask_train_tensor = tf.cast(Mask_train_tensor, tf.float32)

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)
        #c_x = c
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)
#        print(logits)
        return logits

class Discriminator2(nn.Module):
    def __init__(self, n_h):
        super(Discriminator2, self).__init__()
        # self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.f_k = nn.Linear(n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h_pl, h_mi, s_bias1=None, s_bias2=None): # c_k
        #c_x = torch.unsqueeze(c, 1)
        #c_x = c_x.expand_as(h_pl)
        #c_x = c
        # sc_1 = torch.squeeze(self.f_k(h_pl), 2)
        # sc_2 = torch.squeeze(self.f_k(h_mi), 2)
        sc_1 = h_pl.reshape(h_pl.size(0)*h_pl.size(1), -1)
        sc_2 = h_mi.reshape(h_mi.size(0)*h_mi.size(1), -1)
        # print(sc_2.shape)
        sc_1 = self.f_k(sc_1)
        sc_2 = self.f_k(sc_2)
        sc_1 = sc_1.reshape(h_pl.size(0), h_pl.size(1))
        sc_2 = sc_2.reshape(h_mi.size(0), h_mi.size(1))

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)
#        print(logits)
        return logits

class Model(nn.Module):
    def __init__(self, dropout=0.5, num_class=1, nodes=64):
        super().__init__()

        self.k = 8
        # self.attention_path = attention2d(in_planes=7, ratios=0.25, K=7, temperature=34)
        self.attention_path = nn.Linear(nodes*nodes, 1)

        # origin
        # self.e2e = nn.Sequential(
        #     E2E(1, 8, (nodes, nodes)),
        #     nn.LeakyReLU(0.33),
        #     E2E(8, 8, (nodes, nodes)), # 0.642
        #     nn.LeakyReLU(0.33),
        # )
        self.n2g_global = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        self.e2e_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_2 = nn.LeakyReLU(0.33)
        self.e2e_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_4 = nn.LeakyReLU(0.33)
        
        self.e2n_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_2 = nn.LeakyReLU(0.33)
        
        self.n2g = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # homo
        self.e2e_DMN_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DMN_2 = nn.LeakyReLU(0.33)
        self.e2e_DMN_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DMN_4 = nn.LeakyReLU(0.33)
        
        self.e2n_DMN_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DMN_2 = nn.LeakyReLU(0.33)
        
        self.n2g_DMN = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )
        
        self.e2e_CEN_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_CEN_2 = nn.LeakyReLU(0.33)
        self.e2e_CEN_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_CEN_4 = nn.LeakyReLU(0.33)
        
        self.e2n_CEN_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_CEN_2 = nn.LeakyReLU(0.33)
        
        self.n2g_CEN = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        self.e2e_SN_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_SN_2 = nn.LeakyReLU(0.33)
        self.e2e_SN_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_SN_4 = nn.LeakyReLU(0.33)
        
        self.e2n_SN_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_SN_2 = nn.LeakyReLU(0.33)
        
        self.n2g_SN = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # DMN-CEN_DMN
        self.e2e_DMN_CEN_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DMN_CEN_2 = nn.LeakyReLU(0.33)
        self.e2e_DMN_CEN_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DMN_CEN_4 = nn.LeakyReLU(0.33)
        
        self.e2n_DMN_CEN_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DMN_CEN_2 = nn.LeakyReLU(0.33)
        
        self.n2g_DMN_CEN = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # DMN_SN_DMN
        self.e2e_DMN_SN_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DMN_SN_2 = nn.LeakyReLU(0.33)
        self.e2e_DMN_SN_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DMN_SN_4 = nn.LeakyReLU(0.33)
        
        self.e2n_DMN_SN_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DMN_SN_2 = nn.LeakyReLU(0.33)
        
        self.n2g_DMN_SN = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # CEN_SN_CEN
        self.e2e_CEN_SN_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_CEN_SN_2 = nn.LeakyReLU(0.33)
        self.e2e_CEN_SN_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_CEN_SN_4 = nn.LeakyReLU(0.33)
        
        self.e2n_CEN_SN_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_CEN_SN_2 = nn.LeakyReLU(0.33)
        
        self.n2g_CEN_SN = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # D_C_S
        self.e2e_DCS_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DCS_2 = nn.LeakyReLU(0.33)
        self.e2e_DCS_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DCS_4 = nn.LeakyReLU(0.33)
        
        self.e2n_DCS_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DCS_2 = nn.LeakyReLU(0.33)
        
        self.n2g_DCS = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # D_S_C
        self.e2e_DSC_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DSC_2 = nn.LeakyReLU(0.33)
        self.e2e_DSC_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DSC_4 = nn.LeakyReLU(0.33)
        
        self.e2n_DSC_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DSC_2 = nn.LeakyReLU(0.33)
        
        self.n2g_DSC = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # C_D_S
        self.e2e_CDS_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_CDS_2 = nn.LeakyReLU(0.33)
        self.e2e_CDS_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_CDS_4 = nn.LeakyReLU(0.33)
        
        self.e2n_CDS_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_CDS_2 = nn.LeakyReLU(0.33)
        
        self.n2g_CDS = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # DDC
        self.e2e_DDC_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DDC_2 = nn.LeakyReLU(0.33)
        self.e2e_DDC_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDC_4 = nn.LeakyReLU(0.33)
        
        self.e2n_DDC_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DDC_2 = nn.LeakyReLU(0.33)
        
        self.n2g_DDC = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # DDS
        self.e2e_DDS_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DDS_2 = nn.LeakyReLU(0.33)
        self.e2e_DDS_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDS_4 = nn.LeakyReLU(0.33)
        
        self.e2n_DDS_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DDS_2 = nn.LeakyReLU(0.33)
        
        self.n2g_DDS = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # CCD
        self.e2e_CCD_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_CCD_2 = nn.LeakyReLU(0.33)
        self.e2e_CCD_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCD_4 = nn.LeakyReLU(0.33)
        
        self.e2n_CCD_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_CCD_2 = nn.LeakyReLU(0.33)
        
        self.n2g_CCD = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # CCS
        self.e2e_CCS_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_CCS_2 = nn.LeakyReLU(0.33)
        self.e2e_CCS_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCS_4 = nn.LeakyReLU(0.33)
        
        self.e2n_CCS_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_CCS_2 = nn.LeakyReLU(0.33)
        
        self.n2g_CCS = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # SSD
        self.e2e_SSD_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_SSD_2 = nn.LeakyReLU(0.33)
        self.e2e_SSD_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_SSD_4 = nn.LeakyReLU(0.33)
        
        self.e2n_SSD_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_SSD_2 = nn.LeakyReLU(0.33)
        
        self.n2g_SSD = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # SSC
        self.e2e_SSC_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_SSC_2 = nn.LeakyReLU(0.33)
        self.e2e_SSC_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_SSC_4 = nn.LeakyReLU(0.33)
        
        self.e2n_SSC_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_SSC_2 = nn.LeakyReLU(0.33)
        
        self.n2g_SSC = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # DDCC
        self.e2e_DDCC_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DDCC_2 = nn.LeakyReLU(0.33)
        self.e2e_DDCC_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDCC_4 = nn.LeakyReLU(0.33)
        
        self.e2n_DDCC_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DDCC_2 = nn.LeakyReLU(0.33)
        
        self.n2g_DDCC = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # DDSS
        self.e2e_DDSS_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DDSS_2 = nn.LeakyReLU(0.33)
        self.e2e_DDSS_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDSS_4 = nn.LeakyReLU(0.33)
        
        self.e2n_DDSS_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DDSS_2 = nn.LeakyReLU(0.33)
        
        self.n2g_DDSS = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # CCSS
        self.e2e_CCSS_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_CCSS_2 = nn.LeakyReLU(0.33)
        self.e2e_CCSS_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCSS_4 = nn.LeakyReLU(0.33)
        
        self.e2n_CCSS_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_CCSS_2 = nn.LeakyReLU(0.33)
        
        self.n2g_CCSS = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # DDCS
        self.e2e_DDCS_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DDCS_2 = nn.LeakyReLU(0.33)
        self.e2e_DDCS_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDCS_4 = nn.LeakyReLU(0.33)
        
        self.e2n_DDCS_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DDCS_2 = nn.LeakyReLU(0.33)
        
        self.n2g_DDCS = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # DDSC
        self.e2e_DDSC_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DDSC_2 = nn.LeakyReLU(0.33)
        self.e2e_DDSC_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DDSC_4 = nn.LeakyReLU(0.33)
        
        self.e2n_DDSC_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DDSC_2 = nn.LeakyReLU(0.33)
        
        self.n2g_DDSC = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # CCDS
        self.e2e_CCDS_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_CCDS_2 = nn.LeakyReLU(0.33)
        self.e2e_CCDS_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCDS_4 = nn.LeakyReLU(0.33)
        
        self.e2n_CCDS_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_CCDS_2 = nn.LeakyReLU(0.33)
        
        self.n2g_CCDS = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # CCSD
        self.e2e_CCSD_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_CCSD_2 = nn.LeakyReLU(0.33)
        self.e2e_CCSD_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_CCSD_4 = nn.LeakyReLU(0.33)
        
        self.e2n_CCSD_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_CCSD_2 = nn.LeakyReLU(0.33)
        
        self.n2g_CCSD = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # SSDC
        self.e2e_SSDC_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_SSDC_2 = nn.LeakyReLU(0.33)
        self.e2e_SSDC_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_SSDC_4 = nn.LeakyReLU(0.33)
        
        self.e2n_SSDC_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_SSDC_2 = nn.LeakyReLU(0.33)
        
        self.n2g_SSDC = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # SSCD
        self.e2e_SSCD_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_SSCD_2 = nn.LeakyReLU(0.33)
        self.e2e_SSCD_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_SSCD_4 = nn.LeakyReLU(0.33)
        
        self.e2n_SSCD_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_SSCD_2 = nn.LeakyReLU(0.33)
        
        self.n2g_SSCD = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        # DSCD
        self.e2e_DSCD_1 = E2E(1, 8, (nodes, nodes))
        self.e2e_DSCD_2 = nn.LeakyReLU(0.33)
        self.e2e_DSCD_3 = E2E(8, 8, (nodes, nodes))
        self.e2e_DSCD_4 = nn.LeakyReLU(0.33)
        
        self.e2n_DSCD_1 = Dynamic_conv2d(in_planes=8, out_planes=48, kernel_size=(1, nodes), stride=1, padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.e2n_DSCD_2 = nn.LeakyReLU(0.33)
        
        self.n2g_DSCD = nn.Sequential(
            nn.Conv2d(48, 64, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )




        self.de_e2n = nn.Sequential(
            nn.ConvTranspose2d(48, 8, (1, nodes)),
            nn.LeakyReLU(0.33),
        )

        self.de_e2e = nn.Sequential(
            DeE2E(8, 8, (nodes, nodes)),
            nn.LeakyReLU(0.33),
            DeE2E(8, 1, (nodes, nodes)),
            nn.LeakyReLU(0.33),
        )

        self.attW = nn.Linear(26, 1)
        self.attW_diff = nn.Linear(26, nodes)
        self.Lrelu = nn.LeakyReLU(0.33)

        self.linear = nn.Sequential(
            nn.Linear(64, 64),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
            nn.Linear(64, 10),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
            nn.Linear(10, num_class)
        )       

        self.nodeclf = nn.Linear(64, 4)
        # fea_len, self.pool_size, False, 0.2, "maxpool"
        self.assign_dim = nodes
        # self.pool_gc = GraphSageLayer(
        #     nodes,
        #     self.assign_dim,
        #     activation=False,
        #     dropout=0.3,
        #     aggregator_type="maxpool")

        self.pool_gc = DenseGCNConv(200, self.assign_dim)
        
        self.assign_tran = None

        self.GC = DenseGCNConv(48, 48)

        self.GT = CrossTransformer()

        # GraphTransformer Encoder
        self.self_attention_norm = nn.LayerNorm(48)
        self.self_attention = MultiHeadAttention()
        self.self_attention_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(48)
        self.ffn = FeedForwardNetwork()
        self.ffn_dropout = nn.Dropout(dropout)

        self.MI = Discriminator2(48)
        self.MI_DMN = Discriminator2(48)
        self.MI_CEN = Discriminator2(48)
        self.MI_SN = Discriminator2(48)
        self.MI_DMN_CEN = Discriminator2(48)
        self.MI_DMN_SN = Discriminator2(48)
        self.MI_CEN_SN = Discriminator2(48)
        self.MI_DCS = Discriminator2(48)
        self.MI_DSC = Discriminator2(48)
        self.MI_CDS = Discriminator2(48)
        self.MI_DDC = Discriminator2(48)
        self.MI_DDS = Discriminator2(48)
        self.MI_CCD = Discriminator2(48)
        self.MI_CCS = Discriminator2(48)
        self.MI_SSD = Discriminator2(48)
        self.MI_SSC = Discriminator2(48)
        self.MI_DDCC = Discriminator2(48)
        self.MI_DDSS = Discriminator2(48)
        self.MI_CCSS = Discriminator2(48)
        self.MI_DDCS = Discriminator2(48)
        self.MI_DDSC = Discriminator2(48)
        self.MI_CCDS = Discriminator2(48)
        self.MI_CCSD = Discriminator2(48)
        self.MI_SSDC = Discriminator2(48)
        self.MI_SSCD = Discriminator2(48)
        self.MI_DSCD = Discriminator2(48)
        # self.MI_L_in_R_out = Discriminator2(48)
        # self.MI_L_out_R_in = Discriminator2(48)
        # self.MI_L_out_R_out = Discriminator2(48)
        # self.MI_R_in_R_out = Discriminator2(48)

        self.MI_path = Discriminator2(48*nodes)

        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.GCN = GCN()

        self.weight_attention = nn.Parameter(torch.Tensor(64, 26))
        self.bias_attention = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()
        self.loss_rec = nn.MSELoss()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_attention, a=0)
        nn.init.zeros_(self.bias_attention)
 
    def MaskAutoEncoder(self, e2n, A, masked_x): # masked_x 32 50 48
        e2n_encoder = torch.squeeze(e2n)
        # print('e2n_encoder ', e2n_encoder.shape) # 16 200
        # print('masked_x ', masked_x.shape)
        masked_x = masked_x.permute(0, 2, 1) # 32 48 50
        e2n_encoder = torch.cat((e2n_encoder, masked_x), -1) # 补上了masked
        # print('e2n_encoder ', e2n_encoder.shape) # 32 48 200
        e2n_encoder_T = e2n_encoder.permute(0, 2, 1) # batch 200 48
        # print(temp.shape)
        # print('A ', A.shape) # 200 200
        # print(A[0])
        # e2n_encoder_T = self.GCN(e2n_encoder_T, A)
        e2n_encoder_T = self.GC(e2n_encoder_T, A)
        e2n_encoder = e2n_encoder_T.permute(0, 2, 1)
        Z = torch.matmul(e2n_encoder_T, e2n_encoder) # batch 200 200
        # print('Z ', Z.shape)
        # Z = nn.sigmoid(Z) # 正相关 负相关分离
        # 哈达姆乘
        Z = Z * Mask_train_tensor
        # print(Mask_train_tensor)
        # print(Z[0][199])

        return Z
        # Z = K.expand_dims(Z, axis=-1)


    def decoder(self):
        self.d2, self.d2_w, self.d2_b = de_e2n(tf.nn.relu(e3),
            [self.batch_size, self.graph_size[0], self.graph_size[0], self.gf_dim*2],k_h=self.graph_size[0], name='g_d2', with_w=True)
        d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
        d2 = tf.concat([d2, e2], 3)
        # d2 is (300 x 300 )
        self.d3, self.d3_w, self.d3_b = de_e2e(tf.nn.relu(d2),
            [self.batch_size,self.graph_size[0], self.graph_size[0], int(self.gf_dim)],k_h=self.graph_size[0], name='g_d3', with_w=True)
        d3 = self.g_bn_d3(self.d3)
        d3 = tf.concat([d3, e1], 3)
        # d3 is (300 x 300 )

        self.d4, self.d4_w, self.d4_b = de_e2e(tf.nn.relu(d3),
            [self.batch_size, self.graph_size[0], self.graph_size[0], self.output_c_dim],k_h=self.graph_size[0], name='g_d4', with_w=True)

    def Pool(self, g, h):
        assign_tensor = self.pool_gc(g, h)
        assign_tensor = F.softmax(assign_tensor, dim=1)
        # assign_tensor = F.relu(assign_tensor)
        # print(assign_tensor[0])
        # assign_tensor = torch.split(assign_tensor, g.batch_num_nodes().tolist())
        # assign_tensor = torch.block_diag(*assign_tensor)

        # adj = g.adjacency_matrix(transpose=True, ctx="cuda")
        adj_new = torch.matmul(h, assign_tensor)
        # print(assign_tensor.shape)
        adj_new = torch.matmul(assign_tensor.reshape(assign_tensor.shape[0], assign_tensor.shape[2], assign_tensor.shape[1]), adj_new)
        
        self.assign_tran = assign_tensor

        # adj_new = F.relu(adj_new)
        # print(adj_new)

        return adj_new

    def forward(self, x, A, masked_x, flag):

        # tong zhi
        x_DMN = x * Brain_DMN_tensor
        x_CEN = x * Brain_CEN_tensor
        x_SN = x * Brain_SN_tensor

        x_DMN_CEN = x * Brain_DMN_CEN_tensor
        x_DMN_SN = x * Brain_DMN_SN_tensor
        x_CEN_SN = x * Brain_CEN_SN_tensor

        x_DCS = x * Brain_D_C_S_tensor
        x_DSC = x * Brain_D_S_C_tensor
        x_CDS = x * Brain_C_D_S_tensor
        x_DDC = x * Brain_D_D_C_tensor
        x_DDS = x * Brain_D_D_S_tensor
        x_CCD = x * Brain_D_D_C_tensor
        x_CCS = x * Brain_C_C_S_tensor
        x_SSD = x * Brain_D_S_S_tensor
        x_SSC = x * Brain_C_S_S_tensor

        x_DDCC = x * Brain_D_D_C_C_tensor
        x_DDSS = x * Brain_D_D_S_S_tensor
        x_CCSS = x * Brain_C_C_S_S_tensor
        x_DDCS = x * Brain_D_D_C_S_tensor
        x_DDSC = x * Brain_D_D_S_C_tensor
        x_CCDS = x * Brain_C_C_D_S_tensor
        x_CCSD = x * Brain_C_C_S_D_tensor
        x_SSDC = x * Brain_S_S_D_C_tensor
        x_SSCD = x * Brain_S_S_C_D_tensor
        x_DSCD = x * Brain_three_hete_tensor

        # con
        x_att = torch.cat((x.reshape(x.size(0), 1, x.size(1), x.size(2)), 
        x_DMN.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_CEN.reshape(x_CEN.size(0), 1, x_CEN.size(1), x_CEN.size(2)), 
        x_SN.reshape(x_SN.size(0), 1, x_SN.size(1), x_SN.size(2)), 
        x_DMN_CEN.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_DMN_SN.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_CEN_SN.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_DCS.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_DSC.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_CDS.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_DDC.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_DDS.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_CCD.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_CCS.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_SSD.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_SSC.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_DDCC.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_DDSS.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_CCSS.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_DDCS.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_DDSC.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_CCDS.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_CCSD.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_SSDC.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_SSCD.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2)), 
        x_DSCD.reshape(x_DMN.size(0), 1, x_DMN.size(1), x_DMN.size(2))), 
        axis=1)
        # print(x_att.shape) # batch 7 nodes nodes
        x_att = self.attention_path(x_att.reshape(x_att.size(0)*x_att.size(1), -1))
        x_att = x_att.reshape(x.size(0), -1)
        x_att = F.sigmoid(x_att)
        # print(x_att) # batch 7

        # print(x[0][0])
        # print(x_L[0][0])
        # x = self.Pool(A, x)
        # print('pool ', x.shape) # 32 32 32

        # print('input', x.shape)
        # x = self.e2e(x, x_att[:, 0]) # 32 8 N N
        x = self.e2e_1(x, x_att[:, 0])
        x = self.e2e_2(x)
        x = self.e2e_3(x, x_att[:, 0])
        x = self.e2e_4(x)
        x = self.e2n_1(x, x_att[:, 0]) # 32 48 N
        x = self.e2n_2(x)
        # print(x.shape) # 32 D 200 1
        # x = self.n2g(x)
        # x = x.view(x.size(0), -1) # all

        # homo graph
        x_DMN = self.e2e_DMN_1(x_DMN, x_att[:, 1])
        x_DMN = self.e2e_DMN_2(x_DMN)
        x_DMN = self.e2e_DMN_3(x_DMN, x_att[:, 1])
        x_DMN = self.e2e_DMN_4(x_DMN)
        x_DMN = self.e2n_DMN_1(x_DMN, x_att[:, 1]) # 32 48 N
        x_DMN = self.e2n_DMN_2(x_DMN)
        # x_DMN = self.n2g_homo(x_DMN)
        # x_DMN = x_DMN.view(x_DMN.size(0), -1) # x_DMN
        x_CEN = self.e2e_CEN_1(x_CEN, x_att[:, 2])
        x_CEN = self.e2e_CEN_2(x_CEN)
        x_CEN = self.e2e_CEN_3(x_CEN, x_att[:, 2])
        x_CEN = self.e2e_CEN_4(x_CEN)
        x_CEN = self.e2n_CEN_1(x_CEN, x_att[:, 2]) # 32 48 N
        x_CEN = self.e2n_CEN_2(x_CEN)
        # x_L_out = self.n2g_homo(x_L_out)
        # x_L_out = x_L_out.view(x_L_out.size(0), -1) # x_L_out
        x_SN = self.e2e_SN_1(x_SN, x_att[:, 3])
        x_SN = self.e2e_SN_2(x_SN)
        x_SN = self.e2e_SN_3(x_SN, x_att[:, 3])
        x_SN = self.e2e_SN_4(x_SN)
        x_SN = self.e2n_SN_1(x_SN, x_att[:, 3]) # 32 48 N
        x_SN = self.e2n_SN_2(x_SN)
        # x_R_in = self.n2g_homo(x_R_in)
        # x_R_in = x_R_in.view(x_R_in.size(0), -1) # x_R_in
        # x_R_out = self.e2e_homo(x_R_out)
        # x_R_out = self.e2n_homo(x_R_out)
        # x_R_out = self.n2g_homo(x_R_out)
        # x_R_out = x_R_out.view(x_R_out.size(0), -1) # x_R_out

        # hetero
        x_DMN_CEN = self.e2e_DMN_CEN_1(x_DMN_CEN, x_att[:, 4])
        x_DMN_CEN = self.e2e_DMN_CEN_2(x_DMN_CEN)
        x_DMN_CEN = self.e2e_DMN_CEN_3(x_DMN_CEN, x_att[:, 4])
        x_DMN_CEN = self.e2e_DMN_CEN_4(x_DMN_CEN)
        x_DMN_CEN = self.e2n_DMN_CEN_1(x_DMN_CEN, x_att[:, 4]) # 32 48 N
        x_DMN_CEN = self.e2n_DMN_CEN_2(x_DMN_CEN)
        # x_DMN_L_out = self.n2g_L_in_L_out(x_DMN_L_out)
        # x_DMN_L_out = x_DMN_L_out.view(x_DMN_L_out.size(0), -1) # x_DMN_L_out
        x_DMN_SN = self.e2e_DMN_SN_1(x_DMN_SN, x_att[:, 5])
        x_DMN_SN = self.e2e_DMN_SN_2(x_DMN_SN)
        x_DMN_SN = self.e2e_DMN_SN_3(x_DMN_SN, x_att[:, 5])
        x_DMN_SN = self.e2e_DMN_SN_4(x_DMN_SN)
        x_DMN_SN = self.e2n_DMN_SN_1(x_DMN_SN, x_att[:, 5]) # 32 48 N
        x_DMN_SN = self.e2n_DMN_SN_2(x_DMN_SN)
        # x_DMN_R_in = self.n2g_L_in_R_in(x_DMN_R_in)
        # x_DMN_R_in = x_DMN_R_in.view(x_DMN_R_in.size(0), -1) # x_DMN_R_in
        x_CEN_SN = self.e2e_CEN_SN_1(x_CEN_SN, x_att[:, 6])
        x_CEN_SN = self.e2e_CEN_SN_2(x_CEN_SN)
        x_CEN_SN = self.e2e_CEN_SN_3(x_CEN_SN, x_att[:, 6])
        x_CEN_SN = self.e2e_CEN_SN_4(x_CEN_SN)
        x_CEN_SN = self.e2n_CEN_SN_1(x_CEN_SN, x_att[:, 6]) # 32 48 N
        x_CEN_SN = self.e2n_CEN_SN_2(x_CEN_SN)
        # x_DMN_R_out = self.n2g_L_in_R_out(x_DMN_R_out)
        # x_DMN_R_out = x_DMN_R_out.view(x_DMN_R_out.size(0), -1) # x_DMN_R_out
        # x_L_out_R_in = self.e2e_L_out_R_in(x_L_out_R_in)
        # x_L_out_R_in = self.e2n_L_out_R_in(x_L_out_R_in)
        # # x_L_out_R_in = self.n2g_L_out_R_in(x_L_out_R_in)
        # # x_L_out_R_in = x_L_out_R_in.view(x_L_out_R_in.size(0), -1) # x_L_out_R_in
        # x_L_out_R_out = self.e2e_L_out_R_out(x_L_out_R_out)
        # x_L_out_R_out = self.e2n_L_out_R_out(x_L_out_R_out)
        # # x_L_out_R_out = self.n2g_L_out_R_out(x_L_out_R_out)
        # # x_L_out_R_out = x_L_out_R_out.view(x_L_out_R_out.size(0), -1) # x_L_out_R_out
        # x_R_in_R_out = self.e2e_R_in_R_out(x_R_in_R_out)
        # x_R_in_R_out = self.e2n_R_in_R_out(x_R_in_R_out)
        # # x_R_in_R_out = self.n2g_R_in_R_out(x_R_in_R_out)
        # # x_R_in_R_out = x_R_in_R_out.view(x_R_in_R_out.size(0), -1) # x_R_in_R_out

        x_DCS = self.e2e_DCS_1(x_DCS, x_att[:, 7])
        x_DCS = self.e2e_DCS_2(x_DCS)
        x_DCS = self.e2e_DCS_3(x_DCS, x_att[:, 7])
        x_DCS = self.e2e_DCS_4(x_DCS)
        x_DCS = self.e2n_DCS_1(x_DCS, x_att[:, 7]) # 32 48 N
        x_DCS = self.e2n_DCS_2(x_DCS)

        x_DSC = self.e2e_DSC_1(x_DSC, x_att[:, 8])
        x_DSC = self.e2e_DSC_2(x_DSC)
        x_DSC = self.e2e_DSC_3(x_DSC, x_att[:, 8])
        x_DSC = self.e2e_DSC_4(x_DSC)
        x_DSC = self.e2n_DSC_1(x_DSC, x_att[:, 8]) # 32 48 N
        x_DSC = self.e2n_DSC_2(x_DSC)

        x_CDS = self.e2e_CDS_1(x_CDS, x_att[:, 9])
        x_CDS = self.e2e_CDS_2(x_CDS)
        x_CDS = self.e2e_CDS_3(x_CDS, x_att[:, 9])
        x_CDS = self.e2e_CDS_4(x_CDS)
        x_CDS = self.e2n_CDS_1(x_CDS, x_att[:, 9]) # 32 48 N
        x_CDS = self.e2n_CDS_2(x_CDS)

        x_DDC = self.e2e_DDC_1(x_DDC, x_att[:, 10])
        x_DDC = self.e2e_DDC_2(x_DDC)
        x_DDC = self.e2e_DDC_3(x_DDC, x_att[:, 10])
        x_DDC = self.e2e_DDC_4(x_DDC)
        x_DDC = self.e2n_DDC_1(x_DDC, x_att[:, 10]) # 32 48 N
        x_DDC = self.e2n_DDC_2(x_DDC)

        x_DDS = self.e2e_DDS_1(x_DDS, x_att[:, 11])
        x_DDS = self.e2e_DDS_2(x_DDS)
        x_DDS = self.e2e_DDS_3(x_DDS, x_att[:, 11])
        x_DDS = self.e2e_DDS_4(x_DDS)
        x_DDS = self.e2n_DDS_1(x_DDS, x_att[:, 11]) # 32 48 N
        x_DDS = self.e2n_DDS_2(x_DDS)

        x_CCD = self.e2e_CCD_1(x_CCD, x_att[:, 12])
        x_CCD = self.e2e_CCD_2(x_CCD)
        x_CCD = self.e2e_CCD_3(x_CCD, x_att[:, 12])
        x_CCD = self.e2e_CCD_4(x_CCD)
        x_CCD = self.e2n_CCD_1(x_CCD, x_att[:, 12]) # 32 48 N
        x_CCD = self.e2n_CCD_2(x_CCD)

        x_CCS = self.e2e_CCS_1(x_CCS, x_att[:, 13])
        x_CCS = self.e2e_CCS_2(x_CCS)
        x_CCS = self.e2e_CCS_3(x_CCS, x_att[:, 13])
        x_CCS = self.e2e_CCS_4(x_CCS)
        x_CCS = self.e2n_CCS_1(x_CCS, x_att[:, 13]) # 32 48 N
        x_CCS = self.e2n_CCS_2(x_CCS)

        x_SSD = self.e2e_SSD_1(x_SSD, x_att[:, 14])
        x_SSD = self.e2e_SSD_2(x_SSD)
        x_SSD = self.e2e_SSD_3(x_SSD, x_att[:, 14])
        x_SSD = self.e2e_SSD_4(x_SSD)
        x_SSD = self.e2n_SSD_1(x_SSD, x_att[:, 14]) # 32 48 N
        x_SSD = self.e2n_SSD_2(x_SSD)

        x_SSC = self.e2e_SSC_1(x_SSC, x_att[:, 15])
        x_SSC = self.e2e_SSC_2(x_SSC)
        x_SSC = self.e2e_SSC_3(x_SSC, x_att[:, 15])
        x_SSC = self.e2e_SSC_4(x_SSC)
        x_SSC = self.e2n_SSC_1(x_SSC, x_att[:, 15]) # 32 48 N
        x_SSC = self.e2n_SSC_2(x_SSC)

        x_DDCC = self.e2e_DDCC_1(x_DDCC, x_att[:, 16])
        x_DDCC = self.e2e_DDCC_2(x_DDCC)
        x_DDCC = self.e2e_DDCC_3(x_DDCC, x_att[:, 16])
        x_DDCC = self.e2e_DDCC_4(x_DDCC)
        x_DDCC = self.e2n_DDCC_1(x_DDCC, x_att[:, 16]) # 32 48 N
        x_DDCC = self.e2n_DDCC_2(x_DDCC)

        x_DDSS = self.e2e_DDSS_1(x_DDSS, x_att[:, 17])
        x_DDSS = self.e2e_DDSS_2(x_DDSS)
        x_DDSS = self.e2e_DDSS_3(x_DDSS, x_att[:, 17])
        x_DDSS = self.e2e_DDSS_4(x_DDSS)
        x_DDSS = self.e2n_DDSS_1(x_DDSS, x_att[:, 17]) # 32 48 N
        x_DDSS = self.e2n_DDSS_2(x_DDSS)

        x_CCSS = self.e2e_CCSS_1(x_CCSS, x_att[:, 18])
        x_CCSS = self.e2e_CCSS_2(x_CCSS)
        x_CCSS = self.e2e_CCSS_3(x_CCSS, x_att[:, 18])
        x_CCSS = self.e2e_CCSS_4(x_CCSS)
        x_CCSS = self.e2n_CCSS_1(x_CCSS, x_att[:, 18]) # 32 48 N
        x_CCSS = self.e2n_CCSS_2(x_CCSS)

        x_DDCS = self.e2e_DDCS_1(x_DDCS, x_att[:, 19])
        x_DDCS = self.e2e_DDCS_2(x_DDCS)
        x_DDCS = self.e2e_DDCS_3(x_DDCS, x_att[:, 19])
        x_DDCS = self.e2e_DDCS_4(x_DDCS)
        x_DDCS = self.e2n_DDCS_1(x_DDCS, x_att[:, 19]) # 32 48 N
        x_DDCS = self.e2n_DDCS_2(x_DDCS)

        x_DDSC = self.e2e_DDSC_1(x_DDSC, x_att[:, 20])
        x_DDSC = self.e2e_DDSC_2(x_DDSC)
        x_DDSC = self.e2e_DDSC_3(x_DDSC, x_att[:, 20])
        x_DDSC = self.e2e_DDSC_4(x_DDSC)
        x_DDSC = self.e2n_DDSC_1(x_DDSC, x_att[:, 20]) # 32 48 N
        x_DDSC = self.e2n_DDSC_2(x_DDSC)

        x_CCDS = self.e2e_CCDS_1(x_CCDS, x_att[:, 21])
        x_CCDS = self.e2e_CCDS_2(x_CCDS)
        x_CCDS = self.e2e_CCDS_3(x_CCDS, x_att[:, 21])
        x_CCDS = self.e2e_CCDS_4(x_CCDS)
        x_CCDS = self.e2n_CCDS_1(x_CCDS, x_att[:, 21]) # 32 48 N
        x_CCDS = self.e2n_CCDS_2(x_CCDS)

        x_CCSD = self.e2e_CCSD_1(x_CCSD, x_att[:, 22])
        x_CCSD = self.e2e_CCSD_2(x_CCSD)
        x_CCSD = self.e2e_CCSD_3(x_CCSD, x_att[:, 22])
        x_CCSD = self.e2e_CCSD_4(x_CCSD)
        x_CCSD = self.e2n_CCSD_1(x_CCSD, x_att[:, 22]) # 32 48 N
        x_CCSD = self.e2n_CCSD_2(x_CCSD)

        x_SSDC = self.e2e_SSDC_1(x_SSDC, x_att[:, 23])
        x_SSDC = self.e2e_SSDC_2(x_SSDC)
        x_SSDC = self.e2e_SSDC_3(x_SSDC, x_att[:, 23])
        x_SSDC = self.e2e_SSDC_4(x_SSDC)
        x_SSDC = self.e2n_SSDC_1(x_SSDC, x_att[:, 23]) # 32 48 N
        x_SSDC = self.e2n_SSDC_2(x_SSDC)

        x_SSCD = self.e2e_SSCD_1(x_SSCD, x_att[:, 24])
        x_SSCD = self.e2e_SSCD_2(x_SSCD)
        x_SSCD = self.e2e_SSCD_3(x_SSCD, x_att[:, 24])
        x_SSCD = self.e2e_SSCD_4(x_SSCD)
        x_SSCD = self.e2n_SSCD_1(x_SSCD, x_att[:, 24]) # 32 48 N
        x_SSCD = self.e2n_SSCD_2(x_SSCD)

        x_DSCD = self.e2e_DSCD_1(x_DSCD, x_att[:, 25])
        x_DSCD = self.e2e_DSCD_2(x_DSCD)
        x_DSCD = self.e2e_DSCD_3(x_DSCD, x_att[:, 25])
        x_DSCD = self.e2e_DSCD_4(x_DSCD)
        x_DSCD = self.e2n_DSCD_1(x_DSCD, x_att[:, 25]) # 32 48 N
        x_DSCD = self.e2n_DSCD_2(x_DSCD)



        # global corr
        x_global = self.n2g_global(x) # batch 64 1 1 
        x_DMN_global = self.n2g_global(x_DMN)
        x_CEN_global = self.n2g_global(x_CEN)
        x_SN_global = self.n2g_global(x_SN)
        x_DMN_CEN_global = self.n2g_global(x_DMN_CEN)
        x_DMN_SN_global = self.n2g_global(x_DMN_SN)
        x_CEN_SN_global = self.n2g_global(x_CEN_SN)
        x_DCS_global = self.n2g_global(x_DCS)
        x_DSC_global = self.n2g_global(x_DSC)
        x_CDS_global = self.n2g_global(x_CDS)
        x_DDC_global = self.n2g_global(x_DDC)
        x_DDS_global = self.n2g_global(x_DDS)
        x_CCD_global = self.n2g_global(x_CCD)
        x_CCS_global = self.n2g_global(x_CCS)
        x_SSD_global = self.n2g_global(x_SSD)
        x_SSC_global = self.n2g_global(x_SSC)
        x_DDCC_global = self.n2g_global(x_DDCC)
        x_DDSS_global = self.n2g_global(x_DDSS)
        x_CCSS_global = self.n2g_global(x_CCSS)
        x_DDCS_global = self.n2g_global(x_DDCS)
        x_DDSC_global = self.n2g_global(x_DDSC)
        x_CCDS_global = self.n2g_global(x_CCDS)
        x_CCSD_global = self.n2g_global(x_CCSD)
        x_SSDC_global = self.n2g_global(x_SSDC)
        x_SSCD_global = self.n2g_global(x_SSCD)
        x_DSCD_global = self.n2g_global(x_DSCD)
        # print(x_global.shape)
        x_corr = torch.cat((x_global, x_DMN_global, x_CEN_global, x_SN_global, x_DMN_CEN_global, x_DMN_SN_global, x_CEN_SN_global,
                            x_DCS_global, x_DSC_global, x_CDS_global, x_DDC_global, x_DDS_global, x_CCD_global, x_CCS_global, x_SSD_global, x_SSC_global,
                            x_DDCC_global, x_DDSS_global, x_CCSS_global, x_DDCS_global, x_DDSC_global, x_CCDS_global, x_CCSD_global, x_SSDC_global, x_SSCD_global, x_DSCD_global), axis=2)
        x_corr = torch.squeeze(x_corr)
        x_corr =torch.matmul(x_corr.permute(0, 2, 1), x_corr)
        x_corr = F.sigmoid(x_corr)
        # print(x_corr.shape) # batch 7 7 


        # 将不同元路径的组合起来
        x_cmb = x.permute(0, 2, 3, 1) # b 200 1 48
        x_DMN_cmb = x_DMN.permute(0, 2, 3, 1)
        x_CEN_cmb = x_CEN.permute(0, 2, 3, 1)
        x_SN_cmb = x_SN.permute(0, 2, 3, 1)
        x_DMN_CEN_cmb = x_DMN_CEN.permute(0, 2, 3, 1)
        x_DMN_SN_cmb = x_DMN_SN.permute(0, 2, 3, 1)
        x_CEN_SN_cmb = x_CEN_SN.permute(0, 2, 3, 1)
        x_DCS_cmb = x_DCS.permute(0, 2, 3, 1)
        x_DSC_cmb = x_DSC.permute(0, 2, 3, 1)
        x_CDS_cmb = x_CDS.permute(0, 2, 3, 1)
        x_DDC_cmb = x_DDC.permute(0, 2, 3, 1)
        x_DDS_cmb = x_DDS.permute(0, 2, 3, 1)
        x_CCD_cmb = x_CCD.permute(0, 2, 3, 1)
        x_CCS_cmb = x_CCS.permute(0, 2, 3, 1)
        x_SSD_cmb = x_SSD.permute(0, 2, 3, 1)
        x_SSC_cmb = x_SSC.permute(0, 2, 3, 1)
        x_DDCC_cmb = x_DDCC.permute(0, 2, 3, 1)
        x_DDSS_cmb = x_DDSS.permute(0, 2, 3, 1)
        x_CCSS_cmb = x_CCSS.permute(0, 2, 3, 1)
        x_DDCS_cmb = x_DDCS.permute(0, 2, 3, 1)
        x_DDSC_cmb = x_DDSC.permute(0, 2, 3, 1)
        x_CCDS_cmb = x_CCDS.permute(0, 2, 3, 1)
        x_CCSD_cmb = x_CCSD.permute(0, 2, 3, 1)
        x_SSDC_cmb = x_SSDC.permute(0, 2, 3, 1)
        x_SSCD_cmb = x_SSCD.permute(0, 2, 3, 1)
        x_DSCD_cmb = x_DSCD.permute(0, 2, 3, 1)
        

        # x_DMN_R_out_cmb = x_DMN_R_out.permute(0, 2, 3, 1)
        # x_L_out_R_in_cmb = x_L_out_R_in.permute(0, 2, 3, 1)
        # x_L_out_R_out_cmb = x_L_out_R_out.permute(0, 2, 3, 1)
        # x_R_in_R_out_cmb = x_R_in_R_out.permute(0, 2, 3, 1)
        X2 = torch.cat((x_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_DMN_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_CEN_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_SN_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_DMN_CEN_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_DMN_SN_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_CEN_SN_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_DCS_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_DSC_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_CDS_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_DDC_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_DDS_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_CCD_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_CCS_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_SSD_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_SSC_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_DDCC_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_DDSS_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_CCSS_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_DDCS_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_DDSC_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_CCDS_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_CCSD_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_SSDC_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_SSCD_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1), 
                    x_DSCD_cmb.reshape(x_cmb.size(0) * x_cmb.size(1), 1, -1)), 
                    axis=1)
        
        a = X2
        y = self.self_attention_norm(a)
        # print(y)
        y, loss_corr = self.self_attention(y, x_corr)
        # print(y) # nan
        y = self.self_attention_dropout(y)
        a = a + y

        y = self.ffn(y)
        y = self.ffn_dropout(y)
        a = a + y
        # print(x.shape)
        # a = a + X
        # print(a.shape) 
        ########################################
        DMN_tsne = a[:, 1, :]
        CEN_tsne = a[:, 2, :]
        SN_tsne = a[:, 3, :]
        DMN_CEN_tsne = a[:, 4, :]
        ##########################################################

        # print(x.shape) # B200 11 48

        b_xent = nn.BCEWithLogitsLoss()
        lbl_x1 = torch.ones(x_DMN.size(0), x_DMN.size(2)).to(device)
        lbl_x2 = torch.zeros(x_DMN.size(0), x_DMN.size(2)).to(device)
        lbl = torch.cat((lbl_x1, lbl_x2), axis=1) # bacth 400
        # ret = 0

        

        #############################
        a = a.reshape(a.size(0) * a.size(2), a.size(1))
        # print(a.shape)
        # for param in self.weight_attention.parameters():
        #     param = F.relu(param)
        #     print(param)
        # a = self.attW(a)
        # a = self.attW_diff(a)
        a = F.linear(a, self.weight_attention, self.bias_attention)
        # print(self.weight_attention)
        path_att = torch.mean(self.weight_attention, dim=0)
        # print(path_att)
        node_att = self.weight_attention[:, 25] # 64 1
        node_arg = torch.argsort(node_att)
        # print(node_arg)

        



        a = a.reshape(x.size(0) * x.size(1), x.size(2), -1) # B*D N N
        a = torch.diagonal(a, dim1=-2, dim2=-1)
        # print(a.shape)
        a = self.Lrelu(a)
        t = self.weight_attention.permute(1,0)
        # print(t.shape)
        weight_cont = torch.matmul(self.weight_attention, self.weight_attention.permute(1,0))
        pos = (Weight_con_tensor * weight_cont)
        neg = (Weight_con_tensor_T * weight_cont)
        
        nce_loss = -torch.log(torch.exp(pos) / torch.sum(torch.exp(neg), dim=1)).mean()
        # print(nce_loss)

        # print(weight_cont.shape)
        # # weight_relu = F.relu(self.weight_attention)
        # weight_relu = self.weight_attention
        # # weight_relu_avg = weight_relu / weight_relu.sum()
        # a = F.linear(a, weight_relu, self.bias_attention)
        # # print(weight_relu_avg)
        # loss_l1 = torch.norm(weight_relu, p=1)
        # a=self.Lrelu(a)
        # print(loss_l1)

        #####################################
        # print(a.shape) 
        XX = a.reshape(x_cmb.size(0), a.size(1), -1) # B N D
        XX = torch.mean(XX, dim=1)
        # print(x_DMN_cmb.shape) # B N 1 D
        
        DMN_tsne = DMN_tsne.reshape(x_cmb.size(0), a.size(1), -1) # B N D
        DMN_tsne = torch.mean(DMN_tsne, dim=1)
        CEN_tsne = CEN_tsne.reshape(x_cmb.size(0), a.size(1), -1) # B N D
        CEN_tsne = torch.mean(CEN_tsne, dim=1)
        SN_tsne = SN_tsne.reshape(x_cmb.size(0), a.size(1), -1) # B N D
        SN_tsne = torch.mean(SN_tsne, dim=1)
        DMN_CEN_tsne = DMN_CEN_tsne.reshape(x_cmb.size(0), a.size(1), -1) # B N D
        DMN_CEN_tsne = torch.mean(DMN_CEN_tsne, dim=1)
        
        # DMN_tsne = XX[:, 1, :]
        # CEN_tsne = XX[:, 2, :]
        # SN_tsne = XX[:, 3, :]
        # DMN_CEN_tsne = XX[:, 4, :]
        tsne = torch.cat((DMN_tsne.reshape(DMN_tsne.size(0), 1, DMN_tsne.size(1)), 
                    CEN_tsne.reshape(CEN_tsne.size(0), 1, CEN_tsne.size(1)),
                    SN_tsne.reshape(SN_tsne.size(0), 1, SN_tsne.size(1)),
                    DMN_CEN_tsne.reshape(SN_tsne.size(0), 1, SN_tsne.size(1)),
                    XX.reshape(XX.size(0), 1, XX.size(1))), axis=1)
        # print(tsne.shape)
        np.save('tsne.npy', tsne.detach().cpu().numpy())
        ######################################

        a = a.reshape(x_DMN.size(0), x_DMN.size(1), -1, 1) # 32 48 N 1
        # for param in self.attW.parameters():
        #     print(param)
        # print(x.shape) # B 200 48 1

        
        
        logits = self.MI(x.reshape(x.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        # print(logits.shape)
        ret = b_xent(logits,lbl)
        logits = self.MI_DMN(x_DMN.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_CEN(x_CEN.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_SN(x_SN.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_DMN_CEN(x_DMN_CEN.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_DMN_SN(x_DMN_SN.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_CEN_SN(x_CEN_SN.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_DCS(x_DCS.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_DSC(x_DSC.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_CDS(x_CDS.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_DDC(x_DDC.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_DDS(x_DDS.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_CCD(x_CCD.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_CCS(x_CCS.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_SSD(x_SSD.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_SSC(x_SSC.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_DDCC(x_DDCC.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_DDSS(x_DDSS.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_CCSS(x_CCSS.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_DDCS(x_DDCS.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_DDSC(x_DDSC.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_CCDS(x_CCDS.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_CCSD(x_CCSD.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_SSDC(x_SSDC.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_SSCD(x_SSCD.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)
        logits = self.MI_DSCD(x_DSCD.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        ret += b_xent(logits,lbl)

        # print(x.shape) # B D N 1
        path_com = torch.cat((x,x_DMN,x_CEN,x_SN,x_DMN_CEN,x_DMN_SN,x_CEN_SN,x_DCS,x_DSC,x_CDS,x_DDC,x_DDS,x_CCD,x_CCS,x_SSD,x_SSC,x_DDCC,x_DDSS,x_CCSS,x_DDCS,x_DDSC,x_CCDS, x_CCSD,x_SSDC,x_SSCD,x_DSCD), axis=3)
        path_com = path_com.reshape(path_com.size(0), path_com.size(3), path_com.size(2), path_com.size(1)) # B P N D
        for p in range(25):
            for p2 in range(p+1, 26):
                path1 = path_com[:, p, :, :]
                path2 = path_com[:, p2, :, :]
                # print(path.shape)
                logits = self.MI(path1, path2)
                loss_c = b_xent(logits, lbl)
                ret += 1/loss_c

        
        # logits = self.MI_L_in_R_out(x_DMN_R_out.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        # ret += b_xent(logits,lbl)
        # logits = self.MI_L_out_R_in(x_L_out_R_in.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        # ret += b_xent(logits,lbl)
        # logits = self.MI_L_out_R_out(x_L_out_R_out.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        # ret += b_xent(logits,lbl)
        # logits = self.MI_R_in_R_out(x_R_in_R_out.reshape(x_DMN.size(0), x.size(2), -1), a.reshape(x_DMN.size(0), x_DMN.size(2), -1))
        # ret += b_xent(logits,lbl)

        ret = ret * (1/26) # 
        # print(ret) # 7.0

        ret = ret + nce_loss * 0.01


        #######################
        # if flag == 'pre': # x  e2e输出dim
        #     x = a
        #     # x_atlas2 = x_atlas2_common + x_atlas2_spec
        # else:
        #     x = a
        #     # x_atlas2 = x_atlas2_common
        #######################

        de_x = self.de_e2n(a) # 32 8 N N
        de_x = self.de_e2e(de_x) # 32 1 N N
        de_x = torch.squeeze(de_x)

        # de_x_atlas2 = self.de_e2n(x_atlas2) # 32 8 N N
        # de_x_atlas2 = self.de_e2e(de_x_atlas2) # 32 1 N N
        # de_x_atlas2 = torch.squeeze(de_x_atlas2)

        # print(de_x.shape) # B 200 200
        de_x_nodeclf = self.nodeclf(de_x.reshape(de_x.size(0)*de_x.size(1), -1))
        # de_x_nodeclf = de_x_nodeclf.reshape(de_x.size(0), de_x.size(1), -1)
        # print(de_x_nodeclf.shape) # B 200 4

        
        de_x_T = de_x.permute(0, 2, 1)
        de_x = torch.matmul(de_x, de_x_T) 

        # de_x2 = torch.squeeze(a)
        # de_x2_T = de_x2.permute(0, 2, 1)
        # de_x2 = torch.matmul(de_x2_T, de_x2) # 
        # de_x += de_x2

        # de_x2_atlas2 = torch.squeeze(x_atlas2)
        # de_x2_atlas2_T = de_x2_atlas2.permute(0, 2, 1)
        # de_x2_atlas2 = torch.matmul(de_x2_atlas2_T, de_x2_atlas2) # 
        # de_x_atlas2 += de_x2_atlas2

        de_x = torch.sigmoid(de_x)
        
        # de_x_atlas2 = torch.sigmoid(de_x_atlas2)
        # print(de_x.shape) # batch * 200 * 200 


        #####################################################################################################################

        a = self.n2g(a) # 48-64
        # x = x.reshape(x_DMN.size(0) * x_DMN.size(1) * x.size(2), x.size(1), x.size(2)) # B 200 11 48(D)
        # x = x.permute(11)
        # print(x.shape)
        a = a.reshape(a.size(0), -1)
        ############################

        


        # x = torch.mean(x, axis=1) # mean  # x batch * Meta * N * D
        

        # x = (x + x_L + x_R + x_LR) / 4.0

        
        # print(x.shape)
        a = self.linear(a)
        a = F.softmax(a, dim=-1)
        # print('output', x.shape)

        return a, de_x, ret, de_x_nodeclf, loss_corr
    
    def get_A(self, x):
        x = self.e2e(x)
#         print(x.shape)
        x = torch.mean(x, dim=1)
        return x
        

# ABIDE load
print('loading ABIDE data...')
# X = np.load('./pcc_correlation_871_cc200.npy')
# Y = np.load('/kaggle/input/cc200-pytorch/871_label_cc200.npy')
# Data_atlas1 = scio.loadmat('./data/ABIDE/data/different_mask_correlation&label_with_same_order/pcc_correlation_871_aal_.mat') # 
# Data_atlas2 = scio.loadmat('./data/ABIDE/data/different_mask_correlation&label_with_same_order/pcc_correlation_871_aal_.mat')
# print(cc200.keys()) # connectivity
# X = Data_atlas1['connectivity']
# X_atlas2 = Data_atlas2['connectivity']
# print(X[0][0])
# print(cc200.shape) # 871 200 200
# Y = np.loadtxt('./data/ABIDE/data/different_mask_correlation&label_with_same_order/871_label_aal.txt')

# X = np.load('./MDD_HC.npy')
# Y = np.load('./MDD_HC_label.npy')
# X = np.load('./BP_HC.npy')
# Y = np.load('./BP_HC_label.npy')
X = np.load('./asd_aal_NYU.npy')
Y = np.load('./asd_aal_label_NYU.npy')

X_atlas2 = X

Y_atlas2 = Y

Nodes_Atlas1 = X.shape[1] 
Nodes_Atlas2 = X_atlas2.shape[-1]
N_subjects = X.shape[0]

XX = np.ones((N_subjects, X.shape[1], X.shape[1]))
for bb in range(0, N_subjects):
    # print('1')
    t = np.corrcoef(X[bb])
    XX[bb] = t
X = XX
# X_hofc = X.copy()
# for bb in range(0, N_subjects):
#     # print('1')
#     t = np.corrcoef(X[bb])
#     X[bb] = t
print('X shape', X.shape)


where_are_nan = np.isnan(X)  # 找出数据中为nan的
where_are_inf = np.isinf(X)  # 找出数据中为inf的
for bb in range(0, N_subjects):
    # print('1')
    # t = np.corrcoef(X[bb])
    # X[bb] = t
    for i in range(0, Nodes_Atlas1):
        for j in range(0, Nodes_Atlas1):
            if where_are_nan[bb][i][j]:
                X[bb][i][j] = 0
            if where_are_inf[bb][i][j]:
                X[bb][i][j] = 1
            # t = np.corrcoef(X[bb][i], X[bb][j])[0,1]
            # X_hofc[bb][i][j] = t
# print(X_hofc[0])
# where_are_nan = np.isnan(X_atlas2)  # 找出数据中为nan的
# where_are_inf = np.isinf(X_atlas2)  # 找出数据中为inf的
# for bb in range(0, N_subjects):
#     for i in range(0, Nodes_Atlas2):
#         for j in range(0, Nodes_Atlas2):
#             if where_are_nan[bb][i][j]:
#                 X[bb][i][j] = 0
#             if where_are_inf[bb][i][j]:
#                 X[bb][i][j] = 1

##################################################计算节点是否具有显著差异
# 先分出 HC & Patient
# print('--------------------', X.shape) # N 116 116
# class_0_list = []
# class_1_list = []
# for i in range(Y.shape[0]):
#     if Y[i] == 0:
#         class_0_list.append(X[i])
#     elif Y[i] == 1:
#         class_1_list.append(X[i])
# class_0_arr = np.array(class_0_list) # N 116 116
# class_1_arr = np.array(class_1_list)

# node_id = 10
# A_node = class_0_arr[:, :, node_id-1]
# B_node = class_1_arr[:, :, node_id-1]
# A_node = np.mean(A_node, axis=1)
# B_node = np.mean(B_node, axis=1)
# # t检验
# import numpy as np
# from scipy import stats
# t_stat, p_value = stats.ttest_ind(A_node, B_node)

# print(node_id)
# print("t-statistic:", t_stat)
# print("p-value:", p_value)
# quit()
# MDD
# 5:0.0007470710927892357 65:0.01324897611725735  67:0.046114839781624515 32:0.18030032463336734 78:0.01671436367048121 8:0.01901482738204021 10:0.04523935482216855 41:0.704488114347143 85:0.9383737352823162 6:0.03501873841910788
# BD
# 65:0.007034522989857381 39:0.04275671093067688 9:0.0062200295019139195 51:0.27010740703659075 64:0.028270210113946644 6:0.0086826009246694 31:0.001669684444741718 67:0.32483645122731775 24:0.04032795920920102 10:0.4164721328353689
# ASD
# 6:0.017715210579355392 31:0.0011645599969787922 23:0.00017908115381979722 62:0.021871932903662017 10:0.11923159142331093 64:0.9233026489110184 39:0.2323995238943011 85:0.0040200472618119425 65:0.831923829230051 68:0.013230085616024892

# adjust
# ASD
# 7:0.0177 64:0.00438 32:0.00116 62:0.0257 10:0.186 64:0.00438 39:0.733 85:0.000937 65:0.9 68:0.367
# MDD
# 5:0.03 65:0.047 67:0.47 32:0.004 78:0.037 8:0.0015 10:0.0003 41:0.53 85:0.913 6:0.0007
# BD
# 65:0.02 39:0.042 9:0.006 51:0.43 64:0.028 6:0.002 31:0.001 67:0.9 24:0.0000000 10:0.006

# subnetwork node
draw_node = []
for i in range(116):
    draw_node.append(i+1)
draw_node = np.array(draw_node)
CC200_sub_file = np.loadtxt('./AAL_sub_new.txt', dtype=str)
CC200_sub_in = np.ones(Nodes_Atlas1)
CC200_sub_out = np.ones(Nodes_Atlas1)
CC200_sub_in[CC200_sub_file=='1'] = 1
CC200_sub_in[CC200_sub_file=='2'] = 2
CC200_sub_in[CC200_sub_file=='3'] = 3
CC200_sub_in[CC200_sub_file=='12'] = 12
CC200_sub_in[CC200_sub_file=='13'] = 13
CC200_sub_in[CC200_sub_file=='23'] = 23
CC200_sub_in[CC200_sub_file=='123'] = 123
CC200_sub_in[CC200_sub_file=='0'] = 0

CC200_sub_draw_D = np.zeros(Nodes_Atlas1) # 为了画图 可解释性
CC200_sub_draw_D[CC200_sub_file=='1'] = 1
CC200_sub_draw_D[CC200_sub_file=='12'] = 1
CC200_sub_draw_D[CC200_sub_file=='13'] = 1
CC200_sub_draw_D[CC200_sub_file=='123'] = 1
CC200_sub_draw_C = np.zeros(Nodes_Atlas1) # 为了画图 可解释性
CC200_sub_draw_C[CC200_sub_file=='2'] = 1
CC200_sub_draw_C[CC200_sub_file=='12'] = 1
CC200_sub_draw_C[CC200_sub_file=='23'] = 1
CC200_sub_draw_C[CC200_sub_file=='123'] = 1
CC200_sub_draw_S = np.zeros(Nodes_Atlas1) # 为了画图 可解释性
CC200_sub_draw_S[CC200_sub_file=='3'] = 1
CC200_sub_draw_S[CC200_sub_file=='23'] = 1
CC200_sub_draw_S[CC200_sub_file=='13'] = 1
CC200_sub_draw_S[CC200_sub_file=='123'] = 1
# CC200_sub_draw_D = np.zeros(Nodes_Atlas1) # 为了画图 可解释性
# CC200_sub_draw_D[CC200_sub_file=='1'] = 1
# CC200_sub_draw_D = np.zeros(Nodes_Atlas1) # 为了画图 可解释性
# CC200_sub_draw_D[CC200_sub_file=='1'] = 1
# CC200_sub_draw_D = np.zeros(Nodes_Atlas1) # 为了画图 可解释性
# CC200_sub_draw_D[CC200_sub_file=='1'] = 1
# CC200_sub_draw_DC = CC200_sub_draw_D + CC200_sub_draw_C
# CC200_sub_draw_DC[CC200_sub_draw_DC!=0] = 1
# CC200_sub_draw_DS = CC200_sub_draw_D + CC200_sub_draw_S
# CC200_sub_draw_DS[CC200_sub_draw_DS!=0] = 1
# CC200_sub_draw_CS = CC200_sub_draw_C + CC200_sub_draw_S
# CC200_sub_draw_CS[CC200_sub_draw_CS!=0] = 1

# D C S DC DS CS DCS DSC CDS DDC DDS CCD CCS SSD SSC DDCC DDSS CCSS DDCS DDSC CCDS CCSD SSDC SSCD DSCD
D_matrix = CC200_sub_draw_D.reshape(Nodes_Atlas1, 1) * CC200_sub_draw_D.reshape(1, Nodes_Atlas1)
print('------D shape ', D_matrix.shape) # 116 116
with open('./sub-draw/D.npy', 'wb') as f:
    np.save(f, D_matrix)
C_matrix = CC200_sub_draw_C.reshape(Nodes_Atlas1, 1) * CC200_sub_draw_C.reshape(1, Nodes_Atlas1)
with open('./sub-draw/C.npy', 'wb') as f:
    np.save(f, C_matrix)
S_matrix = CC200_sub_draw_S.reshape(Nodes_Atlas1, 1) * CC200_sub_draw_S.reshape(1, Nodes_Atlas1)
with open('./sub-draw/S.npy', 'wb') as f:
    np.save(f, S_matrix)

DC_matrix = CC200_sub_draw_D.reshape(Nodes_Atlas1, 1) * CC200_sub_draw_C.reshape(1, Nodes_Atlas1) + CC200_sub_draw_C.reshape(Nodes_Atlas1, 1) * CC200_sub_draw_D.reshape(1, Nodes_Atlas1)
with open('./sub-draw/DC.npy', 'wb') as f:
    np.save(f, DC_matrix)
DS_matrix = CC200_sub_draw_D.reshape(Nodes_Atlas1, 1) * CC200_sub_draw_S.reshape(1, Nodes_Atlas1) + CC200_sub_draw_S.reshape(Nodes_Atlas1, 1) * CC200_sub_draw_D.reshape(1, Nodes_Atlas1)
with open('./sub-draw/DS.npy', 'wb') as f:
    np.save(f, DS_matrix)
CS_matrix = CC200_sub_draw_C.reshape(Nodes_Atlas1, 1) * CC200_sub_draw_S.reshape(1, Nodes_Atlas1) + CC200_sub_draw_S.reshape(Nodes_Atlas1, 1) * CC200_sub_draw_C.reshape(1, Nodes_Atlas1)
with open('./sub-draw/CS.npy', 'wb') as f:
    np.save(f, CS_matrix)
DCS_matrix = DC_matrix + CS_matrix
DCS_matrix[DCS_matrix!=0]=1
with open('./sub-draw/DCS.npy', 'wb') as f:
    np.save(f, DCS_matrix)
DSC_matrix = DS_matrix + CS_matrix
DSC_matrix[DSC_matrix!=0]=1
with open('./sub-draw/DSC.npy', 'wb') as f:
    np.save(f, DSC_matrix)
# print('-------------------------', DSC_matrix.shape) # 116 116 这是模板矩阵
CDS_matrix = DC_matrix + DS_matrix
CDS_matrix[CDS_matrix!=0]=1
with open('./sub-draw/CDS.npy', 'wb') as f:
    np.save(f, CDS_matrix)
DDC_matrix = D_matrix + DC_matrix
DDC_matrix[DDC_matrix!=0]=1
with open('./sub-draw/DDC.npy', 'wb') as f:
    np.save(f, DDC_matrix)
DDS_matrix = D_matrix + DS_matrix
DDS_matrix[DDS_matrix!=0]=1
with open('./sub-draw/DDS.npy', 'wb') as f:
    np.save(f, DDS_matrix)
CCD_matrix = C_matrix + DC_matrix
CCD_matrix[CCD_matrix!=0]=1
with open('./sub-draw/CCD.npy', 'wb') as f:
    np.save(f, CCD_matrix)
CCS_matrix = C_matrix + CS_matrix
CCS_matrix[CCS_matrix!=0]=1
with open('./sub-draw/CCS.npy', 'wb') as f:
    np.save(f, CCS_matrix)
SSD_matrix = S_matrix + DS_matrix
SSD_matrix[SSD_matrix!=0]=1
with open('./sub-draw/SSD.npy', 'wb') as f:
    np.save(f, SSD_matrix)
SSC_matrix = S_matrix + CS_matrix
SSC_matrix[SSC_matrix!=0]=1
with open('./sub-draw/SSC.npy', 'wb') as f:
    np.save(f, SSC_matrix)
DDCC_matrix = D_matrix + C_matrix + DC_matrix
DDCC_matrix[DDCC_matrix!=0]=1
with open('./sub-draw/DDCC.npy', 'wb') as f:
    np.save(f, DDCC_matrix)
DDSS_matrix = D_matrix + S_matrix + DS_matrix
DDSS_matrix[DDSS_matrix!=0]=1
with open('./sub-draw/DDSS.npy', 'wb') as f:
    np.save(f, DDSS_matrix)
CCSS_matrix = C_matrix + S_matrix + CS_matrix
CCSS_matrix[CCSS_matrix!=0]=1
with open('./sub-draw/CCSS.npy', 'wb') as f:
    np.save(f, CCSS_matrix)
DDCS_matrix = D_matrix + DC_matrix + CS_matrix
DDCS_matrix[DDCS_matrix!=0]=1
with open('./sub-draw/DDCS.npy', 'wb') as f:
    np.save(f, DDCS_matrix)
DDSC_matrix = D_matrix + DS_matrix + CS_matrix
DDSC_matrix[DDSC_matrix!=0]=1
with open('./sub-draw/DDSC.npy', 'wb') as f:
    np.save(f, DDSC_matrix)
CCDS_matrix = C_matrix + DC_matrix + DS_matrix
CCDS_matrix[CCDS_matrix!=0]=1
with open('./sub-draw/CCDS.npy', 'wb') as f:
    np.save(f, CCDS_matrix)
CCSD_matrix = C_matrix + CS_matrix + DS_matrix
CCSD_matrix[CCSD_matrix!=0]=1
with open('./sub-draw/CCSD.npy', 'wb') as f:
    np.save(f, CCSD_matrix)
SSDC_matrix = S_matrix + DC_matrix + DS_matrix
SSDC_matrix[SSDC_matrix!=0]=1
with open('./sub-draw/SSDC.npy', 'wb') as f:
    np.save(f, SSDC_matrix)
SSCD_matrix = S_matrix + DC_matrix + CS_matrix
SSCD_matrix[SSCD_matrix!=0]=1
with open('./sub-draw/SSCD.npy', 'wb') as f:
    np.save(f, SSCD_matrix)
DSCD_matrix = DS_matrix + CS_matrix + DC_matrix
DSCD_matrix[DSCD_matrix!=0]=1
with open('./sub-draw/DSCD.npy', 'wb') as f:
    np.save(f, DSCD_matrix)






# print(X.shape) # numsubject 46 46
for i in range(CC200_sub_in.shape[0]):
    if CC200_sub_in[i] == 12:
        temp = X[:, i, :]
        temp_dim = temp[:, np.newaxis, :]
        add_0 = np.ones(X.shape[0])
        add_0 = add_0[:, np.newaxis] # n 1
        temp_add = np.concatenate((temp, add_0), axis=1)
        temp_add_dim = temp_add[:, :, np.newaxis] # n 47 1
        X = np.concatenate((X, temp_dim), axis=1) # n 47 46
        X = np.concatenate((X, temp_add_dim), axis=2) # n 47 47
        CC200_sub_in[i] = 1
        CC200_sub_in=np.append(CC200_sub_in,2)
        draw_node = np.append(draw_node, i+1)
    if CC200_sub_in[i] == 13:
        temp = X[:, i, :]
        temp_dim = temp[:, np.newaxis, :]
        add_0 = np.ones(X.shape[0])
        add_0 = add_0[:, np.newaxis] # n 1
        temp_add = np.concatenate((temp, add_0), axis=1)
        temp_add_dim = temp_add[:, :, np.newaxis] # n 47 1
        X = np.concatenate((X, temp_dim), axis=1) # n 47 46
        X = np.concatenate((X, temp_add_dim), axis=2) # n 47 47
        CC200_sub_in[i] = 1
        CC200_sub_in=np.append(CC200_sub_in,3)
        draw_node = np.append(draw_node, i+1)
    if CC200_sub_in[i] == 23:
        temp = X[:, i, :]
        temp_dim = temp[:, np.newaxis, :]
        add_0 = np.ones(X.shape[0])
        add_0 = add_0[:, np.newaxis] # n 1
        temp_add = np.concatenate((temp, add_0), axis=1)
        temp_add_dim = temp_add[:, :, np.newaxis] # n 47 1
        X = np.concatenate((X, temp_dim), axis=1) # n 47 46
        X = np.concatenate((X, temp_add_dim), axis=2) # n 47 47
        CC200_sub_in[i] = 2
        CC200_sub_in=np.append(CC200_sub_in,3)
        draw_node = np.append(draw_node, i+1)
    if CC200_sub_in[i] == 123:
        temp = X[:, i, :]
        temp_dim = temp[:, np.newaxis, :]
        add_0 = np.ones(X.shape[0])
        add_0 = add_0[:, np.newaxis] # n 1
        temp_add = np.concatenate((temp, add_0), axis=1)
        temp_add_dim = temp_add[:, :, np.newaxis] # n 47 1
        X = np.concatenate((X, temp_dim), axis=1) # n 47 46
        X = np.concatenate((X, temp_add_dim), axis=2) # n 47 47
        draw_node = np.append(draw_node, i+1)

        temp = X[:, i, :]
        temp_dim = temp[:, np.newaxis, :]
        add_0 = np.ones(X.shape[0])
        add_0 = add_0[:, np.newaxis] # n 1
        temp_add = np.concatenate((temp, add_0), axis=1)
        temp_add_dim = temp_add[:, :, np.newaxis] # n 47 1
        X = np.concatenate((X, temp_dim), axis=1) # n 47 46
        X = np.concatenate((X, temp_add_dim), axis=2) # n 47 47
        CC200_sub_in[i] = 1
        CC200_sub_in=np.append(CC200_sub_in,2)
        CC200_sub_in=np.append(CC200_sub_in,3)
        draw_node = np.append(draw_node, i+1)

delete_nodes = [i for i in range(Nodes_Atlas1) if CC200_sub_in[i] == 0]
for node in delete_nodes[::-1]:
    CC200_sub_in = np.delete(CC200_sub_in, node, axis=0)
    X = np.delete(X, node, axis=1)
    X = np.delete(X, node, axis=2)
    draw_node = np.delete(draw_node, node, axis=0)
# print(CC200_sub_in.shape) # 46 现在0全都去除了

print('draw node ', draw_node)
# [ 3  4  5  6  7  8  9 10 11 15 16 19 20 23 24 25 26 29 30 31 32 33 34 35
#  36 37 38 39 40 41 42 51 52 59 61 62 64 65 66 67 68 72 77 78 85 89  5  6
#   7  8  9  9 10 10 23 24 31 32 65 66 67 77 77 78]

# print(CC200_sub_in.shape) 64
# X # n 64 64
Nodes_Atlas1_new = X.shape[-1]

# 现在子网络指示矩阵 1 2 3 三种
CC200_sub_1 = np.zeros(Nodes_Atlas1_new)
CC200_sub_2 = np.zeros(Nodes_Atlas1_new)
CC200_sub_3 = np.zeros(Nodes_Atlas1_new)
CC200_sub_1[CC200_sub_in==1] = 1
CC200_sub_2[CC200_sub_in==2] = 1
CC200_sub_3[CC200_sub_in==3] = 1

Brain_DMN = CC200_sub_1.reshape(Nodes_Atlas1_new, 1) * CC200_sub_1.reshape(1, Nodes_Atlas1_new)
Brain_CEN = CC200_sub_2.reshape(Nodes_Atlas1_new, 1) * CC200_sub_2.reshape(1, Nodes_Atlas1_new)
Brain_SN = CC200_sub_3.reshape(Nodes_Atlas1_new, 1) * CC200_sub_3.reshape(1, Nodes_Atlas1_new)

Brain_DMN_CEN = CC200_sub_1.reshape(Nodes_Atlas1_new, 1) * CC200_sub_2.reshape(1, Nodes_Atlas1_new) + CC200_sub_2.reshape(Nodes_Atlas1_new, 1) * CC200_sub_1.reshape(1, Nodes_Atlas1_new)
Brain_DMN_SN = CC200_sub_1.reshape(Nodes_Atlas1_new, 1) * CC200_sub_3.reshape(1, Nodes_Atlas1_new) + CC200_sub_3.reshape(Nodes_Atlas1_new, 1) * CC200_sub_1.reshape(1, Nodes_Atlas1_new)
Brain_CEN_SN = CC200_sub_2.reshape(Nodes_Atlas1_new, 1) * CC200_sub_3.reshape(1, Nodes_Atlas1_new) + CC200_sub_3.reshape(Nodes_Atlas1_new, 1) * CC200_sub_2.reshape(1, Nodes_Atlas1_new)
# BrainDMNCEN N * N 0 1 Matrix
from scipy.stats import mannwhitneyu
def clustering_coefficient2(adj_matrix):
    num_nodes = len(adj_matrix)
    clustering_coeffs = []

    for i in range(num_nodes):
        neighbors = np.nonzero(adj_matrix[i])[0]
        num_neighbors = len(neighbors)

        if num_neighbors < 2:
            clustering_coeffs.append(0)
            continue

        num_edges = 0
        for j in range(num_neighbors):
            for k in range(j + 1, num_neighbors):
                if adj_matrix[neighbors[j]][neighbors[k]] == 1:
                    num_edges += 1

        clustering_coeffs.append(2 * num_edges / (num_neighbors * (num_neighbors - 1)))

    return np.mean(clustering_coeffs)
def random_graph(num_nodes, edge_probability):
    random_adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_probability:
                random_adj_matrix[i][j] = 1
                random_adj_matrix[j][i] = 1
    return random_adj_matrix

def small_world_coefficient(adj_matrix, num_random_graphs=10, edge_probability=0.5):
    # Calculate actual clustering coefficient
    actual_clustering_coeff = clustering_coefficient2(adj_matrix)

    # Calculate average clustering coefficient of random graphs
    random_clustering_coeffs = []
    num_nodes = len(adj_matrix)
    for _ in range(num_random_graphs):
        random_graph_adj_matrix = random_graph(num_nodes, edge_probability)
        random_clustering_coeffs.append(clustering_coefficient2(random_graph_adj_matrix))
    average_random_clustering_coeff = np.mean(random_clustering_coeffs)

    # print(average_random_clustering_coeff) # 0

    # Calculate small world coefficient
    if average_random_clustering_coeff == 0:
        return 0
    else:
        return actual_clustering_coeff / average_random_clustering_coeff
def remove_isolated_nodes(adj_matrix, isolated_nodes):
    num_nodes = len(adj_matrix)

    # Find isolated nodes (nodes with no connections)
    # if isolated_nodes == None:
    # print(isolated_nodes)
    # isolated_nodes2 = []
    Nodes_Atlas1 = 64
    isolated_nodes2 = [i for i in range(Nodes_Atlas1) if isolated_nodes[i] == 0]

    # Remove isolated nodes and their related rows/columns
    for node in isolated_nodes2[::-1]:
        adj_matrix = np.delete(adj_matrix, node, axis=0)
        adj_matrix = np.delete(adj_matrix, node, axis=1)

    return adj_matrix

# X_temp = X.copy()
# for bb in range(0, N_subjects):
#     for i in range(0, 64):
#         for j in range(0, 64):
#             if abs(X[bb][i][j]) > 0.5:
#                 X_temp[bb][i][j] = 1
#             else:
#                 X_temp[bb][i][j] = 0
# print(Brain_DMN)
'''
T=0.33

for l in range(Brain_DMN.shape[0]):
    Brain_DMN[l][l] = 0
print('DMN', np.count_nonzero(Brain_DMN)) # 702 /2 = 351
For_draw_DMN = X * Brain_DMN
# For_draw_DMN = X_hofc * Brain_DMN
# for s in range(For_draw_DMN.shape[0]):
#     for i in range(For_draw_DMN.shape[1]):
#         for j in range(For_draw_DMN.shape[1]):
#             if abs(For_draw_DMN[s][i][j]) < T:
#                 For_draw_DMN[s][i][j] = 0
#             else:
#                 For_draw_DMN[s][i][j] = 1
For_draw_label0 = For_draw_DMN[Y==0] # 246
For_draw_label1 = For_draw_DMN[Y==1]
total_pcc_DMN_hc = np.sum(For_draw_label0,axis=2)
total_pcc_DMN_hc = np.sum(total_pcc_DMN_hc,axis=1)
# total_pcc_DMN_hc = For_draw_label0.flatten()
total_pcc_DMN_mdd = np.sum(For_draw_label1,axis=2)
total_pcc_DMN_mdd = np.sum(total_pcc_DMN_mdd,axis=1)
total_pcc_DMN_hc_max = np.max(For_draw_label0, axis=2)
total_pcc_DMN_hc_max = np.max(total_pcc_DMN_hc_max, axis=1)
total_pcc_DMN_mdd_max = np.max(For_draw_label1,axis=2)
total_pcc_DMN_mdd_max = np.max(total_pcc_DMN_mdd_max,axis=1)
# print(total_pcc_DMN_hc_max)
# print(total_pcc_DMN_mdd_max)
# total_pcc_DMN_mdd = For_draw_label1.flatten()
# print('DMN hc', total_pcc_DMN_hc.s) # 79.22
# print('DMN mdd', total_pcc_DMN_mdd / 182)
con_pro_hc = total_pcc_DMN_hc / 351
con_pro_mdd = total_pcc_DMN_mdd / 351
stat, p_value_pro = mannwhitneyu(con_pro_hc, con_pro_mdd)
stat, p_value = mannwhitneyu(total_pcc_DMN_hc, total_pcc_DMN_mdd)
stat_max, p_value_max = mannwhitneyu(total_pcc_DMN_hc_max, total_pcc_DMN_mdd_max)
print('DMN p value ', p_value)
# print('DMN pro p ', p_value_pro)
# print('DMN p value max', p_value_max)
# hc_small = []
# mdd_small = []
# for bb in range(For_draw_label0.shape[0]):
#     # DMN = remove_isolated_nodes(For_draw_label0[bb], CC200_sub_1)
#     hc_small.append(small_world_coefficient(For_draw_label0[bb]))
# for bb in range(For_draw_label1.shape[0]):
#     # DMN = remove_isolated_nodes(For_draw_label1[bb], CC200_sub_1)
#     mdd_small.append(small_world_coefficient(For_draw_label1[bb]))
# stat, p_value = mannwhitneyu(np.array(hc_small), np.array(mdd_small))
# print('DMN small', p_value)




for l in range(Brain_CEN.shape[0]):
    Brain_CEN[l][l] = 0
For_draw_CEN = X * Brain_CEN
# For_draw_CEN = X_hofc * Brain_CEN
# for s in range(For_draw_CEN.shape[0]):
#     for i in range(For_draw_CEN.shape[1]):
#         for j in range(For_draw_CEN.shape[1]):
#             if abs(For_draw_CEN[s][i][j]) < T:
#                 For_draw_CEN[s][i][j] = 0
#             else:
#                 For_draw_CEN[s][i][j] = 1
For_draw_label0 = For_draw_CEN[Y==0] # 246
For_draw_label1 = For_draw_CEN[Y==1]
total_pcc_CEN_hc = np.sum(For_draw_label0,axis=2)
total_pcc_CEN_hc = np.sum(total_pcc_CEN_hc,axis=1)
total_pcc_CEN_mdd = np.sum(For_draw_label1,axis=2)
total_pcc_CEN_mdd = np.sum(total_pcc_CEN_mdd,axis=1)
total_pcc_CEN_hc_max = np.max(For_draw_label0, axis=2)
total_pcc_CEN_hc_max = np.max(total_pcc_CEN_hc_max, axis=1)
total_pcc_CEN_mdd_max = np.max(For_draw_label1,axis=2)
total_pcc_CEN_mdd_max = np.max(total_pcc_CEN_mdd_max,axis=1)
# print('CEN hc', total_pcc_CEN_hc) # 79.22
# print('CEN mdd', total_pcc_CEN_mdd / 182)
stat, p_value = mannwhitneyu(total_pcc_CEN_hc, total_pcc_CEN_mdd)
stat_max, p_value_max = mannwhitneyu(total_pcc_CEN_hc_max, total_pcc_CEN_mdd_max)

print('CEN p value', p_value)
# print('CEN p value max', p_value_max)
# hc_small = []
# mdd_small = []
# # print(For_draw_label0[0])
# for bb in range(For_draw_label0.shape[0]):
#     hc_small.append(small_world_coefficient(For_draw_label0[bb]))
# for bb in range(For_draw_label1.shape[0]):
#     mdd_small.append(small_world_coefficient(For_draw_label1[bb]))
# stat, p_value = mannwhitneyu(np.array(hc_small), np.array(mdd_small))
# print(hc_small) # 0
# print('CEN small', p_value)

for l in range(Brain_SN.shape[0]):
    Brain_SN[l][l] = 0
For_draw_SN = X * Brain_SN
# For_draw_SN = X_hofc * Brain_SN
# for s in range(For_draw_SN.shape[0]):
#     for i in range(For_draw_SN.shape[1]):
#         for j in range(For_draw_SN.shape[1]):
#             if abs(For_draw_SN[s][i][j]) < T:
#                 For_draw_SN[s][i][j] = 0
#             else:
#                 For_draw_SN[s][i][j] = 1
For_draw_label0 = For_draw_SN[Y==0] # 246
For_draw_label1 = For_draw_SN[Y==1]
total_pcc_SN_hc = np.sum(For_draw_label0,axis=2)
total_pcc_SN_hc = np.sum(total_pcc_SN_hc,axis=1)
total_pcc_SN_mdd = np.sum(For_draw_label1,axis=2)
total_pcc_SN_mdd = np.sum(total_pcc_SN_mdd,axis=1)
total_pcc_SN_hc_max = np.max(For_draw_label0, axis=2)
total_pcc_SN_hc_max = np.max(total_pcc_SN_hc_max, axis=1)
total_pcc_SN_mdd_max = np.max(For_draw_label1,axis=2)
total_pcc_SN_mdd_max = np.max(total_pcc_SN_mdd_max,axis=1)
# print('CEN hc', total_pcc_CEN_hc) # 79.22
# print('CEN mdd', total_pcc_CEN_mdd / 182)
stat, p_value = mannwhitneyu(total_pcc_SN_hc, total_pcc_SN_mdd)
stat_max, p_value_max = mannwhitneyu(total_pcc_SN_hc_max, total_pcc_SN_mdd_max)


print('SN p value', p_value)
# print('SN p value max', p_value_max)
# hc_small = []
# mdd_small = []
# for bb in range(For_draw_label0.shape[0]):
#     hc_small.append(small_world_coefficient(For_draw_label0[bb]))
# for bb in range(For_draw_label1.shape[0]):
#     mdd_small.append(small_world_coefficient(For_draw_label1[bb]))
# stat, p_value = mannwhitneyu(np.array(hc_small), np.array(mdd_small))
# print('SN small', p_value)



for l in range(Brain_DMN_CEN.shape[0]):
    Brain_DMN_CEN[l][l] = 0
For_draw_DMN_CEN = X * Brain_DMN_CEN
# for s in range(For_draw_DMN_CEN.shape[0]):
#     for i in range(For_draw_DMN_CEN.shape[1]):
#         for j in range(For_draw_DMN_CEN.shape[1]):
#             if abs(For_draw_DMN_CEN[s][i][j]) < T:
#                 For_draw_DMN_CEN[s][i][j] = 0
#             else:
#                 For_draw_DMN_CEN[s][i][j] = 1
# print('DMN-CEN-------------', For_draw_DMN_CEN.shape)
For_draw_label0 = For_draw_DMN_CEN[Y==0] # 246
For_draw_label1 = For_draw_DMN_CEN[Y==1]
# print('DMN-CEN-------------label0', For_draw_label0.shape)
total_pcc_DMN_CEN_hc = np.sum(For_draw_label0,axis=2)
total_pcc_DMN_CEN_hc = np.sum(total_pcc_DMN_CEN_hc,axis=1)
total_pcc_DMN_CEN_mdd = np.sum(For_draw_label1,axis=2)
total_pcc_DMN_CEN_mdd = np.sum(total_pcc_DMN_CEN_mdd,axis=1)
total_pcc_DMN_CEN_hc_max = np.max(For_draw_label0, axis=2)
total_pcc_DMN_CEN_hc_max = np.max(total_pcc_DMN_CEN_hc_max, axis=1)
total_pcc_DMN_CEN_mdd_max = np.max(For_draw_label1,axis=2)
total_pcc_DMN_CEN_mdd_max = np.max(total_pcc_DMN_CEN_mdd_max,axis=1)
# print('CEN hc', total_pcc_CEN_hc) # 79.22
# print('CEN mdd', total_pcc_CEN_mdd / 182)
stat, p_value = mannwhitneyu(total_pcc_DMN_CEN_hc, total_pcc_DMN_CEN_mdd)
stat, p_value_max = mannwhitneyu(total_pcc_DMN_CEN_hc_max, total_pcc_DMN_CEN_mdd_max)

print('DMN_CEN p value', p_value)
# print('DMN_CEN p value max', p_value_max)
# hc_small = []
# mdd_small = []
# for bb in range(For_draw_label0.shape[0]):
#     hc_small.append(small_world_coefficient(For_draw_label0[bb]))
# for bb in range(For_draw_label1.shape[0]):
#     mdd_small.append(small_world_coefficient(For_draw_label1[bb]))
# stat, p_value = mannwhitneyu(np.array(hc_small), np.array(mdd_small))
# print('DMNCEN small', p_value)

for l in range(Brain_DMN_SN.shape[0]):
    Brain_DMN_SN[l][l] = 0
For_draw_DMN_SN = X * Brain_DMN_SN
# for s in range(For_draw_DMN_SN.shape[0]):
#     for i in range(For_draw_DMN_SN.shape[1]):
#         for j in range(For_draw_DMN_SN.shape[1]):
#             if abs(For_draw_DMN_SN[s][i][j]) < T:
#                 For_draw_DMN_SN[s][i][j] = 0
#             else:
#                 For_draw_DMN_SN[s][i][j] = 1
For_draw_label0 = For_draw_DMN_SN[Y==0] # 246
For_draw_label1 = For_draw_DMN_SN[Y==1]
total_pcc_DMN_SN_hc = np.sum(For_draw_label0,axis=2)
total_pcc_DMN_SN_hc = np.sum(total_pcc_DMN_SN_hc,axis=1)
total_pcc_DMN_SN_mdd = np.sum(For_draw_label1,axis=2)
total_pcc_DMN_SN_mdd = np.sum(total_pcc_DMN_SN_mdd,axis=1)
total_pcc_DMN_SN_hc_max = np.max(For_draw_label0, axis=2)
total_pcc_DMN_SN_hc_max = np.max(total_pcc_DMN_SN_hc_max, axis=1)
total_pcc_DMN_SN_mdd_max = np.max(For_draw_label1,axis=2)
total_pcc_DMN_SN_mdd_max = np.max(total_pcc_DMN_SN_mdd_max,axis=1)
# print('CEN hc', total_pcc_CEN_hc) # 79.22
# print('CEN mdd', total_pcc_CEN_mdd / 182)
stat, p_value = mannwhitneyu(total_pcc_DMN_SN_hc, total_pcc_DMN_SN_mdd)
stat, p_value_max = mannwhitneyu(total_pcc_DMN_SN_hc_max, total_pcc_DMN_SN_mdd_max)

print('DMN_SN p value', p_value)
# print('DMN_SN p value max', p_value_max)
# hc_small = []
# mdd_small = []
# for bb in range(For_draw_label0.shape[0]):
#     hc_small.append(small_world_coefficient(For_draw_label0[bb]))
# for bb in range(For_draw_label1.shape[0]):
#     mdd_small.append(small_world_coefficient(For_draw_label1[bb]))
# stat, p_value = mannwhitneyu(np.array(hc_small), np.array(mdd_small))
# print('DMNSN small', p_value)

for l in range(Brain_CEN_SN.shape[0]):
    Brain_CEN_SN[l][l] = 0
For_draw_CEN_SN = X * Brain_CEN_SN
# for s in range(For_draw_CEN_SN.shape[0]):
#     for i in range(For_draw_CEN_SN.shape[1]):
#         for j in range(For_draw_CEN_SN.shape[1]):
#             if abs(For_draw_CEN_SN[s][i][j]) < T:
#                 For_draw_CEN_SN[s][i][j] = 0
#             else:
#                 For_draw_CEN_SN[s][i][j] = 1
For_draw_label0 = For_draw_CEN_SN[Y==0] # 246
For_draw_label1 = For_draw_CEN_SN[Y==1]
total_pcc_CEN_SN_hc = np.sum(For_draw_label0,axis=2)
total_pcc_CEN_SN_hc = np.sum(total_pcc_CEN_SN_hc,axis=1)
total_pcc_CEN_SN_mdd = np.sum(For_draw_label1,axis=2)
total_pcc_CEN_SN_mdd = np.sum(total_pcc_CEN_SN_mdd,axis=1)
total_pcc_CEN_SN_hc_max = np.max(For_draw_label0, axis=2)
total_pcc_CEN_SN_hc_max = np.max(total_pcc_CEN_SN_hc_max, axis=1)
total_pcc_CEN_SN_mdd_max = np.max(For_draw_label1,axis=2)
total_pcc_CEN_SN_mdd_max = np.max(total_pcc_CEN_SN_mdd_max,axis=1)
# print('CEN hc', total_pcc_CEN_hc) # 79.22
# print('CEN mdd', total_pcc_CEN_mdd / 182)
stat, p_value = mannwhitneyu(total_pcc_CEN_SN_hc, total_pcc_CEN_SN_mdd)
stat, p_value_max = mannwhitneyu(total_pcc_CEN_SN_hc_max, total_pcc_CEN_SN_mdd_max)

print('CEN_SN p value', p_value)
# print('CEN_SN p value max', p_value_max)
# hc_small = []
# mdd_small = []
# for bb in range(For_draw_label0.shape[0]):
#     hc_small.append(small_world_coefficient(For_draw_label0[bb]))
# for bb in range(For_draw_label1.shape[0]):
#     mdd_small.append(small_world_coefficient(For_draw_label1[bb]))
# stat, p_value = mannwhitneyu(np.array(hc_small), np.array(mdd_small))
# print('CENSN small', p_value)

'''


# 两条边的
Brain_D_C_S = Brain_DMN_CEN + Brain_CEN_SN
Brain_D_S_C = Brain_DMN_SN + Brain_CEN_SN
Brain_C_D_S = Brain_DMN_CEN + Brain_DMN_SN

Brain_D_D_C = Brain_DMN_CEN + Brain_DMN
Brain_D_D_S = Brain_DMN_SN + Brain_DMN
Brain_D_C_C = Brain_DMN_CEN + Brain_CEN
Brain_C_C_S = Brain_CEN_SN + Brain_CEN
Brain_D_S_S = Brain_DMN_SN + Brain_SN
Brain_C_S_S = Brain_CEN_SN + Brain_SN

#三条边的
Brain_three_hete = Brain_DMN_CEN + Brain_DMN_SN + Brain_CEN_SN

Brain_D_D_C_C = Brain_DMN + Brain_DMN_CEN + Brain_CEN
Brain_D_D_S_S = Brain_DMN + Brain_DMN_SN + Brain_SN
Brain_C_C_S_S = Brain_CEN + Brain_CEN_SN + Brain_SN
Brain_D_D_C_S = Brain_DMN + Brain_DMN_CEN + Brain_CEN_SN
Brain_D_D_S_C = Brain_DMN + Brain_DMN_SN + Brain_CEN_SN
Brain_C_C_D_S = Brain_CEN + Brain_DMN_CEN + Brain_DMN_SN
Brain_C_C_S_D = Brain_CEN + Brain_CEN_SN + Brain_DMN_SN
Brain_S_S_D_C = Brain_SN + Brain_DMN_SN + Brain_DMN_CEN
Brain_S_S_C_D = Brain_SN + Brain_CEN_SN + Brain_DMN_CEN

# tensor
Brain_DMN_tensor = torch.from_numpy(Brain_DMN).float().to(device)
Brain_CEN_tensor = torch.from_numpy(Brain_CEN).float().to(device)
Brain_SN_tensor = torch.from_numpy(Brain_SN).float().to(device)
Brain_DMN_CEN_tensor = torch.from_numpy(Brain_DMN_CEN).float().to(device)
Brain_DMN_SN_tensor = torch.from_numpy(Brain_DMN_SN).float().to(device)
Brain_CEN_SN_tensor = torch.from_numpy(Brain_CEN_SN).float().to(device)

Brain_D_C_S_tensor = torch.from_numpy(Brain_D_C_S).float().to(device)
Brain_D_S_C_tensor = torch.from_numpy(Brain_D_S_C).float().to(device)
Brain_C_D_S_tensor = torch.from_numpy(Brain_C_D_S).float().to(device)
Brain_D_D_C_tensor = torch.from_numpy(Brain_D_D_C).float().to(device)
Brain_D_D_S_tensor = torch.from_numpy(Brain_D_D_S).float().to(device)
Brain_D_C_C_tensor = torch.from_numpy(Brain_D_C_C).float().to(device)
Brain_C_C_S_tensor = torch.from_numpy(Brain_C_C_S).float().to(device)
Brain_D_S_S_tensor = torch.from_numpy(Brain_D_S_S).float().to(device)
Brain_C_S_S_tensor = torch.from_numpy(Brain_C_S_S).float().to(device)

Brain_three_hete_tensor = torch.from_numpy(Brain_three_hete).float().to(device) # DSCD
Brain_D_D_C_C_tensor = torch.from_numpy(Brain_D_D_C_C).float().to(device)
# print('----------------------------', Brain_D_D_C_C.shape) # 64 * 64
Brain_D_D_S_S_tensor = torch.from_numpy(Brain_D_D_S_S).float().to(device)
Brain_C_C_S_S_tensor = torch.from_numpy(Brain_C_C_S_S).float().to(device)
Brain_D_D_C_S_tensor = torch.from_numpy(Brain_D_D_C_S).float().to(device)
Brain_D_D_S_C_tensor = torch.from_numpy(Brain_D_D_S_C).float().to(device)
Brain_C_C_D_S_tensor = torch.from_numpy(Brain_C_C_D_S).float().to(device)
Brain_C_C_S_D_tensor = torch.from_numpy(Brain_C_C_S_D).float().to(device)
Brain_S_S_D_C_tensor = torch.from_numpy(Brain_S_S_D_C).float().to(device)
Brain_S_S_C_D_tensor = torch.from_numpy(Brain_S_S_C_D).float().to(device)

# Weight_con = np.zeros((Nodes_Atlas1_new, Nodes_Atlas1_new))
Weight_con = Brain_DMN + Brain_CEN + Brain_SN
Weight_con_T = np.ones((Nodes_Atlas1_new, Nodes_Atlas1_new))
Weight_con_T = Weight_con_T - Weight_con
Weight_con_tensor = torch.from_numpy(Weight_con).float().to(device)
Weight_con_tensor_T = torch.from_numpy(Weight_con_T).float().to(device)
# print(Weight_con.shape)

##

from statsmodels.multivariate.manova import MANOVA

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
def MANOVA_3fea(x1,x2):
    # print(x1)
    # print(x2)
    df1 = pd.DataFrame(x1, columns=['Feature1', 'Feature2', 'Feature3'])
    df1['Group'] = 'Group1'
    df2 = pd.DataFrame(x2, columns=['Feature1', 'Feature2', 'Feature3'])
    df2['Group'] = 'Group2'

    df = pd.concat([df1, df2], ignore_index=True)

    maov = MANOVA.from_formula('Feature1 + Feature2 + Feature3 ~ Group', data=df)
    print('-----------')
    print(maov.mv_test()) 

def MANOVA_2fea(x1,x2):
    # print(x1)
    # print(x2)
    df1 = pd.DataFrame(x1, columns=['Feature1', 'Feature2'])
    df1['Group'] = 'Group1'
    df2 = pd.DataFrame(x2, columns=['Feature1', 'Feature2'])
    df2['Group'] = 'Group2'

    df = pd.concat([df1, df2], ignore_index=True)

    maov = MANOVA.from_formula('Feature1 + Feature2 ~ Group', data=df)
    print('-----------')
    print(maov.mv_test()) 

def ANOVA_2fea(x1, x2):
    
    df1 = pd.DataFrame(x1, columns=['Feature'])
    df1['Group'] = 'Group1'
    df2 = pd.DataFrame(x2, columns=['Feature'])
    df2['Group'] = 'Group2'

    df = pd.concat([df1, df2], ignore_index=True)

    model = ols('Feature ~ Group', data=df).fit()
    anova_results = anova_lm(model)

    print('-----------')
    print(anova_results)

##### X Y 
# 先分出 HC & Patient
# print('--------------------', X.shape) # N 116 116
class_0_list = []
class_1_list = []
for i in range(Y.shape[0]):
    if Y[i] == 0:
        class_0_list.append(X[i])
    elif Y[i] == 1:
        class_1_list.append(X[i])
class_0_arr = np.array(class_0_list)
class_1_arr = np.array(class_1_list)
# print('------------', class_0_arr.shape) # 94
# print('------------', class_1_arr.shape) # 117




DD_ASD_class0 = class_0_arr * Brain_CEN # 94 64 64
DC_ASD_class0 = class_0_arr * Brain_CEN_SN
CC_ASD_class0 = class_0_arr * Brain_SN
DD_ASD_class0 = np.mean(DD_ASD_class0, axis=2)
DD_ASD_class0 = np.mean(DD_ASD_class0, axis=1)
# print('----------------', DD_ASD_class0) # 94
DC_ASD_class0 = np.mean(DC_ASD_class0, axis=2)
DC_ASD_class0 = np.mean(DC_ASD_class0, axis=1)
CC_ASD_class0 = np.mean(CC_ASD_class0, axis=2)
CC_ASD_class0 = np.mean(CC_ASD_class0, axis=1)
DDCC_arr_class0 = np.concatenate((np.expand_dims(DD_ASD_class0, axis=1), np.expand_dims(DC_ASD_class0, axis=1), np.expand_dims(CC_ASD_class0, axis=1)), axis=1)
# print('-----------', DDCC_arr.shape) # 94 3

DD_ASD_class1 = class_1_arr * Brain_CEN
DC_ASD_class1 = class_1_arr * Brain_CEN_SN
CC_ASD_class1 = class_1_arr * Brain_SN
DD_ASD_class1 = np.mean(DD_ASD_class1, axis=2)
DD_ASD_class1 = np.mean(DD_ASD_class1, axis=1)
DC_ASD_class1 = np.mean(DC_ASD_class1, axis=2)
DC_ASD_class1 = np.mean(DC_ASD_class1, axis=1)
CC_ASD_class1 = np.mean(CC_ASD_class1, axis=2)
CC_ASD_class1 = np.mean(CC_ASD_class1, axis=1)
DDCC_arr_class1 = np.concatenate((np.expand_dims(DD_ASD_class1, axis=1), np.expand_dims(DC_ASD_class1, axis=1), np.expand_dims(CC_ASD_class1, axis=1)), axis=1)

MANOVA_3fea(DDCC_arr_class0, DDCC_arr_class1)


# DC_ASD_class0 = class_0_arr * Brain_SN
# DS_ASD_class0 = class_0_arr * Brain_DMN_SN
# DC_ASD_class0 = np.mean(DC_ASD_class0, axis=2)
# DC_ASD_class0 = np.mean(DC_ASD_class0, axis=1)
# DS_ASD_class0 = np.mean(DS_ASD_class0, axis=2)
# DS_ASD_class0 = np.mean(DS_ASD_class0, axis=1)
# CDS_arr_class0 = np.concatenate((np.expand_dims(DC_ASD_class0, axis=1), np.expand_dims(DS_ASD_class0, axis=1)), axis=1)

# DC_ASD_class1 = class_1_arr * Brain_SN
# DS_ASD_class1 = class_1_arr * Brain_DMN_SN
# DC_ASD_class1 = np.mean(DC_ASD_class1, axis=2)
# DC_ASD_class1 = np.mean(DC_ASD_class1, axis=1)
# DS_ASD_class1 = np.mean(DS_ASD_class1, axis=2)
# DS_ASD_class1 = np.mean(DS_ASD_class1, axis=1)
# CDS_arr_class1 = np.concatenate((np.expand_dims(DC_ASD_class1, axis=1), np.expand_dims(DS_ASD_class1, axis=1)), axis=1)

# MANOVA_2fea(CDS_arr_class0, CDS_arr_class1)


# DS_ASD_class0 = class_0_arr * Brain_DMN_CEN
# DS_ASD_class0 = np.mean(DS_ASD_class0, axis=2)
# DS_ASD_class0 = np.mean(DS_ASD_class0, axis=1)
# # DS_ASD_class0 = np.concatenate((np.expand_dims(DS_ASD_class0, axis=1), np.expand_dims(DS_ASD_class0, axis=1)), axis=1)

# DS_ASD_class1 = class_1_arr * Brain_DMN_CEN
# DS_ASD_class1 = np.mean(DS_ASD_class1, axis=2)
# DS_ASD_class1 = np.mean(DS_ASD_class1, axis=1)
# # DS_ASD_class1 = np.concatenate((np.expand_dims(DS_ASD_class1, axis=1), np.expand_dims(DS_ASD_class1, axis=1)), axis=1)
# # print(DS_ASD_class0)
# # print(DS_ASD_class1)
# ANOVA_2fea(DS_ASD_class0, DS_ASD_class1)

###

# print('Brain DMN shape matrix', Brain_DMN.shape) # 64 64 
# 在这里 保存想要的元路径的模板
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# DDSC
# with open('./meta-paths/ASD-DS.npy', 'wb') as f:
#     np.save(f, X * Brain_DMN_SN) 
# quit()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

###

# 计算各种指标 Lin Lout Rin Rout 四个区域
# CC
def clustering_coefficient(adj_matrix):
    num_nodes = len(adj_matrix)
    clustering_coeffs = []

    for i in range(num_nodes):
        neighbors = np.nonzero(adj_matrix[i])[0]
        num_neighbors = len(neighbors)

        if num_neighbors < 2:
            clustering_coeffs.append(0)
            continue

        num_edges = 0
        for j in range(num_neighbors):
            for k in range(j + 1, num_neighbors):
                if adj_matrix[neighbors[j]][neighbors[k]] == 1:
                    num_edges += 1

        clustering_coeffs.append(2 * num_edges / (num_neighbors * (num_neighbors - 1)))

    return clustering_coeffs

Brain_DMN = X * Brain_DMN
Brain_CEN = X * Brain_CEN
Brain_SN = X * Brain_SN

Brain_DMN_CEN = X * Brain_DMN_CEN
Brain_DMN_SN = X * Brain_DMN_SN
Brain_CEN_SN = X * Brain_CEN_SN
# Brain_L_out_R_in = X * Brain_L_out_R_in
# Brain_L_out_R_out = X * Brain_L_out_R_out
# Brain_R_in_R_out = X * Brain_R_in_R_out

def draw(D):
    D = np.mean(D, axis=0)
    D = D.reshape(-1)
    D = D[D != 0]
    return D
Brain_DMN_CEN_draw = draw(Brain_DMN_CEN)
Brain_DMN_SN_draw = draw(Brain_DMN_SN)
Brain_CEN_SN_draw = draw(Brain_CEN_SN)
# Brain_L_out_R_in_draw = draw(Brain_L_out_R_in)
# Brain_L_out_R_out_draw = draw(Brain_L_out_R_out)
# Brain_R_in_R_out_draw = draw(Brain_R_in_R_out)
# print(Brain_L_in_L_out_draw.shape)
with open('Brain_DMN_CEN.npy', 'wb') as f:
    np.save(f, Brain_DMN_CEN_draw) # 1958
print(Brain_DMN_CEN_draw.shape)
with open('Brain_DMN_SN.npy', 'wb') as f:
    np.save(f, Brain_DMN_SN_draw) # 132
print(Brain_DMN_SN_draw.shape)
with open('Brain_CEN_SN.npy', 'wb') as f:
    np.save(f, Brain_CEN_SN_draw) # 2068
print(Brain_CEN_SN_draw.shape)
# with open('Brain_L_out_R_in.npy', 'wb') as f:
#     np.save(f, Brain_L_out_R_in_draw) # 1068
# print(Brain_L_out_R_out_draw.shape)
# with open('Brain_L_out_R_out.npy', 'wb') as f:
#     np.save(f, Brain_L_out_R_out_draw) # 16732
# print(Brain_R_in_R_out_draw.shape)
# with open('Brain_R_in_R_out.npy', 'wb') as f:
#     np.save(f, Brain_R_in_R_out_draw) # 1128

Nodes_Atlas1 = Nodes_Atlas1_new
for bb in range(0, N_subjects):
    for i in range(0, Nodes_Atlas1):
        for j in range(0, Nodes_Atlas1):
            if abs(Brain_DMN[bb][i][j]) > 0.03:
                Brain_DMN[bb][i][j] = 1
            else:
                Brain_DMN[bb][i][j] = 0

            if abs(Brain_CEN[bb][i][j]) > 0.03:
                Brain_CEN[bb][i][j] = 1
            else:
                Brain_CEN[bb][i][j] = 0

            if abs(Brain_SN[bb][i][j]) > 0.03:
                Brain_SN[bb][i][j] = 1
            else:
                Brain_SN[bb][i][j] = 0

            # if abs(Brain_R_out[bb][i][j]) > 0.05:
            #     Brain_R_out[bb][i][j] = 1
            # else:
            #     Brain_R_out[bb][i][j] = 0

def random_graph(num_nodes, edge_probability):
    random_adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_probability:
                random_adj_matrix[i][j] = 1
                random_adj_matrix[j][i] = 1
    return random_adj_matrix

def small_world_coefficient(adj_matrix, num_random_graphs=100, edge_probability=0.5):
    # Calculate actual clustering coefficient
    actual_clustering_coeff = clustering_coefficient2(adj_matrix)

    # Calculate average clustering coefficient of random graphs
    random_clustering_coeffs = []
    num_nodes = len(adj_matrix)
    for _ in range(num_random_graphs):
        random_graph_adj_matrix = random_graph(num_nodes, edge_probability)
        random_clustering_coeffs.append(clustering_coefficient2(random_graph_adj_matrix))
    average_random_clustering_coeff = np.mean(random_clustering_coeffs)

    # print(average_random_clustering_coeff) # 0

    # Calculate small world coefficient
    if average_random_clustering_coeff == 0:
        return 0
    else:
        return actual_clustering_coeff / average_random_clustering_coeff

# remove node
def remove_isolated_nodes(adj_matrix, isolated_nodes):
    num_nodes = len(adj_matrix)

    # Find isolated nodes (nodes with no connections)
    # if isolated_nodes == None:
    # print(isolated_nodes)
    # isolated_nodes2 = []
    isolated_nodes2 = [i for i in range(Nodes_Atlas1) if isolated_nodes[i] == 0]

    # Remove isolated nodes and their related rows/columns
    for node in isolated_nodes2[::-1]:
        adj_matrix = np.delete(adj_matrix, node, axis=0)
        adj_matrix = np.delete(adj_matrix, node, axis=1)

    return adj_matrix

def characteristic_path_length(adj_matrix):
    num_nodes = len(adj_matrix)
    total_distance = 0
    num_pairs = 0

    # Floyd-Warshall algorithm to compute all-pairs shortest paths
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i][j] > adj_matrix[i][k] + adj_matrix[k][j]:
                    adj_matrix[i][j] = adj_matrix[i][k] + adj_matrix[k][j]

    # Calculate total_distance and num_pairs
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i][j] < np.inf:
                total_distance += adj_matrix[i][j]
                num_pairs += 1

    # Calculate characteristic path length
    if num_pairs == 0:
        return 0
    else:
        return total_distance / num_pairs

def average_shortest_path_length(adj_matrix):
    num_nodes = len(adj_matrix)
    dist_matrix = np.copy(adj_matrix)

    # Set diagonal elements to 0 (distance to itself is 0)
    np.fill_diagonal(dist_matrix, 0)

    # Apply Floyd-Warshall algorithm to compute all-pairs shortest paths
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if dist_matrix[i][j] > dist_matrix[i][k] + dist_matrix[k][j]:
                    dist_matrix[i][j] = dist_matrix[i][k] + dist_matrix[k][j]

    # Calculate average shortest path length
    total_distance = np.sum(dist_matrix)
    num_pairs = num_nodes * (num_nodes - 1)
    if num_pairs == 0:
        return 0
    else:
        return total_distance / num_pairs

def average_node_degree(adj_matrix, directed=False):
    num_nodes = len(adj_matrix)
    total_degree = 0

    if not directed:
        # For undirected graph, degree is the sum of row or column elements (they are the same)
        for i in range(num_nodes):
            total_degree += np.sum(adj_matrix[i])
    else:
        # For directed graph, calculate both out-degree and in-degree
        for i in range(num_nodes):
            total_degree += np.sum(adj_matrix[i])  # Out-degree
            total_degree += np.sum(adj_matrix[:, i])  # In-degree

    # Calculate average degree
    average_degree = total_degree / num_nodes

    return average_degree

def clustering_coefficient2(adj_matrix):
    num_nodes = len(adj_matrix)
    clustering_coeffs = []

    for i in range(num_nodes):
        neighbors = np.nonzero(adj_matrix[i])[0]
        num_neighbors = len(neighbors)

        if num_neighbors < 2:
            clustering_coeffs.append(0)
            continue

        num_edges = 0
        for j in range(num_neighbors):
            for k in range(j + 1, num_neighbors):
                if adj_matrix[neighbors[j]][neighbors[k]] == 1:
                    num_edges += 1

        clustering_coeffs.append(2 * num_edges / (num_neighbors * (num_neighbors - 1)))

    return np.mean(clustering_coeffs)

# adjacency_matrix = np.array([[0, 1, 1, 0, 0],
#                              [1, 0, 1, 1, 0],
#                              [1, 1, 0, 1, 1],
#                              [0, 1, 1, 0, 1],
#                              [0, 0, 1, 1, 0]])

# degrees = np.sum(adjacency_matrix, axis=1)
# def gini_coefficient(data):
#     sorted_data = np.sort(data)
#     n = len(data)
#     denominator = np.sum(data)
#     if denominator == 0:
#         return 0  # Handle the case of total degree being zero
#     numerators = (np.arange(1, n + 1) * sorted_data).sum()
#     gini = (2 * numerators) / (n * denominator) - (n + 1) / n
#     return gini
# def find_neighbors(node_index):
#     return np.where(adjacency_matrix[node_index] == 1)[0]

# neighbors_degrees = [degrees[find_neighbors(i)] for i in range(len(degrees))]
# local_gini_coefficients = [gini_coefficient(neigh_degrees) if len(neigh_degrees) > 0 else 0.0 for neigh_degrees in neighbors_degrees]

# print("Local Gini Coefficients:", local_gini_coefficients)

import networkx as nx
def node_efficiency(adjacency_matrix):
    G = nx.Graph(adjacency_matrix)
    node_efficiencies = {}

    for node in G.nodes():
        shortest_paths = nx.single_source_shortest_path_length(G, node)
        # print(shortest_paths)
        efficiencies = [1 / (length+1) for length in shortest_paths.values()]
        if len(efficiencies) > 0:
            node_efficiencies[node] = sum(efficiencies) / len(efficiencies)
        else:
            node_efficiencies[node] = 0.0  # Handle isolated nodes

    return node_efficiencies

# Replace this with your actual adjacency matrix
# adjacency_matrix = np.array([[1, 1, 1, 0, 0],
#                              [1, 0, 1, 1, 0],
#                              [1, 1, 0, 1, 1],
#                              [0, 1, 1, 0, 1],
#                              [0, 0, 1, 1, 0]])

# efficiencies = node_efficiency(adjacency_matrix)
# print("Node Efficiencies:", efficiencies)



CC_DMN_all = 0
CC_CEN_all = 0
CC_SN_all = 0
# CC_R_out_all = 0

cpl_DMN_all = 0
cpl_CEN_all = 0
cpl_SN_all = 0
# cpl_R_out_all = 0

degree_DMN_all = 0
degree_CEN_all = 0
degree_SN_all = 0
# degree_R_out_all = 0

smallworld_DMN_all = 0
smallworld_CEN_all = 0
smallworld_SN_all = 0
# smallworld_R_out_all = 0

node_eff_DMN_all = 0
node_eff_CEN_all = 0
node_eff_SN_all = 0

# for bb in range(0, N_subjects):
#     np.sum(matrix)
DMN_node_num = 27
CEN_node_num = 21
SN_node_num = 16

conti = 0
cal = 0
print(Y)
for bb in range(0, N_subjects):
    if Y[bb] == 0:
        print(conti)
        conti+=1
        continue
    # print('ca')
    cal += 1
    # print(Brain_L_in[bb].shape) # 200 200
    DMN = remove_isolated_nodes(Brain_DMN[bb], CC200_sub_1)
    cc_DMN = clustering_coefficient(DMN)
    cc_DMN = cc_DMN[cc_DMN!=0]
    CC_DMN_all += np.mean(cc_DMN)
    cpl_DMN_all += characteristic_path_length(DMN)   
    degree_DMN_all += average_node_degree(DMN) 
    smallworld_DMN_all += small_world_coefficient(DMN)
    # print(sum(node_efficiency(DMN).values())/DMN_node_num)
    node_eff_DMN = sum(node_efficiency(DMN).values())/DMN_node_num
    node_eff_DMN_all += node_eff_DMN

    CEN = remove_isolated_nodes(Brain_CEN[bb], CC200_sub_2)    
    cc_CEN = clustering_coefficient(CEN)
    cc_CEN = cc_CEN[cc_CEN!=0]
    CC_CEN_all += np.mean(cc_CEN)
    cpl_CEN_all += characteristic_path_length(CEN)  
    degree_CEN_all += average_node_degree(CEN) 
    smallworld_CEN_all += small_world_coefficient(CEN)
    # print(sum(node_efficiency(DMN).values())/DMN_node_num)
    node_eff_CEN = sum(node_efficiency(CEN).values())/CEN_node_num
    node_eff_CEN_all += node_eff_CEN
    
    SN = remove_isolated_nodes(Brain_SN[bb], CC200_sub_3)
    cc_SN = clustering_coefficient(SN)
    cc_SN = cc_SN[cc_SN!=0]
    CC_SN_all += np.mean(cc_SN)
    cpl_SN_all += characteristic_path_length(SN) 
    degree_SN_all += average_node_degree(SN)  
    smallworld_SN_all += small_world_coefficient(SN)
    # node_eff_SN_all += node_efficiency(SN)
    # print(sum(node_efficiency(DMN).values())/DMN_node_num)
    node_eff_SN = sum(node_efficiency(SN).values())/SN_node_num
    node_eff_SN_all += node_eff_SN

    # R_out = remove_isolated_nodes(Brain_R_out[bb], CC200_R_out)
    # cc_R_out = clustering_coefficient(R_out)
    # cc_R_out = cc_R_out[cc_R_out!=0]
    # CC_R_out_all += np.mean(cc_R_out)
    # degree_R_out_all += average_node_degree(R_out) 
    # cpl_R_out_all += characteristic_path_length(R_out) 
    # smallworld_R_out_all += small_world_coefficient(R_out)


# print('CC DMN ', CC_DMN_all/N_subjects)
# print('CC CEN ', CC_CEN_all/N_subjects)
# print('CC SN ', CC_SN_all/N_subjects)
# # print('CC Rout ', CC_DMN_all/N_subjects)

# print('cpl DMN ', cpl_DMN_all/N_subjects)
# print('cpl CEN ', cpl_CEN_all/N_subjects)
# print('cpl SN ', cpl_SN_all/N_subjects)
# # print('cpl Rout ', cpl_R_out_all/N_subjects)

# print('avg_D DMN ', degree_DMN_all/N_subjects)
# print('avg_D CEN ', degree_CEN_all/N_subjects)
# print('avg_D SN ', degree_SN_all/N_subjects)
# # print('avg_D Rout ', degree_R_out_all/N_subjects)

# print('small_W DMN ', smallworld_DMN_all/N_subjects)
# print('small_W CEN ', smallworld_CEN_all/N_subjects)
# print('small_W SN ', smallworld_SN_all/N_subjects)
# # print('small_W Rout ', smallworld_R_out_all/N_subjects)

# print('node_eff DMN ', node_eff_DMN_all/N_subjects)
# print('node_eff CEN ', node_eff_CEN_all/N_subjects)
# print('node_eff SN ', node_eff_SN_all/N_subjects)

print('-----------------------over-----------------------')

print('CC DMN ', CC_DMN_all/cal)
print('CC CEN ', CC_CEN_all/cal)
print('CC SN ', CC_SN_all/cal)
# print('CC Rout ', CC_DMN_all/N_subjects)

print('cpl DMN ', cpl_DMN_all/cal)
print('cpl CEN ', cpl_CEN_all/cal)
print('cpl SN ', cpl_SN_all/cal)
# print('cpl Rout ', cpl_R_out_all/N_subjects)

print('avg_D DMN ', degree_DMN_all/cal)
print('avg_D CEN ', degree_CEN_all/cal)
print('avg_D SN ', degree_SN_all/cal)
# print('avg_D Rout ', degree_R_out_all/N_subjects)

print('small_W DMN ', smallworld_DMN_all/cal)
print('small_W CEN ', smallworld_CEN_all/cal)
print('small_W SN ', smallworld_SN_all/cal)
# print('small_W Rout ', smallworld_R_out_all/N_subjects)

print('node_eff DMN ', node_eff_DMN_all/cal)
print('node_eff CEN ', node_eff_CEN_all/cal)
print('node_eff SN ', node_eff_SN_all/cal)

#################
# 0


# 1
# CC DMN  0.7859181855889957
# CC CEN  0.8017083246291724
# CC SN  0.8191969660376246
# cpl DMN  0.006089352243198396
# cpl CEN  0.019178440607012046
# cpl SN  0.07834249084249083
# avg_D DMN  0.16137566137566137
# avg_D CEN  0.3932496075353215
# avg_D SN  1.213598901098901
# small_W DMN  0.010660784086802553
# small_W CEN  0.030986126151496284
# small_W SN  0.2504311391581794
# node_eff DMN  0.9515945873970567
# node_eff CEN  0.8914097995730644
# node_eff SN  0.7855926625457876

# print('CC DMN ', CC_DMN_all/DMN_node_num)
# print('CC CEN ', CC_CEN_all/CEN_node_num)
# print('CC SN ', CC_SN_all/SN_node_num)
# # print('CC Rout ', CC_DMN_all/N_subjects)

# print('cpl DMN ', cpl_DMN_all/DMN_node_num)
# print('cpl CEN ', cpl_CEN_all/CEN_node_num)
# print('cpl SN ', cpl_SN_all/SN_node_num)
# # print('cpl Rout ', cpl_R_out_all/N_subjects)

# print('avg_D DMN ', degree_DMN_all/DMN_node_num)
# print('avg_D CEN ', degree_CEN_all/CEN_node_num)
# print('avg_D SN ', degree_SN_all/SN_node_num)
# # print('avg_D Rout ', degree_R_out_all/N_subjects)

# print('small_W DMN ', smallworld_DMN_all/DMN_node_num)
# print('small_W CEN ', smallworld_CEN_all/CEN_node_num)
# print('small_W SN ', smallworld_SN_all/SN_node_num)
# print('small_W Rout ', smallworld_R_out_all/N_subjects)


N_subjects = X.shape[0]
Nodes_Atlas1 = Nodes_Atlas1_new

# where_are_nan = np.isnan(X)  # 找出数据中为nan的
# where_are_inf = np.isinf(X)  # 找出数据中为inf的
# for bb in range(0, N_subjects):
#     for i in range(0, Nodes_Atlas1):
#         for j in range(0, Nodes_Atlas1):
#             if where_are_nan[bb][i][j]:
#                 X[bb][i][j] = 0
#             if where_are_inf[bb][i][j]:
#                 X[bb][i][j] = 1

# where_are_nan = np.isnan(X_atlas2)  # 找出数据中为nan的
# where_are_inf = np.isinf(X_atlas2)  # 找出数据中为inf的
# for bb in range(0, N_subjects):
#     for i in range(0, Nodes_Atlas2):
#         for j in range(0, Nodes_Atlas2):
#             if where_are_nan[bb][i][j]:
#                 X[bb][i][j] = 0
#             if where_are_inf[bb][i][j]:
#                 X[bb][i][j] = 1

print('---------------------')
print('X Atlas1', X.shape) # N M M
print('Y Atlas1', Y.shape)
print('X Atlas2', X_atlas2.shape) # N M M
print('Y Atlas2', Y_atlas2.shape)

print('---------------------')

epochs_rec = 0 # 10-724 20
epochs = 100 + epochs_rec # 200 671

# if True:
#     epochs = 0

batch_size = 32 # 64 0.660
dropout = 0.5
lr = 0.003 #0.005
decay = 0.01
result = []
acc_final = 0
result_final = []

from sklearn.model_selection import KFold
for ind in range(1):
    setup_seed(ind)

    # Masked
    nodes_number = Nodes_Atlas1 # 暂时不用mask
    nums = np.ones(Nodes_Atlas1) # 制作mask模板
    nums[:Nodes_Atlas1-nodes_number] = 0 # 根据设置的nodes number 决定多少是mask 即mask比例
    np.random.seed(1)
    np.random.shuffle(nums) # 200 75%1 25%0 shuffle打散
    # print(nums)
    # print('nums----------')
    Mask = nums.reshape(nums.shape[0], 1) * nums # 200 200
    # print('X before ', X.shape)
    Masked_X = X * Mask # 将部分转换为 0（masked）
    # print('X after ', X.shape)
    X0=X
    Masked_X_rest = X - Masked_X
    # print('Masked_X_rest ', Masked_X_rest[0][])
    J = nodes_number # J 拷贝出一份
    for i in range(0, J):
        ind = i
        # ind = nums.shape[0] - 1 - i
        if nums[ind] == 0:
            for j in range(J, Nodes_Atlas1):
                if nums[j] == 1:
                    Masked_X[:, [ind, j], :] = Masked_X[:, [j, ind], :]
                    Masked_X[:, :, [ind, j]] = Masked_X[:, :, [j, ind]]
                    Masked_X_rest[:, [ind, j], :] = Masked_X_rest[:, [j, ind], :]
                    Masked_X_rest[:, :, [ind, j]] = Masked_X_rest[:, :, [j, ind]]
                    X0[:, [ind, j], :] = X0[:, [j, ind], :]
                    X0[:, :, [ind, j]] = X0[:, :, [j, ind]]
                    J = j + 1
                    break
                
            # Masked_X = np.delete(Masked_X, ind, axis=1)
            # Masked_X = np.delete(Masked_X, ind, axis=2)
    # print(Masked_X[0,1,:])
    # print(Masked_X_rest[0,1,:]) # 只有后面的有值
    X_0 = Masked_X # X_0 是残缺的 只有前面unmasked的有值
    Masked_X = Masked_X[:, :nodes_number, :nodes_number]
    unMasked_X = Masked_X[:, nodes_number:, nodes_number:] # 这个是被mask的那些pcc

    X = Masked_X 
    X_unmasked = Masked_X_rest  

    acc_all = 0
    CV = 5
    kf = KFold(n_splits=5, shuffle=True)
    kfold_index = 0
    for trainval_index, test_index in kf.split(X, Y):
        kfold_index += 1
        print('kfold_index:', kfold_index)
        X_trainval, X_test = X[trainval_index], X[test_index]
        Y_trainval, Y_test = Y[trainval_index], Y[test_index]

        X_trainval_masked_rest = X_unmasked[trainval_index]
        X_test_masked_rest = X_unmasked[test_index]

        X_trainval_0, X_test_0 = X0[trainval_index], X0[test_index]
        for train_index, val_index in kf.split(X_trainval, Y_trainval):
            X_train, X_val = X_trainval[:], X_trainval[:]
            Y_train, Y_val = Y_trainval[:], Y_trainval[:]

            X_train_masked_rest = X_trainval_masked_rest[:]
            X_val_masked_rest = X_trainval_masked_rest[:]

            X_train_0 = X_trainval_0[:] # 完整的A
            X_val_0 = X_trainval_0[:]
        print('X_train', X_train.shape)
        print('X_val', X_val.shape)
        print('X_test', X_test.shape)
        print('Y_train', Y_train.shape)
        print('Y_val', Y_val.shape)
        print('Y_test', Y_test.shape)
        
        # train dataset average
        X_train_number = X_train.shape[0]
        X_avg = X_train_0.sum(axis=0)
        # X_avg = X_avg.sum(axis=2)
        # print('------X avg-------', X_avg.shape)
        X_avg = X_avg / X_train_number

        for k in range(X_train_0.shape[0]):
            for i in range(X_train_0.shape[1]):
                for j in range(X_train_0.shape[2]):
                    if abs(X_train_0[k][i][j] < 0.2):
                        X_train_0[k][i][j] = 0
                    else:
                        X_train_0[k][i][j] = 1
        for k in range(X_test_0.shape[0]):
            for i in range(X_test_0.shape[1]):
                for j in range(X_test_0.shape[2]):
                    if abs(X_test_0[k][i][j] < 0.2): #
                        X_test_0[k][i][j] = 0
                    else:
                        X_test_0[k][i][j] = 1

        for k in range(X_avg.shape[0]):
            for i in range(X_avg.shape[1]):
                if abs(X_avg[k][i] < 0.15): 
                    X_avg[k][i] = 0
                else:
                    X_avg[k][i] = 1
        print('-----')
        print(X_avg.sum(axis=0).sum(axis=0))
        # for k in range(X_avg_test.shape[0]):
        #     for i in range(X_avg_test.shape[1]):
        #         if abs(X_avg_test[k][i] < 0.5):
        #             X_avg_test[k][i] = 0
        #         else:
        #             X_avg_test[k][i] = 1

        print(X_avg)
        X_avg = X_avg.reshape(1, X_avg.shape[0], X_avg.shape[1]) #  
        X_avg = np.repeat(X_avg, X_train.shape[0], 0)
        X_avg_test = np.repeat(X_avg, X_test.shape[0], 0)

        X_train_0 = X_train_0
        X_test_0 = X_test_0

        # X_train_0 = X_avg # 注释了 这里是平均的A


        # X_test_0 = X_avg_test
        # print(X_test_0)

        X_masked = np.zeros([X_train.shape[0], Nodes_Atlas1-nodes_number, 48])
        X_masked_test = np.zeros([X_test.shape[0], Nodes_Atlas1-nodes_number, 48])
        print('')
        for i in range(X_masked.shape[0]):
            for j in range(X_masked.shape[1]):
                X_masked[i][j] = np.random.normal(loc=0.0, scale=1.0, size=48)
        for i in range(X_masked_test.shape[0]):
            for j in range(X_masked_test.shape[1]):
                X_masked_test[i][j] = np.random.normal(loc=0.0, scale=1.0, size=48)

        # model
        model = Model(dropout=dropout, num_class=2)
        model.to(device)
        

        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
        optimizer2 = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
        # optimizer2 = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    #     lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 200, 250], gamma=0.8)
        loss_fn = nn.CrossEntropyLoss()
        loss_rec = nn.MSELoss()
        # loss_fn = nn.MSELoss()
        
        best_val = 0
        best_loss = 1000
        best_acc = 0
        
        # train
        for epoch in range(1, epochs+1):
            model.train()

            idx_batch = np.random.permutation(int(X_train.shape[0]))
            num_batch = X_train.shape[0] // int(batch_size)
            
            loss_train = 0
            for bn in range(num_batch):
                if bn == num_batch - 1:
                    batch = idx_batch[bn * int(batch_size):]
                else:
                    batch = idx_batch[bn * int(batch_size) : (bn+1) * int(batch_size)]
                train_data_batch = X_train[batch]
                train_label_batch = Y_train[batch]
                train_data_batch_A = X_train_0[batch]
                train_data_batch_rest = X_train_masked_rest[batch]
                train_data_batch_maskedX = X_masked[batch]
                # print(train_data_batch[0])

                train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
                train_label_batch_dev = torch.from_numpy(train_label_batch).long().to(device)
                train_data_batch_A_dev = torch.from_numpy(train_data_batch_A).float().to(device)
                train_data_batch_rest_dev = torch.from_numpy(train_data_batch_rest).float().to(device)
                train_data_batch_maskedX_dev = torch.from_numpy(train_data_batch_maskedX).float().to(device)

                optimizer.zero_grad()
                optimizer2.zero_grad()
                if epoch < epochs_rec:
                    outputs, rec, ret, nodeclf, loss_corr = model(train_data_batch_dev, train_data_batch_A_dev, train_data_batch_maskedX_dev, 'pre')
                else:
                    outputs, rec, ret, _, loss_corr = model(train_data_batch_dev, train_data_batch_A_dev, train_data_batch_maskedX_dev, 'clf')
                
                loss1 = loss_fn(outputs, train_label_batch_dev) + ret * 0.1 + loss_corr * 0.001
                # print(loss_corr)
                # print(train_data_batch_rest_dev[0][0])
                # loss2 = loss_rec(rec, train_data_batch_rest_dev)

                loss_nodeclf = nn.CrossEntropyLoss()
                # targets = Node_type_list_tensor.repeat(train_data_batch_dev.size(0), 1)

                if epoch < epochs_rec:
                    loss2 = loss_rec(rec, train_data_batch_rest_dev)
                    # loss3 = loss_nodeclf(nodeclf, targets.argmax(axis=1).long())
                    loss3 = 0
                    # print(loss3)
                    loss = loss2 + loss3
                    loss_train += loss
                    loss.backward()
                    optimizer.step()
                else:
                    loss = loss1 # + loss2
                    loss_train += loss
                    loss.backward()
                    optimizer.step()

                
            
            loss_train /= num_batch
            if epoch % 10 == 0:
                print('epoch:', epoch, 'train loss:', loss_train.item())
                
            # val
            if epoch % 1 == 0 and epoch > epochs_rec:
                # model.eval()
                
                # val_data_batch_dev = torch.from_numpy(X_val).float().to(device)
                # val_label_batch_dev = torch.from_numpy(Y_val).long().to(device)
                # outputs, rec = model(val_data_batch_dev)
                # loss1 = loss_fn(outputs, val_label_batch_dev)
                # loss2 = loss_rec(rec, val_data_batch_dev)
                # loss = loss1 + loss2
                # _, indices = torch.max(outputs, dim=1)
                # preds = indices.cpu()
                # # print(preds)
                # acc_val = metrics.accuracy_score(preds, Y_val)
                if loss_train < best_loss: #True: #acc_val > best_val:
                    # best_val = acc_val
                    best_loss = loss_train
                    model.eval()
                    test_data_batch_dev = torch.from_numpy(X_test).float().to(device)
                    test_data_batch_A_dev = torch.from_numpy(X_test_0).float().to(device)
                    test_data_batch_maskedX_dev = torch.from_numpy(X_masked_test).float().to(device)

                    print(test_data_batch_dev.shape)

                    # time
                    import time
                    # torch.cuda.synchronize()
                    start = time.time()

                    outputs, _, _, _, _ = model(test_data_batch_dev, test_data_batch_A_dev, test_data_batch_maskedX_dev, 'clf')
                    
                    # torch.cuda.synchronize()
                    end = time.time()

                    # print('Test Time: ', end-start)

                    _, indices = torch.max(outputs, dim=1)
                    preds = indices.cpu()
                    # print(preds)
                    acc = metrics.accuracy_score(preds, Y_test)
                    print('Test acc', acc)
                    if best_acc < acc:
                        best_acc = acc

                # print('Test acc', acc_val)
        print(best_acc)
        # if epoch % 1 == 0:
            
        torch.save(model.state_dict(), './models/' + str(kfold_index) + '.pt')
        result.append([kfold_index, acc])
        acc_all += acc
    temp = acc_all / CV
    acc_final += temp
    result_final.append(temp)
    print(result)

ACC = acc_final / CV
# In[]

print(result_final)
print(ACC)

# 0.652