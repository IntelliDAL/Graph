from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import warnings
import sklearn.metrics as metrics
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.autograd import Variable
import time
import math
from torch.nn import init
import torch
import random

warnings.filterwarnings("ignore")


def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


def AUC(label, pre):
    pos = []
    neg = []
    auc = 0
    for index, l in enumerate(label):
        if l == 0:
            neg.append(index)
        else:
            pos.append(index)
    for i in pos:
        for j in neg:
            if pre[i] > pre[j]:
                auc += 1
            elif pre[i] == pre[j]:
                auc += 0.5
    return auc * 1.0 / (len(pos) * len(neg))


def call_indexes(labels, preds):
    result = np.zeros(len(labels))
    result[(labels == 0) * (preds == 0)] = 1
    result[(labels == 0) * (preds == 1)] = 2
    result[(labels == 1) * (preds == 0)] = 3
    result[(labels == 1) * (preds == 1)] = 4
    # TN FP FN TP, FP:实际为负 预测为正
    TN = sum(result == 1)
    FP = sum(result == 2)
    FN = sum(result == 3)
    TP = sum(result == 4)
    print("TN={}, FP={}, FN={}, TP={}".format(TN, FP, FN, TP))

    acc = (TN + TP) / (TN + TP + FN + FP)
    pre = TP / (TP + FP) if (TP + FP) > 0 else -1
    sen = TP / (TP + FN) if (TP + FN) > 0 else -1  # sen = recall
    spe = TN / (TN + FP) if (TN + FP) > 0 else -1
    F1 = 2 * pre * sen / (pre + sen)
    auc = AUC(labels, preds)

    _index = [acc, pre, sen, spe, F1, auc]
    _matrix = [TN, FP, FN, TP]

    return _index, _matrix


def cal_result_ave(result, zhe):
    ave_result = {'acc': 0,
                  'pre': 0,
                  'sen/recall': 0,
                  'spe': 0,
                  'F1': 0,
                  'auc': 0}

    std_result = {'acc': [],
                  'pre': [],
                  'sen/recall': [],
                  'spe': [],
                  'F1': [],
                  'auc': []}

    for z in range(0, zhe):
        for key in ave_result.keys():
            ave_result[str(key)] += result[z][str(key)]
            std_result[str(key)].append(result[z][str(key)])

    for key in ave_result.keys():
        ave_result[str(key)] /= zhe
        std_temp = std_result[str(key)]
        std_result[str(key)] = 0
        std_result[str(key)] = np.std(std_temp, ddof=1)

    return ave_result, std_result


# 模型部分
class tcn_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, p):
        super(tcn_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=p)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class tcn_Networks(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super(tcn_Networks, self).__init__()
        self.block1 = tcn_block(1, 8, 7, 2, 1, 0.2)
        self.block2 = tcn_block(8, 16, 7, 2, 1, 0.2)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """
        :param x:   N V T
        :return:
        """
        N, V, T = x.size()
        x = x.reshape(N * V, 1, T)  # N*V  1 T
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(N, V, -1)

        x = self.fc(x)
        return x


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.in_dim = in_dim  # 输入的维度
        self.out_dim = out_dim  # 输出的维度
        # self.neg_penalty = neg_penalty  # 负值
        self.kernel = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        init.xavier_normal_(self.kernel)
        self.c = 0.85

    def forward(self, x, adj):
        feature_dim = int(adj.shape[-1])
        eye = torch.eye(feature_dim).cuda()  # 生成对角矩阵 feature_dim * feature_dim
        if x is None:  # 如果没有初始特征
            AXW = torch.tensordot(adj, self.kernel, [[-1], [0]])  # batch_size * num_node * feature_dim
        else:
            XW = torch.tensordot(x, self.kernel, [[-1], [0]])  # batch *  num_node * feature_dim
            AXW = torch.matmul(adj, XW)  # batch *  num_node * feature_dim
        I_cAXW = AXW
        y_relu = torch.nn.functional.relu(I_cAXW)
        temp = torch.mean(input=y_relu, dim=-2, keepdim=True) + 1e-6
        col_mean = temp.repeat([1, feature_dim, 1])
        y_norm = torch.divide(y_relu, col_mean)  # 正则化后的值
        output = torch.nn.functional.softplus(y_norm)
        # if self.neg_penalty != 0:
        #     neg_loss = torch.multiply(torch.tensor(self.neg_penalty),
        #                               torch.sum(torch.nn.functional.relu(1e-6 - self.kernel)))
        return output


class Layer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_node, num_reduce, skip):
        super(Layer, self).__init__()
        self.num_reduce = num_reduce  # 5
        self.num_node = num_node  # 200
        # self.kernel_p = nn.ParameterList(
        #     [nn.Parameter(torch.FloatTensor(self.num_node, self.num_reduce)) for i in range(22)])
        # self.kernel_n = nn.ParameterList(
        #     [nn.Parameter(torch.FloatTensor(self.num_node, self.num_reduce)) for i in range(22)])
        self.kernel = nn.Parameter(torch.FloatTensor(self.num_node, self.num_reduce))
        self.in_features = in_features  # 输入特征  5
        self.hidden_features = hidden_features  # 卷積之後的特徵
        self.out_features = out_features  # lstm 輸出的特徵
        # 這一套是 lstm的參數
        self.skip = skip
        self.tcn = tcn_Networks(528, 128)
        self.Wi = nn.Parameter(torch.FloatTensor(hidden_features, out_features))
        # init.xavier_normal_(self.Wi)
        self.Wf = nn.Parameter(torch.FloatTensor(hidden_features, out_features))
        # init.xavier_normal_(self.Wf)
        self.Wo = nn.Parameter(torch.FloatTensor(hidden_features, out_features))
        # init.xavier_normal_(self.Wo)
        self.Wc = nn.Parameter(torch.FloatTensor(hidden_features, out_features))
        # init.xavier_normal_(self.Wc)
        self.bi = nn.Parameter(torch.FloatTensor(out_features, 1))
        # init.zeros_(self.bi)
        self.bf = nn.Parameter(torch.FloatTensor(out_features, 1))
        # init.zeros_(self.bf)
        self.bo = nn.Parameter(torch.FloatTensor(out_features, 1))
        # init.zeros_(self.bo)
        self.bc = nn.Parameter(torch.FloatTensor(out_features, 1))
        # init.zeros_(self.bc)
        # 這是一套skip的參數
        self.Wi_skip1 = nn.Parameter(torch.FloatTensor(hidden_features, out_features))
        # init.xavier_normal_(self.Wi_skip1)
        self.Wf_skip1 = nn.Parameter(torch.FloatTensor(hidden_features, out_features))
        # init.xavier_normal_(self.Wf_skip1)
        self.Wo_skip1 = nn.Parameter(torch.FloatTensor(hidden_features, out_features))
        # init.xavier_normal_(self.Wo_skip1)
        self.Wc_skip1 = nn.Parameter(torch.FloatTensor(hidden_features, out_features))
        # init.xavier_normal_(self.Wc_skip1)
        self.bi_skip1 = nn.Parameter(torch.FloatTensor(out_features, 1))
        # init.zeros_(self.bi_skip1)
        self.bf_skip1 = nn.Parameter(torch.FloatTensor(out_features, 1))
        # init.zeros_(self.bf_skip1)
        self.bo_skip1 = nn.Parameter(torch.FloatTensor(out_features, 1))
        # init.zeros_(self.bo_skip1)
        self.bc_skip1 = nn.Parameter(torch.FloatTensor(out_features, 1))
        # init.zeros_(self.bc_skip1)
        self.gcn_networks = GCN(self.in_features + self.out_features, self.hidden_features)
        self.fc = nn.Linear((self.skip + 1) * out_features, out_features)
        self.dropout = nn.Dropout(p=0.5)
        self.losses = []

        self.reset_weigths()

    def reset_weigths(self):
        """reset weights
        """
        stdv = 1.0 / math.sqrt(256)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def dim_reduce(self, adj_matrix, num_reduce,
                   ortho_penalty, variance_penalty, neg_penalty, kernel):
        kernel_p = torch.nn.functional.relu(kernel)
        AF = torch.tensordot(adj_matrix, kernel_p, [[-1], [0]])
        reduced_adj_matrix = torch.transpose(
            torch.tensordot(kernel_p, AF, [[0], [1]]),  # num_reduce*batch*num_reduce
            1, 0)  # num_reduce*batch*num_reduce*num_reduce
        kernel_p_tran = kernel_p.transpose(-1, -2)  # num_reduce * column_dim
        gram_matrix = torch.matmul(kernel_p_tran, kernel_p)
        diag_elements = gram_matrix.diag()

        if ortho_penalty != 0:
            ortho_loss_matrix = torch.square(gram_matrix - torch.diag(diag_elements))
            ortho_loss = torch.multiply(torch.tensor(ortho_penalty), torch.sum(ortho_loss_matrix))
            self.losses.append(ortho_loss)

        if variance_penalty != 0:
            variance = diag_elements.var()
            variance_loss = torch.multiply(torch.tensor(variance_penalty), variance)
            self.losses.append(variance_loss)

        if neg_penalty != 0:
            neg_loss = torch.multiply(torch.tensor(neg_penalty),
                                      torch.sum(torch.nn.functional.relu(torch.tensor(1e-6) - kernel)))
            self.losses.append(neg_loss)
        self.losses.append(0.05 * torch.sum(torch.abs(kernel_p)))
        return reduced_adj_matrix

    
    def forward(self, X, A, state, state_skip, device='cpu'):
        # 维度下降
        # x 代表的是原始输入的时序信号
        # A 代表的是adj 也就是graph
        (h, c) = state
        h_t = h  # (N, V, H) 64*5*256
        c_t = c  # (N, V, H)
        (h_skip, c_skip) = state_skip
        h_t_skip = h_skip
        c_t_skip = c_skip
        seq_size = A.shape[1]  # 子图个数
        batch = X.shape[0]
        lstm_outs = []
        # lstm 运算
        features = []
        A_pool = []
        # lstm 运算
        # t是时间序列片段的数量 N是受试者数量
        for t in range(seq_size):
            # 看一下t的变换，也就是数值范围
            adj_t = A[:, t, :, :]  # pcc值組成的圖
            adj_t = self.dim_reduce(adj_t, self.num_reduce, 0.2, 0.3, 0.1,
                                    self.kernel)  # 维度下降后的连通矩阵
            # adj_t : batch_size * 5 * 5

            # adj_t 最后的5*5是一个图，之后的5*256应该看一下这个5是不是ensemble的过程
            A_pool.append(adj_t.unsqueeze(1))  # unsqueeze(1)是在第1维添加一个维度
            feature_t = X[:, t, :, :]  # 原始信號
            feature_t = self.tcn(feature_t)  # 用tcn學習原始信號
            feature_t = torch.matmul(self.kernel.t(), feature_t)  # 轉職    batch num_reduce in_dim
            features.append(feature_t.unsqueeze(1))  # 64*5*256
            combined = torch.cat([feature_t, h_t], 2)  # 64*5*384
            # x进行了gcn操作，这里运行了graphembedding以及gcn操作，所以x就是时态graph
            x = self.gcn_networks(combined, adj_t)  # 64*5*100
            it = torch.sigmoid((torch.matmul(x, self.Wi) + self.bi.t()))  # (batch_size,N,hidden_size)
            ft = torch.sigmoid((torch.matmul(x, self.Wf) + self.bf.t()))
            ot = torch.sigmoid((torch.matmul(x, self.Wo) + self.bo.t()))
            ut = torch.relu((torch.matmul(x, self.Wc) + self.bc.t()))
            c_t = ft * c_t + it * ut  # shape=
            h_t = ot * torch.tanh(c_t)  # shape = (batch_size, num_node, hidden_dim) 64*5*256
            # print("h_t.shape", h_t.shape) # 64*5*256
            # lstm_outs.append(h_t.unsqueeze(1))
            # else:
            #     adj_t = A[:, t, :, :]  # pcc值为正的子图
            #     adj_t_before = A[:, t - 1, :, :]
            #     adj_t_diff = torch.abs(adj_t - adj_t_before)
            #     diff_t = self.gcn_difference(None, adj_t_diff)  # score得分
            #     feature_t = X[:, t, :, :]
            #     feature_t = self.tcn(feature_t)
            #     features.append(feature_t.unsqueeze(1))
            #     combined = torch.cat([feature_t, h_t], 2)
            #     x = self.gcn_networks(combined, adj_t)
            #     x = x * diff_t
            #     it = torch.sigmoid((torch.matmul(x, self.Wi) + self.bi.t()))  # (batch_size,N,hidden_size)
            #     ft = torch.sigmoid((torch.matmul(x, self.Wf) + self.bf.t()))
            #     ot = torch.sigmoid((torch.matmul(x, self.Wo) + self.bo.t()))
            #     ut = torch.relu((torch.matmul(x, self.Wc) + self.bc.t()))
            #     c_t = ft * c_t + it * ut  # shape=
            #     h_t = ot * torch.tanh(c_t)  # shape = (batch_size, num_node, hidden_dim)
        # skip-lstm 运算    x_after_gcn [64, 4, num_node, dim]
        features = torch.cat(features, 1).contiguous()
        num_p = int(seq_size / self.skip)
        # 此处的features是拿到的最后几处时间段的特征
        features = features[:, -(num_p * self.skip):, :, :]
        features = features.view(features.shape[0], -1, self.skip, features.shape[2], features.shape[3])  # B 3 3 N V
        features = features.permute([0, 2, 1, 3, 4]).contiguous()
        features = features.view(-1, features.shape[2], features.shape[3], features.shape[4])
        A_pool = torch.cat(A_pool, 1)

        A_pool = A_pool[:, -(num_p * self.skip):, :, :]
        A_pool = A_pool.view(A_pool.shape[0], -1, self.skip, A_pool.shape[2], A_pool.shape[3])  # B 3 3 N V
        A_pool = A_pool.permute([0, 2, 1, 3, 4]).contiguous()
        A_pool = A_pool.view(-1, A_pool.shape[2], A_pool.shape[3], A_pool.shape[4])

        for j in range(features.shape[1]):
            # 在经过之前的维度变换之后，此处相连的其实就是原本应该跳跃相隔的图
            feature_t_skip = features[:, j, :, :]
            adj_t_skip = A_pool[:, j, :, :]
            combined_skip = torch.cat([feature_t_skip, h_t_skip], 2)
            x_skip = self.gcn_networks(combined_skip, adj_t_skip)

            it_skip = torch.sigmoid(
                (torch.matmul(x_skip, self.Wi_skip1) + self.bi_skip1.t()))  # (batch_size,N,hidden_size)
            ft_skip = torch.sigmoid((torch.matmul(x_skip, self.Wf_skip1) + self.bf_skip1.t()))
            ot_skip = torch.sigmoid((torch.matmul(x_skip, self.Wo_skip1) + self.bo_skip1.t()))
            ut_skip = torch.relu((torch.matmul(x_skip, self.Wc_skip1) + self.bc_skip1.t()))
            c_t_skip = ft_skip * c_t_skip + it_skip * ut_skip  # shape=
            h_t_skip = ot_skip * torch.tanh(c_t_skip)  # shape = (batch_size, num_node, hidden_dim)
            # print("h_t_skip.shape", h_t_skip.shape) # 128 * 5 * 256
            lstm_outs.append(h_t_skip.unsqueeze(1))
        
        # skip-lstm 运算    x_after_gcn [64, 4, num_node, dim]
        h_t_skip = h_t_skip.view(batch, -1, h_t_skip.shape[1], h_t_skip.shape[2])  # 64*2*5*256
        h_t_skip = h_t_skip.permute([0, 2, 1, 3]).contiguous()  # 64*5*2*256
        h_t_skip = h_t_skip.view(h_t_skip.shape[0], h_t_skip.shape[1], -1)  # 64*5*512
        # h_t 是之前的循环的也就是lstm部分的，后边这个h_t_ship是lstm_skip部分的，从论文中可以得知，第二部分为跳跃，第一部分为整体放入lstm
        h_t_all = torch.cat([h_t, h_t_skip], 2)  # 64*5*256
        h_t_all = self.fc(h_t_all)
        h_t_all = self.dropout(h_t_all)
        # 此处的losses为超图聚合的loss，不需要动
        # 只需要对最后返回的loss加上一个时序的loss即可，这个loss需要在此模型中写入一个函数
        loss = torch.sum(torch.tensor(self.losses))
        self.losses.clear()
        return h_t_all, loss


class BrainNet(nn.Module):
    def __init__(self):
        super(BrainNet, self).__init__()
        """
        """
        self.LG_Networks = Layer(128, 100, 256, 200, 5, 2)
        self.lin = nn.Linear(256 * 5, 2)

    def forward(self, x, adj):
        state = (torch.zeros([adj.shape[0], 5, 256]).cuda(),
                 torch.zeros([adj.shape[0], 5, 256]).cuda())
        state_skip = (
            torch.zeros([adj.shape[0] * 2, 5, 256]).cuda(),
            torch.zeros([adj.shape[0] * 2, 5, 256]).cuda())
        h, loss = self.LG_Networks(x, adj, state, state_skip)
        h = h.reshape(h.shape[0], -1)
        x = self.lin(h)
        return x, loss


def data_augment_train(data, label):
    window = 300
    new_data = []
    new_label = []
    for i in range(len(data)):
        for j in range(3):
            new_data.append(data[i, :, j * window:(j + 1) * window])
            new_label.append(label[i])
    new_data = np.array(new_data)
    new_label = np.array(new_label)
    # print(new_data.shape)     # num_sample 3 22 400
    return new_data, new_label


def data_augment_test(data, label):
    window = 300
    new_data = []
    new_label = []
    for i in range(len(data)):
        temp_data = []
        for j in range(3):
            temp_data.append(data[i, :, j * window:(j + 1) * window])
        new_label.append(label[i])
        new_data.append(temp_data)
    new_data = np.array(new_data)
    new_label = np.array(new_label)
    # print(new_data.shape)     # num_sample 3 22 400
    return new_data, new_label


def load_data(fold, split_stride_value, split_window_value):
    dir_path = "ASDData/"
    # dir_path = "ASDData/"

    train_data_path = dir_path + 'train_data_' + str(fold) + '.npy'
    train_label_path = dir_path + 'train_label_' + str(fold) + '.npy'
    test_data_path = dir_path + 'test_data_' + str(fold) + '.npy'
    test_label_path = dir_path + 'test_label_' + str(fold) + '.npy'

    # train_data_path = 'data/train_data_' + str(fold) + '.npy'
    # test_data_path = 'data/test_data_' + str(fold) + '.npy'
    # test_label_path = 'data/test_label_' + str(fold) + '.npy'
    # train_label_path = 'data/train_label_' + str(fold) + '.npy'

    train_data = np.load(train_data_path)
    train_label = np.load(train_label_path)
    test_data = np.load(test_data_path)
    test_label = np.load(test_label_path)

    # print(np.shape(train_data))
    # print(np.shape(train_label))
    # print(np.shape(test_data))
    # print(np.shape(test_label))
    # print("##############################")

    new_train_data, new_train_label = data_augment_train(train_data, train_label)
    new_test_data, new_test_label = data_augment_test(test_data, test_label)

    print(np.shape(new_train_data))
    print(np.shape(new_train_label))
    print(np.shape(new_test_data))
    print(np.shape(new_test_label))
    print("##############################")

    time_train_data = []
    time_train_graph = []
    time_test_data = []
    time_test_graph = []

    for i in range(len(new_train_data)):
        temp_train = split(new_train_data[i], split_stride_value, split_window_value)
        time_train_data.append(temp_train[0])
        time_train_graph.append(temp_train[1])

    for j in range(len(new_test_data)):
        temp_test_graph = []
        temp_test_data = []
        for k in range(len(new_test_data[0])):
            temp_test = split(new_test_data[j][k], split_stride_value, split_window_value)
            temp_test_data.append(temp_test[0])
            temp_test_graph.append(temp_test[1])
        time_test_graph.append(temp_test_graph)
        time_test_data.append(temp_test_data)

    time_train_data = np.array(time_train_data)
    time_train_graph = np.array(time_train_graph)
    time_test_data = np.array(time_test_data)
    time_test_graph = np.array(time_test_graph)

    print(np.shape(time_train_data))
    print(np.shape(time_train_graph))
    print(np.shape(time_test_data))
    print(np.shape(time_test_graph))
    print("##############################")

    dataset_sampler = datasets(time_train_data, time_train_graph, new_train_label)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=64,
        shuffle=True,
        num_workers=0)
    dataset_sampler = datasets(time_test_data, time_test_graph, new_test_label)
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=64,
        shuffle=False,
        num_workers=0)
    return train_dataset_loader, test_dataset_loader

    # 生成A


def gen_graph(data):
    """

    :param data:   raw_data
    :return:
    """
    adj = np.corrcoef(data)
    where_are_nan = np.isnan(adj)  # 找出数据中为nan的
    where_are_inf = np.isinf(adj)  # 找出数据中为inf
    for i in range(0, 200):
        for j in range(0, 200):
            if where_are_nan[i][j]:
                adj[i][j] = 0
            if where_are_inf[i][j]:
                adj[i][j] = 0.8
    return adj


def split(data, split_stride_value, split_window_value):
    """
    生成10個圖
    :param data:    (22 * 1200)
    :return:
    """
    stride = split_stride_value  # 25
    window = split_window_value  # 150
    time_data = []
    time_graph = []
    size = data.shape[1]
    num_seq = int((size - window) / stride) + 1
    for i in range(num_seq):
        time_graph.append(gen_graph(data[:, i * stride: i * stride + window]))
        time_data.append(data[:, i * stride: i * stride + window])
    time_data = np.array(time_data)
    time_graph = np.array(time_graph)
    return time_data, time_graph


class datasets(Dataset):
    def __init__(self, x, adj, label):
        self.adj_all = adj
        self.labels = label
        self.x = x

    def __getitem__(self, idx):
        # h0 为每个节点的特征矩阵
        return_dic = {'adj': self.adj_all[idx],
                      'h0': self.x[idx],
                      'label': self.labels[idx]
                      }

        return return_dic

    def __len__(self):
        return len(self.labels)


def evaluate(dataset, model, name='Validation', max_num_examples=None, device='cpu'):
    model.eval()
    avg_loss = 0.0
    preds = []
    labels = []
    if name == 'Train':
        with torch.no_grad():
            for batch_idx, data in enumerate(dataset):
                adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
                h0 = Variable(data['h0'].to(torch.float32), requires_grad=False).to(device)
                label = Variable(data['label'].long()).to(device)
                labels.append(data['label'].long().numpy())
                ypred, _ = model(h0, adj)
                loss = F.cross_entropy(ypred, label, size_average=True)
                avg_loss += loss
                _, indices = torch.max(ypred, 1)
                preds.append(indices.cpu().data.numpy())

                if max_num_examples is not None:
                    if (batch_idx + 1) * 64 > max_num_examples:
                        break
        avg_loss /= batch_idx + 1

        labels = np.hstack(labels)
        preds = np.hstack(preds)

        _index, _matrix = call_indexes(labels, preds)
        result = {'acc': _index[0],
                  'pre': _index[1],
                  'sen/recall': _index[2],
                  'spe': _index[3],
                  'F1': _index[4],
                  'auc': _index[5],
                  'matrix': _matrix}

    else:
        with torch.no_grad():
            for batch_idx, data in enumerate(dataset):
                adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
                h0 = Variable(data['h0'].to(torch.float32), requires_grad=False).to(device)
                labels.append(data['label'].long().numpy())
                ypred_all = torch.zeros([adj.shape[0], 2]).cuda()
                for i in range(adj.shape[1]):
                    ypred, _ = model(h0[:, i, :, :, :], adj[:, i, :, :, :])
                    ypred_soft = torch.softmax(ypred, 1)
                    ypred_all += ypred_soft
                ypred_avg = ypred_all / adj.shape[1]
                _, indices = torch.max(ypred_avg, 1)
                preds.append(indices.cpu().data.numpy())
                if max_num_examples is not None:
                    if (batch_idx + 1) * 64 > max_num_examples:
                        break
        labels = np.hstack(labels)
        preds = np.hstack(preds)

        _index, _matrix = call_indexes(labels, preds)
        result = {'acc': _index[0],
                  'pre': _index[1],
                  'sen/recall': _index[2],
                  'spe': _index[3],
                  'F1': _index[4],
                  'auc': _index[5],
                  'matrix': _matrix}

    return result


def train(dataset, model,
          split_stride_value,
          split_window_value,
          epoch_v, test_dataset=None, fold=0, lr_v=0.005, device='cpu'):
    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_v, weight_decay=0.001)
    scheduler_1 = CosineAnnealingWarmRestarts(optimizer2, T_0=50, T_mult=2)
    iter = 0

    for epoch in range(epoch_v):
        avg_loss = 0.0
        avg_lstm = 0.0
        model.train()
        print("epoch =", epoch)
        for batch_idx, data in enumerate(dataset):
            model.zero_grad()
            adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
            h0 = Variable(data['h0'].float(), requires_grad=False).to(device)
            label = Variable(data['label'].long()).to(device)
            pred, reg_loss = model(h0, adj)
            loss = F.cross_entropy(pred, label, size_average=True)
            all_loss = loss + reg_loss
            avg_lstm += reg_loss.item()
            avg_loss += loss.item()
            all_loss.backward()


            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer2.step()
            iter += 1
        avg_loss /= batch_idx + 1
        avg_lstm /= batch_idx + 1

        print("avg_loss =", avg_loss)

        if (epoch + 1) % 10 == 0:
            test_result = evaluate(test_dataset, model, name='test', device=device)
            print("test:\n", test_result)
            print("##############################")

        scheduler_1.step()

        if (epoch + 1) % 10 == 0:
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer2.state_dict(),
                          "epoch": epoch}
            path_checkpoint = "model\\m_aug_{}_{}\\{}\\checkpoint_{}_epoch.pkl".format(split_stride_value,
                                                                                       split_window_value,
                                                                                       fold, epoch)
            # torch.save(checkpoint, path_checkpoint)

    return model


###########################################################################################
###########################################################################################
# 主函数


def main():
    # seed = 4
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    # torch.manual_seed(seed)
    seed_v = 4
    set_seed(seed=seed_v)

    # 导入数据
    print('finished')
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Device: ', device)
    result = []

    ##################################################
    choose_zhe = np.array([1, 2, 3, 4, 5])
    # choose_zhe = np.array([2])  # 0.01
    lr_value = 0.001
    zhe = 5
    split_stride_value = 25
    split_window_value = 150
    epoch_value = 1000
    ##################################################

    for i in range(0, zhe):
        print("zhe =", zhe)
        train_data_loader, test_data_loader = load_data(i + 1, split_stride_value, split_window_value)
        model = BrainNet()
        model.to(device)

        if (i + 1) in choose_zhe:
            model = train(train_data_loader, model,
                          split_stride_value,
                          split_window_value,
                          epoch_v=epoch_value,
                          test_dataset=test_data_loader,
                          fold=i + 1, lr_v=lr_value, device='cuda')

        else:
            path_checkpoint = "model\\m_aug_{}_{}\\{}\\checkpoint_{}_epoch.pkl".format(split_stride_value,
                                                                                   split_window_value,
                                                                                   i + 1, epoch_value - 1)
            check_point = torch.load(path_checkpoint)
            model.load_state_dict(check_point['model_state_dict'])

        test_result = evaluate(test_data_loader, model, name='Test', device='cuda')

        result.append(test_result)
        print(test_result)

    print("##################### results #####################")

    for z in range(0, zhe):
        print("zhe_{}:".format(z + 1))
        print(result[z])

    # cal result mean value
    ave_result, std_result = cal_result_ave(result, zhe)
    print("##################### average #####################")
    print("result ave value:")
    for key in ave_result.keys():
        print(str(key), ": {}±{}".format(ave_result[str(key)], std_result[str(key)]))


if __name__ == '__main__':
    main()
