import sys

from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import warnings
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.autograd import Variable
import time
import math
from torch.nn import init
import os
import torch
import random
from torch.backends import cudnn
from sklearn.metrics import confusion_matrix

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


def load_data_noSTGCN(path1, path2):
    temp_data = np.load(path1)
    label = np.load(path2)
    all_data = []
    for i in range(0, len(temp_data)):
        all_data.append(temp_data[i, 0, :, :, 0].T)

    all_data = np.array(all_data)
    print(all_data.shape)

    return all_data, label


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
        # print(x.shape)

        # x = self.fc(x) # 128 所以直接注释了
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
            # print("XW, " + str(np.shape(x)) + ", " + str(np.shape(self.kernel)))
            XW = torch.tensordot(x, self.kernel, [[-1], [0]])  # batch *  num_node * feature_dim
            AXW = torch.matmul(adj, XW)  # batch *  num_node * feature_dim

        I_cAXW = AXW
        y_relu = torch.nn.functional.relu(I_cAXW)
        temp = torch.mean(input=y_relu, dim=-2, keepdim=True) + 1e-6
        col_mean = temp.repeat([1, feature_dim, 1])
        y_norm = torch.divide(y_relu, col_mean)  # 正则化后的值
        output = torch.nn.functional.softplus(y_norm)
        return output


class Layer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_node, num_reduce, skip):
        super(Layer, self).__init__()
        self.num_reduce = num_reduce
        self.num_node = num_node

        self.kernel = nn.Parameter(torch.FloatTensor(self.num_node, self.num_reduce))  # F 原图点数量，超图点数量 调整self.num_reduce
        self.in_features = in_features  # 输入特征  5
        self.hidden_features = hidden_features  # 卷積之後的特徵
        self.out_features = out_features  # lstm 輸出的特徵
        # 這一套是 lstm的參數
        self.skip = skip
        self.tcn = tcn_Networks(528, in_features)  # 150-528 100-336 75-240 50-128
        self.Wi = nn.Parameter(torch.FloatTensor(hidden_features, out_features))
        self.Wf = nn.Parameter(torch.FloatTensor(hidden_features, out_features))
        self.Wo = nn.Parameter(torch.FloatTensor(hidden_features, out_features))
        self.Wc = nn.Parameter(torch.FloatTensor(hidden_features, out_features))
        self.bi = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.bf = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.bo = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.bc = nn.Parameter(torch.FloatTensor(out_features, 1))
        # 這是一套skip的參數
        self.Wi_skip1 = nn.Parameter(torch.FloatTensor(hidden_features, out_features))
        self.Wf_skip1 = nn.Parameter(torch.FloatTensor(hidden_features, out_features))
        self.Wo_skip1 = nn.Parameter(torch.FloatTensor(hidden_features, out_features))
        self.Wc_skip1 = nn.Parameter(torch.FloatTensor(hidden_features, out_features))
        self.bi_skip1 = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.bf_skip1 = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.bo_skip1 = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.bc_skip1 = nn.Parameter(torch.FloatTensor(out_features, 1))
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

    def reset_weigths_nyaia(self):
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

        (h, c) = state
        h_t = h  # (N, V, H)
        c_t = c  # (N, V, H)
        (h_skip, c_skip) = state_skip
        h_t_skip = h_skip
        c_t_skip = c_skip
        seq_size = A.shape[1]  # 子图个数
        batch = X.shape[0]
        # lstm 运算
        features = []
        A_pool = []
        # lstm 运算
        # print(A.shape) # 64 0
        for t in range(seq_size):
            adj_t = A[:, t, :, :]  # pcc值組成的圖

            adj_t = self.dim_reduce(adj_t, self.num_reduce, 0.2, 0.3, 0.1, self.kernel)

            A_pool.append(adj_t.unsqueeze(1))

            feature_t = X[:, t, :, :]  # 原始信號
            feature_t = self.tcn(feature_t)  # 用tcn學習原始信號
            feature_t = torch.matmul(self.kernel.t(), feature_t)  # 轉職 .t()=转置    batch num_reduce in_dim
            features.append(feature_t.unsqueeze(1))
            combined = torch.cat([feature_t, h_t], 2)
            # print("combined =", combined.shape)
            # print("adj_t =", adj_t.shape)

            x = self.gcn_networks(combined, adj_t)

            it = torch.sigmoid((torch.matmul(x, self.Wi) + self.bi.t()))  # (batch_size,N,hidden_size)
            ft = torch.sigmoid((torch.matmul(x, self.Wf) + self.bf.t()))
            ot = torch.sigmoid((torch.matmul(x, self.Wo) + self.bo.t()))
            ut = torch.relu((torch.matmul(x, self.Wc) + self.bc.t()))
            c_t = ft * c_t + it * ut  # shape=
            h_t = ot * torch.tanh(c_t)  # shape = (batch_size, num_node, hidden_dim)

        # skip-lstm 运算    x_after_gcn [64, 4, num_node, dim]
        # print(features) # []
        features = torch.cat(features, 1).contiguous()
        num_p = int(seq_size / self.skip)
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

        # skip-lstm 运算    x_after_gcn [64, 4, num_node, dim]
        h_t_skip = h_t_skip.view(batch, -1, h_t_skip.shape[1], h_t_skip.shape[2])
        h_t_skip = h_t_skip.permute([0, 2, 1, 3]).contiguous()
        h_t_skip = h_t_skip.view(h_t_skip.shape[0], h_t_skip.shape[1], -1)
        h_t_all = torch.cat([h_t, h_t_skip], 2)
        h_t_all = self.fc(h_t_all)
        # print('h_t_all', h_t_all.shape)
        h_t_all = self.dropout(h_t_all)
        loss = torch.sum(torch.tensor(self.losses))
        self.losses.clear()
        return h_t_all, loss


class BrainNet(nn.Module):
    def __init__(self, tcn_hyper=528):
        super(BrainNet, self).__init__()
        """
        """
        # (in_features, hidden_features, out_features, num_node, num_reduce, skip)
        # self.LG_Networks = Layer(128, 100, 256, 200, 5, 2)
        # 150-528 100-336 75-240 50-128
        # 超图参数 调大一些看影响
        # self.super_nodes = 12
        self.super_nodes = 12
        super_nodes = self.super_nodes
        self.LG_Networks = Layer(tcn_hyper, 100, 256, 200, super_nodes, 2)  # 150-528 100-336 75-240 50-128
        self.lin = nn.Linear(256 * super_nodes, 2)

    def forward(self, x, adj):
        super_nodes = self.super_nodes
        state = (torch.zeros([adj.shape[0], super_nodes, 256]).cuda(),
                 torch.zeros([adj.shape[0], super_nodes, 256]).cuda())
        state_skip = (
            torch.zeros([adj.shape[0] * 2, super_nodes, 256]).cuda(),
            torch.zeros([adj.shape[0] * 2, super_nodes, 256]).cuda())
        # 这里之前 数值基本都一样
        h, loss = self.LG_Networks(x, adj, state, state_skip)

        h = h.reshape(h.shape[0], -1)

        x = self.lin(h)
        # print('x', x.shape)
        return x, loss


def load_data(fold):
    dir_path = "..\\ASD_data_176_250\\"

    train_data_path = dir_path + 'train_data_' + str(fold) + '.npy'
    train_label_path = dir_path + 'train_label_' + str(fold) + '.npy'
    test_data_path = dir_path + 'test_data_' + str(fold) + '.npy'
    test_label_path = dir_path + 'test_label_' + str(fold) + '.npy'

    train_data, train_label = load_data_noSTGCN(train_data_path, train_label_path)
    test_data, test_label = load_data_noSTGCN(test_data_path, test_label_path)

    # 观察数据，测试bug用
    print(np.shape(train_data), np.shape(train_label), np.shape(test_data), np.shape(test_label))
    # sys.exit()
    # 缩小数据，测试bug用
    # train_data = train_data[0:20, :, :]
    # train_label = train_label[0:20]
    # test_data = test_data[0:20, :, :]
    # test_label = test_label[0:20]

    new_test_data, new_test_label = test_data, test_label

    time_test_data1 = []
    time_test_graph1 = []
    time_test_data2 = []
    time_test_graph2 = []
    time_test_data3 = []
    time_test_graph3 = []

    for j in range(len(new_test_data)):
        temp_test1 = split(new_test_data[j], 25, 50)
        time_test_data1.append(temp_test1[0])
        time_test_graph1.append(temp_test1[1])
        temp_test2 = split(new_test_data[j], 25, 100)
        time_test_data2.append(temp_test2[0])
        time_test_graph2.append(temp_test2[1])
        temp_test3 = split(new_test_data[j], 25, 150)
        time_test_data3.append(temp_test3[0])
        time_test_graph3.append(temp_test3[1])

        if j % 50 == 0:
            print(j, len(new_test_data))

    time_test_data1 = np.array(time_test_data1)
    time_test_graph1 = np.array(time_test_graph1)
    time_test_data2 = np.array(time_test_data2)
    time_test_graph2 = np.array(time_test_graph2)
    time_test_data3 = np.array(time_test_data3)
    time_test_graph3 = np.array(time_test_graph3)

    dataset_sampler = datasets(time_test_data1, time_test_graph1,
                               time_test_data2, time_test_graph2,
                               time_test_data3, time_test_graph3,
                               new_test_label)

    test_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=64,
        shuffle=False,
        num_workers=0)

    return test_dataset_loader


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


def split(data, stride_value, window_value):
    """
    生成10個圖
    :param data:    (22 * 1200)
    :return:
    """
    stride = stride_value  # 25 # 15      步长 10 15 20 25 30
    window = window_value  # 150 # 150    window_size c
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
    def __init__(self, x1, adj1, x2, adj2, x3, adj3, label):
        self.x1 = x1
        self.adj1 = adj1
        self.x2 = x2
        self.adj2 = adj2
        self.x3 = x3
        self.adj3 = adj3
        self.labels = label

    def __getitem__(self, idx):
        return_dic = {
            'adj1': self.adj1[idx],
            'adj2': self.adj2[idx],
            'adj3': self.adj3[idx],
            'h1': self.x1[idx],
            'h2': self.x2[idx],
            'h3': self.x3[idx],
            'label': self.labels[idx]
        }

        return return_dic

    def __len__(self):
        return len(self.labels)


def evaluate(dataset, model_1, model_2, model_3, device='cpu'):
    model_1.eval()
    model_2.eval()
    model_3.eval()
    preds = []
    labels = []

    with torch.no_grad():

        for batch_idx, data in enumerate(dataset):
            adj1 = Variable(data['adj1'].to(torch.float32), requires_grad=False).to(device)
            h01 = Variable(data['h1'].to(torch.float32), requires_grad=False).to(device)
            adj2 = Variable(data['adj2'].to(torch.float32), requires_grad=False).to(device)
            h02 = Variable(data['h2'].to(torch.float32), requires_grad=False).to(device)
            adj3 = Variable(data['adj3'].to(torch.float32), requires_grad=False).to(device)
            h03 = Variable(data['h3'].to(torch.float32), requires_grad=False).to(device)
            labels.append(data['label'].long().numpy())

            sub_idx = 0
            ypred_all = torch.zeros([adj1.shape[0], 2]).cuda()

            for i in range(adj1.shape[0]):
                h0_temp, adj_temp = h01[i, :, :, :].unsqueeze(0), adj1[i, :, :, :].unsqueeze(0)
                ypred0, _ = model_1(h0_temp, adj_temp)
                h0_temp, adj_temp = h02[i, :, :, :].unsqueeze(0), adj2[i, :, :, :].unsqueeze(0)
                ypred1, _ = model_2(h0_temp, adj_temp)
                h0_temp, adj_temp = h03[i, :, :, :].unsqueeze(0), adj3[i, :, :, :].unsqueeze(0)
                ypred2, _ = model_3(h0_temp, adj_temp)

                ypred_soft0 = torch.softmax(ypred0, 1)
                ypred_soft1 = torch.softmax(ypred1, 1)
                ypred_soft2 = torch.softmax(ypred2, 1)

                ypred_soft = (ypred_soft0*0.1 + ypred_soft1*0.45 + ypred_soft2*0.45)
                # ypred_soft = (ypred_soft1 + ypred_soft2)
                ypred_soft = torch.softmax(ypred_soft, 1)
                ypred_all[sub_idx] += ypred_soft[0]
                sub_idx = sub_idx + 1

            # ypred_avg = ypred_all / 3
            # print("ypred_all", ypred_all)
            # print("ypred_avg", ypred_avg)

            _, indices = torch.max(ypred_all, 1)
            preds.append(indices.cpu().data.numpy())

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
        std_temp = []
        for key in ave_result.keys():
            ave_result[str(key)] += result[z][str(key)]
            std_result[str(key)].append(result[z][str(key)])

    for key in ave_result.keys():
        ave_result[str(key)] /= zhe
        std_temp = std_result[str(key)]
        std_result[str(key)] = 0
        std_result[str(key)] = np.std(std_temp, ddof=1)

    return ave_result, std_result


###########################################################################################
###########################################################################################
# 主函数


def main():
    # 设置种子
    seed_v = 1
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

    ################################## 调参 ##################################
    zhe = 5
    lr_value = 0.01
    epoch_value = 150  # 100
    split_stride_value = 25  # 步长 10 15 20 25 30
    # 修改 split_window_value 时记得修改以下参数：
    # self.tcn = tcn_Networks(528, XXX)
    # self.LG_Networks = Layer(XXX, 100, 256, 9, 5, 2)
    # XXX 处值对应：150-528 100-336 75-240 50-128
    # 改保存文件名
    split_window_value = 100  # 窗口 50 100 150
    nodes = 12  # self.super_nodes = 12
    print(zhe, lr_value, epoch_value, split_stride_value, split_window_value, nodes)
    ################################## 调参 ##################################

    for i in range(0, zhe):
        print("zhe={}".format(i + 1))
        set_seed(seed=seed_v)
        model_1 = BrainNet(tcn_hyper=128)
        model_2 = BrainNet(tcn_hyper=336)
        model_3 = BrainNet(tcn_hyper=528)

        test_dataset_loader = load_data(i + 1)

        path_checkpoint = "model\\m_{}_{}\\{}\\checkpoint_{}_epoch.pkl".format(25, 50, i + 1, epoch_value - 1)
        check_point = torch.load(path_checkpoint)
        model_1.load_state_dict(check_point['model_state_dict'])
        model_1.to(device)
        path_checkpoint = "model\\m_{}_{}\\{}\\checkpoint_{}_epoch.pkl".format(25, 100, i + 1, epoch_value - 1)
        check_point = torch.load(path_checkpoint)
        model_2.load_state_dict(check_point['model_state_dict'])
        model_2.to(device)
        path_checkpoint = "model\\m_{}_{}\\{}\\checkpoint_{}_epoch.pkl".format(25, 150, i + 1, epoch_value - 1)
        check_point = torch.load(path_checkpoint)
        model_3.load_state_dict(check_point['model_state_dict'])
        model_3.to(device)

        test_result = evaluate(test_dataset_loader, model_1, model_2, model_3, device='cuda')
        result.append(test_result)
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

# ssh -p 1267 user@202.199.7.80
# watch -n 0.2 nvidia-smi
