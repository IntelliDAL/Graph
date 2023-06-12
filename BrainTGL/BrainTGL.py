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
import os
import torch
import random

warnings.filterwarnings("ignore")


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
        self.num_reduce = num_reduce  # 5
        self.num_node = num_node  # 116
        # self.kernel_p = nn.ParameterList(
        #     [nn.Parameter(torch.FloatTensor(self.num_node, self.num_reduce)) for i in range(22)])
        # self.kernel_n = nn.ParameterList(
        #     [nn.Parameter(torch.FloatTensor(self.num_node, self.num_reduce)) for i in range(22)])
        self.kernel = nn.Parameter(torch.FloatTensor(self.num_node, self.num_reduce))  # F 原图点数量，超图点数量 调整self.num_reduce
        self.in_features = in_features  # 输入特征  5
        self.hidden_features = hidden_features  # 卷積之後的特徵
        self.out_features = out_features  # lstm 輸出的特徵
        # 這一套是 lstm的參數
        self.skip = skip
        self.tcn = tcn_Networks(528, 528)  # 150-528 100-336 75-240 50-128
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
            adj_t = self.dim_reduce(adj_t, self.num_reduce, 0.2, 0.3, 0.1,
                                    self.kernel)  # 维度下降后的连通矩阵
            A_pool.append(adj_t.unsqueeze(1))
            feature_t = X[:, t, :, :]  # 原始信號
            feature_t = self.tcn(feature_t)  # 用tcn學習原始信號
            feature_t = torch.matmul(self.kernel.t(), feature_t)  # 轉職    batch num_reduce in_dim
            features.append(feature_t.unsqueeze(1))
            combined = torch.cat([feature_t, h_t], 2)
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
    def __init__(self):
        super(BrainNet, self).__init__()
        """
        """
        # (in_features, hidden_features, out_features, num_node, num_reduce, skip)
        # self.LG_Networks = Layer(128, 100, 256, 200, 5, 2)
        # 150-528 100-336 75-240 50-128
        # 超图参数 调大一些看影响
        self.super_nodes = 12
        super_nodes = self.super_nodes
        self.LG_Networks = Layer(528, 100, 256, 200, super_nodes, 2)  # 150-528 100-336 75-240 50-128
        self.lin = nn.Linear(256 * super_nodes, 2)

    def forward(self, x, adj):
        super_nodes = self.super_nodes
        state = (torch.zeros([adj.shape[0], super_nodes, 256]).cuda(),
                 torch.zeros([adj.shape[0], super_nodes, 256]).cuda())
        state_skip = (
            torch.zeros([adj.shape[0] * 2, super_nodes, 256]).cuda(),
            torch.zeros([adj.shape[0] * 2, super_nodes, 256]).cuda())
        h, loss = self.LG_Networks(x, adj, state, state_skip)
        h = h.reshape(h.shape[0], -1)

        x = self.lin(h)
        # print('x', x.shape)
        return x, loss


def load_data(fold, split_stride_value, split_window_value):
    dir_path = "CC200_ALL_Data_longer_and_equal_176\\"

    train_data_path = dir_path + 'train_data_' + str(fold) + '.npy'
    train_label_path = dir_path + 'train_label_' + str(fold) + '.npy'
    test_data_path = dir_path + 'test_data_' + str(fold) + '.npy'
    test_label_path = dir_path + 'test_label_' + str(fold) + '.npy'

    train_data = np.load(train_data_path)
    train_label = np.load(train_label_path)
    test_data = np.load(test_data_path)
    test_label = np.load(test_label_path)

    # 测试bug用
    # print(np.shape(train_data), np.shape(train_label), np.shape(test_data), np.shape(test_label))
    # os.system("pause")

    # train_data = train_data[0:10, :, :]
    # train_label = train_label[0:10]
    # test_data = test_data[0:10, :, :]
    # test_label = test_label[0:10]

    new_train_data, new_train_label = train_data, train_label
    new_test_data, new_test_label = test_data, test_label

    time_train_data = []
    time_test_data = []
    time_train_graph = []
    time_test_graph = []
    for i in range(len(new_train_data)):
        temp_train = split(new_train_data[i], split_stride_value, split_window_value)
        time_train_data.append(temp_train[0])
        time_train_graph.append(temp_train[1])

        if i % 50 == 0:
            print(i, len(new_train_data))

    for j in range(len(new_test_data)):
        temp_test = split(new_test_data[j], split_stride_value, split_window_value)
        time_test_data.append(temp_test[0])
        time_test_graph.append(temp_test[1])

        if j % 50 == 0:
            print(j, len(new_test_data))

    time_train_data = np.array(time_train_data)
    time_test_data = np.array(time_test_data)
    time_test_graph = np.array(time_test_graph)
    time_train_graph = np.array(time_train_graph)
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
    def __init__(self, x, adj, label):
        self.adj_all = adj
        self.labels = label
        self.x = x

    def __getitem__(self, idx):
        return_dic = {'adj': self.adj_all[idx],
                      'h0': self.x[idx],
                      'label': self.labels[idx]
                      }

        return return_dic

    def __len__(self):
        return len(self.labels)


# 训练技巧
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif score >= self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


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
        from sklearn.metrics import confusion_matrix
        # auc = metrics.roc_auc_score(labels, preds, average='macro', sample_weight=None)
        result = {'prec': metrics.precision_score(labels, preds, average='macro'),
                  'recall': metrics.recall_score(labels, preds, average='macro'),
                  'acc': metrics.accuracy_score(labels, preds),
                  'F1': metrics.f1_score(labels, preds, average="macro"),
                  #   'auc': auc,
                  'matrix': confusion_matrix(labels, preds)}
    else:
        with torch.no_grad():
            for batch_idx, data in enumerate(dataset):
                adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
                h0 = Variable(data['h0'].to(torch.float32), requires_grad=False).to(device)
                labels.append(data['label'].long().numpy())
                ypred_all = torch.zeros([adj.shape[0], 2]).cuda()

                # print("H0 shape: " + str(np.shape(h0)) + ", A shape: " + str(np.shape(adj)))

                ypred, _ = model(h0, adj)
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
        from sklearn.metrics import confusion_matrix
        auc = metrics.roc_auc_score(labels, preds, average='macro', sample_weight=None)
        result = {'prec': metrics.precision_score(labels, preds, average='macro'),
                  'recall': metrics.recall_score(labels, preds, average='macro'),
                  'acc': metrics.accuracy_score(labels, preds),
                  'F1': metrics.f1_score(labels, preds, average="macro"),
                  'auc': auc,
                  'matrix': confusion_matrix(labels, preds)}
    return result


def train(dataset, model, lr_value, epoch_value, val_dataset=None, test_dataset=None, fold=0,
          device='cpu'):
    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_value,  # 原：0.005
                                  weight_decay=0.001)  # 0.001
    scheduler_1 = CosineAnnealingWarmRestarts(optimizer2, T_0=50, T_mult=2)
    # for name in model.state_dict():
    #     print(name)
    iter = 0
    best_val_acc = 0
    best_epoch = 0
    #####################################################################################################
    for epoch in range(epoch_value):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        print(epoch)
        for batch_idx, data in enumerate(dataset):
            # for k, v in model.named_parameters():
            #     v.requires_grad = True
            time1 = time.time()
            model.zero_grad()
            adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
            h0 = Variable(data['h0'].float(), requires_grad=False).to(device)
            label = Variable(data['label'].long()).to(device)
            pred, reg_loss = model(h0, adj)
            loss = F.cross_entropy(pred, label, size_average=True)
            all_loss = loss + reg_loss
            all_loss.backward()
            time3 = time.time()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer2.step()
            iter += 1
            avg_loss += loss
        avg_loss /= batch_idx + 1
        print(avg_loss)
        eval_time = time.time()
        if (epoch + 1) % 20 == 0:
            train_result = evaluate(dataset, model, name='Train', device=device)
            test_result = evaluate(test_dataset, model, name='test', device=device)
            print("train:       ", train_result)
            print("test:       ", test_result)
        scheduler_1.step()
        # if (epoch + 1) % 5 == 0:
        if epoch + 1 == 2000:
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer2.state_dict(),
                          "epoch": epoch}

            # path_checkpoint = "./model/第一版模型文件/{}/checkpoint_{}_epoch.pkl".format(fold, epoch)
            path_checkpoint = "D:\\IDAL Projects\\IDALData\\ASD_try_20211223\\model\\longer_and_equal_176"
            path_checkpoint = path_checkpoint + "\\{}\\checkpoint_{}_epoch.pkl".format(fold, epoch)

            torch.save(checkpoint, path_checkpoint)
    return model


###########################################################################################
###########################################################################################
# 主函数


def main():
    # 设置种子
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
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
    lr_value = 0.005
    epoch_value = 100
    split_stride_value = 10  # 步长 10 15 20 25 30
    # 修改 split_window_value 时记得修改以下参数：
    # self.tcn = tcn_Networks(528, XXX)
    # self.LG_Networks = Layer(XXX, 100, 256, 200, 5, 2)
    # XXX 处值对应：150-528 100-336 75-240 50-128
    split_window_value = 150  # 窗口 50 100 150
    nodes = 12  # 此是为了记录，与上面代码无关，记得到上面代码处修改self.super_nodes = XX
    print(zhe, lr_value, epoch_value, split_stride_value, split_window_value, nodes)
    ################################## 调参 ##################################

    for i in range(0, 5):
        model = BrainNet()
        model.to(device)
        # print('model:', model)

        train_data_loader, test_data_loader = load_data(i + 1,
                                                        split_stride_value,
                                                        split_window_value)
        model = train(train_data_loader, model,
                      5e-5, 2000,
                      val_dataset=None,
                      test_dataset=test_data_loader,
                      fold=i + 1,
                      device='cuda')

        test_result = evaluate(test_data_loader, model, name='Test', device='cuda')
        print(test_result)
        result.append(test_result)

    for i in range(0, 5):
        model = BrainNet()
        model.to(device)
        # print('model:', model)

        train_data_loader, test_data_loader = load_data(i + 1,
                                                        split_stride_value,
                                                        split_window_value)
        model = train(train_data_loader, model,
                      5e-6, 5000,
                      val_dataset=None,
                      test_dataset=test_data_loader,
                      fold=i + 1,
                      device='cuda')

        test_result = evaluate(test_data_loader, model, name='Test', device='cuda')
        print(test_result)
        result.append(test_result)

    print(result)


if __name__ == '__main__':
    main()
