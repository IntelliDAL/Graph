import torch
import torch.nn as nn
import torch.nn.init as init
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
import tqdm
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


class BrainNet(nn.Module):
    def __init__(self):
        super(BrainNet, self).__init__()
        """
        """
        self.LG_Networks = GCN(22, 256)
        self.LG_Networks1 = GCN(256, 256)
        self.lin = nn.Linear(256 * 22, 2)

    def forward(self, adj):
        h = self.LG_Networks(None, adj)
        h = self.LG_Networks1(h, adj)
        # h, _ = torch.max(h, 1)
        h = h.reshape([h.shape[0], -1])
        x = self.lin(h)
        return x


def load_data(fold):
    train_data_path = 'data/train_data_' + str(fold) + '.npy'
    test_data_path = 'data/test_data_' + str(fold) + '.npy'
    test_label_path = 'data/test_label_' + str(fold) + '.npy'
    train_label_path = 'data/train_label_' + str(fold) + '.npy'
    train_data = np.load(train_data_path)
    train_label = np.load(train_label_path)
    test_data = np.load(test_data_path)
    test_label = np.load(test_label_path)
    time_train_graph = []
    time_test_graph = []
    for i in range(len(train_data)):
        time_train_graph.append(gen_graph(train_data[i]))
    for j in range(len(test_data)):
        time_test_graph.append(gen_graph(test_data[j]))
    time_test_graph = np.array(time_test_graph)
    time_train_graph = np.array(time_train_graph)
    dataset_sampler = datasets(time_train_graph, train_label)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=64,
        shuffle=True,
        num_workers=0)
    dataset_sampler = datasets(time_test_graph, test_label)
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
    for i in range(0, 22):
        for j in range(0, 22):
            if where_are_nan[i][j]:
                adj[i][j] = 0
            if where_are_inf[i][j]:
                adj[i][j] = 0.8
    return adj


class datasets(Dataset):
    def __init__(self, adj, label):
        self.adj_all = adj
        self.labels = label

    def __getitem__(self, idx):
        return_dic = {'adj': self.adj_all[idx],
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
                label = Variable(data['label'].long()).to(device)
                labels.append(data['label'].long().numpy())
                ypred = model(adj)
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
        auc = metrics.roc_auc_score(labels, preds)
        result = {'prec': metrics.precision_score(labels, preds),
                  'recall': metrics.recall_score(labels, preds),
                  'acc': metrics.accuracy_score(labels, preds),
                  'F1': metrics.f1_score(labels, preds),
                  'auc': auc,
                  'matrix': confusion_matrix(labels, preds)}
    else:
        with torch.no_grad():
            for batch_idx, data in enumerate(dataset):
                adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
                h0 = Variable(data['h0'].to(torch.float32), requires_grad=False).to(device)
                labels.append(data['label'].long().numpy())
                ypred_all = torch.zeros([adj.shape[0], 2]).cuda()
                for i in range(adj.shape[1]):
                    ypred = model(h0[:, i, :, :, :], adj[:, i, :, :, :])
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


def train(dataset, model, val_dataset=None, test_dataset=None, fold=0,
          device='cpu'):
    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005,
                                  weight_decay=0.001)
    scheduler_1 = CosineAnnealingWarmRestarts(optimizer2, T_0=50, T_mult=2)
    for name in model.state_dict():
        print(name)
    iter = 0
    best_val_acc = 0
    best_epoch = 0

    for epoch in range(150):
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
            label = Variable(data['label'].long()).to(device)
            pred = model(adj)
            loss = F.cross_entropy(pred, label, size_average=True)
            loss.backward()
            time3 = time.time()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer2.step()
            iter += 1
            avg_loss += loss
        avg_loss /= batch_idx + 1
        print(avg_loss)
        eval_time = time.time()
        # if (epoch + 1) % 50 == 0:
        #     test_result = evaluate(test_dataset, model, name='test', device=device)
        #     print("test:       ", test_result)
        scheduler_1.step()
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
    for i in range(5):
        train_data_loader, test_data_loader = load_data(i + 1)
        model = BrainNet()
        model.to(device)
        print('model:', model)
        model = train(train_data_loader, model, val_dataset=None, test_dataset=test_data_loader, fold=i + 1,
                      device='cuda')
        test_result = evaluate(test_data_loader, model, name='Train', device='cuda')
        print(test_result)
        result.append(test_result)
    print(result)


if __name__ == '__main__':
    main()
