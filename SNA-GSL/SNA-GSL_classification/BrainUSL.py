import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import numpy as np
import random
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


def corrcoef(data, replace_nan_as=0):
    """
    data: batch * node_num * feature_len
    return: batch * node_num * node_num
    """
    x = data - torch.mean(data, dim=-1, keepdim=True)
    cov = x.matmul(x.transpose(-2, -1)) / (x.shape[1] - 1)
    d = cov[:, range(cov.shape[1]), range(cov.shape[1])]
    stddev = torch.sqrt(d)
    cov /= stddev[:, :, None]
    cov /= stddev[:, None, :]
    cov = torch.where(torch.isnan(cov), torch.full_like(cov, replace_nan_as), cov)
    return cov


def confusion(g_turth, predictions):
    # print(g_turth)
    tn, fp, fn, tp = confusion_matrix(g_turth, predictions).ravel()
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    sensitivity = (tp) / (tp + fn)
    specificty = (tn) / (tn + fp)
    # print(g_turth)
    auc = metrics.roc_auc_score(g_turth, predictions, average='macro', sample_weight=None)
    return accuracy, sensitivity, specificty, auc


def cos_similar(p, q):
    sim_matrix = p.matmul(q.transpose(-2, -1))
    a = torch.norm(p, p=2, dim=-1)
    b = torch.norm(q, p=2, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    sim_matrix = torch.where(torch.isnan(sim_matrix), torch.full_like(sim_matrix, 0), sim_matrix)
    return sim_matrix


def sameLoss(x, x_aug):
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    loss = 2 - cosine_similarity(x, x_aug, dim=-1).mean() - cosine_similarity(x_abs, x_aug_abs, dim=-1).mean()
    return loss


#######################对比策略函数################
def unsupervisedGroupContrast(x, x_aug, label, T=0.4):  # MDD's T is setting to 0.6, and bd's T is 0.4
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    sim_matrix1 = torch.einsum('ik,jk->ij', x, x) / torch.einsum('i,j->ij', x_abs, x_abs)
    sim_matrix1 = torch.exp(sim_matrix1 / T)
    sim_matrix2 = torch.einsum('ik,jk->ij', x_aug, x_aug) / torch.einsum('i,j->ij', x_aug_abs, x_aug_abs)
    sim_matrix2 = torch.exp(sim_matrix2 / T)
    sim_matrix = torch.concat([sim_matrix1, sim_matrix2], dim=0)
    pos_label = label.tile(dims=[2, 1])
    neg_label = (1 - pos_label)
    pos = sim_matrix[pos_label.bool()].mean() + 1
    neg = sim_matrix[neg_label.bool()].sum() / (neg_label.sum() - neg_label.shape[0]) + 1
    loss = - torch.log(pos / (neg + pos))
    return loss


def false_unsupervisedGroupContrast(x, pos_label, T=0.5):
    batch = x.shape[0]
    x_abs = x.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x, x) / torch.einsum('i,j->ij', x_abs, x_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_matrix = sim_matrix[pos_label.bool()]
    neg_label = (1 - pos_label.int())
    neg_label[range(batch), range(batch)] = 0
    neg_matrix = sim_matrix[neg_label.bool()]
    pos = pos_matrix.mean() + 1
    neg = neg_matrix.mean() + 1
    loss = - torch.log(pos / neg)
    return loss


##################graph_kernel——开始########################
def rbfKernel(bag1, bag2,gamma):
    n, m = bag1.shape[0], bag2.shape[0]
    bag1_i_norm = torch.sum(bag1 ** 2, dim=1)
    bag2_i_norm = torch.sum(bag2 ** 2, dim=1)
    # bag1_i_norm = torch.cat([torch.sum(bag1 ** 2, dim=1).unsqueeze(0), torch.sum(x, dim=1).unsqueeze(0)],dim=0)
    # bag2_i_norm = torch.cat([torch.sum(bag2 ** 2, dim=1).unsqueeze(0), torch.sum(x, dim=1).unsqueeze(0)],dim=0)
    distMat = torch.as_tensor(
        torch.tile(bag1_i_norm, [m, 1]).T + torch.tile(bag2_i_norm, [n, 1]) - 2 * bag1.matmul(bag2.T))

    kMat = torch.exp(-gamma * distMat)
    return kMat


def kernelEntry(bag1, bag2, weightMatrix1, weightMatrix2, gamma):
    n, m = bag1.shape[0], bag2.shape[0]
    activeEdgesCount1 = weightMatrix1.sum(dim=1)
    activeEdgesCount2 = weightMatrix2.sum(dim=1)
    activeEdgesCoef1 = 1. / (activeEdgesCount1 + 1e-3)
    activeEdgesCoef2 = 1. / (activeEdgesCount2 + 1e-3)

    k = rbfKernel(bag1, bag2,gamma=gamma)
    k = torch.tile(activeEdgesCoef1, dims=[m, 1]).T * torch.tile(activeEdgesCoef2, dims=[n, 1]) * k
    k = torch.sum(k) / torch.sqrt(torch.sum(activeEdgesCoef1)) / torch.sqrt(torch.sum(activeEdgesCoef2))

    return k


def distMatrix(bag, method="gaussian", gamma=1):
    n = bag.shape[0]
    bag_i_norm = torch.sum(bag ** 2, dim=1)
    distMat = torch.tile(bag_i_norm, [n, 1]) + torch.tile(bag_i_norm, [n, 1]).T - 2 * bag.matmul(bag.T)

    if method == "gaussian":
        distMat = 1 - torch.exp(-gamma * distMat)
    return distMat


def graph_simliarty(bag,method="gaussian", gamma=1):
    n = bag.shape[0]
    graph_kernel = torch.zeros(size=(n, n))
    w = torch.zeros_like(bag)
    for wi in range(n):
        w[wi] = distMatrix(bag[wi])
    for i in range(n):
        for j in range(i + 1, n):
            mat = kernelEntry(bag[i], bag[j], w[i], w[j], 0.5)
            graph_kernel[i][j] = mat
            graph_kernel[j][i] = mat
    graph_kernel = graph_kernel
    return graph_kernel


##################graph_kernel——结束########################


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class SigCNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SigCNN, self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=10)
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=15)
        self.bn = nn.BatchNorm1d(116, affine=True)
        self.bn2 = nn.BatchNorm1d(116, affine=True)

        self.alpha = 1
        self.beta = 1
        self.w = nn.Parameter(torch.FloatTensor(116, 116), requires_grad=True)
        torch.nn.init.uniform_(self.w, a=0, b=1)

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx

    def get_w_data(self):
        w = self.grad_w()
        return w.data

    def grad_w(self):
        w = self._normalize(self.w)
        w = F.relu(self.bn2(w))
        w = (w + w.T) / 2
        return w

    def forward(self, x):
        x = self.bn(x)
        x = self.conv1(x)
        x = self.conv2(x).reshape(x.shape[0], self.in_channel, -1)
        x = cos_similar(x, x)

        w = self._normalize(self.w)
        w = F.relu(self.bn2(w))
        w = (w + w.T) / 2
        l1 = torch.norm(w, p=1, dim=1).mean()
        x = w * x
        return x, l1


class E2E(nn.Module):
    def __init__(self, in_channel, out_channel, input_shape, **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.d = input_shape[0]
        self.conv1xd = nn.Conv2d(in_channel, out_channel, (self.d, 1))
        self.convdx1 = nn.Conv2d(in_channel, out_channel, (1, self.d))

    def forward(self, A):
        A = A.view(-1, self.in_channel, 116, 116)
        a = self.conv1xd(A)
        b = self.convdx1(A)
        concat1 = torch.cat([a] * self.d, 2)
        concat2 = torch.cat([b] * self.d, 3)
        return concat1 + concat2


class Model(nn.Module):
    def __init__(self, dropout=0.5, num_class=1, out_channel=116 * 2):
        super().__init__()

        self.e2e = nn.Sequential(
            E2E(1, 8, (116, 116)),
            nn.LeakyReLU(0.33),
            E2E(8, 8, (116, 116)),
            nn.LeakyReLU(0.33),
        )

        self.e2n2g = nn.Sequential(
            nn.Conv2d(8, 48, (1, 116)),
            nn.LeakyReLU(0.33),
            nn.Conv2d(48, 116, (116, 1)),
            nn.LeakyReLU(0.33),
        )
        self.e2n = nn.Sequential(
            nn.Conv2d(8, 48, (1, 116)),
            nn.LeakyReLU(0.33)
        )
        self.n2g = nn.Sequential(
            nn.Conv2d(48, 116, (116, 1)),
            nn.LeakyReLU(0.33)
        )
        self.linear = nn.Sequential(
            nn.Linear(116, 64),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
            nn.Linear(64, 32),
            nn.Linear(32, 10),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
            nn.Linear(10, num_class)
        )
        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.sigcnn = SigCNN(in_channel=116, out_channel=out_channel)

    def net(self, x):
        x = self.e2e(x)
        x = self.e2n(x)
        return x

    def Line(self, x):
        x = self.n2g(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        F.softmax(x, dim=-1)
        return x

    # 调用graph_kernel##########，通过设定阈值phi决定哪些相似哪些不相似，通常是根据一个batch下正样本的比例设置的。
    # y设置为0.5则取一个batch下最相似的一半为正样本，不需要阈值，只针对当前batch数据，但是我没使用
    def get_label_matrix_from_sim(self, adj, y=None, phi=0.002):

        sim_matrix = graph_simliarty(adj)
        if y is not None:
            rate = y
            node_num = adj.shape[0] ** 2
            # value = sim_matrix.sort(descending=True)[0]
            value = sim_matrix.reshape(1, -1).squeeze().sort(descending=True)[0]
            index = int(rate * node_num)
            phi = value[index]
            if phi == 0:
                phi = 0.001
            sim_matrix = (sim_matrix >= phi).int()
        else:
            sim_matrix = (sim_matrix >= phi).int()
        # print('比例：', sim_matrix.sum() / (sim_matrix.shape[0] ** 2))
        return sim_matrix

    def gen_phi(self, adj, phi):
        return adj * (torch.abs(adj) > phi)

    def pre_train(self, x):
        x1 = corrcoef(x)
        # mdd's phi is setting to 0.0017, bd was default
        # 调用kernel，生成01正负样本标签矩阵
        group_label = self.get_label_matrix_from_sim(adj=x1, phi=0.0017)
        x, loss = self.sigcnn(x)
        x = self.net(x)
        x1 = self.net(x1)
        x = x.view(x.shape[0], -1)
        x1 = x1.view(x1.shape[0], -1)
        same = sameLoss(x, x1)
        # 通过原始图、增强图、标签矩阵，进行对比学习
        super_loss = false_unsupervisedGroupContrast(x, group_label) + false_unsupervisedGroupContrast(x1, group_label)
        return [super_loss, same, 0.3 * loss]

    def base_net(self, x):  # S
        x = self.net(x)
        x = self.Line(x)
        return x

    def forward(self, x, y=None, pre=False):
        if pre:
            return self.pre_train(x)
        else:
            return self.base_net(x)

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# print(device)
# # Set_seed
# setup_seed(10)
# pre_epo = 10
# epochs = 50 + pre_epo  # 116 671
# batch_size = 32  # 64 0.660
# pre_batch = 32
# dropout = 0.5
# lr = 0.001
# decay = 0.01
# result = []
# acc_final = 0
# n_split = 5
# result_final = []
# result_auc = []
# result_sen = []
# result_spec = []
# source_data = "MDD"
#
# print('loading data...')
# X = np.load(f'./Dataset/new/{source_data}_HC_sig.npy')
# X = torch.from_numpy(X).permute([0, 2, 1]).numpy()
#
# Y = np.load(f'./Dataset/new/{source_data}_HC_label.npy')
# # 1 indicates BD/MDD ,0 is HC
# print('---------------------')
# print('X', X.shape)  # N M M
# print('Y', Y.shape)
# print('---------------------')
#
# for ind in range(5):
#     setup_seed(ind)
#     kf = KFold(n_splits=n_split, shuffle=True)
#     kfold_index = 0
#     for trainval_index, test_index in kf.split(X, Y):
#         kfold_index += 1
#         print('kfold_index:', kfold_index)
#         X_trainval, X_test = X[trainval_index], X[test_index]
#         Y_trainval, Y_test = Y[trainval_index], Y[test_index]
#         for train_index, val_index in kf.split(X_trainval, Y_trainval):
#             X_train, X_val = X_trainval[:], X_trainval[:]
#             Y_train, Y_val = Y_trainval[:], Y_trainval[:]
#         print('X_train', X_train.shape)
#         print('X_val', X_val.shape)
#         print('X_test', X_test.shape)
#         print('Y_train', Y_train.shape)
#         print('Y_val', Y_val.shape)
#         print('Y_test', Y_test.shape)
#
#         # model
#         model = Model(dropout=dropout, num_class=2)
#         model.to(device)
#         optimizer1 = optim.SGD(model.parameters(), lr=lr / 5, weight_decay=decay, momentum=0.9,
#                                nesterov=True)  # for pre-train
#         optimizer2 = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)  # for train
#         loss_fn = nn.CrossEntropyLoss()
#         time = 0
#         best_val = 0
#         best_loss = 1000
#         acc_all = 0
#         # train
#         for epoch in range(1, epochs + 1):
#             # torch.cuda.empty_cache()
#
#             model.train()
#             idx_batch = np.random.permutation(int(X_train.shape[0]))
#             num_batch = X_train.shape[0] // int(batch_size)
#             pre_num_batch = X_train.shape[0] // int(pre_batch)
#
#             loss_train = 0
#
#             if epoch <= pre_epo:
#                 # pre-trained stage
#                 for bn in range(pre_num_batch):
#                     if bn == pre_num_batch - 1:
#                         batch = idx_batch[bn * int(pre_batch):]
#                     else:
#                         batch = idx_batch[bn * int(pre_batch): (bn + 1) * int(pre_batch)]
#                     train_data_batch = X_train[batch]
#                     train_label_batch = Y_train[batch]
#                     train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
#                     train_label_batch_dev = torch.from_numpy(train_label_batch).long().to(device)
#                     semi_loss = model(train_data_batch_dev, pre=True)
#                     loss = semi_loss
#                     semi_loss = sum(semi_loss)
#
#                     optimizer1.zero_grad()
#                     semi_loss.backward()
#                     optimizer1.step()
#
#                 print(
#                     f"epoc:{epoch} semi loss: {loss} sparse rate :{(model.sigcnn.get_w_data() != 0).int().sum() / (116 * 116)}")
#
#             # mini_net is the pre-trained GGM for generated graphs, and not involved in fine tuning
#             mini_net = nn.Sequential(
#                 model.sigcnn
#             )
#
#             # training stage
#             if epoch > pre_epo:
#                 for bn in range(num_batch):
#                     if bn == num_batch - 1:
#                         batch = idx_batch[bn * int(batch_size):]
#                     else:
#                         batch = idx_batch[bn * int(batch_size): (bn + 1) * int(batch_size)]
#                     train_data_batch = X_train[batch]
#                     train_label_batch = Y_train[batch]
#                     train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
#                     train_label_batch_dev = torch.from_numpy(train_label_batch).long().to(device)
#
#                     model.eval()
#                     train_data_batch_dev, _ = mini_net(train_data_batch_dev)
#                     model.train()
#
#                     outputs = model(train_data_batch_dev)
#                     optimizer2.zero_grad()
#                     loss = loss_fn(outputs, train_label_batch_dev)
#                     loss_train += loss
#                     loss.backward()
#                     optimizer2.step()
#
#             loss_train /= num_batch
#             if epoch % 10 == 0 and epoch > pre_epo:
#                 print('epoch:', epoch, 'train loss:', loss_train)
#                 if torch.isnan(loss):
#                     print(f"model out {outputs[0]}, label {train_label_batch_dev[0]}")
#                 # print loss
#                 # for name, parms in model.named_parameters():
#                 #    print('-->name:', name, ' -->grad_value:', parms.grad)
#
#             # val
#             if epoch > pre_epo:
#                 with torch.no_grad():
#                     if loss_train < best_loss:
#                         best_loss = loss_train
#                         model.eval()
#                         test_data_batch_dev = torch.from_numpy(X_test).float().to(device)
#                         test_data_batch_dev, _ = mini_net(test_data_batch_dev)
#
#                         outputs = model(test_data_batch_dev)
#                         _, indices = torch.max(outputs, dim=1)
#                         preds = indices.cpu()
#                         # print(preds )
#                         y_arr = np.array(Y_test, dtype=np.int32)
#                         # print(y_arr)
#                         y_true, y_pred = [], []
#                         # y_true.extend(y_arr.tolist())
#                         y_pred.extend(preds.tolist())
#                         acc = metrics.accuracy_score(preds, Y_test)
#                         acc, sen, spec, auc = confusion(y_arr, y_pred)
#                         print('Test acc', acc)
#                         acc_all += acc
#
#         temp = acc
#         final_auc = auc
#         result.append(temp)
#         acc_final += temp
#         result_final.append(temp)
#         result_auc.append(auc)
#         result_spec.append(spec)
#         result_sen.append(sen)
# ACC = acc_final / n_split
#
# print(result_final)
# print(f'acc : {np.array(result).mean()}')
# print(f'auc : {np.array(result_auc).mean()}')
# print(f'sen : {np.array(result_sen).mean()}')
# print(f'spec : {np.array(result_spec).mean()}')
# print(len(result))
