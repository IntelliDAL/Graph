import pickle
import warnings
from torch.utils.data import Dataset
import torch.nn as nn
import Sequence_Time_LSTM_1
import torch.nn.functional as F
import math
import scipy
from lstm import LSTM_1
from torch.nn import init
import dill
import random
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
import torch.optim as optim
import numpy as np
import torch
from sklearn.metrics import r2_score
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
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
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
        eye = torch.eye(feature_dim) # 生成对角矩阵 feature_dim * feature_dim
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
        # print(I_cAXW)
        return I_cAXW
class E2E(nn.Module):

    def __init__(self, in_channel, out_channel, input_shape,hidden_dim, remain_node_num,**kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_dim = hidden_dim
        self.d1 = input_shape[0]
        self.d2 = input_shape[1]
        self.conv1xd = nn.Conv2d(in_channel, out_channel, (self.d1, 1))
        self.convdx1 = nn.Conv2d(in_channel, out_channel, (1, self.d2))
        self.nodes = remain_node_num

    def forward(self, A):
        #         print(A.shape)
        A = A.view(-1, self.in_channel, self.nodes, self.nodes+self.hidden_dim)
        # A = A.view(-1, self.in_channel, self.nodes, self.nodes)
        a = self.conv1xd(A)
        b = self.convdx1(A)
        concat1 = torch.cat([a] * self.d1, 2)
        concat2 = torch.cat([b] * self.d2, 3)

        # A = torch.mean(concat1+concat2, 1)
        # print('e2e', (concat1+concat2).shape)
        return concat1 + concat2
class Model(nn.Module):
    def __init__(self, dropout=0.5, num_class=1,remain_node_num = 70,
                 lstm_hidden =64,eton_outputchanel = 48,linear_dim1 = 64,linear_dim2=10,
                 ):
        super().__init__()
        self.feature_dim = remain_node_num
        # self.hiddend_dim = hiddend_dim
        self.lstm_hidden = lstm_hidden
        # self.gcn_hidden_dim = gcn_hidden_dim
        self.linear_dim1 = linear_dim1
        self.linear_dim2 = linear_dim2
        self.eton_outputchanel = eton_outputchanel
        # self.gcn_networks1 = GCN(self.feature_dim + self.hiddend_dim, self.gcn_hidden_dim)
        # self.Wi = nn.Parameter(torch.FloatTensor(hiddend_dim,self.lstm_hidden))
        # self.Wf = nn.Parameter(torch.FloatTensor(hiddend_dim,self.lstm_hidden))
        # self.Wo = nn.Parameter(torch.FloatTensor(hiddend_dim,self.lstm_hidden))
        # self.Wc = nn.Parameter(torch.FloatTensor(hiddend_dim,self.lstm_hidden))
        # self.bi = nn.Parameter(torch.FloatTensor(self.lstm_hidden, 1))
        # self.bf = nn.Parameter(torch.FloatTensor(self.lstm_hidden, 1))
        # self.bo = nn.Parameter(torch.FloatTensor(self.lstm_hidden, 1))
        # self.bc = nn.Parameter(torch.FloatTensor(self.lstm_hidden, 1))
        self.cell = LSTM_1(self.eton_outputchanel, self.lstm_hidden)
        # self.e2e = nn.Sequential(
        #     E2E(1, 8, (self.feature_dim, self.feature_dim),self.lstm_hidden,remain_node_num),
        #     nn.LeakyReLU(0.33),
        #     E2E(8, 8, (self.feature_dim, self.feature_dim),self.lstm_hidden,remain_node_num),  # 0.642
        #     nn.LeakyReLU(0.33),
        # )
        self.e2e = nn.Sequential(
            E2E(1, 8, (self.feature_dim, self.feature_dim+self.lstm_hidden),self.lstm_hidden,remain_node_num),
            nn.LeakyReLU(0.33),
            E2E(8, 8, (self.feature_dim, self.feature_dim+self.lstm_hidden),self.lstm_hidden,remain_node_num),  # 0.642
            nn.LeakyReLU(0.33),
        )

        self.e2n = nn.Sequential(
            # nn.Conv2d(8, eton_outputchanel, (1, self.feature_dim)),
            nn.Conv2d(8, eton_outputchanel, (1, self.feature_dim+self.lstm_hidden)),  # 32 652
            nn.LeakyReLU(0.33),
        )

        self.n2g = nn.Sequential(
            nn.Conv2d(eton_outputchanel, self.feature_dim, (self.feature_dim, 1)),
            nn.LeakyReLU(0.33),
        )
        self.dropout = nn.Dropout(p=0.5)
        self.reset_weigths()
        self.lstm = Sequence_Time_LSTM_1
        self.linear1 = nn.Sequential(
            nn.Linear(self.lstm_hidden * self.feature_dim, linear_dim1),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
            nn.Linear(linear_dim1, linear_dim2),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
        )
        self.linear2 = nn.Linear(linear_dim2, num_class)

        self.linear3 = nn.Linear(linear_dim2, 9)

        self.GC = DenseGCNConv(eton_outputchanel, eton_outputchanel)

        for layer in self.linear1:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        nn.init.kaiming_normal_(self.linear3.weight)
        nn.init.zeros_(self.linear3.bias)

        # self.GCN = GCN()
    def reset_weigths(self):
        """reset weights
        """
        stdv = 1.0 / math.sqrt(256)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)
    def MaskAutoEncoder(self, e2n, A, masked_x,Mask_train_tensor):  # masked_x 32 50 48
        e2n_encoder = torch.squeeze(e2n)  ###150*150的X 经过brainnetcnn得到150*48的embedding，
        masked_x = masked_x.permute(0, 2, 1)  # 32 48 50
        e2n_encoder = torch.cat((e2n_encoder, masked_x), -1)  # 补上了masked  masked的特征是
        e2n_encoder_T = e2n_encoder.permute(0, 2, 1)  # batch 200 48
        e2n_encoder_T = self.GC(e2n_encoder_T, A)
        e2n_encoder = e2n_encoder_T.permute(0, 2, 1)
        Z = torch.matmul(e2n_encoder_T, e2n_encoder)  # batch 200 200
        # Z = nn.sigmoid(Z) # 正相关 负相关分离
        # 哈达姆乘
        Z = Z * Mask_train_tensor
        return Z
    def forward(self, x, A, masked_x,time_point,Mask_train_tensor):
        state = (torch.zeros([A.shape[0], self.feature_dim,self.lstm_hidden]),
                 torch.zeros([A.shape[0],  self.feature_dim,self.lstm_hidden]))
        (hidden , c) = state
        state_shared = (torch.zeros([A.shape[0], self.feature_dim, self.lstm_hidden]),
                 torch.zeros([A.shape[0], self.feature_dim, self.lstm_hidden]))
        (hidden_shared, c_shared) = state_shared
        # state = (torch.zeros([A.shape[0], self.lstm_hidden]),
        #          torch.zeros([A.shape[0], self.lstm_hidden]))
        # (hidden, c) = state
        # h = hidden  # (N, V, H)
        # state_shared = (torch.zeros([A.shape[0], self.lstm_hidden]),
        #                 torch.zeros([A.shape[0], self.lstm_hidden]))
        # (hidden_shared, c_shared) = state_shared
        seq_size = x.shape[1]  # 子图个数
        pred_clas_sequence = []
        pred_reg_sequence = []
        encoder_sequence = []
        features_learned = []
        for t in range(seq_size):
            # if time_point=="48":
            #     print("time point ==48")
            adj_t = A[:, t, :, :]  #
            feature_t = x[:, t, :, :]  #
            combined = torch.cat([feature_t, hidden], 2)
            # combined = feature_t
            masked_x_t =masked_x[:, t, :, :]
            # x = self.gcn_networks1(combined, adj_t)
            # x = self.gcn_networks3(x, adj_t)
            combined = self.e2e(combined)
            combined = self.e2n(combined) #combined 32*48*70*1
            graph_feature = self.n2g(combined)
            graph_feature = graph_feature.squeeze()
            features_learned.append(graph_feature)
            z = self.MaskAutoEncoder(combined, adj_t, masked_x_t,Mask_train_tensor)
            encoder_sequence.append(z)
            # combined = self.n2g(combined)
            combined = torch.squeeze(combined)
            combined = combined.permute(0, 2, 1)
            if time_point=="24":
                hidden_shared,c_shared, hidden, c = Sequence_Time_LSTM_1.predict_multiletask(combined,hidden_shared,c_shared,hidden,c,self.cell,0)
            if time_point=="36":
                hidden_shared,c_shared,hidden, c = Sequence_Time_LSTM_1.predict_multiletask(combined,hidden_shared,c_shared,hidden,c,self.cell,1)
            if time_point=="48":
                hidden_shared,c_shared, hidden, c = Sequence_Time_LSTM_1.predict_multiletask(combined,hidden_shared,c_shared,hidden,c,self.cell,2)
            h_t = self.dropout(hidden)
            h_t = h_t.reshape(h_t.shape[0], -1)
            pred_clas = self.linear2(self.linear1(h_t))
            pred_reg = self.linear3(self.linear1(h_t))
            pred_clas= F.softmax(pred_clas, dim=-1)
            pred_clas_sequence.append(pred_clas)
            pred_reg_sequence.append(pred_reg)
        return pred_clas_sequence, encoder_sequence,pred_reg_sequence,features_learned

    def get_A(self, x):
        x = self.e2e(x)
        #         print(x.shape)
        x = torch.mean(x, dim=1)
        return x
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def process_data(X,Y,all_node_num,remain_node_num):

    for ind in range(1):
        # setup_seed(ind)
        # Masked
        nums = np.ones(all_node_num)  # 制作mask模板
        nums[:all_node_num - remain_node_num] = 0  # 根据设置的nodes number 决定多少是mask 即mask比例
        # np.random.seed(1)
        np.random.shuffle(nums)  # 200 75%1 25%0 shuffle打散
        Mask = nums.reshape(nums.shape[0], 1) * nums  # 200 200
        Masked_X = X * Mask  # 将部分转换为 0（masked）
        X0 = X
        Masked_X_rest = X - Masked_X
        J = remain_node_num  # J 拷贝出一份
        for i in range(0, J):
            ind = i
            if nums[ind] == 0:
                for j in range(J, all_node_num):
                    if nums[j] == 1:
                        Masked_X[:,:, [ind, j], :] = Masked_X[:,:,  [j, ind], :]
                        Masked_X[:,:, :, [ind, j]] = Masked_X[:, :, :, [j, ind]]
                        Masked_X_rest[:,:, [ind, j], :] = Masked_X_rest[:,:,  [j, ind], :]
                        Masked_X_rest[:, :,:,  [ind, j]] = Masked_X_rest[:, :, :, [j, ind]]
                        X0[:, :, [ind, j], :] = X0[:, :, [j, ind], :]
                        X0[:,:,  :, [ind, j]] = X0[:, :, :, [j, ind]]
                        J = j + 1
                        break

        X_0 = Masked_X  # X_0 是残缺的 只有前面unmasked的有值
        Masked_X = Masked_X[:,:,  :remain_node_num, :remain_node_num]
        unMasked_X = Masked_X[:,:,  remain_node_num:, remain_node_num:]  # 这个是被mask的那些pcc

        X = Masked_X
        X_unmasked = Masked_X_rest
    return X,X_unmasked
from infonce_loss import info_nce
def constract_loss(features_learned):
    features_learned = torch.stack(features_learned)
    features_learned = features_learned.squeeze()
    features_learned = features_learned.transpose(-1,-2)
    loss_constract1 = info_nce(features_learned[0],features_learned[1])
    loss_constract2 = info_nce(features_learned[1],features_learned[2])
    loss_constract_ave = loss_constract1+loss_constract2
    return loss_constract_ave
def rec_loss(pred,true):

    true = true.permute([1,0, 2, 3])  #
    pred = torch.stack(pred)

    num_subject = pred.shape[1]
    loss_rec = nn.MSELoss()
    loss = loss_rec(pred,true)
    # loss_constract1 = infonce_manual(pred[0],pred[1],0.1)
    # loss_constract2 = infonce_manual(pred[1],pred[2],0.1)
    return loss/num_subject
def entro_test(pred, true):
    pred = torch.stack(pred)
    true = np.array(true)
    nb_subjects = true.shape[0]
    pred = pred[-1]
    true = true[-1][:,2]
    mask = ~torch.isnan(true)
    mae = torch.nn.functional.l1_loss(
        pred[mask], true[mask], reduction='sum') / nb_subjects
def ccc(x,y):
    ''' Concordance Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc
import pickle
import pandas as pd
def evaluate_test(outputs,preds_reg, true,time_point):

    with open('mean_std.pickle', 'rb') as file:
        mean_std_dict = pickle.load(file)
    mean = mean_std_dict["mean"]
    std = mean_std_dict["std"]

    labels_true_class= true[:,3]
    labels_true_reg = true[:,-9:]

    nb_subjects = true.shape[0]
    preds_class_prob = outputs[-1]
    preds_reg = preds_reg[-1]
    _, indices = torch.max(preds_class_prob, dim=1)
    preds_class_prob = preds_class_prob.detach().numpy()
    preds_reg = preds_reg.detach().numpy()
    preds_class = indices.cpu()

    result_rmse = {
        # "BCA": balanced_accuracy_score(labels_true_class, preds_class),
        #       " mauc": roc_auc_score(labels_true_class, preds_class_prob, average='macro', multi_class='ovo'),
        #       'matrix': confusion_matrix(labels_true_class, preds_class),
        "CDRSB": np.nanmean(
            np.abs(labels_true_reg[:, 0] - (preds_reg[:, 0]))),
        "ADAS11": np.nanmean(
            np.abs((labels_true_reg[:, 1] ) - (preds_reg[:, 1]) )),
        "MMSE": np.nanmean(
            np.abs(labels_true_reg[:, 2]  - (preds_reg[:, 2]))),
        "RAVLT_immediate": np.nanmean(
            np.abs((labels_true_reg[:, 3] ) - (preds_reg[:, 3]) )),
        "RAVLT_learning": np.nanmean(
            np.abs((labels_true_reg[:, 4]) - (preds_reg[:, 4]) )),
        "ADAS13": np.nanmean(
            np.abs((labels_true_reg[:, 5] ) - (preds_reg[:, 5]) )),
        "RAVLT_forgetting": np.nanmean(
            np.abs((labels_true_reg[:, 6] ) - (preds_reg[:, 6]) )),
        "RAVLT_perc_forgetting": np.nanmean(
            np.abs((labels_true_reg[:, 7] ) - (preds_reg[:, 7]) )),
        "MOCA": np.nanmean(
            np.abs((labels_true_reg[:, 8] ) - (preds_reg[:, 8]) ))}
    mask_0 = ~np.isnan(labels_true_reg[:, 0])
    mask_1 = ~np.isnan(labels_true_reg[:, 1])
    mask_2 = ~np.isnan(labels_true_reg[:, 2])
    mask_3 = ~np.isnan(labels_true_reg[:, 3])
    mask_4 = ~np.isnan(labels_true_reg[:, 4])
    mask_5 = ~np.isnan(labels_true_reg[:, 5])
    mask_6 = ~np.isnan(labels_true_reg[:, 6])
    mask_7 = ~np.isnan(labels_true_reg[:, 7])
    mask_8 = ~np.isnan(labels_true_reg[:, 8])
    result_ccc = {
        "CDRSB": ccc(labels_true_reg[:, 0][mask_0], (preds_reg[:, 0][mask_0])),

        "ADAS11": ccc(labels_true_reg[:, 1][mask_1], (preds_reg[:, 1][mask_1])),
        "MMSE": ccc(labels_true_reg[:, 2][mask_2], (preds_reg[:, 2][mask_2])),
        "RAVLT_immediate": ccc(labels_true_reg[:, 3][mask_3], (preds_reg[:, 3][mask_3])),
        "RAVLT_learning": ccc(labels_true_reg[:, 4][mask_4], (preds_reg[:, 4][mask_4])),
        "ADAS13": ccc(labels_true_reg[:, 5][mask_5], (preds_reg[:, 5][mask_5])),
        "RAVLT_forgetting": ccc(labels_true_reg[:, 6][mask_6], (preds_reg[:, 6][mask_6])),
        "RAVLT_perc_forgetting": ccc(labels_true_reg[:, 7][mask_7], (preds_reg[:, 7][mask_7])),
        "MOCA": ccc(labels_true_reg[:, 8][mask_8], (preds_reg[:, 8][mask_8]))}

    result_r2 = {
        "CDRSB": r2_score(labels_true_reg[:, 0][mask_0], (preds_reg[:, 0][mask_0])),

        "ADAS11": r2_score(labels_true_reg[:, 1][mask_1], (preds_reg[:, 1][mask_1])),
        "MMSE": r2_score(labels_true_reg[:, 2][mask_2], (preds_reg[:, 2][mask_2])),
        "RAVLT_immediate": r2_score(labels_true_reg[:, 3][mask_3], (preds_reg[:, 3][mask_3])),
        "RAVLT_learning": r2_score(labels_true_reg[:, 4][mask_4], (preds_reg[:, 4][mask_4])),
        "ADAS13": r2_score(labels_true_reg[:, 5][mask_5], (preds_reg[:, 5][mask_5])),
        "RAVLT_forgetting": r2_score(labels_true_reg[:, 6][mask_6], (preds_reg[:, 6][mask_6])),
        "RAVLT_perc_forgetting": r2_score(labels_true_reg[:, 7][mask_7], (preds_reg[:, 7][mask_7])),
        "MOCA": r2_score(labels_true_reg[:, 8][mask_8], (preds_reg[:, 8][mask_8]))}

    result_cc = {
       "CDRSB": scipy.stats.pearsonr(labels_true_reg[:, 0][mask_0], (preds_reg[:, 0][mask_0]))[0],

        "ADAS11": scipy.stats.pearsonr(labels_true_reg[:, 1][mask_1], (preds_reg[:, 1][mask_1]))[0],
        "MMSE": scipy.stats.pearsonr(labels_true_reg[:, 2][mask_2], (preds_reg[:, 2][mask_2]))[0],
        "RAVLT_immediate": scipy.stats.pearsonr(labels_true_reg[:, 3][mask_3], (preds_reg[:, 3][mask_3]))[0],
        "RAVLT_learning": scipy.stats.pearsonr(labels_true_reg[:, 4][mask_4], (preds_reg[:, 4][mask_4]))[0],
        "ADAS13": scipy.stats.pearsonr(labels_true_reg[:,5][mask_5], (preds_reg[:, 5][mask_5]))[0],
       "RAVLT_forgetting": scipy.stats.pearsonr(labels_true_reg[:, 6][mask_6], (preds_reg[:, 6][mask_6]))[0],
        "RAVLT_perc_forgetting": scipy.stats.pearsonr(labels_true_reg[:, 7][mask_7], (preds_reg[:, 7][mask_7]))[0],
        "MOCA": scipy.stats.pearsonr( labels_true_reg[:, 8][mask_8], (preds_reg[:, 8][mask_8]))[0]}

    return result_rmse,result_cc,result_ccc
def mae_loss_reg(pred, true):
    """
    Calculate mean absolute error (MAE)
    Args:
        pred: predicted values, [nb_timpoints, nb_subjects, nb_features]
        true: true values, [nb_timpoints, nb_subjects, nb_features]
        mask: values to evaluate, [nb_timpoints, nb_subjects, nb_features]
    Returns:
        MAE loss
    """

    pred = torch.stack(pred)
    true = np.array(true)
    nb_subjects = true.shape[0]
    true_multi = []
    true_multi.append(true[:,4:13])
    true_multi.append(true[:,13:22])
    true_multi.append(true[:, 22:31])
    true_multi2 = np.stack(true_multi)
    true_multi2 = torch.from_numpy(true_multi2)
    ## 只考虑ADAS11 ADAS13 MMSE ["CDRSB", "ADAS11", "MMSE", "RAVLT_immediate","RAVLT_learning", "ADAS13", "RAVLT_forgetting","RAVLT_perc_forgetting", "MOCA"]

    loss_temporal_smooth = np.nansum(np.nansum(np.nansum(abs(pred[1:,:,:].detach().numpy()  - pred[:-1,:,:].detach().numpy() ))))
    mask = ~torch.isnan(true_multi2)
    return torch.nn.functional.l1_loss(
        pred[mask], true_multi2[mask], reduction='sum') / nb_subjects
    return torch.nn.functional.l1_loss(
        pred[mask], true_multi2[mask], reduction='sum') / nb_subjects #+ loss_temporal_smooth/nb_subjects
def ent_loss(pred, true):
    """
    Calculate cross-entropy loss
    Args:
        pred: predicted probability distribution,
              [nb_timpoints, nb_subjects, nb_classes]
        true: true class, [nb_timpoints, nb_subjects, 1]
        mask: timepoints to evaluate, [nb_timpoints, nb_subjects, 1]
    Returns:
        cross-entropy loss
    """
    nb_subjects = true[0].shape[0]
    true_06 = true[:,1]
    true_12 = true[:,2]
    true_pre  = true[:,3]
    loss = torch.nn.functional.cross_entropy(pred[0], true_06.long(), reduction='sum')+\
           torch.nn.functional.cross_entropy(pred[1], true_12.long(), reduction='sum')+\
           torch.nn.functional.cross_entropy(pred[2], true_pre.long(), reduction='sum')
    return loss / nb_subjects
def deal_dataset(kfold_index,kf,X_masked,X_unmasked,X,Y,trainval_index,test_index,all_node_num,remain_node_num):
    kfold_index += 1
    print('kfold_index:', kfold_index)
    X_trainval, X_test = X_masked[trainval_index], X_masked[test_index]
    Y_trainval, Y_test = Y[trainval_index], Y[test_index]

    X_trainval_masked_rest = X_unmasked[trainval_index]
    X_trainval_0, X_test_0 = X[trainval_index], X[test_index]
    for train_index, val_index in kf.split(X_trainval, Y_trainval):
        # 取消验证集
        X_train, X_val = X_trainval[:], X_trainval[:]
        Y_train, Y_val = Y_trainval[:], Y_trainval[:]

        X_train_masked_rest = X_trainval_masked_rest[:]
        X_test_masked_rest = X_trainval_masked_rest[:]

        X_train_0 = X_trainval_0[:]  # 完整的A
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
    X_avg = X_avg / X_train_number
    X_train_0[X_train_0 < 0.2] = 0
    X_train_0[X_train_0 >= 0.2] = 1
    X_test_0[X_test_0 < 0.2] = 0
    X_test_0[X_test_0 >= 0.2] = 1
    X_avg[abs(X_avg) < 0.15] = 0
    X_avg[abs(X_avg) >= 0.15] = 1

    X_avg = X_avg.reshape(1, X_avg.shape[0], X_avg.shape[1], X_avg.shape[2])  #
    X_avg = np.repeat(X_avg, X_train.shape[0], 0)
    X_train_0 = X_avg
    X_masked = np.zeros([X_train.shape[0], X_train.shape[1], all_node_num - remain_node_num, 48])
    X_masked_test = np.zeros([X_test.shape[0], X_test.shape[1], all_node_num - remain_node_num, 48])
    for i in range(X_masked.shape[0]):
        for j in range(X_masked.shape[1]):
            for k in range(X_masked.shape[2]):
                X_masked[i][j][k] = np.random.normal(loc=0.0, scale=1.0, size=48)
    for i in range(X_masked_test.shape[0]):
        for j in range(X_masked_test.shape[1]):
            for k in range(X_masked_test.shape[2]):
                X_masked_test[i][j][k] = np.random.normal(loc=0.0, scale=1.0, size=48)
    return X_train,X_train_0,X_masked,X_train_masked_rest,Y_train,X_test,X_test_0,X_masked_test,Y_test
class data_strue:
    def __init__(self,data,time_point):
        self.data = data
        self.time_point = time_point
    def item(self):
        return self.data
    def time(self):
        return self.time_point
def to_strue(X,timepoint):
    X_strue = []
    for i in range(0,X.shape[0]):
        data = data_strue(X[i],timepoint)
        X_strue.append(data)
    X_strue = np.array(X_strue)
    return X_strue
def fusion_batch(num_batch,idx_batch,train_data_batch_all,train_label_batch_all,
                 train_data_batch_A_all,train_data_batch_rest_all,train_data_batch_maskedX_all,
                 X_train,X_train_0,X_masked,X_train_masked_rest,Y_train,batch_size):
    num_batch = num_batch+1
    for bn in range(num_batch):
        if bn == num_batch - 1:
            batch = idx_batch[bn * int(batch_size):]
        else:
            batch = idx_batch[bn * int(batch_size): (bn + 1) * int(batch_size)]
        train_data_batch_24 = X_train[batch]
        train_label_batch_24 = Y_train[batch]
        train_data_batch_A_24 = X_train_0[batch]
        if  not X_train_masked_rest is None:
            train_data_batch_rest_24 = X_train_masked_rest[batch]
        else:
            train_data_batch_rest_24 = None
        train_data_batch_maskedX_24 = X_masked[batch]
        train_data_batch_all.append(train_data_batch_24)
        train_label_batch_all.append(train_label_batch_24)
        train_data_batch_A_all.append(train_data_batch_A_24)
        train_data_batch_rest_all.append(train_data_batch_rest_24)
        train_data_batch_maskedX_all.append(train_data_batch_maskedX_24)
    return train_data_batch_all,train_label_batch_all,train_data_batch_A_all,\
           train_data_batch_rest_all,train_data_batch_maskedX_all
import csv
# def find_index(Y_array):
#     index = -1
#     ###002_s_4654   1	14	27	19	1	18	4	100	24
#     ###002_s_4557   1	12	28	31	3	18	5
#     ### oo2_s_4473  2	6	29	42	6	9	2
#     ###002_s_4521   3	20	24	27	2	27	6
#     ###019_S_4293   4	21	25	21	2	29	1
#     ###130_S_4817   2	 13	27	25	4	18	6
#     ## 002_S_4262   0	 4	29	54	8	7
#     ## 006_S_4346  2.5	13	30	39	7	20
#     ## 013_S_4395 1	5	29	29	4	10
#     ### 018_S_4889 1.5	9	29	41	6	17
#     ### 006_S_4960 1.5	6	26	37	3	8
#     ### 018_S_4868 1	13	25	21	2	20
#     ### 002_S_4229 0.5	6	29	22	1	13
#     ### 002_S_4746 1	15	27	30	3	22
#     ## 019_S_4548 0.5	6	28	38	6	12
#     ##100_S_4556  0	8	28	19	4	10
#     ##053_S_4813  1	4	29	56	11	4
#     ##002_S_4799  0.5	6	30	27	5	10
#     ##018_S_4809  1	16	26	27	2	23
#     for  i in range(Y_array.shape[0]):
#         if (Y_array[i][6] == 1.000 and Y_array[i][7] ==14.000 \
#                 and Y_array[i][8] == 27.000 and Y_array[i][9] ==19.000)\
#             or  (Y_array[i][6] == 1.000 and Y_array[i][7] ==12.000 \
#                 and Y_array[i][8] == 28.000 and Y_array[i][9] ==31.000) \
#             or (Y_array[i][6] == 2.000 and Y_array[i][7] == 6.000 \
#                     and Y_array[i][8] == 29.000 and Y_array[i][9] == 42.000) \
#             or (Y_array[i][6] ==3.000 and Y_array[i][7] == 20.000 \
#                     and Y_array[i][8] == 24.000 and Y_array[i][9] == 27.000) \
#             or (Y_array[i][6] == 4.000 and Y_array[i][7] == 21.000 \
#                     and Y_array[i][8] == 25.000 and Y_array[i][9] == 21.000) \
#             or (Y_array[i][6] == 2.000 and Y_array[i][7] == 13.000 \
#                     and Y_array[i][8] == 27.000 and Y_array[i][9] == 25.000) \
#             or (Y_array[i][6] == 0.000 and Y_array[i][7] == 4.000 \
#                     and Y_array[i][8] == 29.000 and Y_array[i][9] == 54.000) \
#             or (Y_array[i][6] == 2.500 and Y_array[i][7] == 13.000 \
#                     and Y_array[i][8] == 30.000 and Y_array[i][9] == 39.000) \
#             or (Y_array[i][6] == 1.000 and Y_array[i][7] == 5.000 \
#                     and Y_array[i][8] == 29.000 and Y_array[i][9] == 29.000) \
#             or (Y_array[i][6] == 1.500 and Y_array[i][7] == 9.000 \
#                     and Y_array[i][8] == 29.000 and Y_array[i][9] == 41.000) \
#             or (Y_array[i][6] == 1.500 and Y_array[i][7] == 6.000 \
#                     and Y_array[i][8] == 26.000 and Y_array[i][9] == 37.000) \
#             or (Y_array[i][6] == 1.500 and Y_array[i][7] == 6.000 \
#                     and Y_array[i][8] == 26.000 and Y_array[i][9] == 37.000):
#
#             index =  i
#             break
#     return index
# def write_predict_results(output_reg_24,output_reg_36,output_reg_48,Y_24_array,Y_36_array,Y_48_array):
#     header = ["CDRSB", "ADAS11", "MMSE", "RAVLT_immediate", "RAVLT_learning",
#               "ADAS13", "RAVLT_forgetting", "RAVLT_perc_forgetting", "MOCA"]  # 数据列名
#
#     index_24 = find_index(Y_24_array)
#     index_36 = find_index(Y_36_array)
#     index_48 = find_index(Y_48_array)
#
#     if index_24!=-1:
#         result_24_6 = output_reg_24[0][index_24,:].detach().numpy()
#         result_24_12 = output_reg_24[1][index_24,:].detach().numpy()
#         result_24_24 = output_reg_24[2][index_24,:].detach().numpy()
#     if index_36!=-1:
#         result_36_6 = output_reg_36[0][index_36,:].detach().numpy()
#         result_36_12 = output_reg_36[1][index_36,:].detach().numpy()
#         result_36_36 = output_reg_36[2][index_36,:].detach().numpy()
#     if index_48!=-1:
#         result_48_6 = output_reg_48[0][index_48,:].detach().numpy()
#         result_48_12 = output_reg_48[1][index_48,:].detach().numpy()
#         result_48_48 = output_reg_48[2][index_48,:].detach().numpy()
#
#     file_path = 'predict_results_our.csv'
#
#     with open(file_path, mode='a', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         if index_24 != -1:
#             writer.writerow(np.array([str(result_24_6[i]) for i in range(0, len(result_24_6))]))
#             writer.writerow(np.array([str(result_24_12[i]) for i in range(0, len(result_24_12))]))
#             writer.writerow(np.array([str(result_24_24[i]) for i in range(0, len(result_24_24))]))
#         else:
#             writer.writerows(["1"])  # 写入数据
#             writer.writerows(["1"])  # 写入数据
#             writer.writerows(["1"])  # 写入数据
#         if index_36 != -1:
#             writer.writerow(np.array([str(result_36_6[i]) for i in range(0, len(result_36_6))]))
#             writer.writerow(np.array([str(result_36_12[i]) for i in range(0, len(result_36_12))]))
#             writer.writerow(np.array([str(result_36_36[i]) for i in range(0, len(result_36_36))]))
#         else:
#             writer.writerows(["1"])  # 写入数据
#             writer.writerows(["1"])  # 写入数据
#             writer.writerows(["1"])  # 写入数据
#         if index_48 != -1:
#             writer.writerow(np.array([str(result_48_6[i]) for i in range(0, len(result_48_6))]))
#             writer.writerow(np.array([str(result_48_12[i]) for i in range(0, len(result_48_12))]))
#             writer.writerow(np.array([str(result_48_48[i]) for i in range(0, len(result_48_48))]))
#         else:
#             writer.writerows(["1"])  # 写入数据
#             writer.writerows(["1"])  # 写入数据
#             writer.writerows(["1"])  # 写入数据
def write_para(para):
    header = ["all_node_num", "remain_node_num", "lstm_hidden", "linear_dim1", "linear_dim2",
              "eton_outputchanel", "dropout", "lr", "decay", "epochs_rec", "epochs"]

    with open('result_图特征约束项_10times.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
        writer.writeheader()  # 写入列名

        writer.writerows([para])  # 写入数据
def write_to_csv(datas):
    header = ["CDRSB", "ADAS11", "MMSE", "RAVLT_immediate", "RAVLT_learning",
               "ADAS13", "RAVLT_forgetting", "RAVLT_perc_forgetting", "MOCA"]     # 数据列名

    # test.csv表示如果在当前目录下没有此文件的话，则创建一个csv文件
    # a表示以“追加”的形式写入，如果是“w”的话，表示在写入之前会清空原文件中的数据
    # newline是数据之间不加空行
    # encoding='utf-8'表示编码格式为utf-8，如果不希望在excel中打开csv文件出现中文乱码的话，将其去掉不写也行。
    with open('result_图特征约束项_10times.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
        # writer.writeheader()  # 写入列名

        writer.writerows([datas])  # 写入数据
def main():
    # 导入数据
    for i in range(10):

        all_node_num = 90
        folders = 5
        eton_outputchanel = 48  ##eton 的输出通道数量
        dropout = 0.3
        decay = 0.01
        kf = KFold(n_splits=folders, shuffle=True)
        kfold_index = 0
        batch_size = 16  #  70 50 /32 64 128/ 32 64 128/ 32 64/
        # 90, 70, 32, 128, 64, 48, 0.3, 0.0001, 0.01, 20, 320

        for remain_node_num in [70]:
            for lstm_hidden in [64]:
                for linear_dim1 in [128]:
                    for linear_dim2 in [64]:
                        if linear_dim1 ==linear_dim2 or linear_dim1<linear_dim2:
                            continue
                        for lr in [0.0001]:
                            for epochs_rec in [20]:
                                for epochs in [300]:
                                    para = {}
                                    para["all_node_num"] = all_node_num
                                    para["remain_node_num"] = remain_node_num
                                    para["lstm_hidden"] = lstm_hidden
                                    para["linear_dim1"] = linear_dim1
                                    para["linear_dim2"] = linear_dim2
                                    para["eton_outputchanel"] = eton_outputchanel
                                    para["dropout"] = dropout
                                    para["lr"] = lr
                                    para["decay"] = decay
                                    para["epochs_rec"] = epochs_rec
                                    para["epochs"] = epochs

                                    print(para)
                                    nums_train = np.ones(all_node_num)  # 制作mask模板
                                    Mask_train = nums_train.reshape(nums_train.shape[0], 1) * nums_train  # 90*90
                                    for i in range(remain_node_num):
                                        Mask_train[i][:remain_node_num] = 0
                                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                                    Mask_train_tensor = torch.from_numpy(Mask_train).float().to(device)

                                    print(torch.cuda.is_available())
                                    if torch.cuda.is_available():
                                        device = 'cuda'
                                    else:
                                        device = 'cpu'
                                    print('Device: ', device)
                                    filename_adj_data_24 = '../dataset/train_test/train_data_adj_all_24.npy'
                                    filename_labels_24 = '../dataset/train_test/train_label_all_24.npy'

                                    filename_adj_data_36 = '../dataset/train_test/train_data_adj_all_36.npy'
                                    filename_labels_36 = '../dataset/train_test/train_label_all_36.npy'

                                    filename_adj_data_48 = '../dataset/train_test/train_data_adj_all_48.npy'
                                    filename_labels_48 = '../dataset/train_test/train_label_all_48.npy'

                                    X_24 = np.load(filename_adj_data_24)
                                    Y_24 = np.load(filename_labels_24)
                                    X_36 = np.load(filename_adj_data_36)
                                    Y_36 = np.load(filename_labels_36)
                                    X_48 = np.load(filename_adj_data_48)
                                    Y_48 = np.load(filename_labels_48)

                                    X_masked_24, X_unmasked_24 = process_data(X_24, Y_24, all_node_num, remain_node_num)
                                    X_masked_36, X_unmasked_36 = process_data(X_36, Y_36, all_node_num, remain_node_num)
                                    X_masked_48, X_unmasked_48 = process_data(X_48, Y_48, all_node_num, remain_node_num)

                                    # model
                                    feature_dim = remain_node_num
                                    X_train_24_array = [];X_train_0_24_array = []; X_train_masked_24_array = [];X_train_masked_rest_24_array = []; Y_train_24_array = []; \
                                    X_test_24_array = []; X_test_0_24_array = []; X_masked_test_24_array = []; Y_test_24_array = []

                                    X_train_36_array = []; X_train_0_36_array = [];X_train_masked_36_array = []; X_train_masked_rest_36_array = [];Y_train_36_array = []; \
                                    X_test_36_array = []; X_test_0_36_array = []; X_masked_test_36_array = []; Y_test_36_array = []

                                    X_train_48_array = []; X_train_0_48_array = []; X_train_masked_48_array = [];X_train_masked_rest_48_array = []; Y_train_48_array = []; \
                                    X_test_48_array = []; X_test_0_48_array = []; X_masked_test_48_array = []; Y_test_48_array = []

                                    for trainval_index, test_index in kf.split(X_masked_24, Y_24):
                                        X_train_24,X_train_0_24,X_train_masked_24,X_train_masked_rest_24,Y_train_24,\
                                        X_test_24,X_test_0_24,X_masked_test_24,Y_test_24 \
                                            = deal_dataset(kfold_index,kf,X_masked_24,X_unmasked_24,X_24,Y_24,trainval_index,test_index,all_node_num , remain_node_num)
                                        X_train_24 = to_strue(X_train_24, "24")
                                        X_test_24 = to_strue(X_test_24, "24")

                                        X_train_24_array.append(X_train_24)
                                        X_train_0_24_array.append(X_train_0_24)
                                        X_train_masked_24_array.append(X_train_masked_24)
                                        X_train_masked_rest_24_array.append(X_train_masked_rest_24)
                                        Y_train_24_array.append(Y_train_24[:,[0,1,2,3]+[i for i in range(15,42)]])
                                        X_test_24_array.append(X_test_24)
                                        X_test_0_24_array.append(X_test_0_24)
                                        X_masked_test_24_array.append(X_masked_test_24)
                                        Y_test_24_array.append(Y_test_24[:,[0,1,2,3]+[i for i in range(15,42)]])
                                    for trainval_index, test_index in kf.split(X_masked_36, Y_36):
                                        X_train_36,X_train_0_36,X_train_masked_36,X_train_masked_rest_36,Y_train_36,\
                                        X_test_36,X_test_0_36,X_masked_test_36,Y_test_36 \
                                            = deal_dataset(kfold_index,kf,X_masked_36,X_unmasked_36,X_36,Y_36,trainval_index,test_index,all_node_num , remain_node_num)

                                        X_train_36 = to_strue(X_train_36, "36")
                                        X_test_36 = to_strue(X_test_36, "36")

                                        X_train_36_array.append(X_train_36)
                                        X_train_0_36_array.append(X_train_0_36)
                                        X_train_masked_36_array.append(X_train_masked_36)

                                        X_train_masked_rest_36_array.append(X_train_masked_rest_36)
                                        Y_train_36_array.append(Y_train_36[:,[0,1,2,4]+[i for i in range(15,33)]+[i for i in range(42,51)]])
                                        X_test_36_array.append(X_test_36)
                                        X_test_0_36_array.append(X_test_0_36)
                                        X_masked_test_36_array.append(X_masked_test_36)
                                        Y_test_36_array.append(Y_test_36[:,[0,1,2,4]+[i for i in range(15,33)]+[i for i in range(42,51)]])
                                    for trainval_index, test_index in kf.split(X_masked_48, Y_48):
                                        X_train_48,X_train_0_48,X_train_masked_48,X_train_masked_rest_48,Y_train_48,\
                                        X_test_48,X_test_0_48,X_masked_test_48,Y_test_48 \
                                            = deal_dataset(kfold_index,kf,X_masked_48,X_unmasked_48,X_48,Y_48,trainval_index,test_index,all_node_num , remain_node_num)

                                        X_train_48 = to_strue(X_train_48, "48")
                                        X_test_48 = to_strue(X_test_48, "48")

                                        X_train_48_array.append(X_train_48)
                                        X_train_0_48_array.append(X_train_0_48)
                                        X_train_masked_48_array.append(X_train_masked_48)

                                        X_train_masked_rest_48_array.append(X_train_masked_rest_48)
                                        Y_train_48_array.append(Y_train_48[:,[0,1,2,5]+[i for i in range(15,33)]+[i for i in range(51,60)]])
                                        X_test_48_array.append(X_test_48)
                                        X_test_0_48_array.append(X_test_0_48)
                                        X_masked_test_48_array.append(X_masked_test_48)
                                        Y_test_48_array.append(Y_test_48[:,[0,1,2,5]+[i for i in range(15,33)]+[i for i in range(51,60)]])
                                    result_total_24_rmse = []
                                    result_total_36_rmse = []
                                    result_total_48_rmse = []
                                    result_total_24_cc = []
                                    result_total_36_cc = []
                                    result_total_48_cc = []
                                    result_total_24_ccc = []
                                    result_total_36_ccc = []
                                    result_total_48_ccc = []
                                    for folder in range(folders):
                                        model = Model(dropout=dropout, num_class=3, remain_node_num=remain_node_num,
                                                  lstm_hidden = lstm_hidden,
                                                  eton_outputchanel = eton_outputchanel,linear_dim1=linear_dim1,linear_dim2=linear_dim2)
                                        model.to(device)
                                        params = list(model.parameters())
                                        k = 0
                                        # for i in params:
                                        #     l = 1
                                        #     print("该层的结构：" + str(list(i.size())))
                                        #     for j in i.size():
                                        #         l *= j
                                        #     print("该层参数和：" + str(l))
                                        #     k = k + l
                                        # print("总参数数量和：" + str(k))
                                        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
                                        optimizer2 = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
                                        # train
                                        for epoch in range(1, epochs + 1):
                                            model.train()
                                            test_data_all = [];test_label_all = [];test_data_A_all = [];test_data_batch_rest_all = [];test_data_maskedX_all = []

                                            idx_batch_24 = np.random.permutation(int(X_train_24_array[folder].shape[0]))
                                            num_batch_24 = X_train_24_array[folder].shape[0] // int(batch_size)

                                            idx_batch_36 = np.random.permutation(int(X_train_36_array[folder].shape[0]))
                                            num_batch_36 = X_train_36_array[folder].shape[0] // int(batch_size)

                                            idx_batch_48 = np.random.permutation(int(X_train_48_array[folder].shape[0]))
                                            num_batch_48 = X_train_48_array[folder].shape[0] // int(batch_size)

                                            loss_train = 0
                                            train_data_batch_all =[]
                                            train_label_batch_all = []
                                            train_data_batch_A_all = []
                                            train_data_batch_rest_all = []
                                            train_data_batch_maskedX_all = []

                                            train_data_batch_all, train_label_batch_all, train_data_batch_A_all, \
                                            train_data_batch_rest_all, train_data_batch_maskedX_all\
                                                = fusion_batch(num_batch_24,idx_batch_24,train_data_batch_all,train_label_batch_all,
                                                 train_data_batch_A_all,train_data_batch_rest_all,train_data_batch_maskedX_all,
                                                 X_train_24_array[folder],X_train_0_24_array[folder],X_train_masked_24_array[folder],
                                                 X_train_masked_rest_24_array[folder],Y_train_24_array[folder],batch_size)

                                            train_data_batch_all, train_label_batch_all, train_data_batch_A_all, \
                                            train_data_batch_rest_all, train_data_batch_maskedX_all= fusion_batch(num_batch_36,idx_batch_36,
                                                 train_data_batch_all,train_label_batch_all,
                                                 train_data_batch_A_all,train_data_batch_rest_all,train_data_batch_maskedX_all,
                                                 X_train_36_array[folder],X_train_0_36_array[folder],X_train_masked_36_array[folder]
                                                    ,X_train_masked_rest_36_array[folder],Y_train_36_array[folder],batch_size)

                                            train_data_batch_all, train_label_batch_all, train_data_batch_A_all, \
                                            train_data_batch_rest_all, train_data_batch_maskedX_all= fusion_batch(num_batch_48,idx_batch_48,
                                                 train_data_batch_all,train_label_batch_all,
                                                 train_data_batch_A_all,train_data_batch_rest_all,train_data_batch_maskedX_all,
                                                 X_train_48_array[folder],X_train_0_48_array[folder],X_train_masked_48_array[folder],
                                                 X_train_masked_rest_48_array[folder],Y_train_48_array[folder],batch_size)

                                            # if epoch == 85:
                                            #     print("error will happen")

                                            for bn in range(len(train_data_batch_all)):
                                                time_point = train_data_batch_all[bn][0].time_point
                                                dict = {}
                                                for name, p in model.named_parameters():
                                                    dict[name] = p
                                                    # print(p.requires_grad)
                                                key_list = ["e2e.0.conv1xd.weight", "e2e.0.conv1xd.bias", "e2e.0.convdx1.weight", "e2e.0.convdx1.bias",
                                                            "e2e.2.conv1xd.weight",
                                                            "e2e.2.conv1xd.bias", "e2e.2.convdx1.weight", "e2e.2.convdx1.bias", "e2n.0.weight",
                                                            "e2n.0.bias", "n2g.0.weight", "n2g.0.bias"]

                                                value = model.named_parameters([key_list])
                                                train_data_batch_dev = torch.from_numpy(np.array([train_data_batch_all[bn][i].data for
                                                                                 i in range(len(train_data_batch_all[bn]))])).float().to(device)
                                                train_label_batch_dev = torch.from_numpy(train_label_batch_all[bn]).float().to(device)
                                                train_data_batch_A_dev = torch.from_numpy(train_data_batch_A_all[bn]).float().to(device)
                                                train_data_batch_rest_dev = torch.from_numpy(train_data_batch_rest_all[bn]).float().to(device)
                                                train_data_batch_maskedX_dev = torch.from_numpy(train_data_batch_maskedX_all[bn]).float().to(device)
                                                optimizer.zero_grad()
                                                optimizer2.zero_grad()
                                                outputs, rec,pred_reg_sequence,features_learned = model(train_data_batch_dev, train_data_batch_A_dev, train_data_batch_maskedX_dev,time_point,Mask_train_tensor)

                                                loss1 = ent_loss(outputs, train_label_batch_dev)
                                                loss_constract = constract_loss(features_learned)
                                                loss_reconstrcu = rec_loss(rec, train_data_batch_rest_dev)
                                                loss3 =mae_loss_reg(pred_reg_sequence,train_label_batch_dev)

                                                if epoch < epochs_rec:
                                                    # loss = loss_reconstrcu+loss_constract*0.1
                                                    loss = loss_reconstrcu+loss_constract

                                                    loss_train += loss
                                                    loss.backward()
                                                    optimizer.step()
                                                else:
                                                    # loss = +loss3 +loss_reconstrcu+loss_constract*0.1
                                                    loss = loss3+loss_reconstrcu+loss_constract
                                                    loss_train += loss
                                                    loss.backward()
                                                    optimizer2.step()

                                            # if epoch % 10 == 0:
                                            print('epoch:', epoch, 'train loss:', loss_train.item())

                                            # if epoch % 20 == 0 and epoch > epochs_rec:
                                            #
                                            #     if True:  # acc_val > best_val:
                                        # torch.save(model.state_dict(), "./model/our_model_"+str(folder)+".pkl")

                                        model.eval()

                                        test_data_24_dev = torch.from_numpy(np.array([X_test_24_array[folder][i].data for i in range(len(X_test_24_array[folder]))])).float().to(device)
                                        test_data_36_dev = torch.from_numpy(np.array([X_test_36_array[folder][i].data for i in range(len(X_test_36_array[folder]))])).float().to(device)
                                        test_data_48_dev = torch.from_numpy(np.array([X_test_48_array[folder][i].data for i in range(len(X_test_48_array[folder]))])).float().to(device)
                                        outputs_24, _,output_reg_24,_ = model(test_data_24_dev, torch.from_numpy(X_test_0_24_array[folder]).float().to(device), torch.from_numpy(X_masked_test_24_array[folder]).float().to(device),"24",Mask_train_tensor)
                                        outputs_36, _,output_reg_36,_ = model(test_data_36_dev, torch.from_numpy(X_test_0_36_array[folder]).float().to(device), torch.from_numpy(X_masked_test_36_array[folder]).float().to(device),"36",Mask_train_tensor)
                                        outputs_48, _,output_reg_48,_ = model(test_data_48_dev, torch.from_numpy(X_test_0_48_array[folder]).float().to(device), torch.from_numpy(X_masked_test_48_array[folder]).float().to(device),"48",Mask_train_tensor)

                                        # train_data_24_dev = torch.from_numpy(np.array(
                                        #     [X_train_24_array[folder][i].data for i in
                                        #      range(len(X_train_24_array[folder]))])).float().to(device)
                                        # train_data_36_dev = torch.from_numpy(np.array(
                                        #     [X_train_36_array[folder][i].data for i in
                                        #      range(len(X_train_36_array[folder]))])).float().to(device)
                                        # train_data_48_dev = torch.from_numpy(np.array(
                                        #     [X_train_48_array[folder][i].data for i in
                                        #      range(len(X_train_48_array[folder]))])).float().to(device)


                                        # outputs_24_train, _, output_reg_24_train, _ = model(train_data_24_dev, torch.from_numpy(
                                        #     X_train_0_24_array[folder]).float().to(device), torch.from_numpy(
                                        #     X_train_masked_24_array[folder]).float().to(device), "24", Mask_train_tensor)
                                        # outputs_36_train, _, output_reg_36_train, _ = model(train_data_36_dev, torch.from_numpy(
                                        #     X_train_0_36_array[folder]).float().to(device), torch.from_numpy(
                                        #     X_train_masked_36_array[folder]).float().to(device), "36", Mask_train_tensor)
                                        # outputs_48_train, _, output_reg_48_train, _ = model(train_data_48_dev, torch.from_numpy(
                                        #     X_train_0_48_array[folder]).float().to(device), torch.from_numpy(
                                        #     X_train_masked_48_array[folder]).float().to(device), "48", Mask_train_tensor)
                                        # write_predict_results(output_reg_24_train,
                                        #                       output_reg_36_train,output_reg_48_train,Y_train_24_array[folder],Y_train_36_array[folder],Y_train_48_array[folder])
                                        # write_predict_results(output_reg_24,output_reg_36,output_reg_48,Y_test_24_array[folder],Y_test_36_array[folder],Y_test_48_array[folder])

                                        result_24_rmse,result_24_cc,result_24_ccc = evaluate_test(outputs_24,output_reg_24,Y_test_24_array[folder],"24")
                                        result_36_rmse,result_36_cc,result_36_ccc = evaluate_test(outputs_36,output_reg_36,Y_test_36_array[folder],"36")
                                        result_48_rmse,result_48_cc,result_48_ccc = evaluate_test(outputs_48,output_reg_48,Y_test_48_array[folder],"48")
                                        # write_predict_results(output_reg_24,Y_test_24_array[folder])
                                        # write_predict_results(output_reg_36,Y_test_36_array[folder])
                                        # write_predict_results(output_reg_48,Y_test_48_array[folder])
                                        # pickle.dump(model, open("./model/our_model.dat", "wb"))

                                        # print('result at time 24', result_24)
                                        # print('result at time 36', result_36)
                                        # print('result at time 48', result_48)
                                        result_total_24_rmse.append(result_24_rmse)
                                        result_total_36_rmse.append(result_36_rmse)
                                        result_total_48_rmse.append(result_48_rmse)

                                        result_total_24_cc.append(result_24_cc)
                                        result_total_36_cc.append(result_36_cc)
                                        result_total_48_cc.append(result_48_cc)

                                        result_total_24_ccc.append(result_24_ccc)
                                        result_total_36_ccc.append(result_36_ccc)
                                        result_total_48_ccc.append(result_48_ccc)
                                        print("result_24_rmse: ",result_24_rmse)
                                        print("result_36_rmse: ",result_36_rmse)
                                        print("result_48_rmse: ",result_48_rmse)
                                        print("result_24_cc: ",result_24_cc)
                                        print("result_36_cc: ",result_24_cc)
                                        print("result_48_cc: ",result_24_cc)
                                        print("result_24_ccc: ",result_24_ccc)
                                        print("result_36_ccc: ",result_24_ccc)
                                        print("result_48_ccc: ",result_24_ccc)

                                    result_ave_24_rmse = {
                                        "CDRSB": 0,
                                        "ADAS11": 0,
                                        "MMSE": 0,
                                        "RAVLT_immediate": 0,
                                        "RAVLT_learning": 0,
                                        "ADAS13": 0,
                                        "RAVLT_forgetting":0,
                                        "RAVLT_perc_forgetting": 0,
                                        "MOCA": 0}
                                    result_ave_36_rmse = {
                                        "CDRSB": 0,
                                        "ADAS11": 0,
                                        "MMSE": 0,
                                        "RAVLT_immediate": 0,
                                        "RAVLT_learning": 0,
                                        "ADAS13": 0,
                                        "RAVLT_forgetting": 0,
                                        "RAVLT_perc_forgetting": 0,
                                        "MOCA": 0}
                                    result_ave_48_rmse = {
                                        "CDRSB": 0,
                                        "ADAS11": 0,
                                        "MMSE": 0,
                                        "RAVLT_immediate": 0,
                                        "RAVLT_learning": 0,
                                        "ADAS13": 0,
                                        "RAVLT_forgetting": 0,
                                        "RAVLT_perc_forgetting": 0,
                                        "MOCA": 0}
                                    result_ave_24_cc = {
                                        "CDRSB": 0,
                                        "ADAS11": 0,
                                        "MMSE": 0,
                                        "RAVLT_immediate": 0,
                                        "RAVLT_learning": 0,
                                        "ADAS13": 0,
                                        "RAVLT_forgetting": 0,
                                        "RAVLT_perc_forgetting": 0,
                                        "MOCA": 0}
                                    result_ave_36_cc = {
                                        "CDRSB": 0,
                                        "ADAS11": 0,
                                        "MMSE": 0,
                                        "RAVLT_immediate": 0,
                                        "RAVLT_learning": 0,
                                        "ADAS13": 0,
                                        "RAVLT_forgetting": 0,
                                        "RAVLT_perc_forgetting": 0,
                                        "MOCA": 0}
                                    result_ave_48_cc = {
                                        "CDRSB": 0,
                                        "ADAS11": 0,
                                        "MMSE": 0,
                                        "RAVLT_immediate": 0,
                                        "RAVLT_learning": 0,
                                        "ADAS13": 0,
                                        "RAVLT_forgetting": 0,
                                        "RAVLT_perc_forgetting": 0,
                                        "MOCA": 0}
                                    result_ave_24_ccc = {
                                        "CDRSB": 0,
                                        "ADAS11": 0,
                                        "MMSE": 0,
                                        "RAVLT_immediate": 0,
                                        "RAVLT_learning": 0,
                                        "ADAS13": 0,
                                        "RAVLT_forgetting": 0,
                                        "RAVLT_perc_forgetting": 0,
                                        "MOCA": 0}
                                    result_ave_36_ccc = {
                                        "CDRSB": 0,
                                        "ADAS11": 0,
                                        "MMSE": 0,
                                        "RAVLT_immediate": 0,
                                        "RAVLT_learning": 0,
                                        "ADAS13": 0,
                                        "RAVLT_forgetting": 0,
                                        "RAVLT_perc_forgetting": 0,
                                        "MOCA": 0}
                                    result_ave_48_ccc = {
                                        "CDRSB": 0,
                                        "ADAS11": 0,
                                        "MMSE": 0,
                                        "RAVLT_immediate": 0,
                                        "RAVLT_learning": 0,
                                        "ADAS13": 0,
                                        "RAVLT_forgetting": 0,
                                        "RAVLT_perc_forgetting": 0,
                                        "MOCA": 0}
                                    for i in range(len(result_total_24_rmse)):
                                        for key in result_total_24_rmse[i].keys():
                                            result_ave_24_rmse[key]+=result_total_24_rmse[i][key]
                                            result_ave_36_rmse[key]+=result_total_36_rmse[i][key]
                                            result_ave_48_rmse[key]+=result_total_48_rmse[i][key]

                                            result_ave_24_cc[key] += result_total_24_cc[i][key]
                                            result_ave_36_cc[key] += result_total_36_cc[i][key]
                                            result_ave_48_cc[key] += result_total_48_cc[i][key]

                                            result_ave_24_ccc[key] += result_total_24_ccc[i][key]
                                            result_ave_36_ccc[key] += result_total_36_ccc[i][key]
                                            result_ave_48_ccc[key] += result_total_48_ccc[i][key]
                                    for key in result_ave_24_rmse.keys():
                                        result_ave_24_rmse[key]= result_ave_24_rmse[key]/5
                                        result_ave_24_cc[key] = result_ave_24_cc[key] / 5
                                        result_ave_24_ccc[key] = result_ave_24_ccc[key] / 5


                                        result_ave_36_rmse[key] = result_ave_36_rmse[key]/5
                                        result_ave_36_cc[key] = result_ave_36_cc[key] / 5
                                        result_ave_36_ccc[key] = result_ave_36_ccc[key] / 5

                                        result_ave_48_rmse[key] =result_ave_48_rmse[key]/5
                                        result_ave_48_cc[key] = result_ave_48_cc[key] / 5
                                        result_ave_48_ccc[key] = result_ave_48_ccc[key] / 5
                                    print("result_ave_24_rmse",result_ave_24_rmse)
                                    print("result_ave_36_rmse",result_ave_36_rmse)
                                    print("result_ave_48_rmse",result_ave_48_rmse)

                                    print("result_ave_24_cc",result_ave_24_cc)
                                    print("result_ave_36_cc",result_ave_36_cc)
                                    print("result_ave_48_cc",result_ave_48_cc)

                                    print("result_ave_24_ccc",result_ave_24_ccc)
                                    print("result_ave_36_ccc",result_ave_36_ccc)
                                    print("result_ave_48_ccc",result_ave_48_ccc)

                                    write_para(para)

                                    write_to_csv(result_ave_24_rmse)
                                    write_to_csv(result_ave_36_rmse)
                                    write_to_csv(result_ave_48_rmse)
                                    write_to_csv(result_ave_24_cc)
                                    write_to_csv(result_ave_36_cc)
                                    write_to_csv(result_ave_48_cc)
                                    write_to_csv(result_ave_24_ccc)
                                    write_to_csv(result_ave_36_ccc)
                                    write_to_csv(result_ave_48_ccc)

if __name__ == '__main__':
    main()
