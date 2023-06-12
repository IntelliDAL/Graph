from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import warnings
import sklearn.metrics as metrics
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.autograd import Variable
import datetime
import math
from torch.nn import init
import torch
import random
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import visdom
from S_Dbw.S_Dbw import S_Dbw

warnings.filterwarnings("ignore")

from torch.backends import cudnn


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

        # x = self.fc(x)
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
        self.num_node = num_node  # 116
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
        self.tcn = tcn_Networks(528, 528)  # 150-528 100-336 75-240
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
            # else:
            #     feature_t_skip = features[:, j, :, :]
            #     adj_t_skip = A[:, j, :, :]
            #     adj_t_skip_before = A[:, j - 1, :, :]
            #     adj_t_skip_diff = torch.abs(adj_t_skip - adj_t_skip_before)
            #     diff_t_skip = self.gcn_difference(None, adj_t_skip_diff)
            #     combined_skip = torch.cat([feature_t_skip, h_t_skip], 2)
            #     x_skip = self.gcn_networks(combined_skip, adj_t_skip)
            #     x_skip = x_skip * diff_t_skip
            #     it_skip = torch.sigmoid(
            #         (torch.matmul(x_skip, self.Wi_skip1) + self.bi_skip1.t()))  # (batch_size,N,hidden_size)
            #     ft_skip = torch.sigmoid((torch.matmul(x_skip, self.Wf_skip1) + self.bf_skip1.t()))
            #     ot_skip = torch.sigmoid((torch.matmul(x_skip, self.Wo_skip1) + self.bo_skip1.t()))
            #     ut_skip = torch.relu((torch.matmul(x_skip, self.Wc_skip1) + self.bc_skip1.t()))
            #     c_t_skip = ft_skip * c_t_skip + it_skip * ut_skip  # shape=
            #     h_t_skip = ot_skip * torch.tanh(c_t_skip)  # shape = (batch_size, num_node, hidden_dim)
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


from sklearn import manifold


def t_sne(X, y, _iter, _epoch, pcaN=2):
    tsne = manifold.TSNE(n_components=pcaN, init='pca')
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
    # print(X_tsne)

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    print('X_norm.shape', X_norm.shape)
    plt.close('all')
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    # plt.savefig("Pics_method1_sklPCA\\Pic_{}_{}_Pic2D.png".format(_iter, _epoch))
    plt.savefig("Pics_method1_sklPCA/Pic_{}_{}_Pic2D.png".format(_iter, _epoch))
    # plt.show()
    return


from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# estimator = KMeans(n_clusters=2, init="k-means++", random_state=420)  # 构造聚类器


# estimator = KMeans(n_clusters=2, n_init=100, init="random", random_state=420)  # 构造聚类器
estimator = AgglomerativeClustering(n_clusters=2)


# estimator = GaussianMixture(n_components=2)


# 修改中心数量记得改标签 temp[

from sklearn.decomposition import PCA
def feature_preprocessing(features, pca_dim=2):
    _, feature_dim = features.shape
    # print(feature_dim)
    # features = features.astype('float32')
    features = features.cpu().detach().numpy()
    features = features.astype('float32')

    # PCA_generator = faiss.PCAMatrix(feature_dim, pca_dim, eigen_power=-0.5)
    # PCA_generator.train(features)
    pca = PCA(n_components=pca_dim)  # n_components can be integer or float in (0,1)
    features = pca.fit_transform(features)

    # PCA-Whitening
    # assert PCA_generator.is_trained
    # features = PCA_generator.apply_py(features)

    # L2 normalization
    row_sum = np.linalg.norm(features, axis=1)
    featrues = features / row_sum[:, np.newaxis]

    return features


from mpl_toolkits.mplot3d import Axes3D


def show_result_3D(data, C, _iter, _epoch):
    colors_list = ['red', 'blue']
    labels_list = ['A', 'B']
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(data)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    y_pred = []
    # for i in range(len(data)):
    #     if i in C1:
    #         y_pred.append(0)
    #     else:
    #         y_pred.append(1)
    # y_pred = np.array(y_pred)
    y_pred = C
    fig = plt.figure()
    ax = Axes3D(fig)

    # unique_lables = set(labels)
    # colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_lables)))
    for i in range(2):
        ax.scatter3D(X_norm[y_pred == i, 0], X_norm[y_pred == i, 1], X_norm[y_pred == i, 2], c=colors_list[i],
                     label=labels_list[i])

    plt.title('data by make_classification()')
    # for i in range(2):
    #     plt.scatter(X_norm[y_pred == i, 0], X_norm[y_pred == i, 1], s=100, c=colors_list[i], label=labels_list[i])
    # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black',
    #             label='Centroids')
    plt.legend()
    # plt.savefig("Pics_method1_sklPCA\\Pic_{}_{}_Pic3D.png".format(_iter, _epoch))
    plt.savefig("Pics_method1_sklPCA/Pic_{}_{}_Pic3D.png".format(_iter, _epoch))
    # plt.show()


global labels_list
labels_list = []
label_list_or = []
ll = []
from sklearn.metrics import silhouette_score


class BrainNet(nn.Module):
    def __init__(self):
        super(BrainNet, self).__init__()
        """
        """
        super_node = 12
        self.super_node = super_node
        self.LG_Networks = Layer(528, 100, 256, 200, self.super_node, 2)
        self.lin = nn.Linear(256 * self.super_node, 2)

    def forward(self, x, adj, ee):

        state = (torch.zeros([adj.shape[0], self.super_node, 256]).cuda(),
                 torch.zeros([adj.shape[0], self.super_node, 256]).cuda())
        state_skip = (
            torch.zeros([adj.shape[0] * 2, self.super_node, 256]).cuda(),
            torch.zeros([adj.shape[0] * 2, self.super_node, 256]).cuda())
        h, loss = self.LG_Networks(x, adj, state, state_skip)

        h = h.reshape(h.shape[0], -1)
        x = self.lin(h)
        # 特征 聚类
        global labels_list
        # print(h.shape[0]) 方法策略1的
        # kmeans 只用一个 试试

        # starttime = datetime.datetime.now()

        if h.shape[0] != 0:
            out = feature_preprocessing(h, pca_dim=2)
            out1 = feature_preprocessing(h, pca_dim=3)
            out2 = feature_preprocessing(h, pca_dim=4)
            out3 = feature_preprocessing(h, pca_dim=5)
            out4 = feature_preprocessing(h, pca_dim=6)
            out5 = feature_preprocessing(h, pca_dim=7)
            out6 = feature_preprocessing(h, pca_dim=8)
            out7 = feature_preprocessing(h, pca_dim=9)
            out8 = feature_preprocessing(h, pca_dim=10)

            # time_point_1 = datetime.datetime.now()

            # print('cluster')
            # out = x.cpu().detach().numpy()
            # out = h.cpu().detach().numpy()
            # print('out',out.shape)

            ################################## 聚类 ##################################
            # out = h.cpu().detach().numpy()
            # print(out.shape)
            out = out.astype('float32')
            estimator.fit(out)  # 聚类
            label_pred = estimator.labels_  # 获取聚类标签
            # label_pred = estimator.predict(out)

            estimator.fit(out1)
            label_pred1 = estimator.labels_
            # label_pred1 = estimator.predict(out1)

            estimator.fit(out2)
            label_pred2 = estimator.labels_
            # label_pred2 = estimator.predict(out2)

            estimator.fit(out3)
            label_pred3 = estimator.labels_
            # label_pred3 = estimator.predict(out3)

            estimator.fit(out4)
            label_pred4 = estimator.labels_
            # label_pred4 = estimator.predict(out4)

            estimator.fit(out5)
            label_pred5 = estimator.labels_
            # label_pred5 = estimator.predict(out5)

            estimator.fit(out6)
            label_pred6 = estimator.labels_
            # label_pred6 = estimator.predict(out6)

            estimator.fit(out7)
            label_pred7 = estimator.labels_
            # label_pred7 = estimator.predict(out7)

            estimator.fit(out8)
            label_pred8 = estimator.labels_
            # label_pred8 = estimator.predict(out8)

            temp = label_pred + label_pred1 + label_pred2 + label_pred3 + label_pred4 + label_pred5 + label_pred6 + label_pred7 + label_pred8
            # temp = label_pred + label_pred1
            # temp = label_pred
            # print(temp)

            # time_point_2 = datetime.datetime.now()

            ################################## 聚类 ##################################
            # labels_list = label_pred
            # 聚类3簇的标签处理
            # temp[temp < 3] = 0
            # temp[temp == 3] = 1
            # temp[temp == 4] = 1
            # temp[temp == 5] = 1
            # temp[temp > 5] = 2
            # labels_list = temp

            # 聚类2簇的标签处理
            temp[temp < 5] = 0
            temp[temp >= 5] = 1
            labels_list = temp

            # temp[temp <= 2] = 0  # 0 1 2
            # temp[temp >= 3] = 1  # 3 4 5
            # labels_list = temp

            print("result = {}".format(label_pred))
            # print("result_1 = {}".format(label_pred1))

            # if ee == 398:
            #     score = silhouette_score(out, label_pred)
            #     score1 = silhouette_score(out1, label_pred1)
            #     print('score', score)
            #     print('score1', score1)
            # score 0.39718685. score1 0.34682256
            # show_result_3D(out1, label_pred1)
            # a = 1
            # print(label_pred.shape) # type-numpy length-batchsize
            # clusters, cluster_loss = cluster(out)
            # print(np.array(clusters).shape)
            # print(clusters)
            # t_sne(out, label_pred)
            # if e == 299:
            #     t_sne(out, label_pred, pcaN=2)
            #     t_sne(out1, label_pred1, pcaN=3)
            #     t_sne(out2, label_pred2, pcaN=4)
            #     t_sne(out3, label_pred3, pcaN=5)
            #     t_sne(out4, label_pred4, pcaN=6)
            #     t_sne(out5, label_pred5, pcaN=7)
            #     t_sne(out6, label_pred6, pcaN=8)
            #     t_sne(out7, label_pred7, pcaN=9)
            #     t_sne(out8, label_pred8, pcaN=10)
        # 0 1 2 3 4 5 6 7 8
        # 0 1 2 3 4 5 6 7 8 9
        else:
            print(h.shape[0])
        # print(h.shape[0])
        global label_list_or

        # print('x', x.shape)

        # if h.shape[0] != 256:
        #     print(label_pred)
        global ll
        if len(label_list_or) == 0:
            pass

        else:
            diff = np.sum(abs(np.array(label_list_or) - np.array(labels_list)))
            print("diff =", diff)

            ll.append(diff)
            if diff < 0:
                # print('')
                import sys
                sys.exit()

        label_list_or = labels_list

        # if ee == 200:
        #     print(ll)
        cluster_results = [[out, label_pred], [out1, label_pred1]]

        # time_point_3 = datetime.datetime.now()

        # print("########################## time ##########################")
        # print("time_point_1 = {}".format(time_point_1 - starttime))
        # print("time_point_2 = {}".format(time_point_2 - time_point_1))
        # print("time_point_3 = {}".format(time_point_3 - time_point_2))
        # print("########################## time ##########################")

        return x, loss, labels_list, cluster_results


def load_data():
    train_data_path = 'data/ASD_data.npy'
    train_data = np.load(train_data_path)
    train_label = np.ones(np.shape(train_data)[0], )

    # 测试bug用
    # train_data = train_data[0:50, :, :]
    # train_label = train_label[0:50]

    new_train_data, new_train_label = train_data, train_label

    time_train_data = []
    time_train_graph = []
    for i in range(len(new_train_data)):
        temp_train = split(new_train_data[i])
        time_train_data.append(temp_train[0])
        time_train_graph.append(temp_train[1])

        if i % 50 == 0:
            print(i, len(new_train_data))

    time_train_data = np.array(time_train_data)
    time_train_graph = np.array(time_train_graph)

    # 测试bug用
    # print("shape 1 ", np.shape(time_train_data))
    # print("shape 2 ", np.shape(time_train_graph))
    # import sys
    # sys.exit()

    dataset_sampler = datasets(time_train_data, time_train_graph, new_train_label)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=300,
        shuffle=False,
        num_workers=0)

    return train_dataset_loader



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


def split(data):
    """
    生成10個圖
    :param data:    (22 * 1200)
    :return:
    """
    stride = 25  # 25
    window = 150  # 150
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


def train(dataset, model, iternumm, device='cpu'):
    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                         model.parameters()),
                                  lr=0.005,  # 0.005 0.0001
                                  weight_decay=0.001)  # 0.001
    scheduler_1 = CosineAnnealingWarmRestarts(optimizer2, T_0=50, T_mult=2)
    # for name in model.state_dict():
    #     print(name)

    best_val_acc = 0
    best_epoch = 0
    global ll
    loss_list = []
    all_loss_list = []
    ll_15s = []
    ll_15_flag = 0
    for_stop = 0
    viz = visdom.Visdom()

    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
        h0 = Variable(data['h0'].float(), requires_grad=False).to(device)
        # label = Variable(data['label'].long()).to(device)  # data
        # pred, reg_loss, cluster_labels, out, label_pred = model(h0, adj, epoch)

    for epoch in range(1000):
        avg_loss = 0.0
        model.train()
        print("###############################################")
        print("iter=", iternumm, ", epoch=", epoch)

        ############## 原for枚举数据集部分 #################
        model.zero_grad()

        pred, reg_loss, cluster_labels, cluster_results = model(h0, adj, epoch)
        label = torch.Tensor(cluster_labels).long().to(device)
        loss = F.cross_entropy(pred, label, size_average=True)

        loss_list.append(loss.item())
        all_loss = loss + (reg_loss / 100)
        all_loss_list.append(all_loss.item())
        all_loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer2.step()
        avg_loss += loss

        print("-----------------------------------------")
        # print(ll)

        ll_small = 5  # 5 # ll小于ll_small的时候压入ll_15s中 连续50次小于5
        ll_small_len = 50  # ll_15s长度为ll_small_len时认为收敛

        ll_len = len(ll)
        if ll_len == 0:
            continue
        if ll[ll_len - 1] <= ll_small:
            ll_15s.append(ll[ll_len - 1])
            ll_15_flag = 3
            if len(ll_15s) == ll_small_len:
                for_stop = 1
                # break

        elif ll_15_flag > 0:
            ll_15_flag = ll_15_flag - 1
        else:
            ll_15_flag = 0
            ll_15s = []

        print("ll_15_flag = ", ll_15_flag)
        print("ll_15s = ", ll_15s)
        print("ll_len = ", len(ll_15s))
        print("-----------------------------------------")

        # #nyaia# visdom库在网页上动态显示曲线 ######################
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
        axs[0].plot(ll)
        axs[1].plot(loss_list)
        axs[2].plot(all_loss_list)
        # print("loss_list = ", loss_list)
        viz.matplot(plt, opts=dict(title='julei/loss/all_loss'), win='julei/loss/all_loss')
        # #nyaia# visdom库 在网页上动态显示曲线 ######################

        ############## 原for枚举数据集部分 #################

        # avg_loss /= batch_idx + 1
        # print('ok1')
        print("avg_loss : ", avg_loss)
        # print('ok2')
        scheduler_1.step()
        path_checkpoint = ""
        checkpoints = 1

        # save model files
        # if (epoch + 1) % 5 == 100:
        #     checkpoint = {"model_state_dict": model.state_dict(),
        #                   "optimizer_state_dict": optimizer2.state_dict(),
        #                   "epoch": epoch}
        #     path_checkpoint = "./model/Clustermodel/{}/checkpoint_{}_epoch.pkl".format(fold, epoch)
        # torch.save(checkpoint, path_checkpoint)

        # t_sne(result_1[0], result_1[1], iternumm, epoch)
        t_sne(cluster_results[0][0], cluster_results[0][1], iternumm, epoch)
        show_result_3D(cluster_results[1][0], cluster_results[1][1], iternumm, epoch)
        result_num = len(cluster_results)
        for a_result in range(0, result_num):
            SH_score = silhouette_score(cluster_results[a_result][0], cluster_results[a_result][1])
            SD_Value = S_Dbw(cluster_results[a_result][0], cluster_results[a_result][1], centerIdxs=None)
            print("result_{} の [silhouette, S_Dbw] = [{}, {}]".format(a_result, SH_score, SD_Value.result()))



        # print("########################## time ##########################")
        # print("time_point_1 = {}".format(time_point_1 - starttime))
        # print("time_point_2 = {}".format(time_point_2 - time_point_1))
        # print("time_point_3 = {}".format(time_point_3 - time_point_2))
        # print("time_point_4 = {}".format(time_point_4 - time_point_3))
        # print("time_point_5 = {}".format(time_point_5 - time_point_4))
        # print("time_point_6 = {}".format(time_point_6 - time_point_5))
        # print("time_point_7 = {}".format(time_point_7 - time_point_6))
        # print("########################## time ##########################")

        if for_stop == 1:
            # t_sne(result_1[0], result_1[1], iternumm, epoch)
            # show_result_3D(result_1[0], result_1[1], iternumm, epoch)
            # print(cluster_labels)
            break

    # print("-----------------------------------------")
    # print(ll)
    # plt.plot(ll)
    # plt.show()
    # show_result_3D(out, label_pred)
    # SH_score = silhouette_score(result_0[0], result_0[1])
    # SH_score1 = silhouette_score(result_1[0], result_1[1])
    # SD_Value = S_Dbw(result_0[0], result_0[1], centerIdxs=None)
    # SD_Value1 = S_Dbw(result_1[0], result_1[1], centerIdxs=None)
    # print("result 0 [silhouette, S_Dbw] = [{}, {}]".format(SH_score, SD_Value.result()))
    # print("result 1 [silhouette, S_Dbw] = [{}, {}]".format(SH_score1, SD_Value1.result()))
    # ll = []

    return model


###########################################################################################
###########################################################################################
# 主函数


def main():
    # 设置种子
    seed = 1
    set_seed(seed=seed)
    # 导入数据
    print('finished')
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Device: ', device)
    for i in range(0, 1):
        train_data_loader = load_data()
        model = BrainNet()

        model.to(device)
        # print('model:', model)
        model = train(train_data_loader, model, i + 1, device='cuda')


if __name__ == '__main__':
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main()

# 1. ll_small = 5    # ll小于ll_small的时候压入ll_15s中 （收敛条件）
# 2. ll_small_len = 50   # ll_15s长度为ll_small_len时认为收敛
# estimator = AgglomerativeClustering(n_clusters=X)     # 聚类中心n_clusters=X


# 服务器的 visdom 显示
# 先把远程的8097端口定到了本地的18097端口（用cmd，不用pycharm里的终端）
# ssh -L 18097:127.0.0.1:8097 -p 45662 root@region-3.autodl.com
# 密码 45662 —— 0gLnwF7chZ
# # python -m visdom.server
# # 然后在浏览器地址栏输入 127.0.0.1:18097
# # 然后运行代码
# 说明网址：
# https://blog.csdn.net/sxl1399504891/article/details/107998550?utm_term=%E6%9C%AC%E6%9C%BA%E8%AE%BF%E9%97%AE%E6%9C%8D%E5%8A%A1%E5%99%A8%E7%9A%84visdom%E7%BD%91%E9%A1%B5&utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~sobaiduweb~default-0-107998550&spm=3001.4430
