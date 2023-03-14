import matplotlib.pyplot as plt
from torch.autograd.variable import Variable
import pickle
# from train_old_version2 import train, evaluate
import warnings
# from cross_valiad import cross_val
# from proprecess import divide, cal_aal_pcc, cal_aal_partial, preprocess_corr
# from model_version import final_model
# from load_data import load_data
import sklearn.metrics as metrics
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from pytorchtools import EarlyStopping
import time
import math
from torch.nn import init
import pandas as pd
import os
import tqdm
import numpy as np
import torch
import random
from Graph_sample import datasets2
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
from tensorboardX import SummaryWriter

writer1 = SummaryWriter()
###############################################################
###############################################################
# 
dim = 200
path1 = 'cc200/'
path2 = 'Phenotypic_V1_0b_preprocessed1.csv'


def get_key(file_name):
    file_name = file_name.split('_')
    key = ''
    for i in range(len(file_name)):
        if file_name[i] == 'rois':
            key = key[:-1]
            break
        else:
            key += file_name[i]
            key += '_'
    return key





def load_data(path1, path2):
    profix = path1
    dirs = os.listdir(profix)
    all = {}
    labels = {}
    all_data = []
    label = []
    files = open('files.txt', 'r', encoding='utf-8')
    for filename in dirs:
        filename = files.readline().strip()
        print(filename)
        a = np.loadtxt(path1 + filename)
        # print(filename)
        a = a.transpose()
        # a = a.tolist()
        all[filename] = a
        all_data.append(a)
        data = pd.read_csv(path2)
        for i in range(len(data)):
            if get_key(filename) == data['FILE_ID'][i]:
                if int(data['DX_GROUP'][i]) == 1:
                    labels[filename] = int(data['DX_GROUP'][i])
                    label.append(int(data['DX_GROUP'][i]))
                else:
                    labels[filename] = 0
                    label.append(0)
                break
    label = np.array(label)
    return all, labels, all_data, label  # 871 * 116 * ?


def cal_aal_pcc(time_data):
    corr_matrix = []
    for key in range(len(time_data)):
        corr = []
        for key1 in range(len(time_data[key])):
            corr_mat = np.corrcoef(time_data[key][key1])
            corr.append(corr_mat)
        corr_matrix.append(corr)
    data_array = np.array(corr_matrix)
    where_are_nan = np.isnan(data_array) 
    where_are_inf = np.isinf(data_array)  
    for bb in range(0, 871):
        for k in range(data_array.shape[1]):
            for i in range(0, dim):
                for j in range(0, dim):
                    if where_are_nan[bb][k][i][j]:
                        data_array[bb][k][i][j] = 0
                    if where_are_inf[bb][k][i][j]:
                        data_array[bb][k][i][j] = 0.8
    corr_p = np.maximum(data_array, 0)  # pylint: disable=E1101
    corr_n = 0 - np.minimum(data_array, 0)  # pylint: disable=E1101
    data_array = [corr_p, corr_n]
    data_array = np.array(data_array)  # 2 871 4 116 116
    data_array = np.transpose(data_array, (1, 0, 2, 3, 4))
    return data_array


def cal_pcc(data, phi):
    '''
    :param data:     871 * 116 * ?
    :return:  adj
    '''
    corr_matrix = []
    for key in range(len(data)):  
        corr_mat = np.corrcoef(data[key])
        # if key == 5:
        #    print(corr_mat)
        corr_mat = np.arctanh(corr_mat - np.eye(corr_mat.shape[0]))

        corr_matrix.append(corr_mat)
    data_array = np.array(corr_matrix)  # 871 116 116
    data_array_or = data_array.copy()[0]
    where_are_nan = np.isnan(data_array)  
    where_are_inf = np.isinf(data_array) 
    for bb in range(0, 871):
        for i in range(0, dim):
            for j in range(0, dim):
                if where_are_nan[bb][i][j]:
                    data_array[bb][i][j] = 0
                if where_are_inf[bb][i][j]:
                    data_array[bb][i][j] = 1
                if data_array[bb][i][j] > phi:
                    data_array[bb][i][j] = 1
                elif data_array[bb][i][j] < phi*(-1):
                    data_array[bb][i][j] = -1
                else:
                    data_array[bb][i][j] = 0

    # print(data_array[0])
    corr_p = np.maximum(data_array, 0)  # pylint: disable=E1101
    corr_n = 0 - np.minimum(data_array, 0)  # pylint: disable=E1101
    data_array = [corr_p, corr_n]
    data_array = np.array(data_array)  # 2 871 116 116
    data_array = np.transpose(data_array, (1, 0, 2, 3))

    corr = corr_p[0]
    from nilearn import plotting
    from nilearn import datasets

    # lab = atlas['labels']
    lab = []
    for i in range(200):
        lab.append(str(i+1))
    np.fill_diagonal(corr, 1)
    # matrix
    # plotting.plot_matrix(data_array_or, vmax=1, vmin=-1)
    # plt.savefig('matrix_or.jpg')

    # 3D brain connection
    # power = datasets.fetch_coords_power_2011()
    # atlas = datasets.fetch_atlas_aal()
    # coords = plotting.find_parcellation_cut_coords(labels_img=atlas['maps'])
    # plotting.plot_connectome(corr, coords, title='Thresholds', edge_threshold='99.8%', node_size=20, colorbar=True)
    # plt.savefig('3Dbrain.jpg')

    # subnetwork
    rest_dataset = datasets.fetch_development_fmri(n_subjects=20)
    func_filenames = rest_dataset.func
    confounds = rest_dataset.confounds
    from nilearn.decomposition import DictLearning

    
    dict_learn = DictLearning(n_components=8, smoothing_fwhm=6.,
                              memory="nilearn_cache", memory_level=2,
                              random_state=0)
    # 
    dict_learn.fit(func_filenames)
    # 
    components_img = dict_learn.components_img_
    from nilearn.regions import RegionExtractor
    extractor = RegionExtractor(components_img, threshold=0.5,
                                thresholding_strategy='ratio_n_voxels',
                                extractor='local_regions',
                                standardize=True, min_region_size=1350)
    # 
    extractor.fit()
    # 
    regions_extracted_img = extractor.regions_img_
    # 
    regions_index = extractor.index_
    # 
    n_regions_extracted = regions_extracted_img.shape[-1]
    from nilearn.connectome import ConnectivityMeasure
    correlations = []
    # 
    connectome_measure = ConnectivityMeasure(kind='correlation')
    for filename, confound in zip(func_filenames, confounds):
        # 
        timeseries_each_subject = extractor.transform(filename, confounds=confound)
        # 
        correlation = connectome_measure.fit_transform([timeseries_each_subject])
        # 
        correlations.append(correlation)
    mean_correlations = np.mean(correlations, axis=0).reshape(n_regions_extracted,
                                                              n_regions_extracted)
    title = 'Correlation between regions'
    # display = plotting.plot_matrix(mean_correlations, vmax=1, vmin=-1,
    #                                colorbar=True, title=title)
    regions_img = regions_extracted_img
    coords_connectome = plotting.find_probabilistic_atlas_cut_coords(regions_img)

    plotting.plot_connectome(mean_correlations, coords_connectome,
                             edge_threshold='90%', title=title)
    plt.savefig('sub.jpg')
    print('print matrix!')
    return data_array


###############################################################################
###############################################################################

import scipy.io as sio
import scipy.sparse as sp
from sklearn import preprocessing
#################################################################################
###############################################################################
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo().A
def normalize_adj_new(adj):
    """Symmetrically normalize adjacency matrix."""
    A_list = []
    for i in range(len(adj)):
        for j in range(i+1, len(adj)):
            A_list.append(adj[i][j])
    print(max(A_list))
    A_normalized = preprocessing.normalize(np.array(A_list)[:, np.newaxis], axis=0).ravel()
    # A = np.array(A_list)
    # A_normalized = A / np.linalg.norm(A)
    idx = 0
    for i in range(len(adj)):
        for j in range(i+1, len(adj)):
            adj[i][j] = A_normalized[idx]
            adj[j][i] = A_normalized[idx]
            idx+=1
    return adj
    # adj = sp.coo_matrix(adj)
    # rowsum = np.array(adj.sum(1))
    # d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo().A
# 

def cross_val(A, A1, A2, labels):
    # num_per_fold = len(graphs)/n_fold
    # np.random.shuffle(graphs)

    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    zip_list = list(zip(A, A1, A2, labels))
    random.Random(0).shuffle(zip_list)
    A, A1, A2, labels = zip(*zip_list)
    test_data_loader = []
    train_data_loader = []
    valid_data_loader = []
    A = np.array(A)
    A1 = np.array(A1)
    A2 = np.array(A2)
    labels = np.array(labels)
    for kk, (train_index, test_index) in enumerate(kf.split(A, labels)):
        train_val_adj, test_adj = A[train_index], A[test_index]
        train_val_adj1, test_adj1 = A1[train_index], A1[test_index]
        train_val_adj2, test_adj2 = A2[train_index], A2[test_index]
        train_val_labels, test_labels = labels[train_index], labels[test_index]
        # tv_folder = StratifiedKFold(n_splits=5, random_state=0, shuffle=True).split(train_val_adj, train_val_labels)
        # for t_idx, v_idx in tv_folder:
        #     train_adj, train_labels = train_val_adj[t_idx], train_val_labels[t_idx]
        #     val_adj, val_labels = train_val_adj[v_idx], train_val_labels[v_idx]

        dataset_sampler = datasets2(test_adj, test_adj1, test_adj2, test_labels)
        test_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=32,
            shuffle=False,
            num_workers=0)
        test_data_loader.append(test_dataset_loader)
        dataset_sampler = datasets2(train_val_adj, train_val_adj1, train_val_adj2, train_val_labels)
        train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=32,
            shuffle=True,
            num_workers=0)
        train_data_loader.append(train_dataset_loader)
        dataset_sampler = datasets2(test_adj, test_adj1, test_adj2, test_labels)
        val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=32,
            shuffle=False,
            num_workers=0)
        valid_data_loader.append(val_dataset_loader)

    return train_data_loader, valid_data_loader, test_data_loader


#############################################################################
#############################################################################


##############################################################################
##############################################################################

# 

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, neg_penalty):
        super(GCN, self).__init__()
        self.in_dim = in_dim  # 
        self.out_dim = out_dim  # 
        self.neg_penalty = neg_penalty  # 
        self.kernel = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.c = 0.85
        self.losses = []

    def forward(self, x, adj):
        # GCN-node
        feature_dim = int(adj.shape[-1])
        eye = torch.eye(feature_dim).cuda()  # feature_dim * feature_dim
        if x is None:  # 
            AXW = torch.tensordot(adj, self.kernel, [[-1], [0]])  # batch_size * num_node * feature_dim
        else:
            XW = torch.tensordot(x, self.kernel, [[-1], [0]])  # batch *  num_node * feature_dim
            AXW = torch.matmul(adj, XW)  # batch *  num_node * feature_dim
        # I_cAXW = eye+self.c*AXW
        I_cAXW = eye + self.c * AXW
        y_relu = torch.nn.functional.relu(I_cAXW)
        temp = torch.mean(input=y_relu, dim=-2, keepdim=True) + 1e-6
        col_mean = temp.repeat([1, feature_dim, 1])
        y_norm = torch.divide(y_relu, col_mean)  # 
        output = torch.nn.functional.softplus(y_norm)
        # output = y_relu
        # 
        if self.neg_penalty != 0:
            neg_loss = torch.multiply(torch.tensor(self.neg_penalty),
                                      torch.sum(torch.nn.functional.relu(1e-6 - self.kernel)))
            self.losses.append(neg_loss)
        return output


class model_gnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(model_gnn, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        ###################################
        self.gcn1_p = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_p = GCN(hidden_dim, hidden_dim, 0.2)
        # self.gcn3_p = GCN(in_dim, hidden_dim, 0.2)
        self.gcn1_n = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_n = GCN(hidden_dim, hidden_dim, 0.2)
        # self.gcn3_n = GCN(in_dim, hidden_dim, 0.2)
        # ----------------------------------
        self.gcn1_p_1 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_p_1 = GCN(hidden_dim, hidden_dim, 0.2)
        # self.gcn3_p_1 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn1_n_1 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_n_1 = GCN(hidden_dim, hidden_dim, 0.2)
        # self.gcn3_n_1 = GCN(in_dim, hidden_dim, 0.2)
        # ----------------------------------
        self.gcn1_p_2 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_p_2 = GCN(hidden_dim, hidden_dim, 0.2)
        # self.gcn3_p_2 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn1_n_2 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_n_2 = GCN(hidden_dim, hidden_dim, 0.2)
        # self.gcn3_n_2 = GCN(in_dim, hidden_dim, 0.2)
        # ---------------------------------
        self.gcn_p_shared = GCN(in_dim, hidden_dim, 0.2)
        self.gcn_n_shared = GCN(in_dim, hidden_dim, 0.2)
        self.gcn_p_shared1 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn_n_shared1 = GCN(in_dim, hidden_dim, 0.2)
        # --------------------------------- ATT score
        self.Wp_1 = nn.Linear(self.hidden_dim, 1)
        self.Wp_2 = nn.Linear(self.hidden_dim, 1)
        self.Wp_3 = nn.Linear(self.hidden_dim, 1)
        self.Wn_1 = nn.Linear(self.hidden_dim, 1)
        self.Wn_2 = nn.Linear(self.hidden_dim, 1)
        self.Wn_3 = nn.Linear(self.hidden_dim, 1)
        ###################################
        self.kernel_p = nn.Parameter(torch.FloatTensor(dim, in_dim))  #
        self.kernel_n = nn.Parameter(torch.FloatTensor(dim, in_dim))
        self.kernel_p1 = nn.Parameter(torch.FloatTensor(dim, in_dim))  #
        self.kernel_n1 = nn.Parameter(torch.FloatTensor(dim, in_dim))
        self.kernel_p2 = nn.Parameter(torch.FloatTensor(dim, in_dim))  #
        self.kernel_n2 = nn.Parameter(torch.FloatTensor(dim, in_dim))
        # print(self.kernel_p)
        # self.kernel_p = Variable(torch.randn(116, 5)).cuda()  # 116 5
        # self.kernel_n = Variable(torch.randn(116, 5)).cuda()   # 116 5
        ################################################
        self.lin1 = nn.Linear(2 * in_dim * in_dim, 16)
        self.lin2 = nn.Linear(16, self.out_dim)
        self.lin1_1 = nn.Linear(2 * in_dim * in_dim, 16)
        self.lin2_1 = nn.Linear(16, self.out_dim)
        self.lin1_2 = nn.Linear(2 * in_dim * in_dim, 16)
        self.lin2_2 = nn.Linear(16, self.out_dim)
        self.losses = []
        self.losses1 = []
        self.losses2 = []
        self.reset_weigths()

    def dim_reduce(self, adj_matrix, num_reduce,
                   ortho_penalty, variance_penalty, neg_penalty, kernel, tell=None):
        kernel_p = torch.nn.functional.relu(kernel)
        batch_size = int(adj_matrix.shape[0])
        AF = torch.tensordot(adj_matrix, kernel_p, [[-1], [0]])
        reduced_adj_matrix = torch.transpose(
            torch.tensordot(kernel_p, AF, [[0], [1]]),  # num_reduce*batch*num_reduce
            1, 0)  # num_reduce*batch*num_reduce*num_reduce
        kernel_p_tran = kernel_p.transpose(-1, -2)  # num_reduce * column_dim
        gram_matrix = torch.matmul(kernel_p_tran, kernel_p)
        diag_elements = gram_matrix.diag()

        if tell == 'A':
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
        elif tell == 'A1':
            if ortho_penalty != 0:
                ortho_loss_matrix = torch.square(gram_matrix - torch.diag(diag_elements))
                ortho_loss = torch.multiply(torch.tensor(ortho_penalty), torch.sum(ortho_loss_matrix))
                self.losses1.append(ortho_loss)

            if variance_penalty != 0:
                variance = diag_elements.var()
                variance_loss = torch.multiply(torch.tensor(variance_penalty), variance)
                self.losses1.append(variance_loss)

            if neg_penalty != 0:
                neg_loss = torch.multiply(torch.tensor(neg_penalty),
                                          torch.sum(torch.nn.functional.relu(torch.tensor(1e-6) - kernel)))
                self.losses1.append(neg_loss)
            self.losses1.append(0.05 * torch.sum(torch.abs(kernel_p)))
        elif tell == 'A2':
            if ortho_penalty != 0:
                ortho_loss_matrix = torch.square(gram_matrix - torch.diag(diag_elements))
                ortho_loss = torch.multiply(torch.tensor(ortho_penalty), torch.sum(ortho_loss_matrix))
                self.losses2.append(ortho_loss)

            if variance_penalty != 0:
                variance = diag_elements.var()
                variance_loss = torch.multiply(torch.tensor(variance_penalty), variance)
                self.losses2.append(variance_loss)

            if neg_penalty != 0:
                neg_loss = torch.multiply(torch.tensor(neg_penalty),
                                          torch.sum(torch.nn.functional.relu(torch.tensor(1e-6) - kernel)))
                self.losses2.append(neg_loss)
            self.losses2.append(0.05 * torch.sum(torch.abs(kernel_p)))
        return reduced_adj_matrix

    def reset_weigths(self):
        """reset weights
            """
        stdv = 1.0 / math.sqrt(dim)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, A, A1, A2):
        ##############################3
        A = torch.transpose(A, 1, 0)
        s_feature_p = A[0]
        s_feature_n = A[1]
        A1 = torch.transpose(A1, 1, 0)
        s_feature_p1 = A1[0]
        s_feature_n1 = A1[1]
        A2 = torch.transpose(A2, 1, 0)
        s_feature_p2 = A2[0]
        s_feature_n2 = A2[1]
        ###############################

        ###############################
        p_reduce = self.dim_reduce(s_feature_p, 10, 0.2, 0.3, 0.1, self.kernel_p, tell='A')
        # p_conv1_1_shared = self.gcn_p_shared(None, p_reduce) # shared GCN
        p_conv1 = self.gcn1_p(None, p_reduce)
        # p_conv1 = p_conv1 + p_conv1_1_shared # sum
        # p_conv1 = torch.cat((p_conv1, p_conv1_1_shared), -1) # concat
        # p_conv2_1_shared = self.gcn_p_shared1(p_conv1, p_reduce)
        p_conv2 = self.gcn2_p(p_conv1, p_reduce)
        # p_conv2_1_shared = self.gcn_p_shared1(p_conv2, p_reduce)
        # p_conv2 = p_conv2 + p_conv2_1_shared
        # p_conv3 = self.gcn3_p(p_conv2, p_reduce)
        n_reduce = self.dim_reduce(s_feature_n, 10, 0.2, 0.5, 0.1, self.kernel_n, tell='A')
        # n_conv1_1_shared = self.gcn_n_shared(None, n_reduce) # shared GCN
        n_conv1 = self.gcn1_n(None, n_reduce)
        # n_conv1 = n_conv1 + n_conv1_1_shared # sum
        # n_conv1 = torch.cat((n_conv1, n_conv1_1_shared), -1) # concat
        n_conv2 = self.gcn2_n(n_conv1, n_reduce)
        # n_conv2_1_shared = self.gcn_n_shared1(n_conv1, n_reduce)
        # n_conv2_1_shared = self.gcn_n_shared1(n_conv1, n_reduce)
        # n_conv2 = n_conv2 + n_conv2_1_shared
        # n_conv3 = self.gcn3_n(n_conv2, n_reduce)
        # ---------------------------------
        p_reduce1 = self.dim_reduce(s_feature_p1, 10, 0.2, 0.3, 0.1, self.kernel_p1, tell='A1')
        # p_conv1_2_shared = self.gcn_p_shared(None, p_reduce1) # shared GCN
        p_conv1_1 = self.gcn1_p_1(None, p_reduce1)
        # p_conv1_1 = p_conv1_1 + p_conv1_2_shared # sum
        # p_conv1_1 = torch.cat((p_conv1_1, p_conv1_2_shared), -1) # concat
        p_conv2_1 = self.gcn2_p_1(p_conv1_1, p_reduce1)
        # p_conv2_2_shared = self.gcn_p_shared1(p_conv1_1, p_reduce1)
        # p_conv2_2_shared = self.gcn_p_shared1(p_conv1_1, p_reduce1)
        # p_conv2_1 = p_conv2_1 + p_conv2_2_shared
        # p_conv3 = self.gcn3_p(p_conv2, p_reduce)
        n_reduce1 = self.dim_reduce(s_feature_n1, 10, 0.2, 0.5, 0.1, self.kernel_n1, tell='A1')
        # n_conv1_2_shared = self.gcn_n_shared(None, n_reduce1) # shared GCN
        n_conv1_1 = self.gcn1_n_1(None, n_reduce1)
        # n_conv1_1 = n_conv1_1 + n_conv1_2_shared # sum
        # n_conv1_1 = torch.cat((n_conv1_1, n_conv1_2_shared), -1) #concat
        n_conv2_1 = self.gcn2_n_1(n_conv1_1, n_reduce1)
        # n_conv2_2_shared = self.gcn_n_shared1(n_conv1_1, n_reduce1)
        # n_conv2_2_shared = self.gcn_n_shared1(n_conv1_1, n_reduce1)
        # n_conv2_1 = n_conv2_1 + n_conv2_2_shared
        # n_conv3 = self.gcn3_n(n_conv2, n_reduce)
        # ---------------------------------
        p_reduce2 = self.dim_reduce(s_feature_p2, 10, 0.2, 0.3, 0.1, self.kernel_p2, tell='A2')
        # p_conv1_3_shared = self.gcn_p_shared(None, p_reduce2) # shared GCN
        p_conv1_2 = self.gcn1_p_2(None, p_reduce2)
        # p_conv1_2 = p_conv1_2 + p_conv1_3_shared # sum
        # p_conv1_2 = torch.cat((p_conv1_2, p_conv1_3_shared), -1) # concat
        p_conv2_2 = self.gcn2_p_2(p_conv1_2, p_reduce2)
        # p_conv2_3_shared = self.gcn_p_shared1(p_conv1_2, p_reduce2)
        # p_conv2_3_shared = self.gcn_p_shared1(p_conv1_2, p_reduce2)
        # p_conv2_2 = p_conv2_2 + p_conv2_3_shared
        # p_conv3 = self.gcn3_p(p_conv2, p_reduce)
        n_reduce2 = self.dim_reduce(s_feature_n2, 10, 0.2, 0.5, 0.1, self.kernel_n2, tell='A2')
        # n_conv1_3_shared = self.gcn_n_shared(None, n_reduce2)
        n_conv1_2 = self.gcn1_n_2(None, n_reduce2)
        # n_conv1_2 = n_conv1_2 + n_conv1_3_shared # sum
        # n_conv1_2 = torch.cat((n_conv1_2, n_conv1_3_shared), -1) # concat
        n_conv2_2 = self.gcn2_n_2(n_conv1_2, n_reduce2)
        # n_conv2_3_shared = self.gcn_n_shared1(n_conv1_2, n_reduce2)
        # n_conv2_3_shared = self.gcn_n_shared1(n_conv1_2, n_reduce2)
        # n_conv2_2 = n_conv2_2 + n_conv2_3_shared
        # n_conv3 = self.gcn3_n(n_conv2, n_reduce)
        # ----------------------------------
        # p_conv1_1_shared = self.gcn_p_shared(None, p_reduce)
        # p_conv1_2_shared = self.gcn_p_shared(None, p_reduce1)
        # p_conv1_3_shared = self.gcn_p_shared(None, p_reduce2)
        #
        # n_conv1_1_shared = self.gcn_n_shared(None, n_reduce)
        # n_conv1_2_shared = self.gcn_n_shared(None, n_reduce1)
        # n_conv1_3_shared = self.gcn_n_shared(None, n_reduce2)
        # -----------------------------------
        # p_conv = p_conv2 + p_conv2_1 + p_conv2_2
        # n_conv = n_conv2 + n_conv2_1 + n_conv2_2
        ##################################

        # conv_concat = torch.cat([p_conv2, n_conv2], -1).reshape([-1, 2 * self.in_dim * self.in_dim])
        conv_concat = torch.cat([p_conv2, n_conv2], -1).reshape([-1, 2 * self.in_dim * self.in_dim])
        conv_concat1 = torch.cat([p_conv2_1, n_conv2_1], -1).reshape([-1, 2 * self.in_dim * self.in_dim])
        conv_concat2 = torch.cat([p_conv2_2, n_conv2_2], -1).reshape([-1, 2 * self.in_dim * self.in_dim])
        output = self.lin2(self.lin1(conv_concat))
        output1 = self.lin2_1(self.lin1_1(conv_concat1))
        output2 = self.lin2_2(self.lin1_2(conv_concat2))
        # output = torch.softmax(output, dim=1)
        loss = torch.sum(torch.tensor(self.losses))
        self.losses.clear()
        loss1 = torch.sum(torch.tensor(self.losses1))
        self.losses1.clear()
        loss2 = torch.sum(torch.tensor(self.losses2))
        self.losses2.clear()
        return output, output1, output2, loss, loss1, loss2

def evaluate(dataset, model, name='Validation', max_num_examples=None, device='cpu'):
    model.eval()
    avg_loss = 0.0
    preds = []
    preds1 = []
    preds2 = []
    # probs
    labels = []
    ypreds = []
    ypreds1 = []
    ypreds2 = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataset):
            adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
            adj1 = Variable(data['adj1'].to(torch.float32), requires_grad=False).to(device)
            adj2 = Variable(data['adj2'].to(torch.float32), requires_grad=False).to(device)
            label = Variable(data['label'].long()).to(device)
            labels.append(data['label'].long().numpy())
            ypred, ypred1, ypred2, losses, losses1, losses2 = model(adj, adj1, adj2)
            loss = F.cross_entropy(ypred, label, size_average=True)
            loss1 = F.cross_entropy(ypred1, label, size_average=True)
            loss2 = F.cross_entropy(ypred2, label, size_average=True)
            loss += losses
            loss1 += losses1
            loss2 += losses2
            for i in ypred:
                ypreds.append(np.array(i.cpu()))
            for i in ypred1:
                ypreds1.append(np.array(i.cpu()))
            for i in ypred2:
                ypreds2.append(np.array(i.cpu()))

            avg_loss += loss
            _, indices = torch.max(ypred, 1)
            preds.append(indices.cpu().data.numpy())
            _, indices = torch.max(ypred1, 1)
            preds1.append(indices.cpu().data.numpy())
            _, indices = torch.max(ypred2, 1)
            preds2.append(indices.cpu().data.numpy())

            if max_num_examples is not None:
                if (batch_idx + 1) * 32 > max_num_examples:
                    break
    avg_loss /= batch_idx + 1

    ypres_all = np.array(ypreds)
    ypres_all1 = np.array(ypreds1)
    ypres_all2 = np.array(ypreds2)

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    preds1 = np.hstack(preds1)
    preds2 = np.hstack(preds2)

    global xx
    global yy
    from sklearn.metrics import confusion_matrix
    auc = metrics.roc_auc_score(labels, preds, average='macro', sample_weight=None)
    result = {'prec': metrics.precision_score(labels, preds),
              'recall': metrics.recall_score(labels, preds),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="macro"),
              'auc': auc,
              'matrix': confusion_matrix(labels, preds)}
    auc = metrics.roc_auc_score(labels, preds1, average='macro', sample_weight=None)
    result1 = {'prec': metrics.precision_score(labels, preds1),
              'recall': metrics.recall_score(labels, preds1),
              'acc': metrics.accuracy_score(labels, preds1),
              'F1': metrics.f1_score(labels, preds1, average="macro"),
              'auc': auc,
              'matrix': confusion_matrix(labels, preds1)}
    auc = metrics.roc_auc_score(labels, preds2, average='macro', sample_weight=None)
    result2 = {'prec': metrics.precision_score(labels, preds2),
              'recall': metrics.recall_score(labels, preds2),
              'acc': metrics.accuracy_score(labels, preds2),
              'F1': metrics.f1_score(labels, preds2, average="macro"),
              'auc': auc,
              'matrix': confusion_matrix(labels, preds2)}
    xx = preds
    yy = labels

    return avg_loss, result, result1, result2, ypres_all, ypres_all1, ypres_all2


def evaluate_all(dataset, preds):
    labels = []

    with torch.no_grad():
        for batch_idx, data in enumerate(dataset):
            labels.append(data['label'].long().numpy())

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    global xx
    global yy
    from sklearn.metrics import confusion_matrix
    auc = metrics.roc_auc_score(labels, preds, average='macro', sample_weight=None)
    result = {'prec': metrics.precision_score(labels, preds),
              'recall': metrics.recall_score(labels, preds),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="macro"),
              'auc': auc,
              'matrix': confusion_matrix(labels, preds)}
    xx = preds
    yy = labels

    return result, preds


result = [0, 0, 0, 0, 0]
ii = 0


def train(dataset, model, val_dataset=None, test_dataset=None,
          device='cpu', phi=None, e=200, supernode=8):
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001,
    #                              weight_decay=0.0001)
    # optimizer1 = torch.optim.Adam([
    #     {'params': model.gcn1_p.parameters(), 'lr': 0.005},
    #     {'params': model.gcn2_p.parameters(), 'lr': 0.005},
    #     {'params': model.gcn1_n.parameters(), 'lr': 0.005},
    #     {'params': model.gcn2_n.parameters(), 'lr': 0.005},
    #     {'params': model.gcn3_n.parameters(), 'lr': 0.005},
    #     {'params': model.gcn3_p.parameters(), 'lr': 0.005}
    # ])
    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001,
                                  weight_decay=0.001)
    cosinLR2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=20, eta_min=0.000001)
    for name in model.state_dict():
        print(name)
    iter = 0
    best_val_acc = 0.0
    early_stopping = EarlyStopping(patience=40, verbose=True)
    bestVal = []
    best = 0
    global ii
    ii += 1

    for epoch in range(e):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        print(epoch)
        for batch_idx, data in enumerate(dataset):
            if epoch < 0:
                for k, v in model.named_parameters():
                    if k != 'gcn1_p.kernel' and k != 'gcn2_p.kernel' and k != 'gcn3_p.kernel' and k != 'gcn1_n.kernel' and k != 'gcn2_n.kernel' and k != 'gcn3_n.kernel':
                        v.requires_grad = False  # 
                time1 = time.time()
                model.zero_grad()
                adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
                label = Variable(data['label'].long()).to(device)
                pred, losses = model(adj)
                loss = F.cross_entropy(pred, label, size_average=True)
                loss += losses
                loss.backward()
                time3 = time.time()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer1.step()
                iter += 1
                avg_loss += loss
            else:
                for k, v in model.named_parameters():
                    v.requires_grad = True
                time1 = time.time()
                model.zero_grad()
                adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
                adj1 = Variable(data['adj1'].to(torch.float32), requires_grad=False).to(device)
                adj2 = Variable(data['adj2'].to(torch.float32), requires_grad=False).to(device)
                label = Variable(data['label'].long()).to(device)
                pred, pred1, pred2, losses, losses1, losses2 = model(adj, adj1, adj2)
                loss = F.cross_entropy(pred, label, size_average=True)
                loss1 = F.cross_entropy(pred1, label, size_average=True)
                loss2 = F.cross_entropy(pred2, label, size_average=True)
                loss += losses
                loss1 += losses1
                loss2 += losses2

                loss = loss + loss1 + loss2

                loss.backward()
                # loss1.backward()
                # loss2.backward()
                time3 = time.time()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer2.step()
                # cosinLR2.step()
                iter += 1
                avg_loss += loss
                # avg_loss += loss1
                # avg_loss += loss2
        avg_loss /= batch_idx + 1
        # print(avg_loss)
        eval_time = time.time()
        if val_dataset is not None:
            _, train_result, train_result1, train_result2, _, _, _ = evaluate(dataset, model, name='Train', device=device)
            val_loss, val_result, val_result1, val_result2, _, _, _ = evaluate(val_dataset, model, name='Validation', device=device)
            # print('',)
            print('train1', train_result)
            print('val1', val_result)
            print('train2', train_result1)
            print('val2', val_result1)
            print('train3', train_result2)
            print('val3', val_result2)
            writer1.add_scalar('acc' + str(ii), val_result['acc'], global_step=epoch)
            writer1.add_scalar('loss' + str(ii), val_loss, global_step=epoch)
            writer1.add_scalar('train_loss' + str(ii), avg_loss, global_step=epoch)
            # if val_result['acc'] >= best_val_acc:
            #     torch.save(model.state_dict(), './GroupINN_model/checkpoint' + str(phi) + '_' + str(supernode) + '_' + str(ii) + '.pt')
            #
            #     torch.save(model.state_dict(), 'checkpoint' + str(ii) + '.pt')
            #     # print('save pth.......')
            #     best_val_acc = val_result['acc']
            #     # print('val          ', val_result)
            #     bestVal = val_result
            #     best = epoch
            torch.save(model.state_dict(), './GroupINN_model/checkpoint' + str(phi) + '_' + str(supernode) + '_' + str(ii) + '.pt')
        # early_stopping(0-val_result['acc'], model)
        # if test_dataset is not None:
        #     test_loss, test_result = evaluate(test_dataset, model, name='Validation', device=device)
        #     print('test         ', test_result)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
    print(bestVal)
    print(best)
    # model.load_state_dict(torch.load('./GroupINN_model/checkpoint' + str(phi) + '_' + str(ii) + '.pt'))
    return model


###########################################################################################
###########################################################################################

# 


def saveList(paraList, path):
    output = open(path, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(paraList, output)
    output.close()


def loadList(path):
    pkl_file = open(path, 'rb')
    segContent = pickle.load(pkl_file)
    pkl_file.close()
    return segContent


def main():
    data_path = 'cc200/'
    label_path = 'Phenotypic_V1_0b_preprocessed1.csv'

    # 
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 
    _, _, raw_data, labels = load_data(data_path, label_path)  # raw_data [871 116 ?]  labels [871]
    ind = 0
    for i in range(871):
        if labels[i] == 0:
            ind += 1
    print(ind)
    # 

    print('finished')
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print('Device: ', device)

    print('process the A')
    adj = cal_pcc(raw_data, 0.6)
    print('0.6 Done')
    adj1 = cal_pcc(raw_data, 0.4)
    print('0.4 Done')
    adj2 = cal_pcc(raw_data, 0.65)
    print('0.65 Done')
    # adj3 = cal_pcc(raw_data, 0.55)
    # print('0.55 Done')
    # adj4 = cal_pcc(raw_data, 0.5)
    # print('0.5 Done')
    # adj5 = cal_pcc(raw_data, 0.45)
    # print('0.45 Done')
    train_data_loaders, valid_data_loaders, test_data_loaders = cross_val(adj, adj1, adj2, labels)

    result = []
    result1 = []
    result2 = []
    pres = []
    jj=1
    for i in range(len(train_data_loaders)):
        model = model_gnn(8, 8, 2)
        # model.load_state_dict(torch.load('GroupINN_model/checkpoint0.6_' + str(jj) + '.pt'))
        jj+=1
        model.to(device)
        print('model:', model)
        model = train(train_data_loaders[i], model, val_dataset=valid_data_loaders[i],
                      test_dataset=test_data_loaders[i], device=device, phi=0.6, e=200, supernode=8)
        dir = './params' + str(i) + str(8) + '.pth'
        torch.save(model.state_dict(), dir)
        _, test_result, test_result1, test_result2, pre, pre1, pre2 = evaluate(test_data_loaders[i], model, name='Test', device=device)
        pre_all = pre + pre1 + pre2
        pres.append(pre_all)
        print('test1', test_result)
        print('test2', test_result1)
        print('test3', test_result2)
        result.append(test_result)
        result1.append(test_result1)
        result2.append(test_result2)
        del model
        del test_result
        del test_result1
        del test_result2
    print(result)
    print(result1)
    print(result2)
    print('------------------------------------')
    for i in range(len(train_data_loaders)):
        pres[i] = pres[i].tolist()
        temp = np.array(pres[i])
        res =  np.argmax(temp, axis=1)
        # for j in range(len(pres[i])):
        #     if pres[i][j] >= 3:
        #         pres[i][j] = 1
        #     else:
        #         pres[i][j] = 0
        pres[i] = np.array(res)
        test_result, _ = evaluate_all(test_data_loaders[i], pres[i])
        print(test_result)
        result.append(test_result)
    quit()

    # result = []
    # result1 = []
    # result2 = []
    for i in range(len(train_data_loaders)):
        model = model_gnn(7, 7, 2)
        # model.load_state_dict(torch.load('GroupINN_model/checkpoint0.6_' + str(jj) + '.pt'))
        jj+=1
        model.to(device)
        print('model:', model)
        model = train(train_data_loaders[i], model, val_dataset=valid_data_loaders[i],
                      test_dataset=test_data_loaders[i], device=device, phi=0.6, e=200, supernode=7)
        dir = './params' + str(i) + str(6) + '.pth'
        torch.save(model.state_dict(), dir)
        _, test_result, test_result1, test_result2, pre, pre1, pre2 = evaluate(test_data_loaders[i], model, name='Test', device=device)
        pre_all = pre + pre1 + pre2
        pres[i] += pre_all
        print('test1', test_result)
        print('test2', test_result1)
        print('test3', test_result2)
        result.append(test_result)
        result1.append(test_result1)
        result2.append(test_result2)
        del model
        del test_result
        del test_result1
        del test_result2
    print(result)
    print(result1)
    print(result2)


    # result = []
    # result1 = []
    # result2 = []
    for i in range(len(train_data_loaders)):
        model = model_gnn(9, 9, 2)
        # model.load_state_dict(torch.load('GroupINN_model/checkpoint0.6_' + str(jj) + '.pt'))
        jj += 1
        model.to(device)
        print('model:', model)
        model = train(train_data_loaders[i], model, val_dataset=valid_data_loaders[i],
                      test_dataset=test_data_loaders[i], device=device, phi=0.6, e=200, supernode=9)
        dir = './params' + str(i) + str(6) + '.pth'
        torch.save(model.state_dict(), dir)
        _, test_result, test_result1, test_result2, pre, pre1, pre2 = evaluate(test_data_loaders[i], model, name='Test',
                                                                               device=device)
        pre_all = pre + pre1 + pre2
        pres[i] += pre_all
        print('test1', test_result)
        print('test2', test_result1)
        print('test3', test_result2)
        result.append(test_result)
        result1.append(test_result1)
        result2.append(test_result2)
        del model
        del test_result
        del test_result1
        del test_result2
    print(result)
    print(result1)
    print(result2)

    print('------------------------------------')
    for i in range(len(train_data_loaders)):
        pres[i] = pres[i].tolist()
        temp = np.array(pres[i])
        res =  np.argmax(temp, axis=1)
        # for j in range(len(pres[i])):
        #     if pres[i][j] >= 3:
        #         pres[i][j] = 1
        #     else:
        #         pres[i][j] = 0
        pres[i] = np.array(res)
        test_result, _ = evaluate_all(test_data_loaders[i], pres[i])
        print(test_result)
    quit()
    for i in range(len(train_data_loaders)):
        model = model_gnn(9, 9, 2)
        # model.load_state_dict(torch.load('GroupINN_model/checkpoint0.6_' + str(jj) + '.pt'))
        jj+=1
        model.to(device)
        print('model:', model)
        model = train(train_data_loaders[i], model, val_dataset=valid_data_loaders[i],
                      test_dataset=test_data_loaders[i], device=device, phi=0.6, e=200, supernode=9)
        # model = train(train_data_loaders1[i], model, val_dataset=valid_data_loaders1[i],
        #               test_dataset=test_data_loaders1[i], device='cuda')
        # model = train(train_data_loaders2[i], model, val_dataset=valid_data_loaders2[i],
        #               test_dataset=test_data_loaders2[i], device='cuda')
        # model = train(train_data_loaders3[i], model, val_dataset=valid_data_loaders3[i],
        #               test_dataset=test_data_loaders3[i], device='cuda')
        dir = './params' + str(i) + str(10) + '.pth'
        torch.save(model.state_dict(), dir)
        _, test_result, pre = evaluate(test_data_loaders[i], model, name='Test', device=device)
        # pres.append(pre)
        pres[i] += pre
        print(test_result)
        result.append(test_result)
        del model
        del test_result
    print(result)

    for i in range(len(train_data_loaders)):
        model = model_gnn(7, 7, 2)
        # model.load_state_dict(torch.load('GroupINN_model/checkpoint0.6_' + str(jj) + '.pt'))
        jj+=1
        model.to(device)
        print('model:', model)
        model = train(train_data_loaders[i], model, val_dataset=valid_data_loaders[i],
                      test_dataset=test_data_loaders[i], device=device, phi=0.6, e=200, supernode=7)
        # model = train(train_data_loaders1[i], model, val_dataset=valid_data_loaders1[i],
        #               test_dataset=test_data_loaders1[i], device='cuda')
        # model = train(train_data_loaders2[i], model, val_dataset=valid_data_loaders2[i],
        #               test_dataset=test_data_loaders2[i], device='cuda')
        # model = train(train_data_loaders3[i], model, val_dataset=valid_data_loaders3[i],
        #               test_dataset=test_data_loaders3[i], device='cuda')
        dir = './params' + str(i) + str(5) + '.pth'
        torch.save(model.state_dict(), dir)
        _, test_result, pre = evaluate(test_data_loaders[i], model, name='Test', device=device)
        # pres.append(pre)
        pres[i] += pre
        print(test_result)
        result.append(test_result)
        del model
        del test_result
    print(result)

    adj = cal_pcc(raw_data, 0.55)
    train_data_loaders, valid_data_loaders, test_data_loaders = cross_val(adj, labels)
    for i in range(len(train_data_loaders)):
        model = model_gnn(8, 8, 2)
        # model.load_state_dict(torch.load('GroupINN_model/checkpoint0.55_' + str(jj) + '.pt'))
        jj+=1
        model.to(device)
        print('model:', model)
        model = train(train_data_loaders[i], model, val_dataset=valid_data_loaders[i],
                      test_dataset=test_data_loaders[i], device=device, phi=0.55, e=200, supernode=8)
        # model = train(train_data_loaders1[i], model, val_dataset=valid_data_loaders1[i],
        #               test_dataset=test_data_loaders1[i], device='cuda')
        # model = train(train_data_loaders2[i], model, val_dataset=valid_data_loaders2[i],
        #               test_dataset=test_data_loaders2[i], device='cuda')
        # model = train(train_data_loaders3[i], model, val_dataset=valid_data_loaders3[i],
        #               test_dataset=test_data_loaders3[i], device='cuda')
        dir = './params' + str(i) + str(8) + '.pth'
        torch.save(model.state_dict(), dir)
        _, test_result, pre = evaluate(test_data_loaders[i], model, name='Test', device=device)
        pres[i] += pre
        print(test_result)
        result.append(test_result)
    print(result)

    for i in range(len(train_data_loaders)):
        model = model_gnn(9, 9, 2)
        # model.load_state_dict(torch.load('GroupINN_model/checkpoint0.55_' + str(jj) + '.pt'))
        jj+=1
        model.to(device)
        print('model:', model)
        model = train(train_data_loaders[i], model, val_dataset=valid_data_loaders[i],
                      test_dataset=test_data_loaders[i], device=device, phi=0.55, e=200, supernode=9)
        # model = train(train_data_loaders1[i], model, val_dataset=valid_data_loaders1[i],
        #               test_dataset=test_data_loaders1[i], device='cuda')
        # model = train(train_data_loaders2[i], model, val_dataset=valid_data_loaders2[i],
        #               test_dataset=test_data_loaders2[i], device='cuda')
        # model = train(train_data_loaders3[i], model, val_dataset=valid_data_loaders3[i],
        #               test_dataset=test_data_loaders3[i], device='cuda')
        dir = './params' + str(i) + str(10) + '.pth'
        torch.save(model.state_dict(), dir)
        _, test_result, pre = evaluate(test_data_loaders[i], model, name='Test', device=device)
        pres[i] += pre
        print(test_result)
        result.append(test_result)
    print(result)

    for i in range(len(train_data_loaders)):
        model = model_gnn(7, 7, 2)
        # model.load_state_dict(torch.load('GroupINN_model/checkpoint0.55_' + str(jj) + '.pt'))
        jj+=1
        model.to(device)
        print('model:', model)
        model = train(train_data_loaders[i], model, val_dataset=valid_data_loaders[i],
                      test_dataset=test_data_loaders[i], device=device, phi=0.55, e=200, supernode=7)
        # model = train(train_data_loaders1[i], model, val_dataset=valid_data_loaders1[i],
        #               test_dataset=test_data_loaders1[i], device='cuda')
        # model = train(train_data_loaders2[i], model, val_dataset=valid_data_loaders2[i],
        #               test_dataset=test_data_loaders2[i], device='cuda')
        # model = train(train_data_loaders3[i], model, val_dataset=valid_data_loaders3[i],
        #               test_dataset=test_data_loaders3[i], device='cuda')
        dir = './params' + str(i) + str(8) + '.pth'
        torch.save(model.state_dict(), dir)
        _, test_result, pre = evaluate(test_data_loaders[i], model, name='Test', device=device)
        pres[i] += pre
        print(test_result)
        result.append(test_result)
    print(result)

    adj = cal_pcc(raw_data, 0.65)
    train_data_loaders, valid_data_loaders, test_data_loaders = cross_val(adj, labels)
    for i in range(len(train_data_loaders)):
        model = model_gnn(8, 8, 2)
        # model.load_state_dict(torch.load('GroupINN_model/checkpoint/checkpoint0.65_' + str(jj-5) + '.pt'))
        jj+=1
        model.to(device)
        print('model:', model)
        model = train(train_data_loaders[i], model, val_dataset=valid_data_loaders[i],
                      test_dataset=test_data_loaders[i], device=device, phi=0.65, e=200, supernode=8)
        # model = train(train_data_loaders1[i], model, val_dataset=valid_data_loaders1[i],
        #               test_dataset=test_data_loaders1[i], device='cuda')
        # model = train(train_data_loaders2[i], model, val_dataset=valid_data_loaders2[i],
        #               test_dataset=test_data_loaders2[i], device='cuda')
        # model = train(train_data_loaders3[i], model, val_dataset=valid_data_loaders3[i],
        #               test_dataset=test_data_loaders3[i], device='cuda')
        dir = './params' + str(i) + str(8) + '.pth'
        torch.save(model.state_dict(), dir)
        _, test_result, pre = evaluate(test_data_loaders[i], model, name='Test', device=device)
        pres[i] += pre
        print(test_result)
        result.append(test_result)
    print(result)

    for i in range(len(train_data_loaders)):
        model = model_gnn(9, 9, 2)
        # model.load_state_dict(torch.load('GroupINN_model/checkpoint/checkpoint0.65_' + str(jj-5) + '.pt'))
        jj+=1
        model.to(device)
        print('model:', model)
        model = train(train_data_loaders[i], model, val_dataset=valid_data_loaders[i],
                      test_dataset=test_data_loaders[i], device=device, phi=0.65, e=200, supernode=9)
        # model = train(train_data_loaders1[i], model, val_dataset=valid_data_loaders1[i],
        #               test_dataset=test_data_loaders1[i], device='cuda')
        # model = train(train_data_loaders2[i], model, val_dataset=valid_data_loaders2[i],
        #               test_dataset=test_data_loaders2[i], device='cuda')
        # model = train(train_data_loaders3[i], model, val_dataset=valid_data_loaders3[i],
        #               test_dataset=test_data_loaders3[i], device='cuda')
        dir = './params' + str(i) + str(8) + '.pth'
        torch.save(model.state_dict(), dir)
        _, test_result, pre = evaluate(test_data_loaders[i], model, name='Test', device=device)
        pres[i] += pre
        print(test_result)
        result.append(test_result)
    print(result)

    for i in range(len(train_data_loaders)):
        model = model_gnn(7, 7, 2)
        # model.load_state_dict(torch.load('GroupINN_model/checkpoint/checkpoint0.65_' + str(jj-5) + '.pt'))
        jj+=1
        model.to(device)
        print('model:', model)
        model = train(train_data_loaders[i], model, val_dataset=valid_data_loaders[i],
                      test_dataset=test_data_loaders[i], device=device, phi=0.65, e=200, supernode=7)
        # model = train(train_data_loaders1[i], model, val_dataset=valid_data_loaders1[i],
        #               test_dataset=test_data_loaders1[i], device='cuda')
        # model = train(train_data_loaders2[i], model, val_dataset=valid_data_loaders2[i],
        #               test_dataset=test_data_loaders2[i], device='cuda')
        # model = train(train_data_loaders3[i], model, val_dataset=valid_data_loaders3[i],
        #               test_dataset=test_data_loaders3[i], device='cuda')
        dir = './params' + str(i) + str(8) + '.pth'
        torch.save(model.state_dict(), dir)
        _, test_result, pre = evaluate(test_data_loaders[i], model, name='Test', device=device)
        pres[i] += pre
        print(test_result)
        result.append(test_result)
    print(result)
    # quit()
    #
    # adj = cal_pcc(raw_data, 0.45)
    # train_data_loaders, valid_data_loaders, test_data_loaders = cross_val(adj, labels)
    # for i in range(len(train_data_loaders)):
    #     model = model_gnn(8, 8, 2)
    #     model.load_state_dict(torch.load('GroupINN_model/checkpoint0.45_' + str(jj) + '.pt'))
    #     jj+=1
    #     model.to(device)
    #     print('model:', model)
    #     model = train(train_data_loaders[i], model, val_dataset=valid_data_loaders[i],
    #                   test_dataset=test_data_loaders[i], device=device, phi=0.45)
    #     # model = train(train_data_loaders1[i], model, val_dataset=valid_data_loaders1[i],
    #     #               test_dataset=test_data_loaders1[i], device='cuda')
    #     # model = train(train_data_loaders2[i], model, val_dataset=valid_data_loaders2[i],
    #     #               test_dataset=test_data_loaders2[i], device='cuda')
    #     # model = train(train_data_loaders3[i], model, val_dataset=valid_data_loaders3[i],
    #     #               test_dataset=test_data_loaders3[i], device='cuda')
    #     dir = './params' + str(i) + str(8) + '.pth'
    #     torch.save(model.state_dict(), dir)
    #     _, test_result, pre = evaluate(test_data_loaders[i], model, name='Test', device=device)
    #     pres[i] += pre
    #     print(test_result)
    #     result.append(test_result)
    # print(result)
    #
    # adj = cal_pcc(raw_data, 0.4)
    # train_data_loaders, valid_data_loaders, test_data_loaders = cross_val(adj, labels)
    # for i in range(len(train_data_loaders)):
    #     model = model_gnn(8, 8, 2)
    #     model.load_state_dict(torch.load('GroupINN_model/checkpoint0.4_' + str(jj) + '.pt'))
    #     jj+=1
    #     model.to(device)
    #     print('model:', model)
    #     model = train(train_data_loaders[i], model, val_dataset=valid_data_loaders[i],
    #                   test_dataset=test_data_loaders[i], device=device, phi=0.4)
    #     # model = train(train_data_loaders1[i], model, val_dataset=valid_data_loaders1[i],
    #     #               test_dataset=test_data_loaders1[i], device='cuda')
    #     # model = train(train_data_loaders2[i], model, val_dataset=valid_data_loaders2[i],
    #     #               test_dataset=test_data_loaders2[i], device='cuda')
    #     # model = train(train_data_loaders3[i], model, val_dataset=valid_data_loaders3[i],
    #     #               test_dataset=test_data_loaders3[i], device='cuda')
    #     dir = './params' + str(i) + str(8) + '.pth'
    #     torch.save(model.state_dict(), dir)
    #     _, test_result, pre = evaluate(test_data_loaders[i], model, name='Test', device=device)
    #     pres[i] += pre
    #     print(test_result)
    #     result.append(test_result)
    # print(result)

    for i in range(len(train_data_loaders)):
        pres[i] = pres[i].tolist()
        temp = np.array(pres[i])
        res =  np.argmax(temp, axis=1)
        # for j in range(len(pres[i])):
        #     if pres[i][j] >= 3:
        #         pres[i][j] = 1
        #     else:
        #         pres[i][j] = 0
        pres[i] = np.array(res)
        test_result, _ = evaluate_all(test_data_loaders[i], pres[i])
        print(test_result)
        result.append(test_result)

    saveList(result, '../result7.pickle')
    print(result)


if __name__ == "__main__":
    main()
