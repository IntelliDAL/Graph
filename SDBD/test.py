import numpy as np
import scipy.io as scio


import torch
 
# print(torch.__version__)

# a = np.load('./data/MDD_HC.npy')
# b = np.load('./data/MDD_HC_label.npy')
# b = b[15:-15]
# data_x = np.load('./data/MDD_HC_sig.npy')

# a = np.load('./data/BP_HC.npy')
# b = np.load('./data/BP_HC_label.npy')
# b = b[10:-10]
# data_x = np.load('./data/BP_HC_sig.npy')

# a = np.load('./data/635_8A.npy')
# b = np.load('./data/635y.npy')
# data_x = np.load('./data/BP_HC_sig.npy')
# data_x = data_x.transpose((0, 2, 1))
# cnt1, cnt0 = 0, 0
# for y in b:
#     if y == 1:
#         cnt1 += 1
#     else:
#         cnt0 += 1
# print(cnt1, cnt0, b.shape)

# print(b)


# A = np.empty([392, 18, 116, 116])
# for k,x in enumerate(data_x):
#     if k%10==0:
#         print(k)
#     for n, left in enumerate(range(0, 851, 50)):
#         for i in range(116):
#             for j in range(i, 116):
#                 if i==j:
#                     A[k][n][i][j] = 1
#                 else:
#                     A[k][n][i][j] = np.corrcoef(x[i,left:left+100], x[j,left:left+100])[0][1]
#                     A[k][n][j][i] = A[k][n][i][j]

# np.save('./data/BP_HC_392_18A.npy', A)

# data = scio.loadmat('./data/pcc_correlation_871_aal_.mat')

# print(data['connectivity'].shape)

# draw

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
 
# # 绘图设置
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(projection='3d')  # 三维坐标轴
# # X和Y的个数要相同
# X = [0.1,0.2,0.3,0.4,0.5]
# Y = [0.5,0.6,0.7,0.8,0.9]
# # Z = np.random.randint(0, 1000, 16) # 生成16个随机整数
# Z = [0.6724, 0.6755, 0.6629, 0.6866, 0.7041, 0.6755, 0.6755, 0.6755, 0.6771, 0.6629, 0.6787, 0.6771, 0.6834, 0.6850, 0.6692, 0.6755, 0.6771, 0.6818, 0.6614, 0.6724, 0.6771, 0.6724, 0.6803, 0.6645, 0.6866]
# # Z = [0.7147, 0.7335, 0.7335, 0.7335, 0.7600, 0.7255, 0.7335, 0.7335, 0.7335, 0.7282, 0.7335,  0.7335,  0.7282,  0.7335,  0.7335, 0.7282,  0.7147,  0.7335,  0.7335,  0.7335, 0.7524,  0.7335,  0.7473,  0.7335,  0.7335] BD
# # Z = [0.6978,  0.6953,  0.6877,  0.6850,  0.7480,  0.6876,  0.7230,  0.7105,  0.7228,  0.6929,  0.6903,  0.7254,  0.6978,  0.7055,  0.7102,  0.6675,  0.7253,  0.7003,  0.7028,  0.7231,  0.6953,  0.7179,  0.6904,  0.7053,  0.6850]
# # meshgrid把X和Y变成平方长度，比如原来都是4，经过meshgrid和ravel之后，长度都变成了16，因为网格点是16个
# xx, yy = np.meshgrid(X, Y)  # 网格化坐标
# X, Y = xx.ravel(), yy.ravel()  # 矩阵扁平化
# # 设置柱子属性
# # height = np.zeros_like(Z) # 新建全0数组，shape和Z相同，据说是图中底部的位置
# for i, z in enumerate(Z):
#     Z[i] -= 0.5
    
# height = [0.5]*25
# width = depth = 0.08 # 柱子的长和宽
# # 颜色数组，长度和Z一致
# c = ['r']*len(Z)
# ax.set_zlim3d(0.5, 0.7)
# # 开始画图，注意本来的顺序是X, Y, Z, width, depth, height，但是那样会导致不能形成柱子，只有柱子顶端薄片，所以Z和height要互换
# ax.bar3d(X, Y, height, width, depth, Z, shade=True)  # width, depth, height
# ax.set_xlabel('a')
# ax.set_ylabel('b')
# ax.set_zlabel('ACC')
# fig.savefig("test.png")
# plt.show()

# draw end


# t-sne

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import random
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold

from model import my_model
from utils import get_args
import logging

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def plot_embedding(result, label, path, index):
    # x_min, x_max = torch.min(result, 0), np.max(result, 0)  # 分别求出每一列最小值和最大值
    # data = (result - x_min) / (x_max - x_min)  # 将数据进行正则化，分母为数据的总长度，因此分子一定小于分母，生成的矩阵元素都是0-1区间内的
    # result = data
    plt.figure(figsize=(10, 8))  # 创建一个画布
    # central_kind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    central_kind = [0, 1]
    # central_kind = ['UCLA', 'UM', 'USM','NYU']
    data = {}
    for central_kind_ in range(len(central_kind)):
        data_ = []
        for i in range(label.shape[0]):
            if label[i] == central_kind_:
                # if -4 < result[i][0] < 5 and -4 < result[i][1] < 5:  # original graph
                    # if result[i][0] < 10 and result[i][1] < 25: #gen graph (2,3)
                    # if -200 <result[i][0] < 200 and -200 <result[i][1] < 250: #gen graph (3,4)
                    # data_.append(result[i])
                data_.append(result[i])
        data[central_kind_] = data_
    # colors = ['black', 'red', 'green', '#FF00FF', 'blue']
    colors = ['black', 'red', 'green', '#FF00FF', 'blue','black', 'red', 'green', '#FF00FF', 'blue']
    markers = [',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'P', '*', 'h', '+', '*', 'x']
    for k, v in data.items():
        # print(k)
        # print(len(v))
        '''target features'''
        # plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1], s=20, marker='o', c=colors[k], label=central_kind[k])
        '''raw features'''
        # if 0<= k <= 4:
        #     plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1], s=10, marker='o', c=colors[k], label=central_kind[k])
        # else:
        #     plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1], s=10, marker='*', c=colors[k], label=central_kind[k],alpha = 0.6)
        '''UDA features'''
        if k > 0:
            plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1], s=30, marker='o', c=colors[k], label=central_kind[k], alpha = 0.6)
        else:
            # plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1], s=10, marker='o', c=colors[k], label=central_kind[k],alpha = 0.3)
            plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1], s=30, marker='o', c=colors[k], label=central_kind[k], alpha = 0.6)

    plt.xticks(())  # 不显示坐标刻度
    plt.yticks(())
    # plt.title(title)  # 设置标题
    # plt.savefig(str(1) + '_' + 't-SNE.png')
    '''raw_features'''
    # plt.savefig('results/figures/raw_features.png')
    '''UDA_features'''
    # plt.savefig('results/figures/UDA_target.png')
    # plt.savefig('results/figures/UDA_features.png')
    '''SSL_features'''
    # plt.savefig('results/figures/SSL_features.png')
    '''Source_only_features'''
    # plt.savefig('results/figures/Source_only_features.png')
    '''UDAGCN'''
    # plt.savefig('results/figures/UDAGCN.png')
    '''node alignmet and classfication'''
    # plt.savefig('results/figures/alignment_class.png')
    plt.savefig('result/figures/tsne-{}-{:d}.png'.format(path, index))
    plt.show()

device = torch.device("cuda:{}".format(0))
Model = torch.load('/home/fennel/Brain/model_checkpoint/ABIDE_6_4/epochs30-2023-06-04-21:58:22.pth', map_location=device)

# tsne = TSNE(n_components=2, init='random', random_state=123,
#             learning_rate=500, n_iter=1000, perplexity=20,
#             early_exaggeration=20, verbose=0)

tsne = TSNE(n_components=2, init='random', random_state=123,
            learning_rate=500, n_iter=1000, perplexity=30,
            early_exaggeration=20, verbose=0)

X = np.load('./data/635_8A.npy')
Y = np.load('./data/635y.npy')

X = X[:256]
Y = Y[:256]

data = torch.from_numpy(X).float().to(device)
label = torch.from_numpy(Y).long()

data0 = data
data2 = Model.Encoder2(data)
data = Model.Encoder(data)

print('data.shape: ',data.shape,' type(label): ',type(label))
print('label: ',label)

# data = data.view(data.size(0), data.size(1), -1, 32)
data = data.mean(dim=3)
data = data.view(data.size(0), data.size(1), -1)
# data = data[:,:,:64]
data = data.cpu().detach().numpy()
print(data.shape)
np.save('./data/s.npy', data)

data2 = data2.mean(dim=3)
data2 = data2.view(data2.size(0), data2.size(1), -1)
data2 = data2.cpu().detach().numpy()
print(data2.shape)
np.save('./data/d.npy', data2)

data0 = data0.mean(dim=2)
data0 = data0.view(data0.size(0), data0.size(1), -1)
data0 = data0.cpu().detach().numpy()
print(data0.shape)
np.save('./data/0.npy', data0)

# for i in range(8):
#     tmp_data = data[:,i,:]
#     result = tsne.fit_transform(tmp_data)  # 进行降维，[1083,64]-->[1083,2]
#     plot_embedding(result, label, 's', i)

# for i in range(8):
#     tmp_data = data2[:,i,:]
#     result = tsne.fit_transform(tmp_data)  # 进行降维，[1083,64]-->[1083,2]
#     plot_embedding(result, label, 'd', i)

# for i in range(8):
#     tmp_data = data0[:,i,:]
#     result = tsne.fit_transform(tmp_data)  # 进行降维，[1083,64]-->[1083,2]
#     plot_embedding(result, label, '0', i)