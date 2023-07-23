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
# 增加可读性
from thop import clever_format

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

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

class E2E(nn.Module):

    def __init__(self, in_channel, out_channel, input_shape, **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.d = input_shape[0]
        self.conv1xd = nn.Conv2d(in_channel, out_channel, (self.d, 1))
        self.convdx1 = nn.Conv2d(in_channel, out_channel, (1, self.d))
        self.nodes = 93

    def forward(self, A):
        
#         print(A.shape)
        A = A.view(-1, self.in_channel, self.nodes, self.nodes)

        a = self.conv1xd(A)
        b = self.convdx1(A)

        concat1 = torch.cat([a]*self.d, 2)
        concat2 = torch.cat([b]*self.d, 3)
        
        # A = torch.mean(concat1+concat2, 1)
        # print('e2e', (concat1+concat2).shape)
        return concat1+concat2

nums_train = np.ones(116) # 制作mask模板
# nums_train[:93] = 0 # 根据设置的nodes number 决定多少是mask 即mask比例 # 写错了 应该是[:nodes]
Mask_train = nums_train.reshape(nums_train.shape[0], 1) * nums_train # 116 116
for i in range(93):
    Mask_train[i][:93] = 0
# np.repeat(Mask_train, X_train.shape[0], 0)
Mask_train_tensor = torch.from_numpy(Mask_train).float().to(device)
# Mask_train_tensor = tf.cast(Mask_train_tensor, tf.float32)

class Model(nn.Module):
    def __init__(self, dropout=0.5, num_class=1, nodes=93):
        super().__init__()

        self.e2e = nn.Sequential(
            E2E(1, 8, (nodes, nodes)),
            nn.LeakyReLU(0.33),
            E2E(8, 8, (nodes, nodes)), # 0.642
            nn.LeakyReLU(0.33),
        )
        
        self.e2n = nn.Sequential(
            nn.Conv2d(8, 48, (1, nodes)), # 32 652
            nn.LeakyReLU(0.33),
        )
        
        self.n2g = nn.Sequential(
            nn.Conv2d(48, nodes, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        self.linear = nn.Sequential(
            nn.Linear(nodes, 64),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
            nn.Linear(64, 10),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
            nn.Linear(10, num_class)
        )       

        self.GC = DenseGCNConv(48, 48)

        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.GCN = GCN()

    def MaskAutoEncoder(self, e2n, A, masked_x): # masked_x 32 50 48
        e2n_encoder = torch.squeeze(e2n)
        # print('e2n_encoder ', e2n_encoder.shape) # 16 116
        # print('masked_x ', masked_x.shape)
        masked_x = masked_x.permute(0, 2, 1) # 32 48 50
        e2n_encoder = torch.cat((e2n_encoder, masked_x), -1) # 补上了masked
        # print('e2n_encoder ', e2n_encoder.shape) # 32 48 116
        e2n_encoder_T = e2n_encoder.permute(0, 2, 1) # batch 116 48
        # print(temp.shape)
        # print('A ', A.shape) # 116 116
        # print(A[0])
        # e2n_encoder_T = self.GCN(e2n_encoder_T, A)
        e2n_encoder_T = self.GC(e2n_encoder_T, A)
        e2n_encoder = e2n_encoder_T.permute(0, 2, 1)
        Z = torch.matmul(e2n_encoder_T, e2n_encoder) # batch 116 116
        # print('Z ', Z.shape)
        # Z = nn.sigmoid(Z) # 正相关 负相关分离
        # 哈达姆乘
        Z = Z * Mask_train_tensor
        # print(Mask_train_tensor)
        # print(Z[0][199])

        return Z
        # Z = K.expand_dims(Z, axis=-1)

    def forward(self, x, A, masked_x):
        # print('input', x.shape)
        x = self.e2e(x)
        x = self.e2n(x)
        # print('e2n', x.shape) # batch 16 116 1
        z = self.MaskAutoEncoder(x, A, masked_x)
        x = self.n2g(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.linear(x)
        x = F.softmax(x, dim=-1)
        # print('output', x.shape)

        return x, z
    
    def get_A(self, x):
        x = self.e2e(x)
#         print(x.shape)
        x = torch.mean(x, dim=1)
        return x
        

# In[]
# ABIDE load
print('loading ABIDE data...')
# X = np.load('./pcc_correlation_871_cc116.npy')
# Y = np.load('/kaggle/input/cc116-pytorch/871_label_cc116.npy')
# Data = scio.loadmat('./data/ABIDE/data/correlation/pcc_correlation_871_cc116_.mat')
# # print(cc116.keys()) # connectivity
# X = Data['connectivity']
# # print(X[0][0])
# # print(cc116.shape) # 871 116 116
# Y = np.loadtxt('./data/ABIDE/data/labels/871_label_cc116.txt')

Data = scio.loadmat('./data/ABIDE/data/correlation/pcc_correlation_871_aal_.mat')
# print(cc116.keys()) # connectivity
X = Data['connectivity']
# print(X[0][0])
# print(cc116.shape) # 871 116 116
Y = np.loadtxt('./data/ABIDE/data/labels/871_labels.txt')

where_are_nan = np.isnan(X)  # 找出数据中为nan的
where_are_inf = np.isinf(X)  # 找出数据中为inf的
for bb in range(0, 871):
    for i in range(0, 116):
        for j in range(0, 116):
            if where_are_nan[bb][i][j]:
                X[bb][i][j] = 0
            if where_are_inf[bb][i][j]:
                X[bb][i][j] = 1

print('---------------------')
print('X', X.shape) # N M M
print('Y', Y.shape)
print('---------------------')
X_temp = X

epochs_rec = 10 # 20 670 # 10 684 # 0 663 # 30 666 # 15 670 5 666
epochs = 50 + epochs_rec # 116 671
# epochs = 1
# if True:
#     epochs = 0

batch_size = 32 # 64 0.660
dropout = 0.5
lr = 0.005
decay = 0.01
result = []
acc_final = 0
result_final = []

list_common_nodes = []

from sklearn.model_selection import KFold
for ind in range(50):
    setup_seed(ind)
    idd = ind

    # Masked
    nodes_number = 93
    nums = np.ones(116) # 制作mask模板
    nums[:116-nodes_number] = 0 # 根据设置的nodes number 决定多少是mask 即mask比例
    np.random.seed(idd)
    np.random.shuffle(nums) # 116 75%1 25%0 shuffle打散
    # print(nums)




    # ind_list = [100, 68, 86, 96, 91, 88, 58, 82] 50-100 8个
    # ind_list = [2,4,38, 13,23] 0-50  5
    ind_list = [40,48,3,14,43,13,24,8,16,22] # 0-50

    # print(ind)
    # print(nums)
    if (idd+1) in ind_list:
        list_common_nodes.append(nums)
    continue


    # print(nums)
    # print('nums----------')
    Mask = nums.reshape(nums.shape[0], 1) * nums # 116 116
    # print('X before ', X.shape)
    Masked_X = X_temp * Mask # 将部分转换为 0（masked）
    # print('X after ', X.shape)
    X0=X_temp
    Masked_X_rest = X_temp - Masked_X
    # print('Masked_X_rest ', Masked_X_rest[0][])
    J = nodes_number # J 拷贝出一份
    for i in range(0, J):
        ind = i
        # ind = nums.shape[0] - 1 - i
        if nums[ind] == 0:
            for j in range(J, 116):
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
    kf = KFold(n_splits=10, shuffle=True)
    kfold_index = 0
    for trainval_index, test_index in kf.split(X, Y):
        kfold_index += 1
        print('kfold_index:', kfold_index)
        if kfold_index != 1:
            continue
        X_trainval, X_test = X[trainval_index], X[test_index]
        Y_trainval, Y_test = Y[trainval_index], Y[test_index]

        X_trainval_masked_rest = X_unmasked[trainval_index]
        X_test_masked_rest = X_unmasked[test_index]

        X_trainval_0, X_test_0 = X0[trainval_index], X0[test_index]
        for train_index, val_index in kf.split(X_trainval, Y_trainval):
            # 取消验证集
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
                    if abs(X_test_0[k][i][j] < 0.2): # 0.3 679 0.4 684 0.45 661 0.6 671 0.2 675
                        X_test_0[k][i][j] = 0
                    else:
                        X_test_0[k][i][j] = 1

        for k in range(X_avg.shape[0]):
            for i in range(X_avg.shape[1]):
                if abs(X_avg[k][i] < 0.15): # 0.2 674  0.4 646 0.3 677 0.1 673 0.15 696 ！！！！！
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

        X_train_0 = X_avg
        # X_test_0 = X_avg_test
        # print(X_test_0)

        X_masked = np.zeros([X_train.shape[0], 116-nodes_number, 48])
        X_masked_test = np.zeros([X_test.shape[0], 116-nodes_number, 48])
        for i in range(X_masked.shape[0]):
            for j in range(X_masked.shape[1]):
                X_masked[i][j] = np.random.normal(loc=0.0, scale=1.0, size=48)
        for i in range(X_masked_test.shape[0]):
            for j in range(X_masked_test.shape[1]):
                X_masked_test[i][j] = np.random.normal(loc=0.0, scale=1.0, size=48)

        # model
        model = Model(dropout=dropout, num_class=2)
        model.to(device)
        # for p in model.parameters():
        #     if p.requires_grad:
        #         print(p.name, p.data.shape)
        # flops, params = profile(model, inputs=(input, ))
        # flops, params = clever_format([flops, params], "%.3f")
        params = list(model.parameters())
        k = 0
        for i in params:
            l = 1
            print("该层的结构：" + str(list(i.size())))
            for j in i.size():
                l *= j
            print("该层参数和：" + str(l))
            k = k + l
        print("总参数数量和：" + str(k))

        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
        optimizer2 = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
    #     lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 116, 250], gamma=0.8)
        loss_fn = nn.CrossEntropyLoss()
        loss_rec = nn.MSELoss()
        # loss_fn = nn.MSELoss()
        
        best_val = 0
        
        
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
                outputs, rec = model(train_data_batch_dev, train_data_batch_A_dev, train_data_batch_maskedX_dev)
                
                loss1 = loss_fn(outputs, train_label_batch_dev)
                # print(train_data_batch_rest_dev[0][0])
                loss2 = loss_rec(rec, train_data_batch_rest_dev)

                if epoch < epochs_rec:
                    loss = loss2
                    loss_train += loss
                    loss.backward()
                    optimizer.step()
                else:
                    loss = loss1 # + loss2
                    loss_train += loss
                    loss.backward()
                    optimizer2.step()

                
            
            loss_train /= num_batch
            if epoch % 10 == 0:
                print('epoch:', epoch, 'train loss:', loss_train.item())
                
            # val
            if True: #epoch % 1 == 0 and epoch > epochs_rec:
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
                if True: #acc_val > best_val:
                    # best_val = acc_val
                    model.eval()
                    test_data_batch_dev = torch.from_numpy(X_test).float().to(device)
                    test_data_batch_A_dev = torch.from_numpy(X_test_0).float().to(device)
                    test_data_batch_maskedX_dev = torch.from_numpy(X_masked_test).float().to(device)
                    outputs, _ = model(test_data_batch_dev, test_data_batch_A_dev, test_data_batch_maskedX_dev)
                    _, indices = torch.max(outputs, dim=1)
                    preds = indices.cpu()
                    # print(preds)
                    acc = metrics.accuracy_score(preds, Y_test)
                    print('Test acc', acc)

                # print('Test acc', acc_val)

        # if epoch % 1 == 0:
            
        torch.save(model.state_dict(), './models/' + str(kfold_index) + '.pt')
        result.append([kfold_index, acc])
        acc_all += acc
    temp = acc_all / 10
    acc_final += temp
    result_final.append(temp)
    print(result)

    # print(nums)
    # ind_list = [40,48,3,14,43,13,24,8,16,22]
    

ACC = acc_final / 10
# In[]

print(result_final)
print(ACC)

cn = np.array(list_common_nodes)
print(cn.shape)
common_node = np.sum(cn, 0)
print(common_node)
# 684
