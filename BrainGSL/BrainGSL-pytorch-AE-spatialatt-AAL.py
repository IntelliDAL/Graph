# In[]
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as scio
import random
from sklearn import metrics
import cv2
from nilearn import plotting
import matplotlib.pyplot as plt

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

class E2E(nn.Module):

    def __init__(self, in_channel, out_channel, input_shape, **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.d = input_shape[0]
        self.conv1xd = nn.Conv2d(in_channel, out_channel, (self.d, 1))
        self.convdx1 = nn.Conv2d(in_channel, out_channel, (1, self.d))

    def forward(self, A):
        
#         print(A.shape)
        A = A.view(-1, self.in_channel, 116, 116)

        a = self.conv1xd(A)
        b = self.convdx1(A)

        concat1 = torch.cat([a]*self.d, 2)
        concat2 = torch.cat([b]*self.d, 3)
        
        # A = torch.mean(concat1+concat2, 1)
        # print('e2e', (concat1+concat2).shape)
        return concat1+concat2

class SpatialAttn(nn.Module):
    def __init__(self, in_features=8, normalize_attn=False):
        super(SpatialAttn, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1,
            kernel_size=1, padding=0, bias=False)

    def forward(self, l):
        N, C, H, W = l.size()
        c = self.op(l) # (batch_size,1,H,W)
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,H,W)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # (batch_size,C)
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,H,W), g

class SpatialAttentionModul(nn.Module):  # 空间注意力模块
    def __init__(self, in_channel=8):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x维度为 [N, C, H, W] 沿着维度C进行操作, 所以dim=1, 结果为[N, H, W]
        MaxPool = torch.max(x, dim=1).values  # torch.max 返回的是索引和value， 要用.values去访问值才行！
        AvgPool = torch.mean(x, dim=1)

        # 增加维度, 变成 [N, 1, H, W]
        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        # 维度拼接 [N, 2, H, W]
        x_cat = torch.cat((MaxPool, AvgPool), dim=1)  # 获得特征图

        # 卷积操作得到空间注意力结果
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        # 与原图通道进行乘积
        x = Ms * x

        return x, Ms

class Model(nn.Module):
    def __init__(self, dropout=0.5, num_class=1):
        super().__init__()

        self.e2e = nn.Sequential(
            E2E(1, 8, (116, 116)),
            nn.LeakyReLU(0.33),
            E2E(8, 8, (116, 116)), # 0.642
            nn.LeakyReLU(0.33),
        )
        
        self.e2n = nn.Sequential(
            nn.Conv2d(8, 48, (1, 116)), # 32 652
            nn.LeakyReLU(0.33),
        )
        
        self.n2g = nn.Sequential(
            nn.Conv2d(48, 116, (116, 1)),
            nn.LeakyReLU(0.33),
        )

        self.linear = nn.Sequential(
            nn.Linear(116, 64),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
            nn.Linear(64, 10),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
            nn.Linear(10, num_class)
        )       

        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        self.att = SpatialAttn()
        self.spatialatt = SpatialAttentionModul()
    

    # def returnCAM(feature_conv, weight_softmax): # weight batch 1 200 200
    #     # generate the class activation maps upsample to 256x256
    #     size_upsample = (200, 200)
    #     # print(feature_conv.shape)
    #     _, nc, h, w = feature_conv.shape # batch 8 200 200
    #     output_cam = []
    #     # for idx in class_idx:  # 这里可以做一个标签也可以做多个标签。

    #     # print(weight_softmax[idx].unsqueeze(0).shape, feature_conv.reshape((nc, h*w)).shape)
    #     # cam = weight_softmax[idx].unsqueeze(0).dot(feature_conv.reshape((nc, h*w)))  # 这里忽略了bz，往后也是，如果要改，可以加循环
    #     cam = torch.matmul(weight_softmax.unsqueeze(0), feature_conv.reshape((nc, h*w)))
    #     cam = cam.reshape(h, w)  # 7,7
    #     # print(cam.shape)
    #     cam = cam - torch.min(cam)
    #     cam_img = cam / torch.max(cam)
    #     cam_img = cam_img.cpu().data.numpy()
    #     cam_img = np.uint8(255 * cam_img)
    #     # cam_img = int(255 * cam_img)
    #     # print(cam_img.shape)
    #     output_cam.append(cv2.resize(cam_img, size_upsample))

    #     return output_cam
    # 定义计算CAM的函数

    def returnCAM(self, feature_conv, weight_softmax, x): 
        # print(weight_softmax.shape) # batch 1 200 200
        weight_softmax = weight_softmax[0] # 1 200 200
        print(weight_softmax.shape)
        
        def key_function(x):
            return x[0]
        
        # with open("edges.txt", "a") as log_file:
        #     v_cpu = weight_softmax[0].detach().cpu().numpy()
        #     # np.savetxt(log_file, v_cpu)
        #     v1 = np.loadtxt("edges1.txt")
        #     v2 = np.loadtxt("edges2.txt")
        #     v3 = np.loadtxt("edges3.txt")
        #     v4 = np.loadtxt("edges4.txt")
        #     v5 = np.loadtxt("edges5.txt")
        #     v6 = np.loadtxt("edges6.txt")
        #     v7 = np.loadtxt("edges7.txt")
        #     v8 = np.loadtxt("edges8.txt")
        #     v9 = np.loadtxt("edges9.txt")
        #     v10 = np.loadtxt("edges10.txt")
        #     v = v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10
        #     v_cpu = v
        #     # print(v_cpu)
        #     list_cpu = []
        #     for i in range(116):
        #         for j in range(116):
        #             list_cpu.append([v_cpu[i][j], (i+1, j+1)])
        #     list_cpu.sort(key=key_function)
        #     print(list_cpu)
        #     # print(np.argsort(v_cpu))
        #     v_cpu[v_cpu<9.26849] = 0
        #     v_cpu[v_cpu>=9.26849] = 1
        #     np.savetxt(log_file,  v_cpu)
        #     a = np.mean(v_cpu, -1)
        #     b = np.mean(v_cpu, -2)
        #     c = a + b
        #     print(np.argsort(c))
        #     print(a+b)
        with open("nodes.txt", "a") as log_file:
            v_cpu = weight_softmax[0].detach().cpu().numpy() # 200 200
            v1 = np.loadtxt("edges1.txt")
            v2 = np.loadtxt("edges2.txt")
            v3 = np.loadtxt("edges3.txt")
            v4 = np.loadtxt("edges4.txt")
            v5 = np.loadtxt("edges5.txt")
            v6 = np.loadtxt("edges6.txt")
            v7 = np.loadtxt("edges7.txt")
            v8 = np.loadtxt("edges8.txt")
            v9 = np.loadtxt("edges9.txt")
            v10 = np.loadtxt("edges10.txt")
            v = v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10
            v_cpu = v
            # v_cpu[v_cpu<0.925] = 0
            # v_cpu[v_cpu>=0.925] = 1
            
            # np.savetxt(log_file,  v_cpu)
            a = np.mean(v_cpu, -2) + np.mean(v_cpu, -1)
            print(np.argsort(a))
            # print(a)

        # 类激活图上采样到 256 x 256
        feature_conv = feature_conv[0]
        size_upsample = (116, 116)
        nc, h, w = feature_conv.shape
        output_cam = []
        # 将权重赋给卷积层：这里的weigh_softmax.shape为(1000, 512)
        # 				feature_conv.shape为(1, 512, 13, 13)
        # weight_softmax[class_idx]由于只选择了一个类别的权重，所以为(1, 512) # 
        # feature_conv.reshape((nc, h * w))后feature_conv.shape为(512, 169)

        # cam = weight_softmax.dot(feature_conv.reshape((nc, h * w)))  # 8 200 200
        # # print(cam.shape)		# 矩阵乘法之后，为各个特征通道赋值。输出shape为（1，169）
        # cam = cam.reshape(h, w) # 得到单张特征图

        cam = weight_softmax.reshape(h, w)

        # 特征图上所有元素归一化到 0-1
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  
        # 再将元素更改到　0-255
        cam_img = np.uint8(255 * cam_img.detach().cpu().numpy())
        output_cam.append(cv2.resize(cam_img, size_upsample))
        output_cam = np.array(output_cam)
        # print(output_cam.shape)
        output_cam = output_cam[0]

        plotting.plot_matrix(output_cam, vmax=256, vmin=0)
        plt.savefig('./Fig-AAL.jpg')

        # heatmap = cv2.applyColorMap(cv2.resize(output_cam, (w, h)), cv2.COLORMAP_JET)
        # result = heatmap * 0.3 + x[0].detach().cpu().numpy() * 0.7
        # cv2.imwrite('CAM.jpg', result)
        return output_cam

    def AutoEncoder(self, e2n):
        e2n_encoder = torch.squeeze(e2n)
        # print('e2n_encoder', e2n_encoder.shape) # None 200 32
        e2n_encoder_T = e2n_encoder.permute(0, 2, 1) # batch 200 16
        # print(temp.shape)
        Z = torch.matmul(e2n_encoder_T, e2n_encoder)
        # print('Z ', Z.shape)
        # Z = nn.sigmoid(Z) # 正相关 负相关分离
        return Z
        # Z = K.expand_dims(Z, axis=-1)

    def forward(self, x, epoch):
        # print('input', x.shape)
        x_or = x
        x = self.e2e(x)

        # att
        # out1, out2 = self.att(x)
        x_vis = x
        x, Ms = self.spatialatt(x)     # 674
        self.returnCAM(x_vis, Ms, x_or)
        # print(out1.shape)# batch 1 200 200
        # print(out2.shape) # batch 8

        # if epoch > 100:
        # x = x * out1


        x = self.e2n(x)
        # print('e2n', x.shape) # batch 16 200 1
        z = self.AutoEncoder(x)
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
# X = np.load('./pcc_correlation_871_cc200.npy')
# Y = np.load('/kaggle/input/cc200-pytorch/871_label_cc200.npy')
Data = scio.loadmat('./data/ABIDE/data/correlation/pcc_correlation_871_aal_.mat')
# print(cc200.keys()) # connectivity
X = Data['connectivity']
# print(X[0][0])
# print(cc200.shape) # 871 200 200
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

epochs_rec = 10 # 100-50 651 10-40 654 39 673 # 45 651
epochs = 50 + epochs_rec # 200 671

# 画图 TEST
# epochs = 1


batch_size = 32 # 64 0.660
dropout = 0.5
lr = 0.005
decay = 0.01
result = []
acc_final = 0
result_final = []

from sklearn.model_selection import KFold
for ind in range(1):
    setup_seed(ind)
    acc_all = 0
    kf = KFold(n_splits=10, shuffle=True)
    kfold_index = 0
    for trainval_index, test_index in kf.split(X, Y):
        kfold_index += 1

        if kfold_index != 5:
            continue

        print('kfold_index:', kfold_index)
        X_trainval, X_test = X[trainval_index], X[test_index]
        Y_trainval, Y_test = Y[trainval_index], Y[test_index]
        for train_index, val_index in kf.split(X_trainval, Y_trainval):
            # 取消验证集
            X_train, X_val = X_trainval[:], X_trainval[:]
            Y_train, Y_val = Y_trainval[:], Y_trainval[:]
        print('X_train', X_train.shape)
        print('X_val', X_val.shape)
        print('X_test', X_test.shape)
        print('Y_train', Y_train.shape)
        print('Y_val', Y_val.shape)
        print('Y_test', Y_test.shape)
        

        # model
        model = Model(dropout=dropout, num_class=2)
        model.to(device)
        # for p in model.parameters():
        #     if p.requires_grad:
        #         print(p.name, p.data.shape)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
        optimizer2 = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
    #     lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 200, 250], gamma=0.8)
        loss_fn = nn.CrossEntropyLoss()
        loss_rec = nn.MSELoss()
        # loss_fn = nn.MSELoss()
        
        best_val = 0
        
        # train
        for epoch in range(1, epochs+1):
            model.train()

            idx_batch = np.random.permutation(int(X_train.shape[0]))
            num_batch = X_train.shape[0] // int(batch_size)
            
            model.load_state_dict(torch.load('./models/att-AAL/' + str(kfold_index) + '.pt'))

            loss_train = 0
            for bn in range(num_batch):
                if bn == num_batch - 1:
                    batch = idx_batch[bn * int(batch_size):]
                else:
                    batch = idx_batch[bn * int(batch_size) : (bn+1) * int(batch_size)]
                train_data_batch = X_train[batch]
                train_label_batch = Y_train[batch]
                train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
                train_label_batch_dev = torch.from_numpy(train_label_batch).long().to(device)

                optimizer.zero_grad()
                outputs, rec = model(train_data_batch_dev, epoch)
                
                loss1 = loss_fn(outputs, train_label_batch_dev)
                loss2 = loss_rec(rec, train_data_batch_dev)

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
                break # TEST

            break # TEST
            
            loss_train /= num_batch
            if epoch % 10 == 0:
                print('epoch:', epoch, 'train loss:', loss_train.item())
                
            # val
            if epoch % 1 == 0 and epoch > epochs_rec:
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
                    outputs, _ = model(test_data_batch_dev, epoch)
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

ACC = acc_final / 10
# In[]

print(result_final)
print(ACC)

# 669
