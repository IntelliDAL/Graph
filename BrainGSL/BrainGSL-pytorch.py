# In[]
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as scio
import random
from sklearn import metrics

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
        A = A.view(-1, self.in_channel, 200, 200)

        a = self.conv1xd(A)
        b = self.convdx1(A)

        concat1 = torch.cat([a]*self.d, 2)
        concat2 = torch.cat([b]*self.d, 3)
        
        # A = torch.mean(concat1+concat2, 1)
        # print('e2e', (concat1+concat2).shape)
        return concat1+concat2


class Model(nn.Module):
    def __init__(self, dropout=0.5, num_class=1):
        super().__init__()

        self.e2e = nn.Sequential(
            E2E(1, 8, (200, 200)),
            nn.LeakyReLU(0.33),
            E2E(8, 8, (200, 200)),
            nn.LeakyReLU(0.33),
        )
        
        self.e2n2g = nn.Sequential(
            nn.Conv2d(8, 48, (1, 200)),
            nn.LeakyReLU(0.33),
            nn.Conv2d(48, 200, (200, 1)),
            nn.LeakyReLU(0.33),
        )

        self.linear = nn.Sequential(
            nn.Linear(200, 64),
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

    def forward(self, x):
        # print('input', x.shape)
        x = self.e2e(x)
        x = self.e2n2g(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.linear(x)
        x = F.softmax(x, dim=-1)
        # print('output', x.shape)

        return x
    
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
Data = scio.loadmat('./data/ABIDE/data/correlation/pcc_correlation_871_cc200_.mat')
# print(cc200.keys()) # connectivity
X = Data['connectivity']
# print(X[0][0])
# print(cc200.shape) # 871 200 200
Y = np.loadtxt('./data/ABIDE/data/labels/871_label_cc200.txt')

where_are_nan = np.isnan(X)  # 找出数据中为nan的
where_are_inf = np.isinf(X)  # 找出数据中为inf的
for bb in range(0, 871):
    for i in range(0, 200):
        for j in range(0, 200):
            if where_are_nan[bb][i][j]:
                X[bb][i][j] = 0
            if where_are_inf[bb][i][j]:
                X[bb][i][j] = 1

print('---------------------')
print('X', X.shape) # N M M
print('Y', Y.shape)
print('---------------------')


# In[]
epochs = 50 # 200 671
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
        print('kfold_index:', kfold_index)
        X_trainval, X_test = X[trainval_index], X[test_index]
        Y_trainval, Y_test = Y[trainval_index], Y[test_index]
        for train_index, val_index in kf.split(X_trainval, Y_trainval):
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
    #     lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 200, 250], gamma=0.8)
        loss_fn = nn.CrossEntropyLoss()
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
                train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
                train_label_batch_dev = torch.from_numpy(train_label_batch).long().to(device)

                optimizer.zero_grad()
                outputs = model(train_data_batch_dev)
                loss = loss_fn(outputs, train_label_batch_dev)
                loss_train += loss
                loss.backward()
                optimizer.step()
            
            loss_train /= num_batch
            if epoch % 10 == 0:
                print('epoch:', epoch, 'train loss:', loss_train.item())
                
            # val
            if epoch % 1 == 0:
                # model.eval()
                
                # val_data_batch_dev = torch.from_numpy(X_val).float().to(device)
                # val_label_batch_dev = torch.from_numpy(Y_val).long().to(device)
                # outputs = model(val_data_batch_dev)
                # loss = loss_fn(outputs, val_label_batch_dev)
                # _, indices = torch.max(outputs, dim=1)
                # preds = indices.cpu()
                # print(preds)
                # acc_val = metrics.accuracy_score(preds, Y_val)
                if True: #acc_val >= best_val:
                    # best_val = acc_val
                    model.eval()
                    test_data_batch_dev = torch.from_numpy(X_test).float().to(device)
                    outputs = model(test_data_batch_dev)
                    _, indices = torch.max(outputs, dim=1)
                    preds = indices.cpu()
                    # print(preds)
                    acc = metrics.accuracy_score(preds, Y_test)
                    print('Test acc', acc)

                # print('Test acc', acc_val)

        # if epoch % 1 == 0:
            
                
        result.append([kfold_index, acc])
        acc_all += acc
    temp = acc_all / 10
    acc_final += temp
    result_final.append(temp)

ACC = acc_final / 10
# In[]

print(result_final)
print(ACC)

# 0.656