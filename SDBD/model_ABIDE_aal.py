import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


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
        
        concat1 = torch.cat([a]*self.d, 2)
        concat2 = torch.cat([b]*self.d, 3)
        return concat1+concat2


class BrainCNN(nn.Module):
    def __init__(self, f_length, dropout=0.5):
        super().__init__()
        self.f_length = f_length

        self.e2e = nn.Sequential(
            E2E(1, 8, (116, 116)),
            nn.LeakyReLU(0.33),
        )

        self.e2n = nn.Sequential(
            nn.Conv2d(8, f_length, (1, 116)),
            nn.LeakyReLU(0.33),
        )
        
        self.n2g = nn.Sequential(
            nn.Conv2d(3, 116, (116, 1)),
            nn.LeakyReLU(0.33),
        )

        self.linear = nn.Sequential(
            nn.Linear(116, f_length),
        )       

        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.e2e(x)
        x = self.e2n(x)
        x = x.view(x.size(0), 116, -1)

        return x
    
    def get_A(self, x):
        x = self.e2e(x)
#         print(x.shape)
        x = torch.mean(x, dim=1)
        return x


class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, batchnorm=True):
        super(LinearUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU())
        else:
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU())

    def forward(self, x):
        return self.model(x)


class my_model(nn.Module):
    def __init__(self, a, b): 
        super(my_model, self).__init__()  # 200 200
        self.BrainCNN_s = BrainCNN(8)     # 200 64
        self.BrainCNN_d = BrainCNN(8)
        # self.encoder_s = LinearUnit(32, 32, batchnorm=False)# 64
        # self.encoder_d = LinearUnit(32, 32, batchnorm=False) # 64
        # self.decoder = LinearUnit(64, 200) # 200   转置相乘
        # self.rebuild = LinearUnit(1, 200)  # 200 200

        self.margin = 0.7
        self.A_loss_fn = nn.MSELoss()
        # self.X_loss_fn = nn.CosineEmbeddingLoss(margin=0.5) 
        self.a = a
        self.b = b

    def triplet_loss(self, anchor, positive, negative):  # 三元组损失

        if (positive == None):
            pos_dist = 0
        else:
            pos_dist = (anchor - positive).pow(2).sum(1)
            
        if (negative == None):
            neg_dist = 0
        else:
            neg_dist = (anchor - negative).pow(2).sum(1)

        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()

    def forward(self, A):
        A_loss = 0
        s_loss = 0
        d_loss = 0
        
        for i in range(A.size(1)-1):
            A_t0 = A[:, i]
            A_t1 = A[:, i+1]
            
            # 图编码
            A_t0_s_f = self.BrainCNN_s(A_t0)
            A_t0_d_f = self.BrainCNN_d(A_t0)
            A_t1_s_f = self.BrainCNN_s(A_t1)
            A_t1_d_f = self.BrainCNN_d(A_t1)
            # print(A_t0_s_f.shape)

            # 编码器解耦出时变时不变特征   s-相关(时不变) d-无关(时变)
            # s_t0 = self.encoder_s(A_t0_s_f)
            # s_t1 = self.encoder_s(A_t1_s_f)
            # d_t0 = self.encoder_d(A_t0_d_f)
            # d_t1 = self.encoder_d(A_t1_d_f)
            
            # G_dt0_st0 = d_t0 + s_t0
            # G_dt1_st0 = d_t1 + s_t0
            # G_dt0_st1 = d_t0 + s_t1
            # G_dt1_st1 = d_t1 + s_t1

            G_dt0_st0 = A_t0_d_f + A_t0_s_f
            G_dt1_st0 = A_t1_d_f + A_t0_s_f
            G_dt0_st1 = A_t0_d_f + A_t1_s_f
            G_dt1_st1 = A_t1_d_f + A_t1_s_f
            
            # 重建图
            A_dt0_st0 = torch.matmul(G_dt0_st0, G_dt0_st0.transpose(-1,-2).contiguous())
            A_dt1_st0 = torch.matmul(G_dt1_st0, G_dt1_st0.transpose(-1,-2).contiguous())
            A_dt0_st1 = torch.matmul(G_dt0_st1, G_dt0_st1.transpose(-1,-2).contiguous())
            A_dt1_st1 = torch.matmul(G_dt1_st1, G_dt1_st1.transpose(-1,-2).contiguous())
            # print(A_t0.shape, A_dt0_st0.shape)
            
            # 计算图损失
            A_loss_00 = self.A_loss_fn(A_t0, A_dt0_st0)
            A_loss_01 = self.A_loss_fn(A_t0, A_dt0_st1)
            A_loss_10 = self.A_loss_fn(A_t1, A_dt1_st0)
            A_loss_11 = self.A_loss_fn(A_t1, A_dt1_st1)
            
            # A_loss += ((A_loss_00 + A_loss_01 + A_loss_10 + A_loss_11) * self.a)
            A_loss += (A_loss_00 + A_loss_01 + A_loss_10 + A_loss_11)
            
            # 计算时变时不变特征损失
#             print(s_t0.shape, s_t1.shape)
            # s_loss += (self.triplet_loss(A_t0_s_f, A_t1_s_f, None) * self.b)
            # d_loss += (self.triplet_loss(A_t0_d_f, None, A_t1_d_f) * self.b)
            s_loss += self.triplet_loss(A_t0_s_f, A_t1_s_f, None)
            d_loss += self.triplet_loss(A_t0_d_f, None, A_t1_d_f)
        
        return A_loss, s_loss, d_loss
    
    def Encoder(self, A):
        
        batch_size = A.size(0)
        windows = A.size(1)
        A_f = self.BrainCNN_s(A)
        
        return A_f.view(batch_size, windows, 116, -1)
    
    def Encoder2(self, A):

        batch_size = A.size(0)
        windows = A.size(1)
        A_f = self.BrainCNN_d(A)

        return A_f.view(batch_size, windows, 116, -1)

    def Encoder_all(self, A):
        batch_size = A.size(0)
        windows = A.size(1)
        A_s = self.BrainCNN_s(A)
        A_s += self.BrainCNN_d(A)

        return A_s.view(batch_size, windows, 116, -1)