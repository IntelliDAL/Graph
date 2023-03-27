import torch
import torch.nn as nn
import torch.nn.functional as F


class E2E(nn.Module):
    def __init__(self, in_channel, out_channel, input_shape):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.d = input_shape[0]
        self.conv1xd = nn.Conv2d(in_channel, out_channel, (self.d, 1))
        self.convdx1 = nn.Conv2d(in_channel, out_channel, (1, self.d))
        self.nodes = input_shape[0]

    def forward(self, A):
        A = A.view(-1, self.in_channel, self.nodes, self.nodes)
        a = self.conv1xd(A)
        b = self.convdx1(A)
        concat1 = torch.cat([a] * self.d, 2)
        concat2 = torch.cat([b] * self.d, 3)
        
        return concat1 + concat2



class GraphStructureEncoder(nn.Module):
    def __init__(self, pool_size, num_hidden):
        super().__init__()
        self.nodes = pool_size

        self.e2e = nn.Sequential(
            E2E(1, 8, (self.nodes, self.nodes)),        # 1 8
            nn.LeakyReLU(0.33),
            E2E(8, 64, (self.nodes, self.nodes)),       # 8 64
            nn.LeakyReLU(0.33)
        )

        self.e2n = nn.Sequential(
            nn.Conv2d(64, num_hidden, (1, self.nodes)), # 64 num_hidden=256
            nn.LeakyReLU(0.33),
        )


    def forward(self, x):
        x = self.e2e(x)
        x = self.e2n(x)
        
        return x

