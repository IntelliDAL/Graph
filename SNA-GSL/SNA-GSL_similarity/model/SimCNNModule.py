import torch
import torch.nn as nn
import torch.nn.functional as F


class E2EBlock(nn.Module):
    def __init__(self, args, in_channel, out_channel):
        super(E2EBlock, self).__init__()
        self.args = args
        self.d = args.n_max_nodes
        if self.args.dataset == 'IMDBMulti':
            self.d = args.pooling_res
        self.cnn1 = torch.nn.Conv2d(in_channel, out_channel, (1, self.d))
        self.cnn2 = torch.nn.Conv2d(in_channel, out_channel, (self.d, 1))
        torch.nn.init.xavier_uniform_(self.cnn1.weight)
        torch.nn.init.xavier_uniform_(self.cnn2.weight)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


class SimCNN(torch.nn.Module):
    def __init__(self, args):
        super(SimCNN, self).__init__()
        in_planes = args.n_heads * 2
        self.d = args.n_max_nodes
        self.args = args

        if args.dataset == 'IMDBMulti':
            self.d = args.pooling_res

        self.e2econv1 = E2EBlock(args, in_channel=in_planes, out_channel=args.conv_channels_0)
        self.e2econv2 = E2EBlock(args, in_channel=args.conv_channels_0, out_channel=args.conv_channels_1)
        self.E2N = torch.nn.Conv2d(in_channels=args.conv_channels_1, out_channels=args.conv_channels_2, kernel_size=(1, self.d))
        self.N2G = torch.nn.Conv2d(args.conv_channels_2, args.conv_channels_3, (self.d, 1))

        self.fc_1 = torch.nn.Linear(args.conv_channels_3, args.conv_channels_3 // 2)
        self.fc_2 = torch.nn.Linear(args.conv_channels_3 // 2, args.conv_channels_3 // 4)
        self.fc_3 = torch.nn.Linear(args.conv_channels_3 // 4, args.conv_channels_3 // 8)
        self.fc_4 = torch.nn.Linear(args.conv_channels_3 // 8, 1)

        torch.nn.init.xavier_uniform_(self.E2N.weight)
        torch.nn.init.xavier_uniform_(self.N2G.weight)
        torch.nn.init.xavier_uniform_(self.fc_1.weight)
        torch.nn.init.xavier_uniform_(self.fc_2.weight)
        torch.nn.init.xavier_uniform_(self.fc_3.weight)
        torch.nn.init.xavier_uniform_(self.fc_4.weight)

    def forward(self, sim_mat, mask_ij):
        out = F.leaky_relu(self.e2econv1(sim_mat), negative_slope=self.args.conv_l_relu_slope, inplace=True)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=self.args.conv_l_relu_slope, inplace=True)
        # out = torch.einsum('bhij,bij->bhij', F.relu(self.e2econv1(x), inplace=True), mask_ij)
        # out = torch.einsum('bhij,bij->bhij', F.relu(self.e2econv2(out), inplace=True), mask_ij)

        out = F.leaky_relu(self.E2N(out), negative_slope=self.args.conv_l_relu_slope, inplace=True)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=self.args.conv_l_relu_slope, inplace=True), p=self.args.conv_dropout, training=self.training).squeeze()

        out = F.dropout(F.leaky_relu(self.fc_1(out), negative_slope=self.args.conv_l_relu_slope, inplace=True), p=self.args.conv_dropout, training=self.training)
        out = F.dropout(F.leaky_relu(self.fc_2(out), negative_slope=self.args.conv_l_relu_slope, inplace=True), p=self.args.conv_dropout, training=self.training)
        out = F.dropout(F.leaky_relu(self.fc_3(out), negative_slope=self.args.conv_l_relu_slope, inplace=True), p=self.args.conv_dropout, training=self.training)
        out = F.leaky_relu(self.fc_4(out), negative_slope=self.args.conv_l_relu_slope, inplace=True)
        out = torch.sigmoid(out).squeeze(-1)

        return out
