import torch
import torch.nn as nn
import torch.nn.functional as F


class SimMatPooling(nn.Module):
    def __init__(self, args):
        super(SimMatPooling, self).__init__()
        self.args = args

        in_channels = args.n_heads * 2

        self.cnn_1 = nn.Conv2d(in_channels=in_channels, out_channels=args.conv_channels_0, kernel_size=(3, 3))
        self.pooling = nn.AdaptiveAvgPool2d((7, 7))
        self.cnn_2 = nn.Conv2d(in_channels=args.conv_channels_0, out_channels=args.conv_channels_1, kernel_size=(3, 3))
        self.cnn_3 = nn.Conv2d(in_channels=args.conv_channels_1, out_channels=args.conv_channels_2, kernel_size=(3, 3))
        self.cnn_4 = nn.Conv2d(in_channels=args.conv_channels_2, out_channels=args.conv_channels_3, kernel_size=(3, 3))

        self.fc_1 = torch.nn.Linear(args.conv_channels_3, args.conv_channels_3 // 2)
        self.fc_2 = torch.nn.Linear(args.conv_channels_3 // 2, args.conv_channels_3 // 4)
        self.fc_3 = torch.nn.Linear(args.conv_channels_3 // 4, args.conv_channels_3 // 8)
        self.fc_4 = torch.nn.Linear(args.conv_channels_3 // 8, 1)

        torch.nn.init.xavier_uniform_(self.cnn_1.weight)
        torch.nn.init.xavier_uniform_(self.cnn_2.weight)
        torch.nn.init.xavier_uniform_(self.cnn_3.weight)
        torch.nn.init.xavier_uniform_(self.cnn_4.weight)

        torch.nn.init.xavier_uniform_(self.fc_1.weight)
        torch.nn.init.xavier_uniform_(self.fc_2.weight)
        torch.nn.init.xavier_uniform_(self.fc_3.weight)
        torch.nn.init.xavier_uniform_(self.fc_4.weight)

    def forward(self, sim_mat):
        out = F.leaky_relu(self.cnn_1(sim_mat), negative_slope=self.args.conv_l_relu_slope, inplace=True)
        out = self.pooling(out)
        out = F.leaky_relu(self.cnn_2(out), negative_slope=self.args.conv_l_relu_slope, inplace=True)
        out = F.leaky_relu(self.cnn_3(out), negative_slope=self.args.conv_l_relu_slope, inplace=True)
        out = F.leaky_relu(self.cnn_4(out), negative_slope=self.args.conv_l_relu_slope, inplace=True).squeeze()

        out = F.dropout(F.leaky_relu(self.fc_1(out), negative_slope=self.args.conv_l_relu_slope, inplace=True), p=self.args.conv_dropout, training=self.training)
        out = F.dropout(F.leaky_relu(self.fc_2(out), negative_slope=self.args.conv_l_relu_slope, inplace=True), p=self.args.conv_dropout, training=self.training)
        out = F.dropout(F.leaky_relu(self.fc_3(out), negative_slope=self.args.conv_l_relu_slope, inplace=True), p=self.args.conv_dropout, training=self.training)
        out = F.leaky_relu(self.fc_4(out), negative_slope=self.args.conv_l_relu_slope, inplace=True)
        out = torch.sigmoid(out).squeeze(-1)

        return out
