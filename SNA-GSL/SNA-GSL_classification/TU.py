import random

import numpy as np
import torch

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree, Constant
from torch_geometric.utils import degree, to_dense_batch, to_dense_adj
from torch.utils.data import random_split
from aug import TUDataset_aug as TUDataset

np.random.seed(1)


class GraphSimDataset(object):
    def __init__(self, args):
        self.args = args
        self.training_graphs = None
        self.training_set = None
        self.val_set = None
        self.testing_set = None
        self.testing_graphs = None
        self.nged_matrix = None
        self.real_data_size = None
        self.number_features = None
        self.normed_dist_mat_all = None
        self.n_max_nodes = 0
        self.n_all_graphs = 0
        self.process_dataset()

    def process_dataset(self):
        print('\nPreparing dataset.\n')
        print(self.args.data_dir + self.args.dataset)
        self.training_graphs = TUDataset(self.args.data_dir, name=self.args.dataset, aug=self.args.aug).shuffle()
        self.testing_graphs = TUDataset(self.args.data_dir, name=self.args.dataset, aug='none').shuffle()
        self.n_max_nodes = max([g.num_nodes for g in self.training_graphs + self.testing_graphs])
        max_degree = 0
        for g in self.training_graphs + self.testing_graphs:
            if g.edge_index.size(1) > 0:
                max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
        one_hot_degree = OneHotDegree(max_degree, cat=True)
        self.training_graphs.transform = one_hot_degree
        self.testing_graphs.transform = one_hot_degree
        self.args.node_feature_size = self.training_graphs.num_features
        # train_num = len(self.training_graphs) - len(self.testing_graphs)
        # val_num = len(self.testing_graphs)
        # self.training_set, self.val_set = random_split(self.training_graphs, [train_num, val_num])

    def create_batches_(self, graphs):
        # 单个图，做图分类使用
        loader = DataLoader(graphs, batch_size=self.args.batch_size, shuffle=True)
        return loader

    def create_batches_test(self, graphs, batch_size):
        # 测试模块使用，自己设定batch
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
        return loader

    def create_batches(self, graphs):
        # 一对图，图相似性计算
        graphs = graphs.dataset
        source_loader = DataLoader(graphs, batch_size=self.args.batch_size, shuffle=True)
        target_loader = DataLoader(graphs, batch_size=self.args.batch_size, shuffle=True)
        return list(zip(source_loader, target_loader))

    def transform_single(self, data):
        # 单个图变换
        b0 = to_dense_batch(data.x, batch=data.batch, max_num_nodes=self.n_max_nodes)
        adj = to_dense_adj(data.edge_index, batch=data.batch, max_num_nodes=self.n_max_nodes).to(self.args.device),
        g = {
            'adj': adj[0],
            'x': b0[0].to(self.args.device),
            'mask': b0[1].to(self.args.device),
            'batch': data.batch.to(self.args.device),
            'sparse_edge_index': data.edge_index,
            'target': data.y.to(self.args.device)
        }
        return g

    def transform(self, data):
        # 一对图变换
        new_data = dict()
        norm_ged = self.nged_matrix[data[0]['i'].reshape(-1).tolist(), data[1]['i'].reshape(-1).tolist()].tolist()
        b0 = to_dense_batch(data[0].x, batch=data[0].batch, max_num_nodes=self.n_max_nodes)
        g0 = {
            'adj': to_dense_adj(data[0].edge_index, batch=data[0].batch, max_num_nodes=self.n_max_nodes).to(
                self.args.device),
            'x': b0[0].to(self.args.device),
            'mask': b0[1].to(self.args.device),
            'batch': data[0].batch.to(self.args.device),
            'sparse_edge_index': data[0].edge_index
        }
        b1 = to_dense_batch(data[1].x, batch=data[1].batch, max_num_nodes=self.n_max_nodes)
        g1 = {
            'adj': to_dense_adj(data[1].edge_index, batch=data[1].batch, max_num_nodes=self.n_max_nodes).to(
                self.args.device),
            'x': b1[0].to(self.args.device),
            'mask': b1[1].to(self.args.device),
            'batch': data[1].batch.to(self.args.device),
            'sparse_edge_index': data[1].edge_index
        }
        new_data['g0'] = g0
        new_data['g1'] = g1
        new_data['target'] = torch.from_numpy(np.exp([(-el) for el in norm_ged])).view(-1).float().to(self.args.device)
        return new_data

