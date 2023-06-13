import numpy as np
import torch

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree, Constant
from torch_geometric.utils import degree, to_dense_batch, to_dense_adj
from torch.utils.data import random_split


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
        if args.use_dist:
            self._gen_normed_dis_dict()

    def process_dataset(self):
        print('\nPreparing dataset.\n')

        self.training_graphs = GEDDataset(self.args.data_dir + self.args.dataset, self.args.dataset, train=True)
        self.testing_graphs = GEDDataset(self.args.data_dir + self.args.dataset, self.args.dataset, train=False)
        self.n_max_nodes = max([g.num_nodes for g in self.training_graphs + self.testing_graphs])
        self.nged_matrix = self.training_graphs.norm_ged
        self.n_all_graphs = self.nged_matrix.shape[0]

        max_degree = 0
        for g in self.training_graphs + self.testing_graphs:
            if g.edge_index.size(1) > 0:
                max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))

        one_hot_degree = OneHotDegree(max_degree, cat=True)
        self.training_graphs.transform = one_hot_degree
        self.testing_graphs.transform = one_hot_degree

        self.args.node_feature_size = self.training_graphs.num_features

        train_num = len(self.training_graphs) - len(self.testing_graphs)
        val_num = len(self.testing_graphs)
        self.training_set, self.val_set = random_split(self.training_graphs, [train_num, val_num])

    def create_batches(self, graphs):
        source_loader = DataLoader(graphs, batch_size=self.args.batch_size, shuffle=True)
        target_loader = DataLoader(graphs, batch_size=self.args.batch_size, shuffle=True)

        return list(zip(source_loader, target_loader))

    def _get_normed_dis_tensor(self, i_list):
        n = len(i_list)
        tar_dist_mat = np.zeros((n, self.n_max_nodes, self.n_max_nodes))
        for index, i in enumerate(i_list):
            tar_dist_mat[index] = self.normed_dist_mat_all[i]

        return torch.from_numpy(tar_dist_mat)

    def _gen_normed_dis_dict(self):
        try:
            f = np.load(self.args.dist_mat_path)
            self.normed_dist_mat_all = f
            self.normed_dist_mat_all *= self.args.dist_start_decay
            return
        except (FileExistsError, FileNotFoundError):
            print('No distance file, generating...')

        dist_mat = np.zeros((self.n_all_graphs, self.n_max_nodes, self.n_max_nodes))

        train_data = list(DataLoader(self.training_graphs, batch_size=len(self.training_graphs), shuffle=False))[0]
        train_adj = to_dense_adj(train_data.edge_index, batch=train_data.batch, max_num_nodes=self.n_max_nodes)

        for train_index, adj in enumerate(train_adj):
            dis_padded = np.ones((self.n_max_nodes, self.n_max_nodes)) * self.n_max_nodes * 10
            for i in range(adj.shape[1]):
                for j in range(adj.shape[1]):
                    if adj[i, j] == 1:
                        dis_padded[i, i] = 0
                        dis_padded[j, j] = 0
                        dis_padded[i, j] = 1

            for i in range(adj.shape[1]):
                for j in range(adj.shape[1]):
                    for k in range(adj.shape[1]):
                        if dis_padded[i, j] > dis_padded[i, k] + dis_padded[k, j]:
                            dis_padded[i, j] = dis_padded[i, k] + dis_padded[k, j]
            dist_mat[train_index] = dis_padded

        test_data = list(DataLoader(self.testing_graphs, batch_size=len(self.testing_graphs), shuffle=False))[0]
        test_adj = to_dense_adj(test_data.edge_index, batch=test_data.batch, max_num_nodes=self.n_max_nodes)

        n_train_graphs = train_adj.shape[0]
        for test_index, adj in enumerate(test_adj):
            dis_padded = np.ones((self.n_max_nodes, self.n_max_nodes)) * self.n_max_nodes * 10
            for i in range(adj.shape[1]):
                for j in range(adj.shape[1]):
                    if adj[i, j] == 1:
                        dis_padded[i, i] = 0
                        dis_padded[j, j] = 0
                        dis_padded[i, j] = 1

            for i in range(adj.shape[1]):
                for j in range(adj.shape[1]):
                    for k in range(adj.shape[1]):
                        if dis_padded[i, j] > dis_padded[i, k] + dis_padded[k, j]:
                            dis_padded[i, j] = dis_padded[i, k] + dis_padded[k, j]
            dist_mat[test_index + n_train_graphs] = dis_padded

        if self.args.norm_dist_method == 'neg':
            self.normed_dist_mat_all = np.exp(-dist_mat)
        elif self.args.norm_dist_method == 'exp^neg':
            self.normed_dist_mat_all = np.exp(-dist_mat)
        np.save(self.args.dist_mat_path, self.normed_dist_mat_all)
        self.normed_dist_mat_all *= self.args.dist_start_decay

    def transform_single(self, data):
        b0 = to_dense_batch(data.x, batch=data.batch, max_num_nodes=self.n_max_nodes)
        adj = to_dense_adj(data.edge_index, batch=data.batch, max_num_nodes=self.n_max_nodes).to(self.args.device),
        g = {
            'adj': adj[0],
            'x': b0[0].to(self.args.device),
            'mask': b0[1].to(self.args.device),
            'batch': data.batch.to(self.args.device),
            'dist': None if not self.args.use_dist else self._get_normed_dis_tensor(data['i']).to(self.args.device),
            'sparse_edge_index': data.edge_index
        }
        return g

    def transform(self, data):
        new_data = dict()
        norm_ged = self.nged_matrix[data[0]['i'].reshape(-1).tolist(), data[1]['i'].reshape(-1).tolist()].tolist()

        b0 = to_dense_batch(data[0].x, batch=data[0].batch, max_num_nodes=self.n_max_nodes)
        g0 = {
            'adj': to_dense_adj(data[0].edge_index, batch=data[0].batch, max_num_nodes=self.n_max_nodes).to(
                self.args.device),
            'x': b0[0].to(self.args.device),
            'mask': b0[1].to(self.args.device),
            'dist': None if not self.args.use_dist else self._get_normed_dis_tensor(data[0]['i']).to(self.args.device)
        }

        b1 = to_dense_batch(data[1].x, batch=data[1].batch, max_num_nodes=self.n_max_nodes)
        g1 = {
            'adj': to_dense_adj(data[1].edge_index, batch=data[1].batch, max_num_nodes=self.n_max_nodes).to(
                self.args.device),
            'x': b1[0].to(self.args.device),
            'mask': b1[1].to(self.args.device),
            'dist': None if not self.args.use_dist else self._get_normed_dis_tensor(data[1]['i']).to(self.args.device)
        }

        new_data['g0'] = g0
        new_data['g1'] = g1
        new_data['target'] = torch.from_numpy(np.exp([(-el) for el in norm_ged])).view(-1).float().to(self.args.device)
        return new_data
