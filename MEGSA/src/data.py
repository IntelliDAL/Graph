"""Purpose: Load graph edit distance datasets and build pretraining and downstream graph-pair batches.

Notes: B denotes batch size, N node count, M motif count, D embedding dimension, and F node feature dimension.
"""

import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree, to_dense_adj, to_dense_batch
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree
import torch

from src.runtime import resolve_device


class DatasetProcessor:
    """Purpose: Prepare graph datasets, dense graph tensors, and targets derived from normalized graph edit distances.

    Notes: Preserve the existing computations and parameter attribute names; module names follow Figure 3 in the paper.
    """

    def __init__(self, args):
        """Purpose: Initialize parameters and components for DatasetProcessor.

        Args:
            args: Runtime configuration containing dataset settings, network dimensions, and training hyperparameters.

        Returns:
            None.
        """
        self.args = args
        self.dataset_name = args.dataset
        self.batch_size = args.batch_size
        self.process_dataset()
        self.device = resolve_device(args.device)

    def process_dataset(self):
        """Purpose: Load training and test datasets and generate one-hot degree features when node features are absent.

        Args:
            None.

        Returns:
            None. Updates dataset and normalized graph edit distance matrix attributes.
        """
        print("\nPreparing dataset.\n")

        self.training_graphs = GEDDataset(
            str(self.args.data_dir / self.dataset_name), self.dataset_name, train=True
        )
        self.testing_graphs = GEDDataset(
            str(self.args.data_dir / self.dataset_name), self.dataset_name, train=False
        )

        # Use a shared maximum node degree for one-hot encoding when node features are absent.
        if self.training_graphs[0].x is None:
            max_degree = 0
            for g in (self.training_graphs + self.testing_graphs):
                if g.edge_index.size(1) > 0:
                    max_degree = max(
                        max_degree, int(degree(g.edge_index[0]).max().item())
                    )

            one_hot_degree = OneHotDegree(max_degree, cat=False)
            self.training_graphs.transform = one_hot_degree
            self.testing_graphs.transform = one_hot_degree

        self.nged_matrix = self.training_graphs.norm_ged

    def create_pretrain_batches(self):
        """Purpose: Build a shuffled graph data loader for pretraining.

        Args:
            None.

        Returns:
            Graph data loader with batches of size batch_size.
        """
        dataloader = DataLoader(
            self.training_graphs,
            batch_size=self.batch_size,
            shuffle=True,
        )
        return dataloader

    def transform_pretrain(self, data):
        """Purpose: Convert a sparse graph batch into dense tensors with a fixed node capacity.

        Args:
            data: Graph batch containing node features, edge indices, and batch indices.

        Returns:
            Node features (B, N, F), node mask (B, N), and adjacency matrices (B, N, N).
        """
        x, mask = to_dense_batch(data.x, data.batch, max_num_nodes=self.args.max_nodes)
        adj = to_dense_adj(data.edge_index, data.batch, max_num_nodes=self.args.max_nodes)
        return x, mask, adj

    def create_train_batches(self):
        """Purpose: Shuffle source and target graphs independently and create graph-pair training batches.

        Args:
            None.

        Returns:
            List of tuples containing source and target graph batches.
        """
        source_loader = DataLoader(
            self.training_graphs,
            batch_size=self.batch_size,
            shuffle=True,
        )
        target_loader = DataLoader(
            self.training_graphs,
            batch_size=self.batch_size,
            shuffle=True,
        )
        return list(zip(source_loader, target_loader))

    def transform(self, data):
        """Purpose: Convert graph pairs to dense tensors and normalized graph edit distances to similarity targets.

        Args:
            data: Tuple containing source and target graph batches.

        Returns:
            Dictionary containing g1, g2, and target; targets have shape (B,).
        """
        new_data = dict()
        source_graph_batch = data[0]
        target_graph_batch = data[1]

        # Convert graph pairs to dense features, masks, and adjacency matrices with a fixed node capacity.
        x1, mask1 = to_dense_batch(source_graph_batch.x, source_graph_batch.batch, max_num_nodes=self.args.max_nodes)
        x2, mask2 = to_dense_batch(target_graph_batch.x, target_graph_batch.batch, max_num_nodes=self.args.max_nodes)
        adj1 = to_dense_adj(source_graph_batch.edge_index, source_graph_batch.batch, max_num_nodes=self.args.max_nodes)
        adj2 = to_dense_adj(target_graph_batch.edge_index, target_graph_batch.batch, max_num_nodes=self.args.max_nodes)
        new_data["g1"] = x1, mask1, adj1
        new_data["g2"] = x2, mask2, adj2

        # Look up normalized edit distances by graph index and map them exponentially to similarity targets.
        normalized_ged = self.nged_matrix[
            data[0]["i"].reshape(-1).tolist(), data[1]["i"].reshape(-1).tolist()
        ].tolist()
        new_data["target"] = (
            torch.from_numpy(np.exp([(-el) for el in normalized_ged])).view(-1).float()
        )
        return new_data
