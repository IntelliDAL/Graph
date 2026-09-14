"""Purpose: Manage MEGSA pretraining, downstream training, checkpoint storage, and evaluation.

Notes: B denotes batch size, N node count, M motif count, D embedding dimension, and F node feature dimension.
"""

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from torch_geometric.data import Batch
from tqdm import tqdm

from src.utils import *
from src.model import SelfSupervisedMotifExtraction, MEGSA
from src.data import DatasetProcessor
from src.runtime import load_checkpoint, resolve_device


class Trainer:
    """Purpose: Manage training and evaluation for self-supervised motif extraction and downstream MEGSA similarity learning.

    Notes: Preserve the existing computations and parameter attribute names; module names follow Figure 3 in the paper.
    """

    def __init__(self, args):
        """Purpose: Initialize parameters and components for Trainer.

        Args:
            args: Runtime configuration containing dataset settings, network dimensions, and training hyperparameters.

        Returns:
            None.
        """
        self.args = args
        self.device = resolve_device(args.device)
        self.processor = DatasetProcessor(args)

        self.pretrain_model = SelfSupervisedMotifExtraction(args).to(device=self.device)
        self.model = MEGSA(args).to(self.device)

        # Configure separate optimizers and loss functions for the two training stages.
        self.pretrain_optimizer = torch.optim.Adam(self.pretrain_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.pretrain_criterion = torch.nn.MSELoss(reduction='sum')
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def process_pretrain_batch(self, batch, epoch):
        """Purpose: Run the forward pass, backpropagation, and parameter update for one pretraining batch.

        Args:
            batch: One graph batch.
            epoch: Current pretraining epoch; retained but unused.

        Returns:
            Total batch loss as a Python float.
        """
        self.pretrain_optimizer.zero_grad()
        x, mask, adj = self.processor.transform_pretrain(batch)
        x, mask, adj = x.to(self.device), mask.to(self.device), adj.to(self.device)

        # Combine reconstruction error and structural regularization into the pretraining objective.
        pred_adj, label, embedding, assign_loss = self.pretrain_model((x, mask, adj))
        rec_loss = self.pretrain_criterion(pred_adj, adj)
        loss = rec_loss + assign_loss * self.args.alpha

        loss.backward()
        self.pretrain_optimizer.step()
        return loss.item()

    def pretrain(self):
        """Purpose: Train the self-supervised motif extraction model for the configured number of epochs.

        Args:
            None.

        Returns:
            None. Updates pretrained model parameters and prints training loss.
        """
        print("\nModel training.\n")
        self.pretrain_model.train()

        loss_list = []

        # Process all pretraining batches each epoch and average the loss by graph count.
        for epoch in range(self.args.pretrain_epochs):
            batches = self.processor.create_pretrain_batches()
            loss_sum = sum(self.process_pretrain_batch(batch_pair, epoch) for batch_pair in batches)
            loss = loss_sum / sum(batch_pair.num_graphs for batch_pair in batches)
            loss_list.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.args.pretrain_epochs}, Loss: {loss:.5f}")

    def process_batch(self, batch):
        """Purpose: Train similarity prediction on a downstream graph-pair batch.

        Args:
            batch: Tuple containing source and target graph batches.

        Returns:
            Batch loss as a Python float.
        """
        self.model.train()
        self.pretrain_model.eval()
        for param in self.pretrain_model.parameters():
            param.requires_grad = False

        # Prepare downstream graph pairs and move inputs and targets to the model device.
        self.optimizer.zero_grad()
        batch = self.processor.transform(batch)
        x1, mask1, adj1 = batch["g1"]
        x2, mask2, adj2 = batch["g2"]
        x1, mask1, adj1 = x1.to(self.device), mask1.to(self.device), adj1.to(self.device)
        x2, mask2, adj2 = x2.to(self.device), mask2.to(self.device), adj2.to(self.device)
        target = batch["target"].to(self.device)

        # Extract motif structures for both graphs using the pretrained model.
        _, m_adj1, _, _ = self.pretrain_model((x1, mask1, adj1))
        _, m_adj2, _, _ = self.pretrain_model((x2, mask2, adj2))

        pred = self.model((x1, mask1, adj1, m_adj1), (x2, mask2, adj2, m_adj2))
        loss = self.criterion(pred, target)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        """Purpose: Load pretrained weights and train the downstream model for the configured number of epochs.

        Args:
            None.

        Returns:
            None. Updates downstream model parameters and prints training loss.
        """
        print("\nModel training.\n")
        self.model.train()
        self.load_pretrain()

        loss_list = []

        # Train on shuffled graph pairs each epoch and average the loss by graph-pair count.
        for epoch in range(self.args.epochs):
            batches = self.processor.create_train_batches()
            loss_sum = sum(self.process_batch(batch_pair) for batch_pair in batches)
            loss = loss_sum / sum(batch_pair[0].num_graphs for batch_pair in batches)
            loss_list.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.args.epochs}, Loss: {loss:.5f}")

    def save_pretrain(self):
        """Purpose: Save self-supervised motif extraction weights to the existing pretraining checkpoint path.

        Args:
            None.

        Returns:
            None.
        """
        path = self.args.pretrain_dir / f'pretrain_model_{self.args.dataset}_{self.args.motif_num}.pth'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.pretrain_model.state_dict(), path)
        print(f'Pretrained model saved to {path}.')

    def load_pretrain(self):
        """Purpose: Load self-supervised motif extraction weights from the existing pretraining checkpoint path.

        Args:
            None.

        Returns:
            None. Maps checkpoint tensors to the selected runtime device.
        """
        path = self.args.pretrain_dir / f'pretrain_model_{self.args.dataset}_{self.args.motif_num}.pth'
        self.pretrain_model.load_state_dict(load_checkpoint(path, self.device))

    def save(self):
        """Purpose: Save downstream MEGSA weights to the existing model checkpoint path.

        Args:
            None.

        Returns:
            None.
        """
        path = self.args.model_dir / f'model_{self.args.dataset}_{self.args.motif_num}.pth'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}.')

    def load(self):
        """Purpose: Load the pretrained extractor and downstream MEGSA model weights.

        Args:
            None.

        Returns:
            None. Maps checkpoint tensors to the selected runtime device.
        """
        self.load_pretrain()
        path = self.args.model_dir / f'model_{self.args.dataset}_{self.args.motif_num}.pth'
        self.model.load_state_dict(load_checkpoint(path, self.device))
        print(f'Model loaded from {path}.')

    def score(self):
        """Purpose: Evaluate test-to-training graph similarity and summarize error, rank correlations, and retrieval precision.

        Args:
            None.

        Returns:
            None. Updates evaluation metric attributes and prints results.
        """
        print("\n\nModel evaluation.\n")
        self.model.eval()
        self.pretrain_model.eval()
        for param in self.pretrain_model.parameters():
            param.requires_grad = False
        device = next(self.model.parameters()).device
        self.model.to(device)

        # Allocate storage for predictions and statistics for each test-to-training graph pair.
        n_test = len(self.processor.testing_graphs)
        n_train = len(self.processor.training_graphs)
        scores = np.empty((n_test, n_train))
        prediction_mat = np.empty((n_test, n_train))
        ground_truth = np.empty((n_test, n_train))
        rho_list, tau_list, prec_at_10_list, prec_at_20_list = [], [], [], []

        # Pair the current test graph with every training graph to form the retrieval candidate batch.
        for i, g in tqdm(enumerate(self.processor.testing_graphs), total=n_test):
            source_dataset = [g] * n_train
            target_dataset = self.processor.training_graphs
            source_batch = Batch.from_data_list(source_dataset).to(device)
            target_batch = Batch.from_data_list(target_dataset).to(device)
            batch = self.processor.transform((source_batch, target_batch))
            x1, mask1, adj1 = batch["g1"]
            x2, mask2, adj2 = batch["g2"]
            target = batch["target"].to(device)

            # Disable gradient recording, then extract motifs and predict similarity.
            with torch.no_grad():
                # Extract motif structures for both graphs using the pretrained model.
                _, m_adj1, _, _ = self.pretrain_model((x1, mask1, adj1))
                _, m_adj2, _, _ = self.pretrain_model((x2, mask2, adj2))
                prediction = self.model(
                    (x1, mask1, adj1, m_adj1),
                    (x2, mask2, adj2, m_adj2)
                )

            # Convert predictions and targets to CPU arrays for evaluation.
            prediction_np = prediction.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()

            prediction_mat[i] = prediction_np
            ground_truth[i] = target_np
            scores[i] = F.mse_loss(prediction, target, reduction="none").cpu().detach().numpy()

            def safe_corr(func, pred, true):
                """Purpose: Return zero for constant inputs; otherwise call the specified rank correlation function.

                Args:
                    func: Function used to calculate rank correlation.
                    pred: Array of predicted scores.
                    true: Array of ground-truth scores.

                Returns:
                    Correlation coefficient, or zero for constant arrays.
                """
                if np.std(pred) == 0 or np.std(true) == 0:
                    return 0.0
                return func(pred, true).correlation

            # Record rank correlations and precision at k for the current test graph.
            rho_list.append(safe_corr(spearmanr, prediction_np, target_np))
            tau_list.append(safe_corr(kendalltau, prediction_np, target_np))
            prec_at_10_list.append(calculate_prec_at_k(10, prediction_np, target_np))
            prec_at_20_list.append(calculate_prec_at_k(20, prediction_np, target_np))

        # Aggregate error, rank correlations, and retrieval precision across all test graphs.
        self.mse = np.mean(scores)
        self.rho = float(np.mean(rho_list))
        self.tau = float(np.mean(tau_list))
        self.prec_at_10 = float(np.mean(prec_at_10_list))
        self.prec_at_20 = float(np.mean(prec_at_20_list))
        self.model_error = float(np.mean(scores))
        print_evals(self.mse, self.rho, self.tau, self.prec_at_10, self.prec_at_20)
