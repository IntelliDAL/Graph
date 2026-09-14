"""Purpose: Define the self-supervised motif extraction and motif interaction graph similarity learning stages of MEGSA.

Notes: B denotes batch size, N node count, M motif count, D embedding dimension, and F node feature dimension.
"""

import torch

from src.layers import *
import torch.nn as nn


class SelfSupervisedMotifExtraction(nn.Module):
    """Purpose: Perform self-supervised motif extraction through motif decomposition and reconstruction.

    Notes: Preserve the existing computations and parameter attribute names; module names follow Figure 3 in the paper.
    """

    def __init__(self, args):
        """Purpose: Initialize parameters and components for SelfSupervisedMotifExtraction.

        Args:
            args: Runtime configuration containing dataset settings, network dimensions, and training hyperparameters.

        Returns:
            None.
        """
        super(SelfSupervisedMotifExtraction, self).__init__()

        # Configure the node encoder for motif decomposition.
        self.encoder = NodeEncoding(
            dim=args.max_nodes + args.node_feat_dim,
            d_model=64,
            nhead=4,
            num_layers=3,
            dropout=args.dropout
        )

        # Configure the decoder for motif reconstruction.
        self.decoder = MotifDecoding(
            d_model=64,
            max_nodes=args.max_nodes,
            nhead=4,
            num_layers=3,
            dropout=args.dropout
        )

        self.assignment = MotifAssignment(dim=64, motifs_num=args.motif_num)
        self.motif_encoder = GNNEncoder(in_dim=args.node_feat_dim, hidden_dim=64, out_dim=64)

    def forward(self, g):
        """Purpose: Run node encoding, motif assignment, motif encoding, and adjacency reconstruction in sequence.

        Args:
            g: Tuple containing node features, node masks, and adjacency matrices.

        Returns:
            Reconstructed adjacency (B, N, N), motif adjacency (B, M, N, N), motif embeddings (B, M, D), and structural loss.
        """
        x, mask, adj = g

        # Motif decomposition: encode nodes and assign them to motifs.
        embedding = self.encoder(g)
        label, m_adj, assign_loss = self.assignment(embedding, mask, adj)

        # Motif reconstruction: encode motifs, fuse representations, and decode adjacency.
        _, motif_embedding = self.motif_encoder(x, mask, adj, m_adj)
        re_embedding = torch.matmul(label, motif_embedding)
        adj = self.decoder(re_embedding)
        return adj, m_adj, motif_embedding, assign_loss


class MEGSA(nn.Module):
    """Purpose: Implement the MEGSA downstream model for motif interaction graph similarity learning.

    Notes: Preserve the existing computations and parameter attribute names; module names follow Figure 3 in the paper.
    """

    def __init__(self, args):
        """Purpose: Initialize parameters and components for MEGSA.

        Args:
            args: Runtime configuration containing dataset settings, network dimensions, and training hyperparameters.

        Returns:
            None.
        """
        super(MEGSA, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.attention_layers = args.attention_layers

        self.attention = MotifLevelAlignment(dim=64, dropout=self.dropout)
        self.lin = SimilarityPrediction(in_dim=128 + 64, hid_dim=64, dropout=self.dropout)
        self.encoder = GNNEncoder(in_dim=args.node_feat_dim, hidden_dim=64, out_dim=64)
        self.ntn = GraphLevelMatching(in_channel=64, k=64)

    def forward(self, g1, g2):
        """Purpose: Combine motif-level alignment results with graph-level matching features to predict graph-pair similarity.

        Args:
            g1: Tuple of node features, node masks, adjacency, and motif adjacency for the first graph.
            g2: Tuple of node features, node masks, adjacency, and motif adjacency for the second graph.

        Returns:
            Similarity scores; squeeze returns a scalar when the batch contains a single graph pair.
        """
        x1, mask1, adj1, m_adj1 = g1
        x2, mask2, adj2, m_adj2 = g2

        # Use the shared encoder to extract graph-level and motif representations for both graphs.
        g_x1, m_x1 = self.encoder(x1, mask1, adj1, m_adj1)
        g_x2, m_x2 = self.encoder(x2, mask2, adj2, m_adj2)

        # Align motifs through multiple rounds of cross-graph interaction.
        for _ in range(self.attention_layers):
            m_x1, m_x2 = self.attention(m_x1, m_x2)

        # Pool motif representations and combine them with graph-level matching features to predict similarity.
        m_x1, m_x2 = m_x1.mean(dim=1), m_x2.mean(dim=1)
        g = self.ntn(g_x1, g_x2)
        h = torch.cat((g, m_x1, m_x2), dim=-1)
        h = self.lin(h)
        return h.squeeze()
