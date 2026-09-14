"""Purpose: Implement node encoding, motif assignment, GNN encoding, motif decoding, and matching layers for MEGSA.

Notes: B denotes batch size, N node count, M motif count, D embedding dimension, and F node feature dimension.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class MotifAssignment(nn.Module):
    """Purpose: Perform motif assignment to produce soft memberships, motif adjacency matrices, and structural regularization loss.

    Notes: Preserve the existing computations and parameter attribute names; module names follow Figure 3 in the paper.
    """

    def __init__(self, dim, motifs_num):
        """Purpose: Initialize parameters and components for MotifAssignment.

        Args:
            dim: Input representation dimension.
            motifs_num: Maximum number of motifs.

        Returns:
            None.
        """
        super(MotifAssignment, self).__init__()
        self.assignment_layer = nn.Linear(dim, motifs_num)

    def binary_loss(self, s, mask):
        """Purpose: Compute the motif membership confidence penalty normalized by the number of valid nodes.

        Args:
            s: Soft node-to-motif assignment matrix of shape (B, N, M).
            mask: Node validity mask of shape (B, N), with one for valid nodes and zero for padding.

        Returns:
            Scalar confidence loss; the current implementation does not divide by the motif count.
        """
        loss = s * (1 - s)
        mask_exp = mask.unsqueeze(-1)
        loss = loss * mask_exp
        return loss.sum() / mask_exp.sum()

    def connectivity_loss(self, adj, s, eps=1e-8):
        """Purpose: Compute the graph Laplacian quadratic form and average all matrix entries.

        Args:
            adj: Graph adjacency matrices of shape (B, N, N).
            s: Soft node-to-motif assignment matrix of shape (B, N, M).
            eps: Numerical stability constant; retained but unused in this function.

        Returns:
            Scalar connectivity loss; the current implementation averages all entries rather than taking the trace.
        """
        # Construct the graph Laplacian and project node assignments into a motif-level quadratic form.
        D = adj.sum(dim=-1, keepdim=True)
        L = D * torch.eye(adj.size(-1), device=adj.device).unsqueeze(0) - adj
        loss = torch.einsum('bij,bjk,bkl->bil', s.transpose(1, 2), L, s)
        return loss.mean()

    def orthogonality_loss(self, s, mask=None, eps=1e-15):
        """Purpose: Compare the normalized assignment Gram matrix with the identity to encourage motif diversity.

        Args:
            s: Soft node-to-motif assignment matrix of shape (B, N, M).
            mask: Node validity mask of shape (B, N), with one for valid nodes and zero for padding.
            eps: Small constant for numerical stability in normalization or logarithms.

        Returns:
            Scalar orthogonality loss averaged over the batch.
        """
        B, N, K = s.size()
        if mask is not None:
            mask = mask.unsqueeze(-1)
            s = s * mask

        # Compare the normalized assignment Gram matrix with the identity to measure motif overlap.
        ss = torch.matmul(s.transpose(1, 2), s)
        i_s = torch.eye(K, device=s.device).unsqueeze(0)
        ss_norm = ss / (torch.norm(ss, dim=(-1, -2), keepdim=True) + eps)
        i_s_norm = i_s / (torch.norm(i_s, dim=(-1, -2), keepdim=True) + eps)
        ortho_loss = torch.norm(ss_norm - i_s_norm, dim=(-1, -2))
        return ortho_loss.mean()

    def balance_loss(self, s, mask=None, eps=1e-6):
        """Purpose: Penalize the negative log of total membership per motif to encourage balanced assignments.

        Args:
            s: Soft node-to-motif assignment matrix of shape (B, N, M).
            mask: Node validity mask of shape (B, N), with one for valid nodes and zero for padding.
            eps: Small constant for numerical stability in normalization or logarithms.

        Returns:
            Scalar balance loss averaged over batch and motif dimensions.
        """
        if mask is not None:
            mask = mask.unsqueeze(-1)
            s = s * mask

        cluster_sum = s.sum(dim=1)
        loss = -torch.log(cluster_sum + eps)
        return loss.mean()

    def smooth_threshold(self, p, t=0.4, k=10.0):
        """Purpose: Sharpen soft assignment strengths with a smooth threshold function.

        Args:
            p: Soft assignment strengths to sharpen.
            t: Smooth threshold value.
            k: Sharpness coefficient for the smooth threshold.

        Returns:
            Differentiable tensor with the same shape as p and values between zero and one.
        """
        return torch.sigmoid(k * (p - t))

    def forward(self, x, mask, adj):
        """Purpose: Generate soft assignments, compute four regularization terms, and construct motif adjacency matrices.

        Args:
            x: Node features or embeddings of shape (B, N, D).
            mask: Node validity mask of shape (B, N), with one for valid nodes and zero for padding.
            adj: Graph adjacency matrices of shape (B, N, N).

        Returns:
            Soft assignments (B, N, M), motif adjacency matrices (B, M, N, N), and total structural loss.
        """
        B, N, D = x.size()
        s = self.assignment_layer(x)
        s = F.sigmoid(s)

        # Compute confidence, connectivity, diversity, and balance penalties.
        binary_loss = self.binary_loss(s, mask)
        connectivity_loss = self.connectivity_loss(adj, s)
        ortho_loss = self.orthogonality_loss(s, mask)
        balance_loss = self.balance_loss(s, mask)

        # Expand memberships to node pairs, sharpen them, and restrict motif edges to the original adjacency.
        s_t = s.transpose(1, 2).unsqueeze(-1)
        s_t = s_t.expand(B, -1, N, N)
        m_adj = (s_t + s_t.transpose(-1, -2)) / 2
        m_adj = self.smooth_threshold(m_adj) * adj.unsqueeze(1)

        total_loss = binary_loss + connectivity_loss + ortho_loss * 50 + balance_loss
        return s, m_adj, total_loss


class GraphConvolution(nn.Module):
    """Purpose: Apply graph convolution with normalized adjacency aggregation, optional residual connections, and layer normalization.

    Notes: Preserve the existing computations and parameter attribute names; module names follow Figure 3 in the paper.
    """

    def __init__(self, in_dim, out_dim, residual=True):
        """Purpose: Initialize parameters and components for GraphConvolution.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            residual: Whether to enable residual connections when input and output dimensions match.

        Returns:
            None.
        """
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.residual = residual and (in_dim == out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj, mask=None):
        """Purpose: Normalize adjacency with self-loops, aggregate features, and apply linear transformation and activation.

        Args:
            x: Node features or embeddings of shape (B, N, D).
            adj: Graph adjacency matrices of shape (B, N, N).
            mask: Node validity mask of shape (B, N), with one for valid nodes and zero for padding.

        Returns:
            Node representations of shape (B, N, out_dim).
        """
        # Add self-loops and apply symmetric degree normalization to obtain the propagation matrix.
        A = adj + torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        D_inv_sqrt = torch.pow(A.sum(-1, keepdim=True).clamp(min=1e-8), -0.5)
        A_hat = D_inv_sqrt * A * D_inv_sqrt.transpose(1, 2)

        # Aggregate neighbor information, then apply a linear map, masking, and residual updates.
        out = torch.bmm(A_hat, x)
        out = self.lin(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        if self.residual:
            out = out + x
        out = self.norm(out)
        return F.relu(out)


class AttentionReadout(torch.nn.Module):
    """Purpose: Aggregate node representations into graph or motif representations using attention readout.

    Notes: Preserve the existing computations and parameter attribute names; module names follow Figure 3 in the paper.
    """

    def __init__(self, in_channels):
        """Purpose: Initialize parameters and components for AttentionReadout.

        Args:
            in_channels: Input node representation dimension.

        Returns:
            None.
        """
        super(AttentionReadout, self).__init__()
        self.w = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x, mask=None):
        """Purpose: Compute context-dependent attention weights and aggregate node representations.

        Args:
            x: Node features or embeddings of shape (B, N, D).
            mask: Node validity mask of shape (B, N), with one for valid nodes and zero for padding.

        Returns:
            Graph or motif representations with the node dimension reduced.
        """
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        # Compute node attention weights from global context and aggregate node representations.
        x_w = self.w(x)
        x_mean = torch.tanh(x_w.mean(dim=-2).unsqueeze(-1))
        c = F.sigmoid(torch.matmul(x, x_mean))
        h = torch.matmul(x.transpose(-1, -2), c).squeeze(-1)
        return h


class GNNEncoder(nn.Module):
    """Purpose: Encode both graphs and motifs with a GNN; perform motif encoding during pretraining.

    Notes: Preserve the existing computations and parameter attribute names; module names follow Figure 3 in the paper.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, residual=True):
        """Purpose: Initialize parameters and components for GNNEncoder.

        Args:
            in_dim: Input feature dimension.
            hidden_dim: Graph convolution hidden dimension.
            out_dim: Output feature dimension.
            residual: Whether to enable residual connections when input and output dimensions match.

        Returns:
            None.
        """
        super(GNNEncoder, self).__init__()
        self.gcn1 = GraphConvolution(in_dim, hidden_dim, residual)
        self.gcn2 = GraphConvolution(hidden_dim, out_dim, residual)
        self.pool = AttentionReadout(out_dim)

    def motif_mask(self, adj, alpha=10.0):
        """Purpose: Generate soft node masks from motif node degrees.

        Args:
            adj: Motif adjacency matrices of shape (B, M, N, N).
            alpha: Sharpness coefficient for converting motif node degrees to soft masks.

        Returns:
            Soft masks of shape (B, M, N); training and inference use the same formula.
        """
        degree = adj.sum(dim=-1)
        mask = torch.sigmoid(alpha * (degree - 0.5))
        return mask

    def forward(self, x, mask, adj, m_adj):
        """Purpose: Encode the original graph and motifs with shared graph convolutions, then apply attention readout.

        Args:
            x: Node features or embeddings of shape (B, N, D).
            mask: Node validity mask of shape (B, N), with one for valid nodes and zero for padding.
            adj: Graph adjacency matrices of shape (B, N, N).
            m_adj: Motif adjacency tensor of shape (B, M, N, N).

        Returns:
            Graph representations (B, D) and motif representations (B, M, D).
        """
        B, N, D = x.size()
        _, M, _, _ = m_adj.size()

        # Encode original graph nodes to retain global structural information for graph-level matching.
        n_x = self.gcn1(x, adj, mask)
        n_x = self.gcn2(n_x, adj, mask)

        # Merge batch and motif dimensions to encode motifs in parallel with shared graph convolutions.
        m_x = x.unsqueeze(1).expand(B, M, N, D).reshape(B * M, N, D)
        m_mask = self.motif_mask(m_adj).reshape(B * M, N)
        m_adj = m_adj.reshape(B * M, N, N)
        m_x = self.gcn1(m_x, m_adj, m_mask)
        m_x = self.gcn2(m_x, m_adj, m_mask)
        m_x = m_x * m_mask.unsqueeze(-1)

        # Read out graph and motif representations, then restore batch and motif dimensions.
        g_x = self.pool(n_x, mask)
        m_x = self.pool(m_x, m_mask)
        m_x = m_x.view(B, M, -1)
        g_x = g_x.view(B, -1)
        return g_x, m_x


class NodeEncoding(nn.Module):
    """Purpose: Perform node encoding with a Transformer using node features and adjacency information.

    Notes: Preserve the existing computations and parameter attribute names; module names follow Figure 3 in the paper.
    """

    def __init__(self, dim, d_model, nhead, num_layers, dropout=0.2):
        """Purpose: Initialize parameters and components for NodeEncoding.

        Args:
            dim: Input representation dimension.
            d_model: Transformer hidden dimension.
            nhead: Number of attention heads.
            num_layers: Number of stacked Transformer layers.
            dropout: Dropout probability.

        Returns:
            None.
        """
        super(NodeEncoding, self).__init__()
        self.proj = nn.Linear(dim, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, g):
        """Purpose: Concatenate node features with adjacency rows and encode them with a Transformer.

        Args:
            g: Tuple containing node features, node masks, and adjacency matrices.

        Returns:
            Node representations (B, N, D); the current implementation does not pass the mask to the Transformer.
        """
        x, mask, adj = g
        x = torch.cat([x, adj], dim=-1)
        x = self.proj(x)
        x = self.transformer_encoder(x)
        return x


class MotifDecoding(nn.Module):
    """Purpose: Perform motif decoding with a Transformer to reconstruct adjacency matrices from fused representations.

    Notes: Preserve the existing computations and parameter attribute names; module names follow Figure 3 in the paper.
    """

    def __init__(self, d_model, max_nodes, nhead, num_layers, dropout=0.2):
        """Purpose: Initialize parameters and components for MotifDecoding.

        Args:
            d_model: Transformer hidden dimension.
            max_nodes: Maximum padded node count.
            nhead: Number of attention heads.
            num_layers: Number of stacked Transformer layers.
            dropout: Dropout probability.

        Returns:
            None.
        """
        super(MotifDecoding, self).__init__()
        self.decoder_queries = nn.Parameter(torch.randn(max_nodes, d_model))
        self.output_projection = nn.Linear(d_model, max_nodes)

        self.encoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, node_embeddings):
        """Purpose: Decode fused node representations using learnable queries.

        Args:
            node_embeddings: Node representations fused from motif embeddings, with shape (B, N, D).

        Returns:
            Reconstructed adjacency matrices (B, N, N), without a probability activation.
        """
        B = node_embeddings.size(0)

        # Repeat learnable queries for each sample, decode, and project to adjacency matrices.
        decoder_input = self.decoder_queries.unsqueeze(0).repeat(B, 1, 1)
        decoded = self.transformer_decoder(decoder_input, node_embeddings)
        reconstructed_adj = self.output_projection(decoded)
        return reconstructed_adj


class MotifLevelAlignment(nn.Module):
    """Purpose: Perform motif-level alignment through cross-graph attention and gated representation updates.

    Notes: Preserve the existing computations and parameter attribute names; module names follow Figure 3 in the paper.
    """

    def __init__(self, dim, dropout=0.2):
        """Purpose: Initialize parameters and components for MotifLevelAlignment.

        Args:
            dim: Input representation dimension.
            dropout: Dropout probability.

        Returns:
            None.
        """
        super().__init__()
        self.linear_q = nn.Linear(dim, dim, bias=False)
        self.linear_k = nn.Linear(dim, dim, bias=False)
        self.linear_v = nn.Linear(dim, dim, bias=False)
        self.update = nn.GRUCell(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        """Purpose: Compute cross-graph attention and update both motif representations with a GRU, residual connections, and normalization.

        Args:
            x1: Motif representations of the first graph, with shape (B, M, D).
            x2: Motif representations of the second graph, with shape (B, M, D).

        Returns:
            Updated motif representations for both graphs, each of shape (B, M, D).
        """
        B, M, D = x1.size()

        # Compute scaled dot-product attention to establish soft motif correspondences between graphs.
        q1 = self.linear_q(x1)
        k2 = self.linear_k(x2)
        score = torch.matmul(q1, k2.transpose(-1, -2)) / (D ** 0.5)
        attn = torch.softmax(score, dim=-1)

        # Aggregate cross-graph messages in both directions using the alignment matrix and its transpose.
        v2 = self.linear_v(x2)
        msg1 = torch.matmul(attn, v2)
        msg2 = torch.matmul(attn.transpose(-1, -2), self.linear_v(x1))

        # Treat each motif as a GRU sample and restore the batch shape after updating.
        x1_new = self.update(msg1.reshape(-1, D), x1.reshape(-1, D))
        x2_new = self.update(msg2.reshape(-1, D), x2.reshape(-1, D))
        x1_new = x1_new.reshape(B, M, D)
        x2_new = x2_new.reshape(B, M, D)

        # Fuse updates into the original representations with dropout, residual connections, and layer normalization.
        x1 = self.norm(x1 + self.dropout(x1_new))
        x2 = self.norm(x2 + self.dropout(x2_new))
        return x1, x2


class SimilarityPrediction(nn.Module):
    """Purpose: Predict similarity from fused graph-level matching features and motif representations.

    Notes: Preserve the existing computations and parameter attribute names; module names follow Figure 3 in the paper.
    """

    def __init__(self, in_dim, hid_dim, dropout=0.2):
        """Purpose: Initialize parameters and components for SimilarityPrediction.

        Args:
            in_dim: Input feature dimension.
            hid_dim: Prediction layer hidden dimension.
            dropout: Dropout probability.

        Returns:
            None.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x):
        """Purpose: Apply a nonlinear mapping to fused features and predict similarity.

        Args:
            x: Fused matching features of shape (B, in_dim).

        Returns:
            Scores of shape (B, 1) with values between zero and one.
        """
        x = F.relu(self.norm(self.fc1(x)))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))


class GraphLevelMatching(nn.Module):
    """Purpose: Perform graph-level matching with a neural tensor network to compute interactions between graph representations.

    Notes: Preserve the existing computations and parameter attribute names; module names follow Figure 3 in the paper.
    """

    def __init__(self, in_channel, k=16, dropout=0.0):
        """Purpose: Initialize parameters and components for GraphLevelMatching.

        Args:
            in_channel: Input graph representation dimension.
            k: Number of neural tensor network slices.
            dropout: Dropout probability.

        Returns:
            None.
        """
        super().__init__()
        self.dim = in_channel
        self.k = k
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Define the bilinear interaction tensor, concatenated feature projection, and bias.
        self.W = nn.Parameter(torch.Tensor(k, in_channel, in_channel))
        self.V = nn.Parameter(torch.Tensor(k, 2 * in_channel))
        self.b = nn.Parameter(torch.Tensor(k))
        self.activation = torch.tanh
        self.reset_parameters()

    def reset_parameters(self):
        """Purpose: Initialize weights with Xavier initialization and set biases to zero.

        Args:
            None.

        Returns:
            None.
        """
        nn.init.xavier_uniform_(self.W.view(self.k, -1))
        nn.init.xavier_uniform_(self.V.unsqueeze(-1))
        nn.init.zeros_(self.b)

    def forward(self, u, v, mask=None):
        """Purpose: Compute bilinear tensor interactions and a concatenated linear term, then apply activation and dropout.

        Args:
            u: Graph-level representations of the first graph, with shape (B, D).
            v: Graph-level representations of the second graph, with shape (B, D).
            mask: Optional graph-pair mask of shape (B,); unused in the current implementation.

        Returns:
            Graph-level matching features of shape (B, k); mask is currently unused.
        """
        B, D = u.shape
        assert D == self.dim, f"Expected dim {self.dim}, got {D}"
        u_exp = u.unsqueeze(1)
        v_exp = v.unsqueeze(1)

        # Compute bilinear interactions across tensor slices and add the linear term from concatenated features.
        uW_v = torch.einsum('bd,kde,be->bk', u, self.W, v)
        uv = torch.cat([u, v], dim=-1)
        term2 = torch.matmul(uv, self.V.t())
        t = uW_v + term2 + self.b.unsqueeze(0)

        h = self.activation(t)
        h = self.dropout(h)
        return h
