"""Differentiable-pooling encoder that turns a CFG into one function vector.

Two GNNs per pooling layer: one embeds nodes, one predicts a soft assignment
of nodes to a smaller set of clusters. Operates on dense (padded) batches.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv
from torch_geometric.nn import dense_diff_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch


class GNNBlock(nn.Module):
    """Two DenseSAGEConv layers with ReLU."""

    def __init__(self, in_dim, hidden_dim, out_dim, normalize=True):
        super().__init__()
        # normalize=True L2-normalizes each node embedding, which is what keeps
        # a 400-block function from producing activations two orders of
        # magnitude larger than a 3-block one.
        self.conv1 = DenseSAGEConv(in_dim, hidden_dim, normalize=normalize)
        self.conv2 = DenseSAGEConv(hidden_dim, out_dim, normalize=normalize)

    def forward(self, x, adj, mask=None):
        x = F.relu(self.conv1(x, adj, mask))
        x = F.relu(self.conv2(x, adj, mask))
        return x


class DiffPoolNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, max_clusters):
        super().__init__()
        self.embed_gnn = GNNBlock(in_dim, hidden_dim, hidden_dim)
        # the assignment GNN emits cluster logits, so it must not be
        # L2-normalized - the softmax inside dense_diff_pool needs the scale
        self.assign_gnn = GNNBlock(in_dim, hidden_dim, max_clusters,
                                   normalize=False)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, adj, mask=None):
        z = self.embed_gnn(x, adj, mask)          # node embeddings
        s = self.assign_gnn(x, adj, mask)         # soft cluster assignment (logits)

        # pool nodes into clusters; link + entropy losses regularize the assignment
        x_pool, _adj_pool, link_loss, ent_loss = dense_diff_pool(z, adj, s, mask)

        pooled = self.fc(x_pool.mean(dim=1))      # graph-level function embedding
        return pooled, link_loss, ent_loss
