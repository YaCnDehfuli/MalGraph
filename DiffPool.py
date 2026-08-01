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


class FunctionEncoder(nn.Module):
    """Encode a sparse PyG CFG into a single function embedding via DiffPool."""

    def __init__(self, in_dim, hidden_dim=128, max_clusters=16):
        super().__init__()
        # raw masked-mean encoder vectors arrive with very uneven scale across
        # dimensions; normalizing them first is worth more than any extra layer
        self.input_norm = nn.LayerNorm(in_dim)
        self.diffpool = DiffPoolNet(in_dim, hidden_dim, max_clusters)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

    def forward(self, data):
        batch = getattr(data, "batch", None)
        x, mask = to_dense_batch(data.x, batch, max_num_nodes=data.num_nodes)
        # to_dense_adj sizes itself from the largest node id that appears in an
        # edge, so a zero-edge function used to come back as an EMPTY tensor:
        # dense_diff_pool then divided the link loss by adj.numel() == 0 and the
        # whole batch went NaN. Blocks with no incident edge (a lone `ret` tail)
        # also silently shrank adj below x. Pinning both to num_nodes fixes the
        # NaN and the shape mismatch at once, and DenseSAGEConv already clamps
        # zero degrees, so no special case is needed here at all.
        adj = to_dense_adj(data.edge_index, batch, max_num_nodes=data.num_nodes)
        pooled, link_loss, ent_loss = self.diffpool(self.input_norm(x), adj, mask)
        return pooled.squeeze(0), link_loss, ent_loss

    def encode_isolated(self, x):
        """Embed nodes that have no CFG of their own - imported symbols.

        They go through the SAME pooling head as real functions, as one-node
        graphs with a self-loop. An earlier version ran them through a separate
        `nn.Linear`, which quietly put half the call graph's node features in a
        different space from the other half.
        """
        x = self.input_norm(x).unsqueeze(1)          # [K, 1, F]
        adj = x.new_ones((x.size(0), 1, 1))          # self-loop
        pooled, _link, _ent = self.diffpool(x, adj)
        return pooled


def smoke():
    """Forward DiffPoolNet on a few random dense graphs; just checks it runs."""
    in_dim, hidden, clusters = 16, 32, 4
    net = DiffPoolNet(in_dim, hidden, clusters)
    for n_nodes in (1, 3, 8, 12):
        x = torch.randn(1, n_nodes, in_dim)
        adj = (torch.rand(1, n_nodes, n_nodes) > 0.5).float()
        mask = torch.ones(1, n_nodes, dtype=torch.bool)
        pooled, link, ent = net(x, adj, mask)
        print(f"nodes={n_nodes} -> pooled {tuple(pooled.shape)}, "
              f"link={link.item():.3f}, ent={ent.item():.3f}")

    # the case that used to NaN: a function whose blocks have no edges at all
    x = torch.randn(1, 5, in_dim)
    adj = torch.zeros(1, 5, 5)
    pooled, link, ent = net(x, adj, torch.ones(1, 5, dtype=torch.bool))
    assert torch.isfinite(pooled).all() and torch.isfinite(link)
    print(f"zero-edge graph -> finite pooled/link/ent (link={link.item():.3f})")
    print("smoke OK")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    if ap.parse_args().smoke:
        smoke()
