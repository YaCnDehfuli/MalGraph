"""Hierarchical classifier over the inter-function graph.

Function embeddings (from the CFG DiffPool encoder) plus imported-symbol
embeddings are the node features of the call graph. A small GNN and a global
readout feed two heads: binary (malware or not) and family (which family).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class HierClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_families, dropout=0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = dropout
        # mean and max readouts answer different questions: "what does this
        # binary mostly look like" versus "does it contain anything extreme".
        # Malicious behaviour usually lives in a handful of functions, so the
        # mean alone dilutes it away in a 4k-function binary.
        self.readout_norm = nn.LayerNorm(hidden_dim * 2)
        self.bin_head = nn.Linear(hidden_dim * 2, 1)
        self.fam_head = nn.Linear(hidden_dim * 2, num_families)

    def forward(self, node_embeddings, edge_index):
        x = F.relu(self.conv1(self.input_norm(node_embeddings), edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        g = self.readout_norm(
            torch.cat([x.mean(dim=0), x.max(dim=0).values], dim=-1))
        return self.bin_head(g), self.fam_head(g)

    def loss(self, bin_logit, fam_logits, is_malware, family_idx,
             lam_family=1.0, pos_weight=None):
        # new_tensor keeps the target on the logit's own device/dtype instead of
        # hard-coding CPU float32
        target = bin_logit.new_tensor([float(is_malware)])
        bin_loss = F.binary_cross_entropy_with_logits(
            bin_logit, target,
            pos_weight=None if pos_weight is None
            else bin_logit.new_tensor([pos_weight]))
        # benign samples have no family -> only supervise the family head on
        # malware, otherwise the head learns a phantom "benign" class
        if is_malware and family_idx is not None:
            fam_target = torch.tensor([family_idx], device=fam_logits.device)
            fam_loss = F.cross_entropy(fam_logits.unsqueeze(0), fam_target)
        else:
            fam_loss = bin_logit.new_zeros(())
        return bin_loss + lam_family * fam_loss
