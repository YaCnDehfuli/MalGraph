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
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.bin_head = nn.Linear(hidden_dim, 1)
        self.fam_head = nn.Linear(hidden_dim, num_families)

    def forward(self, node_embeddings, edge_index):
        x = F.relu(self.conv1(node_embeddings, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        g = x.mean(dim=0)                       # single-binary graph readout
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
        fam_target = torch.tensor([family_idx or 0], device=fam_logits.device)
        fam_loss = F.cross_entropy(fam_logits.unsqueeze(0), fam_target)
        return bin_loss + lam_family * fam_loss
