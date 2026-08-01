"""Encoder, DiffPool, sample assembly, and the classification heads."""
import torch
import pytest

from DiffPool import DiffPoolNet, FunctionEncoder
from Graph_Loader import cfg_to_pyg
from model import HierClassifier


def _graph(n_nodes, edges, dim=16):
    import networkx as nx
    cfg = nx.DiGraph()
    for i in range(n_nodes):
        cfg.add_node(i, instructions=[[i, "90", "nop", ""]])
    cfg.add_edges_from(edges)
    return cfg_to_pyg(cfg, {i: torch.randn(dim) for i in range(n_nodes)})


def test_edge_index_is_long_and_directed():
    data = _graph(3, [(0, 1), (1, 2)])
    assert data.edge_index.dtype == torch.long
    assert data.edge_index.shape == (2, 2)
    # symmetrizing would double the edges
    assert data.edge_index.shape[1] == 2


def test_edgeless_graph_gets_the_right_empty_shape():
    data = _graph(3, [])
    assert data.edge_index.shape == (2, 0)
    assert data.edge_index.dtype == torch.long


def test_diffpool_forward_shapes():
    net = DiffPoolNet(16, 32, 4)
    for n_nodes in (1, 3, 12):
        x = torch.randn(1, n_nodes, 16)
        adj = (torch.rand(1, n_nodes, n_nodes) > 0.5).float()
        mask = torch.ones(1, n_nodes, dtype=torch.bool)
        pooled, link, ent = net(x, adj, mask)
        assert pooled.shape == (1, 32)
        assert torch.isfinite(link) and torch.isfinite(ent)


def test_zero_edge_function_does_not_produce_nan():
    """dense_diff_pool divides the link loss by adj.numel(); an unpinned
    to_dense_adj returned an EMPTY adjacency for an edge-less function, so the
    division was 0/0 and the whole binary's loss went NaN."""
    encoder = FunctionEncoder(16, hidden_dim=8, max_clusters=4)
    pooled, link, ent = encoder(_graph(5, []))
    assert torch.isfinite(pooled).all()
    assert torch.isfinite(link) and torch.isfinite(ent)


def test_blocks_with_no_incident_edge_keep_the_adjacency_square():
    """Node 4 appears in no edge, so to_dense_adj sized itself to 4x4 while x
    had 5 rows and DenseSAGEConv multiplied mismatched shapes."""
    encoder = FunctionEncoder(16, hidden_dim=8, max_clusters=4)
    pooled, _link, _ent = encoder(_graph(5, [(0, 1), (1, 2), (2, 3)]))
    assert pooled.shape == (8,)
    assert torch.isfinite(pooled).all()


def test_single_node_function_is_handled():
    encoder = FunctionEncoder(16, hidden_dim=8, max_clusters=4)
    pooled, _link, _ent = encoder(_graph(1, []))
    assert pooled.shape == (8,)
    assert torch.isfinite(pooled).all()


def test_isolated_nodes_land_in_the_same_space_as_functions():
    """Imports used to go through a separate nn.Linear, which put half the call
    graph's features in a different representation space."""
    encoder = FunctionEncoder(16, hidden_dim=8, max_clusters=4)
    function_vec, _l, _e = encoder(_graph(4, [(0, 1), (1, 2), (2, 3)]))
    api_vecs = encoder.encode_isolated(torch.randn(3, 16))
    assert api_vecs.shape == (3, 8)
    assert api_vecs.shape[1] == function_vec.shape[0]


def test_family_head_is_only_supervised_on_malware():
    clf = HierClassifier(8, 8, num_families=3)
    bin_logit = torch.zeros(1, requires_grad=True)
    fam_logits = torch.zeros(3, requires_grad=True)

    benign = clf.loss(bin_logit, fam_logits, is_malware=0, family_idx=None)
    benign.backward()
    assert fam_logits.grad is None or torch.count_nonzero(fam_logits.grad) == 0

    fam_logits = torch.zeros(3, requires_grad=True)
    malware = clf.loss(torch.zeros(1, requires_grad=True), fam_logits,
                       is_malware=1, family_idx=2)
    malware.backward()
    assert torch.count_nonzero(fam_logits.grad) > 0


def test_classifier_forward_shapes():
    clf = HierClassifier(8, 16, num_families=4)
    x = torch.randn(6, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    bin_logit, fam_logits = clf(x, edge_index)
    assert bin_logit.shape == (1,)
    assert fam_logits.shape == (4,)


def test_classifier_survives_a_call_graph_with_no_edges():
    clf = HierClassifier(8, 16, num_families=4)
    bin_logit, fam_logits = clf(torch.randn(3, 8),
                                torch.empty((2, 0), dtype=torch.long))
    assert torch.isfinite(bin_logit).all() and torch.isfinite(fam_logits).all()


def test_positive_class_weight_changes_the_loss():
    clf = HierClassifier(8, 8, num_families=2)
    logit = torch.zeros(1)
    plain = clf.loss(logit, torch.zeros(2), 1, None)
    weighted = clf.loss(logit, torch.zeros(2), 1, None, pos_weight=4.0)
    assert weighted > plain


@pytest.mark.parametrize("mlm_prob", [0.0, 0.15, 1.0])
def test_mlm_collator_masks_and_ignores_the_rest(tiny_dataset, mlm_prob):
    from transformer_train import (
        AsmMLMDataset, load_tokenizer, mlm_collator,
    )
    import os

    tokenizer = load_tokenizer(os.path.join(tiny_dataset["encoder_dir"],
                                            "tokenizer.json"))
    lines = ["push rbx [INS] mov rax, IMM [INS] ret"] * 4
    dataset = AsmMLMDataset(lines, tokenizer, max_len=32)
    batch = mlm_collator([dataset[i] for i in range(4)], tokenizer, mlm_prob)

    assert set(batch) >= {"input_ids", "attention_mask", "labels"}
    assert batch["labels"].shape == batch["input_ids"].shape
    # unmasked positions must be -100 so they do not contribute to the loss
    supervised = batch["labels"] != -100
    assert bool(supervised.any())
    assert torch.equal(batch["input_ids"][supervised],
                       torch.full((int(supervised.sum()),),
                                  tokenizer.mask_token_id))


def test_encoder_embeddings_are_deterministic_and_cached(tiny_dataset, tmp_path):
    from embed_blocks import BlockEmbedder

    embedder = BlockEmbedder(tiny_dataset["encoder_dir"], max_len=64)
    lines = ["push rbx [INS] ret", "xor eax, eax", "push rbx [INS] ret"]
    first = embedder.embed(lines)
    second = embedder.embed(lines)
    assert torch.allclose(first, second)
    # identical text must give an identical vector
    assert torch.allclose(first[0], first[2])

    cached = embedder.embed_cached(lines, cache_dir=str(tmp_path / "cache"))
    assert cached.shape == first.shape
    assert torch.allclose(cached, first, atol=1e-5)


def test_empty_block_text_does_not_crash_the_embedder(tiny_dataset):
    from embed_blocks import BlockEmbedder
    embedder = BlockEmbedder(tiny_dataset["encoder_dir"], max_len=64)
    out = embedder.embed(["", "   ", "ret"])
    assert out.shape[0] == 3
    assert torch.isfinite(out).all()
