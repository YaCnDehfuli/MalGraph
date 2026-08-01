"""Assemble one binary into a hierarchical sample.

blocks -> (encoder) block embeddings -> per-function CFG -> (DiffPool)
function embedding -> inter-function graph with function embeddings as nodes.

Imported symbols are nodes too. `fcg` already types `api:` edges, but an earlier
version of this file mapped only function offsets onto rows, so every API node
was dropped on the floor and `apirefs` never reached the classifier. An API
node's feature is the encoder's embedding of its `[API] module symbol`
pseudo-block, which lives in the same representation space as the real blocks.
"""
from dataclasses import dataclass, field
from typing import Optional

import torch

from .CFG_Extractor import build_all_cfgs
from .Graph_Loader import cfg_to_pyg
from .data_tokenizer import canonicalize_operands, api_to_line, INS_SEP
from .fcg import build_call_graph


@dataclass
class HierarchicalSample:
    node_embeddings: torch.Tensor       # [num_nodes, dim]: functions, then APIs
    fcg_edge_index: torch.Tensor        # [2, num_call_edges]
    node_ids: list                      # row i -> function offset or "api:..."
    num_functions: int
    is_malware: int
    family: Optional[str] = None
    aux_loss: dict = field(default=None)

    @property
    def function_embeddings(self):
        """Function rows only (API rows are appended after them)."""
        return self.node_embeddings[:self.num_functions]


def block_line(instructions):
    """One basic block -> the canonical text the encoder was trained on."""
    return INS_SEP.join(
        f"{m} {canonicalize_operands(o)}".strip()
        for _a, _h, m, o in instructions
    )


def _function_vectors(cfgs, embedder, function_encoder, cache_dir):
    """Embed every block in the binary once, then pool each CFG."""
    # one embedder call for the whole binary: embedding per function re-pays the
    # tokenizer/model call overhead thousands of times on a real report.
    order, lines = [], []
    for offset, cfg in cfgs.items():
        for node, instructions in cfg.nodes(data="instructions"):
            order.append((offset, node))
            lines.append(block_line(instructions or []))
    vectors = embedder.embed_cached(lines, cache_dir=cache_dir)

    by_function = {offset: {} for offset in cfgs}
    for (offset, node), vector in zip(order, vectors):
        by_function[offset][node] = vector

    func_vecs, offsets = [], []
    link_total = ent_total = None
    for offset, cfg in cfgs.items():
        if cfg.number_of_nodes() == 0:
            continue
        data = cfg_to_pyg(cfg, by_function[offset])
        fvec, link_loss, ent_loss = function_encoder(data)
        func_vecs.append(fvec)
        offsets.append(offset)
        link_total = link_loss if link_total is None else link_total + link_loss
        ent_total = ent_loss if ent_total is None else ent_total + ent_loss
    return func_vecs, offsets, link_total, ent_total


def build_sample(report, embedder, function_encoder, is_malware, family=None,
                 cache_dir="embeddings_cache", include_apis=True):
    """One SMDA report -> one `HierarchicalSample` ready for the classifier."""
    cfgs = build_all_cfgs(report, parallel=False)
    func_vecs, offsets, link_total, ent_total = _function_vectors(
        cfgs, embedder, function_encoder, cache_dir)

    if not func_vecs:
        raise ValueError("report contains no function with any basic block")

    fcg = build_call_graph(report, include_apis=include_apis)
    row = {offset: i for i, offset in enumerate(offsets)}
    node_ids = list(offsets)

    api_nodes = [n for n, is_api in fcg.nodes(data="is_api") if is_api]
    if api_nodes:
        api_lines = [api_to_line(fcg.nodes[n]["api_name"]) for n in api_nodes]
        api_vecs = embedder.embed_cached(api_lines, cache_dir=cache_dir)
        # an import has no CFG, so it is pooled as a one-node graph through the
        # same head the functions use - same weights, same space
        api_vecs = function_encoder.encode_isolated(api_vecs)
        for node, vector in zip(api_nodes, api_vecs):
            row[node] = len(node_ids)
            node_ids.append(node)
            func_vecs.append(vector)

    node_embeddings = torch.stack(func_vecs)
    edges = [[row[u], row[v]] for u, v in fcg.edges() if u in row and v in row]
    edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                  if edges else torch.empty((2, 0), dtype=torch.long))

    zero = node_embeddings.new_zeros(())
    return HierarchicalSample(
        node_embeddings=node_embeddings,
        fcg_edge_index=edge_index,
        node_ids=node_ids,
        num_functions=len(offsets),
        is_malware=is_malware,
        family=family,
        # kept apart, not pre-summed: train.py weights link and entropy with
        # their own lambdas, which a single blended scalar made impossible.
        aux_loss={
            "link": zero if link_total is None else link_total,
            "entropy": zero if ent_total is None else ent_total,
            "n_functions": len(offsets),
        },
    )
