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

from CFG_Extractor import build_all_cfgs
from Graph_Loader import cfg_to_pyg
from data_tokenizer import canonicalize_operands, api_to_line, INS_SEP
from fcg import build_call_graph


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
    by_function = {}
    for offset, cfg in cfgs.items():
        lines = {n: block_line(i or []) for n, i in cfg.nodes(data="instructions")}
        vectors = embedder.embed(list(lines.values()))
        by_function[offset] = dict(zip(lines, vectors))

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
        aux_loss={
            "total": zero if link_total is None else link_total + ent_total,
            "n_functions": len(offsets),
        },
    )
