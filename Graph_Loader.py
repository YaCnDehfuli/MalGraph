"""Convert a per-function CFG (+ block embeddings) into a PyG Data object."""
import torch
from torch_geometric.data import Data


def cfg_to_pyg(cfg, block_vectors):
    """cfg: nx.DiGraph with int node ids; block_vectors: {node_id: tensor}."""
    nodes = list(cfg.nodes())
    idx = {n: i for i, n in enumerate(nodes)}

    x = torch.stack([block_vectors[n] for n in nodes])

    # both directions so message passing can reach predecessors too
    edges = [[idx[u], idx[v]] for u, v in cfg.edges()]
    edges += [[idx[v], idx[u]] for u, v in cfg.edges()]
    edge_index = torch.tensor(edges).t().contiguous()

    return Data(x=x, edge_index=edge_index, num_nodes=len(nodes))
