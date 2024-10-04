"""Inter-function graph (function call graph) for a binary.

Nodes are functions; edges come from each function's outrefs / apirefs.
outrefs are NOT all calls - they mix direct calls, tail jumps, conditional
transfers and unresolved targets - so edges are typed rather than collapsed.
"""
import networkx as nx

CALL = "call"
TAIL_JUMP = "tail_jump"
API = "api"
UNRESOLVED = "unresolved"


def api_node(api_name):
    """Stable node id for an imported symbol."""
    return f"api:{api_name}"


def _edge_type(src_func, target, func_ids, own_blocks):
    # a transfer whose target is a known function entry is a call - including
    # when that entry is the source itself, which is plain recursion. A target
    # that lands inside the source's own blocks is a tail jump we already have
    # in the CFG. Anything else is an indirect/unresolved target.
    if target in func_ids:
        return CALL
    if target in own_blocks:
        return TAIL_JUMP
    return UNRESOLVED


def build_call_graph(report, include_apis=True):
    """Return a DiGraph over function offsets (+ optional `api:` nodes).

    Edges carry an `etype` so a consumer can keep or drop unresolved targets
    without re-deriving what an outref meant.
    """
    xcfg = report["xcfg"]
    func_ids = {int(fid) for fid in xcfg}

    fcg = nx.DiGraph()
    # keep every function as a node so isolated ones are not dropped
    for fid in func_ids:
        fcg.add_node(fid, is_api=False)

    for fid, func in xcfg.items():
        src = int(fid)
        own_blocks = {int(addr) for addr in func.get("blocks", {})}
        for _site, targets in func.get("outrefs", {}).items():
            for target in targets:
                t = int(target)
                etype = _edge_type(src, t, func_ids, own_blocks)
                if etype == UNRESOLVED:
                    # an unresolved target is not a function; adding it would
                    # invent a node with no code behind it.
                    continue
                fcg.add_edge(src, t, etype=etype)
        if not include_apis:
            continue
        # named API references become typed edges to synthetic api nodes
        for _site, api_name in func.get("apirefs", {}).items():
            node = api_node(api_name)
            fcg.add_node(node, is_api=True, api_name=api_name)
            fcg.add_edge(src, node, etype=API)
    return fcg


def call_graph_stats(fcg):
    """Node/edge counts broken down by edge type - handy for sanity checks."""
    counts = {CALL: 0, TAIL_JUMP: 0, API: 0}
    for _u, _v, etype in fcg.edges(data="etype"):
        counts[etype] = counts.get(etype, 0) + 1
    n_api = sum(1 for _n, is_api in fcg.nodes(data="is_api") if is_api)
    return {
        "nodes": fcg.number_of_nodes(),
        "functions": fcg.number_of_nodes() - n_api,
        "api_nodes": n_api,
        "edges": fcg.number_of_edges(),
        **{f"edges_{k}": v for k, v in counts.items()},
    }
