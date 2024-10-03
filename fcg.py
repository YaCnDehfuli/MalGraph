"""Inter-function graph (function call graph) for a binary.

Nodes are functions; edges come from each function's outrefs / apirefs.
outrefs are NOT all calls - they mix direct calls, tail jumps, conditional
transfers and unresolved targets - so edges are typed rather than collapsed.
"""
import networkx as nx

CALL = "call"
def api_node(api_name):
    """Stable node id for an imported symbol."""
    return f"api:{api_name}"


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
        # every outref is treated as a call edge
        for _site, targets in func.get("outrefs", {}).items():
            for target in targets:
                fcg.add_edge(src, int(target))
        if not include_apis:
            continue
        for _site, api_name in func.get("apirefs", {}).items():
            node = api_node(api_name)
            fcg.add_node(node, is_api=True, api_name=api_name)
            fcg.add_edge(src, node)
    return fcg
