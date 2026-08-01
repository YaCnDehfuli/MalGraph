import json
import time
import argparse
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed


def load_smda_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def build_function_cfg(func_data):
    """Build a directed control-flow graph for one function."""
    cfg = nx.DiGraph()
    # block keys arrive as strings, blockref targets as ints -> normalize both to int
    # so the same address is a single node (otherwise "123" and 123 split into two).
    for block_addr, instructions in func_data["blocks"].items():
        # each instruction is [addr, hex_bytes, mnemonic, operands]
        cfg.add_node(int(block_addr), instructions=instructions)
    for src, targets in func_data["blockrefs"].items():
        for target in targets:
            cfg.add_edge(int(src), int(target))
    return cfg


def extract_functions(smda_data):
    functions = smda_data["xcfg"]
    graphs = {}
    for func_id, func_data in functions.items():
        graphs[func_id] = build_function_cfg(func_data)
    return graphs


def _cfg_parts(func_data):
    """Worker: return plain (nodes, edges) lists instead of an nx graph.

    Shipping the nx.DiGraph (with instruction lists as node attrs) back through
    the pool is slow/fragile to pickle; plain tuples travel cleanly and the
    parent reassembles the graph.
    """
    nodes = [(int(addr), instr) for addr, instr in func_data["blocks"].items()]
    edges = [(int(src), int(t))
             for src, targets in func_data["blockrefs"].items() for t in targets]
    return nodes, edges


def extract_functions_parallel(smda_data):
    """Parallel version so the 3.9k-function report doesn't take forever.

    One task per *function* on a single shared pool; workers return picklable
    node/edge lists and the parent builds the nx graphs.
    """
    functions = smda_data["xcfg"]
    graphs = {}
    with ProcessPoolExecutor() as ex:
        future_to_id = {ex.submit(_cfg_parts, func_data): func_id
                        for func_id, func_data in functions.items()}
        for fut in as_completed(future_to_id):
            func_id = future_to_id[fut]
            nodes, edges = fut.result()
            cfg = nx.DiGraph()
            for addr, instr in nodes:
                cfg.add_node(addr, instructions=instr)
            cfg.add_edges_from(edges)
            graphs[func_id] = cfg
    return graphs


def build_all_cfgs(smda_data, parallel=True):
    """Return {function_offset(int): nx.DiGraph} for every function in the report."""
    raw = (extract_functions_parallel(smda_data) if parallel
           else extract_functions(smda_data))
    return {int(func_id): cfg for func_id, cfg in raw.items()}


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(round((len(sorted_vals) - 1) * q)))
    return sorted_vals[idx]


def graph_stats(graphs):
    """Corpus-level stats over all function CFGs."""
    n_funcs = len(graphs)
    total_nodes = sum(g.number_of_nodes() for g in graphs.values())
    total_edges = sum(g.number_of_edges() for g in graphs.values())
    block_lens = sorted(len(instr)
                        for g in graphs.values()
                        for _, instr in g.nodes(data="instructions") if instr)
    return {
        "functions": n_funcs,
        "blocks": total_nodes,
        "cfg_edges": total_edges,
        "block_len_p50": _percentile(block_lens, 0.50),
        "block_len_p99": _percentile(block_lens, 0.99),
        "block_len_max": block_lens[-1] if block_lens else 0,
    }


def visualize_cfg(cfg, title="", save_to=None):
    """Draw one function CFG. Delegates to `viz` so the layout stays in one place."""
    from .viz import draw_cfg
    return draw_cfg(cfg, title=title, save_to=save_to)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__ or "SMDA report -> CFGs")
    ap.add_argument("--report", default="examples/sample_pid.json")
    ap.add_argument("--serial", action="store_true", help="disable parallel build")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--draw", action="store_true")
    ap.add_argument("--save-to", default=None,
                    help="write the drawing here instead of opening a window")
    args = ap.parse_args(argv)

    smda = load_smda_json(args.report)
    t0 = time.time()
    graphs = build_all_cfgs(smda, parallel=not args.serial)
    print(f"built {len(graphs)} function CFGs in {time.time() - t0:.2f}s")
    if args.stats:
        for k, v in graph_stats(graphs).items():
            print(f"  {k}: {v}")
    if args.draw or args.save_to:
        # biggest function first: a one-block CFG makes a useless picture
        fid = max(graphs, key=lambda f: graphs[f].number_of_nodes())
        visualize_cfg(graphs[fid], title=f"Function 0x{fid:x}",
                      save_to=args.save_to)
    return graphs


if __name__ == "__main__":
    main()
