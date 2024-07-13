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


def _build_one(func_data):
    """Worker: build the whole function graph in the child process."""
    cfg = nx.DiGraph()
    for addr, instr in func_data["blocks"].items():
        cfg.add_node(int(addr), instructions=instr)
    for src, targets in func_data["blockrefs"].items():
        for target in targets:
            cfg.add_edge(int(src), int(target))
    return cfg


def extract_functions_parallel(smda_data):
    """Parallel version so the 3.9k-function report doesn't take forever.

    One task per *function* on a single shared pool - submitting per basic
    block deadlocked on a pool created inside the loop.
    """
    functions = smda_data["xcfg"]
    graphs = {}
    with ProcessPoolExecutor() as ex:
        future_to_id = {ex.submit(_build_one, func_data): func_id
                        for func_id, func_data in functions.items()}
        for fut in as_completed(future_to_id):
            graphs[future_to_id[fut]] = fut.result()
    return graphs


def build_all_cfgs(smda_data, parallel=True):
    """Return {function_offset(int): nx.DiGraph} for every function in the report."""
    raw = (extract_functions_parallel(smda_data) if parallel
           else extract_functions(smda_data))
    return {int(func_id): cfg for func_id, cfg in raw.items()}
