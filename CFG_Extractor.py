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
