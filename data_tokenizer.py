"""Turn an SMDA report into a text corpus of assembly instructions.

Each basic block becomes one training sequence (many instructions per block),
which is what the DistilBERT masked-LM encoder trains on.
"""
import re
import json
import hashlib

# hex constants: wide ones look like absolute addresses, narrow ones like immediates
def load_report(path):
    with open(path, "r") as f:
        return json.load(f)


def iter_blocks(report):
    """Yield the instruction list of every basic block in the report."""
    for func in report["xcfg"].values():
        for instructions in func["blocks"].values():
            yield instructions


def extract_block_sequences(report):
    """One sequence (list of 'mnemonic operands' strings) per basic block."""
    sequences = []
    for instructions in iter_blocks(report):
        seq = [f"{mnemonic} {canonicalize_operands(operands)}".strip()
               for _addr, _hexbytes, mnemonic, operands in instructions]
        sequences.append(seq)
    return sequences
