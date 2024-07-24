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


INS_SEP = " [INS] "  # separates consecutive instructions within a block


def block_to_line(seq):
    """One basic block -> one corpus line, instructions joined by [INS]."""
    return INS_SEP.join(seq)


def iter_block_lines(report):
    """Stream one corpus line per non-empty block (no full materialization)."""
    for instructions in iter_blocks(report):
        seq = [f"{mnemonic} {canonicalize_operands(operands)}".strip()
               for _addr, _hexbytes, mnemonic, operands in instructions]
        yield block_to_line(seq)
