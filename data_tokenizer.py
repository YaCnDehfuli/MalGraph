"""Turn an SMDA report into a text corpus of assembly instructions.

Each basic block becomes one training sequence (many instructions per block),
which is what the DistilBERT masked-LM encoder trains on.
"""
import re
import json
import hashlib

# hex constants: wide ones look like absolute addresses, narrow ones like immediates
_HEX = re.compile(r"0x[0-9a-fA-F]+")
_DEC = re.compile(r"(?<![\w])[0-9]{2,}(?![\w])")

# imports get their own pseudo-block so the encoder learns API names too; the
# call graph attaches them as nodes (see fcg.build_call_graph / sample.py).
API_PREFIX = "[API]"


def canonicalize_operands(operands):
    """Mask concrete values so ASLR-shifted equivalents share tokens.

    Wide hex -> ADDR, narrow hex / plain decimals -> IMM. Registers and memory
    expressions (the [...] parts) are kept as-is.
    """
    def _hex(m):
        return "ADDR" if len(m.group(0)) > 8 else "IMM"  # >6 hex digits ~ address
    text = _HEX.sub(_hex, operands)
    text = _DEC.sub("IMM", text)
    return text


def load_report(path):
    with open(path, "r") as f:
        return json.load(f)


def iter_blocks(report):
    """Yield the instruction list of every basic block in the report."""
    for func in report["xcfg"].values():
        for instructions in func["blocks"].values():
            yield instructions


def api_to_line(api_name):
    """One import -> one corpus line, e.g. 'ws2_32.dll!connect'.

    The '!' and '.' are split by the WordPiece pre-tokenizer's whitespace rule
    only if we help it, so the module and symbol are separated explicitly.
    """
    module, _, symbol = api_name.partition("!")
    return f"{API_PREFIX} {module} {symbol}".strip()


def iter_api_lines(report):
    """Yield one corpus line per distinct import referenced by the report."""
    seen = set()
    for func in report["xcfg"].values():
        for api_name in func.get("apirefs", {}).values():
            if api_name not in seen:
                seen.add(api_name)
                yield api_to_line(api_name)


def extract_block_sequences(report):
    """One sequence (list of 'mnemonic operands' strings) per basic block."""
    sequences = []
    for instructions in iter_blocks(report):
        if not instructions:
            continue  # empty blocks would emit blank corpus lines
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
        if not instructions:
            continue
        seq = [f"{mnemonic} {canonicalize_operands(operands)}".strip()
               for _addr, _hexbytes, mnemonic, operands in instructions]
        yield block_to_line(seq)


def _stable_hash(line):
    """sha1, not builtin hash(): PYTHONHASHSEED randomizes str hashing, which
    would make the deduped corpus differ run to run."""
    return hashlib.sha1(line.encode("utf-8")).hexdigest()


def iter_corpus_lines(report, include_apis=True):
    """Every training line for one report: basic blocks, then imports."""
    yield from iter_block_lines(report)
    if include_apis:
        yield from iter_api_lines(report)


def write_corpus(report, out_path, dedup=True, include_apis=True):
    """Write one basic-block sequence per line, streaming, optionally deduped."""
    with open(out_path, "w") as out:
        return _stream_into(iter_corpus_lines(report, include_apis), out,
                            set() if dedup else None)


def _stream_into(lines, handle, seen):
    n = 0
    for line in lines:
        if seen is not None:
            h = _stable_hash(line)
            if h in seen:
                continue
            seen.add(h)
        handle.write(line + "\n")
        n += 1
    return n


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", default="examples/sample_pid.json",
                        help="a single SMDA report")
    parser.add_argument("--report-dir", default=None,
                        help="a directory of reports (overrides --report)")
    parser.add_argument("--out", default="corpus.txt")
    parser.add_argument("--no-dedup", action="store_true")
    args = parser.parse_args(argv)

    if args.report_dir:
        written, n_reports = write_corpus_from_dir(
            args.report_dir, args.out, dedup=not args.no_dedup)
        print(f"wrote {written} lines from {n_reports} reports to {args.out}")
    else:
        written = write_corpus(load_report(args.report), args.out,
                               dedup=not args.no_dedup)
        print(f"wrote {written} lines to {args.out}")
    return args.out


if __name__ == "__main__":
    main()
