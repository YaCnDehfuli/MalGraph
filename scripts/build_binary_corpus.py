#!/usr/bin/env python3
"""Build a labeled SMDA corpus from real executables on the local system.

The memory-dump corpus this project was built for is not redistributable: it is
captured from live samples inside a lab VM. This script produces a *public,
reproducible* corpus instead, from binaries that already exist on the machine,
so the pipeline can be trained and validated on real disassembly by anyone.

Two labelings are available.

`--mode packing` compresses half the binaries with UPX, the packer family
malware has used for decades: a packed file exposes only its unpacking stub to
static disassembly, and the real code appears only once it has been unpacked in
memory. That makes "packed or not" both a supervised task and a direct
measurement of how much code static analysis loses. It needs the `upx` binary.

`--mode package` labels each binary with its source package instead. Nothing is
packed, and the classes share a compiler, a libc and a coding style, so the
label cannot be read off the function count - it is the harder of the two and
the better test of the learned representation.

    python scripts/build_binary_corpus.py --mode packing --out corpus/ --limit 32
    python scripts/build_binary_corpus.py --mode package --out corpus/ --packages 4

Requires `smda`.
"""
import os
import sys
import json
import time
import shutil
import argparse
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(REPO_ROOT, "src")):
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

DEFAULT_SOURCES = ("/usr/bin", "/bin")


def is_elf64(path):
    try:
        with open(path, "rb") as handle:
            head = handle.read(20)
    except OSError:
        return False
    # \x7fELF, 64-bit class, and an executable/shared object type
    return head[:4] == b"\x7fELF" and len(head) >= 20 and head[4] == 2


def candidate_binaries(sources, min_size, max_size):
    """Deterministically ordered ELF64 executables in the given directories."""
    found = []
    for source in sources:
        if not os.path.isdir(source):
            continue
        for name in sorted(os.listdir(source)):
            path = os.path.join(source, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            if not os.access(path, os.X_OK):
                continue
            size = os.path.getsize(path)
            if not (min_size <= size <= max_size) or not is_elf64(path):
                continue
            found.append(path)
    # de-duplicate identical files reachable under two prefixes
    seen, unique = set(), []
    for path in found:
        key = os.path.realpath(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def pack(source, destination, level="-1"):
    """UPX-compress `source` into `destination`; True if UPX accepted it."""
    shutil.copy2(source, destination)
    result = subprocess.run(["upx", "-q", level, destination],
                            capture_output=True, text=True)
    if result.returncode != 0:
        os.path.exists(destination) and os.remove(destination)
        return False
    return True


def report_stats(report):
    xcfg = report.get("xcfg", {})
    blocks = sum(len(f.get("blocks", {})) for f in xcfg.values())
    instructions = sum(len(b) for f in xcfg.values()
                       for b in f.get("blocks", {}).values())
    apis = sum(len(f.get("apirefs", {})) for f in xcfg.values())
    return {"functions": len(xcfg), "blocks": blocks,
            "instructions": instructions, "api_refs": apis}


def owning_package(path):
    """The dpkg package a file belongs to, or None."""
    try:
        out = subprocess.run(["dpkg", "-S", os.path.realpath(path)],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return out.stdout.split(":", 1)[0].strip() or None


def build_package_corpus(out_dir, per_package=12, packages=4,
                         sources=DEFAULT_SOURCES, min_size=15_000,
                         max_size=400_000, log=print):
    """A harder, packer-free corpus: which source package is this code from?

    Packed-vs-unpacked is nearly separable by function count alone, so it says
    little about whether the model reads structure. Source package is a real,
    verifiable label over binaries that share a compiler, a libc and a coding
    style, which makes it a genuine test of the representation.

    The family head is only supervised on positives, so every sample here is
    labeled positive; the binary head is unused in this mode and its metrics
    are meaningless.
    """
    from smda.Disassembler import Disassembler

    reports_dir = os.path.join(out_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    by_package = {}
    for path in candidate_binaries(sources, min_size, max_size):
        package = owning_package(path)
        if package:
            by_package.setdefault(package, []).append(path)

    ranked = sorted(by_package.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    chosen = [(name, paths) for name, paths in ranked
              if len(paths) >= per_package][:packages]
    log("packages: " + ", ".join(f"{n} ({len(p)})" for n, p in chosen))

    disassembler = Disassembler()
    labels = {}
    for package, paths in chosen:
        taken = 0
        for path in paths:
            if taken >= per_package:
                break
            name = os.path.basename(path)
            try:
                report = disassembler.disassembleFile(path).toDict()
            except Exception as exc:
                log(f"  skip {name}: {exc}")
                continue
            if len(report.get("xcfg", {})) < 20:
                continue                  # too little code to be informative
            filename = f"report_{package}_{name}.json"
            with open(os.path.join(reports_dir, filename), "w") as handle:
                json.dump(report, handle)
            labels[filename] = {"is_malware": 1, "family": package}
            taken += 1
            log(f"  {package:28s} {name:22s} {len(report['xcfg'])} functions")

    with open(os.path.join(reports_dir, "labels.json"), "w") as handle:
        json.dump(labels, handle, indent=2)
    log(f"\n{len(labels)} reports over {len(chosen)} packages in {reports_dir}")
    return labels


def build(out_dir, limit=32, sources=DEFAULT_SOURCES, min_size=15_000,
          max_size=400_000, level="-1", figure=None, log=print):
    from smda.Disassembler import Disassembler

    reports_dir = os.path.join(out_dir, "reports")
    work_dir = os.path.join(out_dir, "work")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    disassembler = Disassembler()
    labels, pairs = {}, []
    candidates = candidate_binaries(sources, min_size, max_size)
    log(f"{len(candidates)} candidate binaries; taking up to {limit}")

    for path in candidates:
        if len(pairs) >= limit:
            break
        name = os.path.basename(path)
        packed_path = os.path.join(work_dir, name + ".upx")
        if not pack(path, packed_path, level):
            continue                      # UPX refuses some binaries; skip them

        try:
            started = time.time()
            plain = disassembler.disassembleFile(path).toDict()
            packed = disassembler.disassembleFile(packed_path).toDict()
        except Exception as exc:          # a binary SMDA cannot handle
            log(f"  skip {name}: {exc}")
            continue

        plain_name = f"report_{name}_plain.json"
        packed_name = f"report_{name}_upx.json"
        for filename, report in ((plain_name, plain), (packed_name, packed)):
            with open(os.path.join(reports_dir, filename), "w") as handle:
                json.dump(report, handle)

        labels[plain_name] = {"is_malware": 0}
        labels[packed_name] = {"is_malware": 1}
        before, after = report_stats(plain), report_stats(packed)
        pairs.append({"binary": name, "size": os.path.getsize(path),
                      "plain": before, "packed": after})
        log(f"  {name:24s} {before['functions']:5d} -> {after['functions']:4d} "
            f"functions, {before['blocks']:6d} -> {after['blocks']:5d} blocks "
            f"({time.time() - started:.1f}s)")

    with open(os.path.join(reports_dir, "labels.json"), "w") as handle:
        json.dump(labels, handle, indent=2)

    summary = summarize(pairs, level)
    with open(os.path.join(out_dir, "packing_stats.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    if figure and pairs:
        from memory_cfg import viz
        viz.draw_recovery_comparison(pairs, save_to=figure)
        log(f"figure: {figure}")
    shutil.rmtree(work_dir, ignore_errors=True)

    log(f"\n{len(labels)} reports ({len(pairs)} pairs) in {reports_dir}")
    log(f"static disassembly recovers "
        f"{summary['recovered_function_ratio'] * 100:.1f}% of the functions "
        f"and {summary['recovered_block_ratio'] * 100:.1f}% of the basic blocks "
        f"once the binary is packed")
    return labels, summary


def summarize(pairs, level="-1"):
    """How much of the code static analysis keeps after packing."""
    if not pairs:
        return {"pairs": 0}
    total = {key: sum(p["plain"][key] for p in pairs)
             for key in ("functions", "blocks", "instructions", "api_refs")}
    packed_total = {key: sum(p["packed"][key] for p in pairs)
                    for key in ("functions", "blocks", "instructions", "api_refs")}
    return {
        "pairs": len(pairs),
        "upx_level": level,
        "plain_totals": total,
        "packed_totals": packed_total,
        "recovered_function_ratio": packed_total["functions"] / max(1, total["functions"]),
        "recovered_block_ratio": packed_total["blocks"] / max(1, total["blocks"]),
        "recovered_instruction_ratio": (packed_total["instructions"]
                                        / max(1, total["instructions"])),
        "median_functions_plain": sorted(p["plain"]["functions"] for p in pairs)[len(pairs) // 2],
        "median_functions_packed": sorted(p["packed"]["functions"] for p in pairs)[len(pairs) // 2],
        "per_binary": pairs,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="corpus", help="output directory")
    ap.add_argument("--mode", choices=("packing", "package"), default="packing",
                    help="packing: packed vs unpacked (needs upx). "
                         "package: which source package the code came from")
    ap.add_argument("--limit", type=int, default=32,
                    help="packing mode: binaries (each yields a plain+packed pair)")
    ap.add_argument("--per-package", type=int, default=12)
    ap.add_argument("--packages", type=int, default=4)
    ap.add_argument("--sources", nargs="*", default=list(DEFAULT_SOURCES))
    ap.add_argument("--min-size", type=int, default=15_000)
    ap.add_argument("--max-size", type=int, default=400_000)
    ap.add_argument("--upx-level", default="-1")
    ap.add_argument("--figure", default=None,
                    help="write the packed/unpacked recovery chart here")
    args = ap.parse_args(argv)

    if args.mode == "package":
        return build_package_corpus(args.out, args.per_package, args.packages,
                                    args.sources, args.min_size, args.max_size)
    if shutil.which("upx") is None:
        raise SystemExit("upx not found; install upx-ucl (or upx) first")
    build(args.out, args.limit, args.sources, args.min_size, args.max_size,
          args.upx_level, args.figure)


if __name__ == "__main__":
    main()
