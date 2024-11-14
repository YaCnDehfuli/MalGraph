"""Synthetic SMDA-style reports so the pipeline can run without a memory dump.

The real workflow consumes SMDA reports carved out of process memory. Those
reports are large, sensitive, and cannot be committed, which makes the
repository impossible to execute end to end from a clean checkout. This module
fabricates reports with the *same schema* so every downstream stage - corpus,
tokenizer, encoder, CFG, FCG, training, evaluation, inference - is exercised on
a laptop with no dump, no VM, and no sample.

Nothing here is malware, and nothing here is a substitute for a real corpus.
Instruction bodies are sampled from a pool harvested out of the committed
fixture; each profile adds a few register-level "motifs" and a Windows API mix
so the classes are separable but not trivially so. Numbers produced from this
generator describe the generator, not malware in the wild.
"""
import json
import random
from dataclasses import dataclass, field

BASE_ADDR = 0x140000000
FUNCTION_STRIDE = 0x800

# Terminators are emitted from the CFG shape, never sampled, so the block's last
# instruction always agrees with its out-degree.
_COND_JUMPS = ("je", "jne", "jg", "jle", "js", "jb", "jae")


@dataclass
class Profile:
    """One class of generated binary: its API mix, idioms, and graph shape."""

    name: str
    is_malware: int
    apis: tuple
    motifs: tuple
    n_functions: tuple = (12, 26)
    blocks_per_function: tuple = (1, 9)
    instructions_per_block: tuple = (2, 9)
    site_density: float = 0.55
    call_density: float = 0.25
    api_density: float = 0.30


# Motifs are ordinary x86-64 idioms. They survive operand canonicalization
# (which erases addresses and immediates) because the signal lives in the
# mnemonics and registers, which is exactly what the encoder gets to see.
PROFILES = (
    Profile(
        name="benign_crt",
        is_malware=0,
        apis=("kernel32.dll!GetLastError", "kernel32.dll!HeapAlloc",
              "kernel32.dll!GetTickCount64", "msvcrt.dll!memcpy",
              "kernel32.dll!EnterCriticalSection", "msvcrt.dll!malloc",
              "kernel32.dll!GetModuleHandleW", "msvcrt.dll!_errno"),
        motifs=(("mov", "rax, qword ptr [rsp + 0x20]"),
                ("lea", "rcx, qword ptr [rbp - 0x40]"),
                ("test", "eax, eax"),
                ("mov", "qword ptr [rsp + 0x8], rbx")),
        call_density=0.22,
        api_density=0.28,
    ),
    Profile(
        name="benign_gui",
        is_malware=0,
        apis=("user32.dll!GetMessageW", "user32.dll!DefWindowProcW",
              "user32.dll!DispatchMessageW", "gdi32.dll!BitBlt",
              "user32.dll!CreateWindowExW", "user32.dll!InvalidateRect",
              "gdi32.dll!SelectObject", "user32.dll!SetWindowTextW"),
        motifs=(("lea", "r8, qword ptr [rsp + 0x50]"),
                ("mov", "edx, dword ptr [rbx + 0x24]"),
                ("cmp", "eax, 0x113"),
                ("movaps", "xmmword ptr [rsp + 0x30], xmm0")),
        n_functions=(14, 28),
        call_density=0.24,
        api_density=0.30,
    ),
    Profile(
        name="benign_service",
        is_malware=0,
        apis=("advapi32.dll!RegQueryValueExW", "advapi32.dll!RegOpenKeyExW",
              "advapi32.dll!ReportEventW", "kernel32.dll!SetServiceStatus",
              "kernel32.dll!WaitForSingleObject", "kernel32.dll!CreateEventW",
              "advapi32.dll!RegCloseKey"),
        motifs=(("mov", "dword ptr [rsp + 0x40], 0x1"),
                ("xor", "r9d, r9d"),
                ("cmp", "qword ptr [rbx + 0x10], 0x0"),
                ("sete", "al")),
        n_functions=(10, 22),
        call_density=0.20,
        api_density=0.32,
    ),
    Profile(
        name="benign_net",
        is_malware=0,
        apis=("winhttp.dll!WinHttpOpen", "winhttp.dll!WinHttpSendRequest",
              "winhttp.dll!WinHttpReadData", "wininet.dll!InternetOpenW",
              "ws2_32.dll!getaddrinfo", "crypt32.dll!CertGetNameStringW",
              "winhttp.dll!WinHttpCloseHandle"),
        motifs=(("mov", "r8d, dword ptr [rdi + 0x18]"),
                ("lea", "rdx, qword ptr [rip + 0x51a2]"),
                ("shr", "eax, 0x10"),
                ("add", "rcx, 0x18")),
        n_functions=(12, 26),
        call_density=0.26,
        api_density=0.34,
    ),
    Profile(
        name="Backdoor",
        is_malware=1,
        apis=("ws2_32.dll!socket", "ws2_32.dll!connect", "ws2_32.dll!recv",
              "ws2_32.dll!send", "ws2_32.dll!WSAStartup",
              "kernel32.dll!CreateProcessW", "kernel32.dll!CreatePipe",
              "advapi32.dll!RegSetValueExW"),
        motifs=(("mov", "dword ptr [rsp + 0x30], 0x2"),
                ("movzx", "edx, word ptr [rbx + 0x8]"),
                ("xchg", "ah, al"),
                ("mov", "r8d, 0x6")),
        call_density=0.30,
        api_density=0.42,
    ),
    Profile(
        name="Rootkit",
        is_malware=1,
        apis=("ntdll.dll!NtQuerySystemInformation", "ntdll.dll!NtOpenProcess",
              "kernel32.dll!DeviceIoControl", "ntdll.dll!ZwMapViewOfSection",
              "advapi32.dll!OpenSCManagerW", "advapi32.dll!CreateServiceW"),
        motifs=(("rdmsr", ""),
                ("wrmsr", ""),
                ("cli", ""),
                ("swapgs", ""),
                ("mov", "cr3, rax")),
        n_functions=(10, 20),
        call_density=0.18,
        api_density=0.36,
    ),
    Profile(
        name="Trojan",
        is_malware=1,
        apis=("kernel32.dll!VirtualAlloc", "kernel32.dll!VirtualProtect",
              "kernel32.dll!WriteProcessMemory", "kernel32.dll!CreateRemoteThread",
              "advapi32.dll!CryptDecrypt", "kernel32.dll!LoadLibraryA"),
        motifs=(("xor", "byte ptr [rdx + rcx], al"),
                ("rol", "eax, 0x7"),
                ("ror", "r9d, 0xd"),
                ("shl", "rdx, 0x5"),
                ("not", "ecx")),
        blocks_per_function=(2, 11),
        call_density=0.28,
        api_density=0.34,
    ),
    Profile(
        name="Worm",
        is_malware=1,
        apis=("mpr.dll!WNetOpenEnumW", "mpr.dll!WNetEnumResourceW",
              "kernel32.dll!CopyFileW", "kernel32.dll!FindFirstFileW",
              "kernel32.dll!FindNextFileW", "netapi32.dll!NetShareEnum"),
        motifs=(("rep movsb", ""),
                ("stosq", ""),
                ("repne scasb", ""),
                ("lodsb", ""),
                ("cmpsb", "")),
        n_functions=(14, 30),
        call_density=0.34,
        api_density=0.30,
    ),
)

PROFILES_BY_NAME = {p.name: p for p in PROFILES}
FAMILIES = tuple(p.name for p in PROFILES if p.is_malware)


def harvest_instruction_pool(report):
    """Collect (mnemonic, operands) pairs from a real report as a sampling pool.

    Control-flow instructions are dropped: block terminators are generated from
    the CFG shape instead, so a block never ends in a branch that disagrees with
    its own out-degree.
    """
    skip = {"jmp", "ret", "int3", "call"} | set(_COND_JUMPS)
    pool, seen = [], set()
    for func in report.get("xcfg", {}).values():
        for instructions in func.get("blocks", {}).values():
            for _addr, _hexbytes, mnemonic, operands in instructions:
                if mnemonic in skip or mnemonic.startswith("j"):
                    continue
                key = (mnemonic, operands)
                if key not in seen:
                    seen.add(key)
                    pool.append(key)
    return pool


_FALLBACK_POOL = (
    ("push", "rbx"), ("push", "rbp"), ("push", "rdi"), ("pop", "rbx"),
    ("sub", "rsp, 0x28"), ("add", "rsp, 0x28"), ("mov", "rbx, rcx"),
    ("mov", "rax, qword ptr [rcx]"), ("mov", "qword ptr [rsp + 0x10], rdx"),
    ("lea", "rdx, qword ptr [rip + 0x2a10]"), ("test", "rax, rax"),
    ("cmp", "dword ptr [rbx + 0x18], 0x0"), ("xor", "eax, eax"),
    ("and", "r8d, 0xff"), ("or", "ecx, edx"), ("inc", "r10"),
    ("dec", "esi"), ("movzx", "eax, byte ptr [rdx]"), ("nop", ""),
)


def _synthetic_bytes(rng, mnemonic, operands):
    """Plausible-looking encoded bytes; length drives the next address."""
    n = 1 + (len(mnemonic) + len(operands)) % 6
    return "".join(f"{rng.randrange(256):02x}" for _ in range(n))


def cfg_shape(rng, n_blocks):
    """Successor lists for a small but realistic CFG (branches + one loop)."""
    succ = {i: [] for i in range(n_blocks)}
    for i in range(n_blocks - 1):
        succ[i].append(i + 1)
    for i in range(max(0, n_blocks - 2)):
        if rng.random() < 0.35:
            succ[i].append(rng.randrange(i + 2, n_blocks))
    if n_blocks >= 4 and rng.random() < 0.5:
        src = rng.randrange(n_blocks // 2, n_blocks)
        succ[src].append(rng.randrange(0, src))
    return succ


def _emit_block(rng, addr, n_body, out_degree, pool, profile, api_slots):
    """One basic block: sampled body, motif injection, shape-consistent tail."""
    instructions = []
    cursor = addr
    for _ in range(n_body):
        if rng.random() < 0.30 and profile.motifs:
            mnemonic, operands = rng.choice(profile.motifs)
        else:
            mnemonic, operands = rng.choice(pool)
        hexbytes = _synthetic_bytes(rng, mnemonic, operands)
        instructions.append([cursor, hexbytes, mnemonic, operands])
        cursor += len(hexbytes) // 2

    # call sites double as the anchors for outrefs / apirefs
    call_sites = []
    for _ in range(api_slots):
        hexbytes = _synthetic_bytes(rng, "call", "")
        instructions.append([cursor, hexbytes, "call", f"0x{rng.randrange(1 << 32):x}"])
        call_sites.append(cursor)
        cursor += len(hexbytes) // 2

    if out_degree >= 2:
        mnemonic = rng.choice(_COND_JUMPS)
        operands = f"0x{rng.randrange(1 << 32):x}"
    elif out_degree == 1:
        mnemonic, operands = "jmp", f"0x{rng.randrange(1 << 32):x}"
    else:
        mnemonic, operands = "ret", ""
    hexbytes = _synthetic_bytes(rng, mnemonic, operands)
    instructions.append([cursor, hexbytes, mnemonic, operands])
    cursor += len(hexbytes) // 2

    return instructions, cursor, call_sites


def build_function(rng, offset, pool, profile):
    """One xcfg entry: blocks, blockrefs, and the call sites feeding the FCG."""
    n_blocks = rng.randint(*profile.blocks_per_function)
    succ = cfg_shape(rng, n_blocks)

    blocks, block_addrs, call_sites = {}, [], []
    cursor = offset
    for i in range(n_blocks):
        block_addr = cursor
        n_body = rng.randint(*profile.instructions_per_block)
        api_slots = 1 if rng.random() < profile.site_density else 0
        instructions, cursor, sites = _emit_block(
            rng, block_addr, n_body, len(succ[i]), pool, profile, api_slots)
        blocks[str(block_addr)] = instructions
        block_addrs.append(block_addr)
        call_sites.extend(sites)

    blockrefs = {}
    for i, targets in succ.items():
        if targets:
            blockrefs[str(block_addrs[i])] = [block_addrs[t] for t in targets]

    function = {
        "offset": offset,
        "blocks": blocks,
        "apirefs": {},
        "stringrefs": {},
        "blockrefs": blockrefs,
        "inrefs": [],
        "outrefs": {},
        "is_exported": False,
        "metadata": {
            "binweight": float(sum(len(v) for v in blocks.values())),
            "characteristics": "i--a-pr--f-",
            "confidence": 1.0,
            "function_name": "",
            "nesting_depth": max(1, n_blocks // 3),
        },
    }
    return function, call_sites


def generate_report(profile, seed, pool, filename=None):
    """A full SMDA-shaped report for one synthetic binary."""
    rng = random.Random(seed)
    n_functions = rng.randint(*profile.n_functions)
    offsets = [BASE_ADDR + i * FUNCTION_STRIDE for i in range(n_functions)]

    xcfg, sites_by_offset = {}, {}
    for offset in offsets:
        function, call_sites = build_function(rng, offset, pool, profile)
        xcfg[str(offset)] = function
        sites_by_offset[offset] = call_sites

    # wire the inter-function graph: a call site is either a named import
    # (apiref), a resolved call to another function, or an indirect target SMDA
    # could not attribute - which is what the UNRESOLVED edge type is for.
    for offset in offsets:
        function = xcfg[str(offset)]
        for site in sites_by_offset[offset]:
            roll = rng.random()
            if roll < profile.api_density:
                function["apirefs"][str(site)] = rng.choice(profile.apis)
            elif roll < 0.5 + profile.call_density:
                target = rng.choice(offsets)
                function["outrefs"][str(site)] = [target]
                xcfg[str(target)]["inrefs"].append(site)
            else:
                function["outrefs"][str(site)] = [BASE_ADDR + rng.randrange(1 << 20)]

    n_blocks = sum(len(f["blocks"]) for f in xcfg.values())
    n_instructions = sum(len(b) for f in xcfg.values() for b in f["blocks"].values())
    return {
        "architecture": "intel",
        "base_addr": BASE_ADDR,
        "binary_size": n_instructions * 4,
        "bitness": 64,
        "code_areas": [[BASE_ADDR, BASE_ADDR + n_functions * FUNCTION_STRIDE]],
        "confidence_threshold": 0.0,
        "disassembly_errors": {},
        "execution_time": 0.0,
        "identified_alignment": 0,
        "metadata": {
            "component": "",
            "family": profile.name if profile.is_malware else "",
            "filename": filename or f"{profile.name}_{seed}.dmp",
            "is_library": False,
            "is_buffer": False,
            "version": "",
            "synthetic": True,
        },
        "message": "Synthetic report - not produced by SMDA.",
        "oep": BASE_ADDR,
        "smda_version": "synthetic-1",
        "statistics": {
            "num_functions": n_functions,
            "num_basic_blocks": n_blocks,
            "num_instructions": n_instructions,
            "num_api_calls": sum(len(f["apirefs"]) for f in xcfg.values()),
            "num_function_calls": sum(len(f["outrefs"]) for f in xcfg.values()),
        },
        "status": "ok",
        "timestamp": f"2024-10-{1 + seed % 28:02d}T12-00-00",
        "xcfg": xcfg,
    }


@dataclass
class DatasetSpec:
    """How many binaries to generate, and from which profiles."""

    n_per_profile: int = 8
    seed: int = 1337
    profiles: tuple = field(default_factory=lambda: PROFILES)


def generate_dataset(out_dir, spec=None, seed_report=None):
    """Write one JSON report per synthetic binary; return the label manifest.

    The manifest is what `train`/`eval` consume: filename -> {is_malware, family}.
    """
    import os

    spec = spec or DatasetSpec()
    pool = harvest_instruction_pool(seed_report) if seed_report else []
    pool = pool + list(_FALLBACK_POOL)

    os.makedirs(out_dir, exist_ok=True)
    labels = {}
    counter = 0
    for profile in spec.profiles:
        for i in range(spec.n_per_profile):
            counter += 1
            seed = spec.seed + counter * 977
            name = f"Graph_PID{4000 + counter}_from_{profile.name}_{i:02d}.json"
            report = generate_report(profile, seed, pool, filename=name)
            with open(os.path.join(out_dir, name), "w") as handle:
                json.dump(report, handle)
            labels[name] = {"is_malware": profile.is_malware}
            if profile.is_malware:
                labels[name]["family"] = profile.name
    return labels


def main(argv=None):
    import os
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="reports/", help="output directory")
    parser.add_argument("--per-profile", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--seed-report", default="examples/sample_pid.json",
                        help="real report used only as an instruction pool")
    parser.add_argument("--manifest", default=None,
                        help="where to write the label manifest (default <out>/labels.json)")
    args = parser.parse_args(argv)

    seed_report = None
    if args.seed_report and os.path.exists(args.seed_report):
        with open(args.seed_report) as handle:
            seed_report = json.load(handle)

    spec = DatasetSpec(n_per_profile=args.per_profile, seed=args.seed)
    labels = generate_dataset(args.out, spec, seed_report)

    manifest = args.manifest or os.path.join(args.out, "labels.json")
    with open(manifest, "w") as handle:
        json.dump(labels, handle, indent=2)
    n_mal = sum(v["is_malware"] for v in labels.values())
    print(f"wrote {len(labels)} synthetic reports to {args.out} "
          f"({n_mal} malicious / {len(labels) - n_mal} benign)")
    print(f"label manifest: {manifest}")
    return labels


if __name__ == "__main__":
    main()
