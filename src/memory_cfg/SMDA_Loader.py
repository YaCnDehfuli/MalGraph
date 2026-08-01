"""Carve a process out of a memory dump (Volatility 3) and disassemble it (SMDA).

This is the only stage that needs the forensics stack, and the only stage that
cannot run from a clean checkout. `smda` is therefore imported lazily, inside
the method that uses it: importing it at module scope made `import memory_cfg`
fail outright on any machine without it, which took the whole pipeline down
along with the one part that genuinely needs a lab host.
"""
import os
import json
import shutil
import logging
import argparse
import subprocess

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def find_volatility(explicit=None):
    """Locate vol.py / vol: explicit path, then $VOLATILITY, then PATH."""
    if explicit:
        return explicit
    env = os.environ.get("VOLATILITY")
    if env:
        return env
    for candidate in ("vol", "vol.py", "volatility3"):
        found = shutil.which(candidate)
        if found:
            return found
    raise FileNotFoundError(
        "Volatility 3 not found. Install it, put `vol` on PATH, set $VOLATILITY, "
        "or pass --volatility /path/to/vol.py")


class Graph_Loader:
    """Carve a process out of a memory dump and disassemble it with SMDA."""

    def __init__(self, dump_dir, dump_name, pid, output_dir=None,
                 volatility=None, architecture="intel.64bit",
                 dump_suffix=".raw"):
        self.pid = pid
        self.dump_name = dump_name
        self.dump_dir = dump_dir
        self.output_dir = output_dir or f"pid{pid}"
        self.volatility = volatility
        self.architecture = architecture
        self.dump_suffix = dump_suffix

    @property
    def dump_path(self):
        return os.path.join(self.dump_dir, f"{self.dump_name}{self.dump_suffix}")

    def process_dump(self):
        if not os.path.exists(self.dump_path):
            raise FileNotFoundError(f"memory dump not found: {self.dump_path}")
        os.makedirs(self.output_dir, exist_ok=True)
        command = [
            find_volatility(self.volatility),
            "-f", self.dump_path,
            "-o", self.output_dir,
            "windows.pslist.PsList",
            "--pid", str(self.pid),
            "--dump",
        ]
        logging.info("Extracting process %s from %s", self.pid, self.dump_name)
        # the original call ignored the exit status, so a failed carve surfaced
        # much later as a confusing "no dumped image" error
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"volatility exited with {result.returncode}; "
                               "command: " + " ".join(command))

    def carved_image(self):
        candidates = [f for f in sorted(os.listdir(self.output_dir))
                      if f.endswith(".dmp") and str(self.pid) in f]
        if not candidates:
            raise FileNotFoundError(
                f"no dumped image for pid {self.pid} in {self.output_dir}")
        return os.path.join(self.output_dir, candidates[0])

    def smda(self, out_path=None):
        from smda.Disassembler import Disassembler
        from smda.SmdaConfig import SmdaConfig

        config = SmdaConfig()
        config.architecture = self.architecture   # guest processes are 64-bit
        disassembler = Disassembler(config)

        carved = self.carved_image()
        logging.info("Disassembling carved image %s for pid %s", carved, self.pid)
        report = disassembler.disassembleFile(carved).toDict()

        out_path = out_path or f"Graph_PID{self.pid}_from_{self.dump_name}.json"
        with open(out_path, "w") as handle:
            json.dump(report, handle, indent=4)
        logging.info("Wrote %s (%d functions)", out_path,
                     len(report.get("xcfg", {})))
        return out_path

    def run(self, out_path=None):
        self.process_dump()
        return self.smda(out_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump-dir", required=True, help="directory holding the dump")
    ap.add_argument("--dump-name", required=True,
                    help="dump filename without its suffix")
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--output-dir", default=None, help="default: pid<PID>/")
    ap.add_argument("--out", default=None, help="path for the SMDA report JSON")
    ap.add_argument("--volatility", default=None, help="path to vol.py")
    ap.add_argument("--architecture", default="intel.64bit")
    ap.add_argument("--dump-suffix", default=".raw")
    args = ap.parse_args(argv)

    loader = Graph_Loader(args.dump_dir, args.dump_name, args.pid,
                          output_dir=args.output_dir,
                          volatility=args.volatility,
                          architecture=args.architecture,
                          dump_suffix=args.dump_suffix)
    print(loader.run(args.out))


if __name__ == "__main__":
    main()
