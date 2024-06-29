import os
import json
import subprocess
import logging

from smda.Disassembler import Disassembler
from smda.SmdaConfig import SmdaConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Graph_Loader():
    """Carve a process out of a memory dump and disassemble it with SMDA."""

    def __init__(self, dump_dir, dump_name, pid):
        self.pid = pid
        self.dump_name = dump_name
        self.dump_dir = dump_dir
        self.output_dir = f"pid{self.pid}"

    def process_dump(self):
        os.makedirs(self.output_dir, exist_ok=True)
        command = [
            "/home/yacn/volatility3/vol.py",
            "-f", os.path.join(self.dump_dir, f"{self.dump_name}.raw"),
            "-o", self.output_dir,
            "windows.pslist.PsList",
            "--pid", str(self.pid),
            "--dump",
        ]
        logging.info("Extracting process %s from %s", self.pid, self.dump_name)
        subprocess.run(command)

    def smda(self):
        config = SmdaConfig()
        config.architecture = "intel.64bit"  # the guest processes are 64-bit
        dis_obj = Disassembler(config)
        # volatility writes one file per dumped region; pick the one for our pid
        candidates = [f for f in os.listdir(self.output_dir)
                      if f.endswith(".dmp") and str(self.pid) in f]
        if not candidates:
            raise FileNotFoundError(f"no dumped image for pid {self.pid} in {self.output_dir}")
        carved = os.path.join(self.output_dir, sorted(candidates)[0])
        logging.info("Disassembling carved image %s for pid %s", carved, self.pid)
        smda_report = dis_obj.disassembleFile(carved)
        json_report = smda_report.toDict()
        output_file = f"Graph_PID{self.pid}_from_{self.dump_name}.json"
        with open(output_file, "w") as f:
            json.dump(json_report, f, indent=4)

    def run(self):
        self.process_dump()
        self.smda()


if __name__ == "__main__":
    loader = Graph_Loader("/home/yacn/Research/dumps/", "wmplayer", 6280)
    loader.run()
