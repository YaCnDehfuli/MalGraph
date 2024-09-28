"""Dataset loader over a directory of normalized SMDA reports.

Two modes are supported:
  - "spatial":  each sample is one binary's static graph (CFGs + call graph).
  - "temporal": the same reports ordered by capture timestamp, for the
                sequence-of-snapshots experiments.
"""
import os
import glob

from CFG_Extractor import load_smda_json

# the label manifest lives next to the reports; it is not a report
MANIFEST_NAMES = {"labels.json", "manifest.json", "split.json"}


class DataLoader:
    """Iterate reports in a directory, lazily, one dict per binary."""

    def __init__(self, report_dir, mode="spatial", names=None):
        if mode not in ("temporal", "spatial"):
            raise ValueError('mode must be "temporal" or "spatial"')
        self.report_dir = report_dir
        self.mode = mode
        if names is not None:
            self.reports = [os.path.join(report_dir, n) for n in names]
        else:
            self.reports = [
                p for p in sorted(glob.glob(os.path.join(report_dir, "*.json")))
                if os.path.basename(p) not in MANIFEST_NAMES
            ]
        if mode == "temporal":
            self.reports = self._order_by_capture_time(self.reports)

    @staticmethod
    def _order_by_capture_time(paths):
        """SMDA stamps each report; temporal mode replays captures in order."""
        def key(path):
            try:
                return (load_smda_json(path).get("timestamp", ""), path)
            except (OSError, ValueError):
                return ("", path)
        return sorted(paths, key=key)

    def __len__(self):
        return len(self.reports)

    def _load_one(self, path):
        # CFGs are deliberately not built here: build_sample needs them anyway,
        # and building twice doubles the most expensive parsing step.
        return {"path": path, "name": os.path.basename(path),
                "report": load_smda_json(path)}

    def __iter__(self):
        for path in self.reports:
            yield self._load_one(path)
