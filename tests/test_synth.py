"""The synthetic generator must produce reports the real pipeline accepts."""
import json

import synth
from CFG_Extractor import build_all_cfgs


REQUIRED_FUNCTION_KEYS = {"offset", "blocks", "apirefs", "stringrefs",
                          "blockrefs", "inrefs", "outrefs", "metadata"}


def test_generated_report_has_the_smda_schema(synthetic_report, real_report):
    # every top-level key the pipeline reads must exist on both
    for key in ("architecture", "bitness", "xcfg", "statistics", "metadata",
                "timestamp"):
        assert key in synthetic_report, key
        assert key in real_report, key

    for func in synthetic_report["xcfg"].values():
        assert REQUIRED_FUNCTION_KEYS <= set(func)


def test_instructions_match_the_real_four_field_layout(synthetic_report):
    for func in synthetic_report["xcfg"].values():
        for instructions in func["blocks"].values():
            assert instructions
            for addr, hexbytes, mnemonic, operands in instructions:
                assert isinstance(addr, int)
                assert isinstance(hexbytes, str) and len(hexbytes) % 2 == 0
                assert isinstance(mnemonic, str) and mnemonic
                assert isinstance(operands, str)


def test_block_terminator_agrees_with_out_degree(synthetic_report):
    """A block with two successors must end in a conditional branch, a block
    with none must end in a return - otherwise the CFG contradicts the code."""
    for func in synthetic_report["xcfg"].values():
        blockrefs = func["blockrefs"]
        for addr, instructions in func["blocks"].items():
            out_degree = len(blockrefs.get(addr, []))
            last = instructions[-1][2]
            if out_degree >= 2:
                assert last.startswith("j") and last != "jmp"
            elif out_degree == 1:
                assert last == "jmp"
            else:
                assert last == "ret"


def test_blockrefs_only_point_at_real_blocks(synthetic_report):
    for func in synthetic_report["xcfg"].values():
        known = {int(a) for a in func["blocks"]}
        for src, targets in func["blockrefs"].items():
            assert int(src) in known
            assert all(int(t) in known for t in targets)


def test_reports_are_deterministic_for_a_seed(instruction_pool):
    profile = synth.PROFILES_BY_NAME["Trojan"]
    first = synth.generate_report(profile, 42, instruction_pool)
    second = synth.generate_report(profile, 42, instruction_pool)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    third = synth.generate_report(profile, 43, instruction_pool)
    assert json.dumps(third, sort_keys=True) != json.dumps(first, sort_keys=True)


def test_the_dataset_is_balanced_and_labelled(tmp_path, real_report):
    spec = synth.DatasetSpec(n_per_profile=2, seed=3)
    labels = synth.generate_dataset(str(tmp_path), spec, real_report)

    n_malicious = sum(m["is_malware"] for m in labels.values())
    # four benign profiles against four malicious ones: a majority-class guess
    # must not already be a good score
    assert n_malicious == len(labels) - n_malicious

    for meta in labels.values():
        if meta["is_malware"]:
            assert meta["family"] in synth.FAMILIES
        else:
            assert "family" not in meta


def test_generated_reports_build_valid_cfgs(tmp_path, real_report):
    spec = synth.DatasetSpec(n_per_profile=1, seed=9)
    labels = synth.generate_dataset(str(tmp_path), spec, real_report)

    for name in labels:
        with open(tmp_path / name) as handle:
            report = json.load(handle)
        cfgs = build_all_cfgs(report, parallel=False)
        assert cfgs
        assert all(cfg.number_of_nodes() > 0 for cfg in cfgs.values())
        assert sum(cfg.number_of_edges() for cfg in cfgs.values()) > 0


def test_harvested_pool_excludes_control_flow(real_report):
    pool = synth.harvest_instruction_pool(real_report)
    assert pool
    mnemonics = {mnemonic for mnemonic, _operands in pool}
    assert not mnemonics & {"jmp", "ret", "call", "je", "jne", "int3"}


def test_profiles_carry_distinct_signal():
    """Two profiles that shared APIs and motifs would make the task unlearnable
    for reasons that have nothing to do with the model."""
    seen_apis, seen_motifs = [], []
    for profile in synth.PROFILES:
        assert profile.apis and profile.motifs
        seen_apis.append(set(profile.apis))
        seen_motifs.append(set(profile.motifs))
    for i, first in enumerate(seen_motifs):
        for second in seen_motifs[i + 1:]:
            assert not first & second
    for i, first in enumerate(seen_apis):
        for second in seen_apis[i + 1:]:
            assert len(first & second) <= 1


def test_cli_writes_reports_and_manifest(tmp_path):
    manifest = tmp_path / "labels.json"
    labels = synth.main(["--out", str(tmp_path), "--per-profile", "1",
                         "--manifest", str(manifest)])
    assert manifest.exists()
    assert json.loads(manifest.read_text()) == labels
    assert len(list(tmp_path.glob("Graph_PID*.json"))) == len(labels)
