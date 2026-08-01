"""The local-binary corpus builder.

Its heavy path needs SMDA and UPX, which are optional; the selection, labeling
and summary logic is pure and always exercised.
"""
import os
import json
import shutil
import importlib.util

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "build_binary_corpus.py")


def _load():
    spec = importlib.util.spec_from_file_location("build_binary_corpus", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    if not os.path.exists(SCRIPT):
        pytest.skip("corpus builder script is missing")
    return _load()


def test_module_imports_without_the_forensics_stack(builder):
    """smda must not be imported at module scope, or the whole tests suite
    becomes uninstallable on a machine that only wants the pipeline."""
    assert hasattr(builder, "build")
    assert hasattr(builder, "build_package_corpus")


def test_is_elf64_accepts_a_real_binary_and_rejects_text(builder, tmp_path):
    real = shutil.which("ls") or "/bin/sh"
    assert builder.is_elf64(real)

    text = tmp_path / "not-a-binary"
    text.write_text("#!/bin/sh\necho hello\n")
    assert not builder.is_elf64(str(text))

    assert not builder.is_elf64(str(tmp_path / "missing"))


def test_candidate_selection_is_deterministic_and_filtered(builder):
    first = builder.candidate_binaries(["/usr/bin", "/bin"], 15_000, 400_000)
    second = builder.candidate_binaries(["/usr/bin", "/bin"], 15_000, 400_000)
    assert first == second                       # order must not drift
    for path in first:
        assert 15_000 <= os.path.getsize(path) <= 400_000
        assert not os.path.islink(path)
    # realpath de-duplication: no binary appears twice
    assert len({os.path.realpath(p) for p in first}) == len(first)


def test_missing_source_directory_is_ignored(builder):
    assert builder.candidate_binaries(["/no/such/place"], 0, 10 ** 9) == []


def test_report_stats_counts_what_it_says(builder):
    report = {"xcfg": {
        "100": {"blocks": {"100": [[1, "90", "nop", ""], [2, "c3", "ret", ""]]},
                "apirefs": {"1": "libc.so!printf"}},
        "200": {"blocks": {"200": [[3, "90", "nop", ""]]}, "apirefs": {}},
    }}
    stats = builder.report_stats(report)
    assert stats == {"functions": 2, "blocks": 2, "instructions": 3,
                     "api_refs": 1}


def test_summarize_reports_the_recovery_ratio(builder):
    pairs = [
        {"binary": "a", "size": 1,
         "plain": {"functions": 100, "blocks": 1000, "instructions": 5000, "api_refs": 40},
         "packed": {"functions": 5, "blocks": 40, "instructions": 200, "api_refs": 2}},
        {"binary": "b", "size": 1,
         "plain": {"functions": 100, "blocks": 1000, "instructions": 5000, "api_refs": 40},
         "packed": {"functions": 5, "blocks": 40, "instructions": 200, "api_refs": 2}},
    ]
    summary = builder.summarize(pairs)
    assert summary["pairs"] == 2
    assert summary["recovered_function_ratio"] == pytest.approx(0.05)
    assert summary["recovered_block_ratio"] == pytest.approx(0.04)
    assert summary["median_functions_plain"] == 100
    assert summary["median_functions_packed"] == 5


def test_summarize_handles_an_empty_run(builder):
    assert builder.summarize([]) == {"pairs": 0}


def test_owning_package_returns_a_name_or_none(builder):
    if shutil.which("dpkg") is None:
        pytest.skip("dpkg not available")
    package = builder.owning_package(shutil.which("ls") or "/bin/ls")
    assert package is None or isinstance(package, str) and package
    assert builder.owning_package("/no/such/file") is None


@pytest.mark.skipif(shutil.which("upx") is None, reason="upx not installed")
def test_pack_shrinks_a_binary(builder, tmp_path):
    source = shutil.which("ls")
    destination = str(tmp_path / "ls.upx")
    if not builder.pack(source, destination):
        pytest.skip("upx refused this binary")
    assert os.path.getsize(destination) < os.path.getsize(source)


@pytest.mark.skipif(shutil.which("upx") is None, reason="upx not installed")
def test_packing_corpus_round_trip(builder, tmp_path):
    pytest.importorskip("smda")
    labels, summary = builder.build(str(tmp_path), limit=2, log=lambda *_: None)
    if not labels:
        pytest.skip("no binary on this machine survived packing + disassembly")

    assert len(labels) % 2 == 0
    assert sum(v["is_malware"] for v in labels.values()) == len(labels) // 2
    manifest = json.load(open(tmp_path / "reports" / "labels.json"))
    assert manifest == labels
    # every named report exists and parses as an SMDA-shaped document
    for name in labels:
        with open(tmp_path / "reports" / name) as handle:
            assert "xcfg" in json.load(handle)
    # packing must not increase the recoverable code overall
    assert summary["recovered_function_ratio"] <= 1.0
    assert not os.path.exists(tmp_path / "work")     # scratch dir cleaned up
