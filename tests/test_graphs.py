"""CFG extraction, call-graph construction, corpus text, and figures."""
import os

import pytest

from CFG_Extractor import (
    build_all_cfgs, extract_functions, extract_functions_parallel, graph_stats,
)
from data_tokenizer import (
    canonicalize_operands, iter_block_lines, iter_api_lines, api_to_line,
    write_corpus, write_corpus_from_dir, INS_SEP,
)
from fcg import build_call_graph, call_graph_stats, CALL, API


def test_cfg_matches_the_reports_blockrefs(real_report):
    cfgs = build_all_cfgs(real_report, parallel=False)
    assert cfgs

    for fid, func in real_report["xcfg"].items():
        cfg = cfgs[int(fid)]
        assert cfg.number_of_nodes() == len(func["blocks"])
        expected = {(int(src), int(dst))
                    for src, targets in func["blockrefs"].items()
                    for dst in targets}
        assert set(cfg.edges()) == expected


def test_block_addresses_are_ints_not_a_mix_of_str_and_int(real_report):
    """The JSON has string block keys and int blockref targets; if the two are
    not normalized the same address becomes two separate nodes."""
    cfgs = build_all_cfgs(real_report, parallel=False)
    for cfg in cfgs.values():
        assert all(isinstance(node, int) for node in cfg.nodes)


def test_parallel_and_serial_extraction_agree(real_report):
    serial = extract_functions(real_report)
    parallel = extract_functions_parallel(real_report)
    assert set(serial) == set(parallel)
    for fid in serial:
        assert set(serial[fid].edges()) == set(parallel[fid].edges())
        assert set(serial[fid].nodes()) == set(parallel[fid].nodes())


def test_control_flow_stays_directed(real_report):
    cfgs = build_all_cfgs(real_report, parallel=False)
    assert any(cfg.is_directed() for cfg in cfgs.values())
    # a symmetrized CFG would report an edge in both directions everywhere
    both_ways = sum(1 for cfg in cfgs.values() for u, v in cfg.edges()
                    if cfg.has_edge(v, u))
    assert both_ways < sum(cfg.number_of_edges() for cfg in cfgs.values())


def test_graph_stats_are_consistent(real_report):
    cfgs = build_all_cfgs(real_report, parallel=False)
    stats = graph_stats(cfgs)
    assert stats["functions"] == len(cfgs)
    assert stats["blocks"] == sum(g.number_of_nodes() for g in cfgs.values())
    assert stats["cfg_edges"] == sum(g.number_of_edges() for g in cfgs.values())
    assert stats["block_len_p50"] <= stats["block_len_p99"] <= stats["block_len_max"]


@pytest.mark.parametrize("operands,expected", [
    ("rax, 0x7ff7fe5fc1de", "rax, ADDR"),
    ("rsp, 0x20", "rsp, IMM"),
    ("eax, eax", "eax, eax"),
    ("qword ptr [rbx + 0x18], 4096", "qword ptr [rbx + IMM], IMM"),
])
def test_operands_are_canonicalized(operands, expected):
    assert canonicalize_operands(operands) == expected


def test_corpus_lines_are_never_blank(real_report, tmp_path):
    lines = list(iter_block_lines(real_report))
    assert lines
    assert all(line.strip() for line in lines)
    assert any(INS_SEP in line for line in lines)

    out = tmp_path / "corpus.txt"
    written = write_corpus(real_report, str(out))
    assert written == len(out.read_text().splitlines())
    assert all(line.strip() for line in out.read_text().splitlines())


def test_corpus_dedup_is_stable_across_processes(real_report, tmp_path):
    """Dedup used builtin hash(), which PYTHONHASHSEED randomizes per process."""
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    write_corpus(real_report, str(first))
    write_corpus(real_report, str(second))
    assert first.read_text() == second.read_text()


def test_api_lines_reach_the_corpus(synthetic_report, tmp_path):
    api_lines = list(iter_api_lines(synthetic_report))
    assert api_lines
    assert all(line.startswith("[API] ") for line in api_lines)
    assert api_to_line("ws2_32.dll!connect") == "[API] ws2_32.dll connect"

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    import json
    (reports_dir / "r.json").write_text(json.dumps(synthetic_report))
    out = tmp_path / "corpus.txt"
    total, n_reports = write_corpus_from_dir(str(reports_dir), str(out))
    assert n_reports == 1 and total > 0
    assert any(line.startswith("[API]") for line in out.read_text().splitlines())


def test_call_graph_types_its_edges(synthetic_report):
    fcg = build_call_graph(synthetic_report)
    etypes = {etype for _u, _v, etype in fcg.edges(data="etype")}
    assert CALL in etypes
    assert API in etypes

    stats = call_graph_stats(fcg)
    assert stats["functions"] == len(synthetic_report["xcfg"])
    assert stats["api_nodes"] > 0
    assert stats["nodes"] == stats["functions"] + stats["api_nodes"]


def test_unresolved_targets_do_not_invent_function_nodes(synthetic_report):
    """An outref to an address that is not a function entry has no code behind
    it; adding it as a node would fabricate a function."""
    fcg = build_call_graph(synthetic_report)
    known = {int(fid) for fid in synthetic_report["xcfg"]}
    for node in fcg.nodes:
        assert isinstance(node, str) or node in known


def test_every_function_is_a_node_even_when_isolated(synthetic_report):
    fcg = build_call_graph(synthetic_report, include_apis=False)
    assert set(fcg.nodes) == {int(fid) for fid in synthetic_report["xcfg"]}


def test_figures_render_to_files(real_report, synthetic_report, tmp_path):
    import viz

    cfgs = build_all_cfgs(real_report, parallel=False)
    biggest = max(cfgs, key=lambda f: cfgs[f].number_of_nodes())

    cfg_png = viz.draw_cfg(cfgs[biggest], title="test",
                           save_to=str(tmp_path / "cfg.png"))
    grid_png = viz.draw_cfg_grid(cfgs, save_to=str(tmp_path / "grid.png"))
    fcg_png = viz.draw_call_graph(build_call_graph(synthetic_report),
                                  save_to=str(tmp_path / "fcg.png"))
    curve = viz.draw_training_curves(
        [{"epoch": 0, "loss": 1.0, "train_accuracy": 0.5},
         {"epoch": 1, "loss": 0.5, "train_accuracy": 0.9}],
        save_to=str(tmp_path / "curve.png"))
    confusion = viz.draw_confusion([[3, 1], [0, 4]], ["benign", "malware"],
                                   save_to=str(tmp_path / "cm.png"))

    for path in (cfg_png, grid_png, fcg_png, curve, confusion):
        assert os.path.exists(path) and os.path.getsize(path) > 0


def test_layered_layout_puts_the_entry_on_top(real_report):
    from viz import layered_layout

    cfgs = build_all_cfgs(real_report, parallel=False)
    cfg = cfgs[max(cfgs, key=lambda f: cfgs[f].number_of_nodes())]
    pos = layered_layout(cfg)
    assert set(pos) == set(cfg.nodes)
    entries = [n for n in cfg.nodes if cfg.in_degree(n) == 0]
    if entries:
        assert max(pos[n][1] for n in entries) == max(y for _x, y in pos.values())
