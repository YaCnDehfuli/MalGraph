# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-05

First public GitHub release of the memory-CFG / hierarchical GNN pipeline.

### Added

- SMDA loader (Volatility `PsList --dump` + disassembly), CFG extractor, typed FCG.
- Assembly corpus writer, WordPiece tokenizer, DistilBERT MLM encoder, cached block embeddings.
- DiffPool function pooling, GraphSAGE hierarchical classifier (binary + family heads).
- Training loop that writes config, split, checkpoint, and predictions; `eval.py` metrics from the saved file only.
- Opcode n-gram and graph-stat baselines; single-report `predict` CLI.
- `scripts/build_binary_corpus.py` (UPX packing measurement) and `scripts/run_pipeline.py`.
- 71 tests; synthetic SMDA fixtures; committed packing stats and figures.
- README, workflow diagram, project metadata, MIT license.

### Changed

- Source laid out as the `memory_cfg` package; legacy timeline experiments moved aside.

### Fixed

- Tokenizer special tokens; pad/truncate to 512; MLM label masking.
- Loss NaN on zero-edge binaries; `dense_diff_pool` mask + aux losses; `edge_index` dtype.
- Outrefs typed as call / tail-jump / unresolved rather than collapsed.
- Family head supervised only when the sample is malicious.
- Classifier node space unified, batched, and rebalanced after a run that barely learned.
- Tokenizer/checkpoint version guard on inference.

[1.0.0]: https://github.com/YaCnDehfuli/MalGraph/releases/tag/v1.0.0
