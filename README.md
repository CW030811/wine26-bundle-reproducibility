# WINE 2026 bundle-pricing reproduction package

An experiment-indexed, auditable release of the source code, synthetic inputs,
training labels, model checkpoints, and reference results used in the bundle-pricing paper.

**Partial verified release, not a claim that the entire paper has been reproduced.**
Seven experiment packages are published below. Five further experiments have source
entry points but no accepted result bundles yet. See [acceptance scope](docs/ACCEPTANCE.md)
before interpreting the word “exact”. Runtime is excluded from numerical acceptance.

## Experiment index

| Paper experiment | Published scope | Guide / data archive |
|---|---|---|
| Table 2 | FCP, PCP, FCPLS, BSP; 12 method/size cells | [Guide](experiments/table2/README.md) · [archive](bundles/table2.tar.gz) |
| Table 5 | Z-corrected FCP/BSP sweep; CPBSD-A is a reused paper baseline | [Guide](experiments/table5/README.md) · [archive](bundles/table5.tar.gz) |
| Table 6 | Ten-seed FCP replay and archived training-loss records | [Guide](experiments/table6/README.md) · [archive](bundles/table6.tar.gz) |
| Table 7 | Four OOD distributions, ten seeds each | [Guide](experiments/table7/README.md) · [archive](bundles/table7.tar.gz) |
| Figure 9 | Cutoff sensitivity; one documented solver-bound exception | [Guide](experiments/figure9/README.md) · [archive](bundles/figure9.tar.gz) |
| Figure 10 | K sensitivity; 630 identical non-runtime replay rows | [Guide](experiments/figure10/README.md) · [archive](bundles/figure10.tar.gz) |
| Figure 11 | 60-instance LP/MILP translation replay, 489/522 | [Guide](experiments/figure11/README.md) · [archive](bundles/figure11.tar.gz) |
| Tables 3–4; Figures 6–8 | Pending full replay and acceptance | [Pending entry points](experiments/pending/README.md) |

Each archive uses repository-relative paths and contains the required data/models
and reference results for its experiment. Shared files are intentionally repeated
so each experiment archive can be downloaded independently. The extractor checks
identical shared files and will not replace different existing files.

## Quick start

```bash
git clone https://github.com/CW030811/wine26-bundle-reproducibility.git
cd wine26-bundle-reproducibility

# Standard-library-only integrity check / extraction (Python 3.9–3.12).
python3 scripts/extract_bundle.py all --check-only
python3 scripts/extract_bundle.py all

# Pinned experiment environment. Install uv separately if it is not available.
uv sync --frozen

# Recompute statistics from the released reference/replay files; no solver needed.
uv run python scripts/verify_reference_results.py
```

For one experiment, use `python3 scripts/extract_bundle.py table2`, then follow
its guide. The combined reference verifier requires **all seven archives** extracted.

Actual optimization requires your own valid **full Gurobi license**. The
size-limited pip license cannot handle the larger MILPs. Set `GRB_LICENSE_FILE`
to your own license file; no license credentials are distributed. Replays can
take hours or days. Keep long jobs in tmux or an equivalent persistent session.

Do not run expensive jobs concurrently if matching time-limited solver behavior
matters. Do not substitute a newly trained checkpoint when checking the archived
paper results. See [training and data generation](docs/TRAINING_AND_DATA.md).

## Layout and integrity

- `src/`: final algorithm sources, generators, labeling, training, evaluators, plotting.
- `scripts/`: replay drivers, archive extraction, and numerical verification.
- `experiments/`: paper-specific commands and scope; pending experiments are separate.
- `bundles/`: seven `.tar.gz` archives plus per-archive/per-file SHA-256 manifest.
- `provenance/`: published target values, export hashes, numerical exceptions.
- `EXPERIMENTS.json`: machine-readable mapping from experiments to sources and bundles.
- `SHA256SUMS.txt`: hashes of the committed release files (excluding itself).

The `.tar.gz` files remain immutable references. Extracted `results/` files may
be overwritten by a replay; run in a fresh checkout if you need to keep another
run. Checkpoint files use PyTorch serialization: only load files from a source
you trust, after checking hashes. A hash checks integrity, not trustworthiness.

The paper PDF, private local history, solver licenses, virtual environments,
machine execution logs, and unverified run outputs are not included.

## Collaboration and licensing

Issues and pull requests are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) describes
the evidence required to add another accepted experiment. The repository owner
can invite collaborators later; none are invited automatically by this release.

**License decision pending owner confirmation.** Public visibility alone does
not grant an MIT or Creative Commons license. No new redistribution license is
asserted in this initial snapshot. Third-party dependencies retain their own licenses.

## 中文说明

当前公开 7 项已验收实验的源码、数据/模型压缩包与参考结果；Table 3、4 和
Figure 6、7、8 仅提供入口，待全量核对通过后补发。运行时间不要求一致；
Table 2 的舍入边界、Table 7 的标准差口径、Figure 9 的单行求解上界差异均有记录。
Table 5 的 CPBSD-A 是沿用的论文基准值，不能视为此次已独立重跑验证。
从头训练的代码和最终训练数据已提供，但不承诺跨硬件训练出逐位相同的模型。
