# WINE 2026 bundle-pricing reproduction package

Source code, synthetic inputs, optimization labels, training records, checkpoints,
and per-sample replay results for **all twelve numerical experiments** in
*Self-Improving Neural Pruning*. This release includes Tables 2–7 and Figures 6–11.
Acceptance follows the [experiment-specific policy](docs/ACCEPTANCE.md);
runtime is machine-dependent and is excluded from numerical acceptance.

## Quick start

```bash
git clone https://github.com/CW030811/wine26-bundle-reproducibility.git
cd wine26-bundle-reproducibility
python3 scripts/extract_bundle.py all --check-only
uv sync --frozen
uv run python scripts/reproduce.py prepare all
uv run python scripts/reproduce.py verify all --reference
```

The reference check recomputes statistics from the packaged results without
running a solver. Each `.tar.gz` archive and every member has a SHA-256 checksum.
Python 3.9–3.12 is supported; install `uv` separately.

For a new replay and data-driven plots:

```bash
uv run python scripts/reproduce.py replay table3 --dry-run
uv run python scripts/reproduce.py replay table3
uv run python scripts/reproduce.py plot table3
# No optimization: draw the packaged replay data directly.
uv run python scripts/reproduce.py plot all --reference
```

Optimization requires a full **Gurobi license** supplied by the user via
`GRB_LICENSE_FILE`. Solver credentials are not distributed. Full replay of all
experiments can take days; run it in tmux or an equivalent persistent session.
New outputs go to `results/reproduced/<experiment>/`, separate from packaged
references. `--output-dir` selects another non-reference output directory.

## Experiment index

| Experiment | Published scope | Guide / archive |
|---|---|---|
| Table 2 | FCP, PCP, FCPLS and BSP on three sizes; 12 method/size cells. | [Guide](experiments/table2/README.md) · [Data](bundles/table2.tar.gz) |
| Table 3 | Base FCP/PCP/FCPLS on six larger sizes; 17 reported method/size cells. | [Guide](experiments/table3/README.md) · [Data](bundles/table3.tar.gz) |
| Table 4 | Self-improved FCP-I/PCP-I/FCPLS-I on six larger sizes; 15 reported cells. | [Guide](experiments/table4/README.md) · [Data](bundles/table4.tar.gz) |
| Table 5 | Corrected FCP/BSP random-valuation sweep; 24 InS/OOS statistics. | [Guide](experiments/table5/README.md) · [Data](bundles/table5.tar.gz) |
| Table 6 | Ten-seed FCP replay plus archived training-loss records; 40 reported values. | [Guide](experiments/table6/README.md) · [Data](bundles/table6.tar.gz) |
| Table 7 | Four OOD distributions, 100 samples and ten base seeds per distribution. | [Guide](experiments/table7/README.md) · [Data](bundles/table7.tar.gz) |
| Figure 6 | Customer-count scalability; fixed 29-sample m=80 evaluation subset and archived BSP normalization. | [Guide](experiments/figure6/README.md) · [Data](bundles/figure6.tar.gz) |
| Figure 7 | Product-count scalability; FCP/PCP/BSP, 12 curve points. | [Guide](experiments/figure7/README.md) · [Data](bundles/figure7.tar.gz) |
| Figure 8 | Base and self-improved comparison; all five curves, 20 curve points. | [Guide](experiments/figure8/README.md) · [Data](bundles/figure8.tar.gz) |
| Figure 9 | Seed-1 cutoff sensitivity with 1,080 reference/replay keys. | [Guide](experiments/figure9/README.md) · [Data](bundles/figure9.tar.gz) |
| Figure 10 | Candidate-budget sensitivity with 630 non-runtime replay rows. | [Guide](experiments/figure10/README.md) · [Data](bundles/figure10.tar.gz) |
| Figure 11 | Sixty-instance LP/MILP path replay; 489 of 522 accepted moves translate. | [Guide](experiments/figure11/README.md) · [Data](bundles/figure11.tar.gz) |

Three additional archives, `training_self_1.tar.gz` through
`training_self_3.tar.gz`, contain the shared 3,000-instance self-improvement
training set. Preparing Table 4 or Figure 8 restores these automatically.
The Table 6 archive supplies the 3,000-instance base training set, and Table 5
supplies 4,000 random-valuation instances, labels and split manifests.
Shared evaluation inputs/checkpoints are repeated where needed so each
experiment can be prepared independently. Identical shared files are checked;
different existing files are preserved and reported as conflicts.

## Repository layout

```text
README.md                       Start here
EXPERIMENTS.json                 Experiment/source/data/command index
SHA256SUMS.txt                   Whole-release file checksums
pyproject.toml, uv.lock          Pinned Python environment
src/
  deterministic/                Generators, MILPs, GNN training and evaluators
  random_valuation/             Random inputs, Z-corrected labels and evaluation
  appendix/, appendix_rerun/    Sensitivity and LP/MILP replay drivers
  report/, test/                Plotting and guarded solver compatibility sources
scripts/
  reproduce.py                  prepare / replay / verify / plot / list
  data_pipeline.py              generate / label / train
  verify_generation.py          Rebuild and compare 7,000 archived random draws
  audit_release.py              Archive, portability and privacy audit
experiments/<table-or-figure>/   Twelve experiment guides
bundles/                        12 experiment archives + 3 training shards
docs/                           Data interface, training recipes, acceptance, coverage
provenance/                     Target values, frozen selections and audit evidence
tests/, .github/workflows/      Regression tests and solver-free CI
```

Raw/native data schemas remain available for inspection. The public command
interface and output statistics schema are shared across experiments; see
[DATA_INTERFACE.md](docs/DATA_INTERFACE.md). Generation, labeling and retraining
commands are in [TRAINING_AND_DATA.md](docs/TRAINING_AND_DATA.md).
Archived checkpoints are used for result comparisons; fresh training may vary
with hardware and library builds.

[PAPER_COVERAGE.md](docs/PAPER_COVERAGE.md) maps the non-experimental tables,
figures and appendices. The paper PDF, historical versions, machine logs,
credentials, and virtual environments are not distributed.

## Collaboration

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
The owner can add collaborators through GitHub repository settings.
The redistribution-license decision remains with the owner; public visibility
alone does not assert a new MIT or Creative Commons license.

## 中文说明

本仓库提供论文全部 12 项数值实验的源码、输入、标签、模型和结果压缩包，
统一使用 `reproduce.py` 准备、重跑、核对与绘图。生成、标注、训练使用
`data_pipeline.py`。各实验按已登记的验收范围核对，运行时间不要求一致。
Figure 6 的 m=80 点采用维护者接受并固定的 29 样本名单；完整 30 样本资料保留。
