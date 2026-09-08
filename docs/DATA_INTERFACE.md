# Public data interface

Run commands from the repository root. Source, archived input and checkpoint
paths are resolved from the wrappers' file locations. Moving or renaming the
checkout does not require editing source files.

| Operation | Command | Output |
|---|---|---|
| Inspect experiments | `python scripts/reproduce.py list` | Status and scope |
| Check integrity | `python scripts/reproduce.py prepare all --check-only` | Archive/member hash checks |
| Restore data | `python scripts/reproduce.py prepare table4` | Data, models, references and training dependencies |
| Inspect a replay plan | `python scripts/reproduce.py replay table4 --dry-run` | Commands without solving |
| Run an experiment | `python scripts/reproduce.py replay table4` | Native results, verification and statistics |
| Check a new run | `python scripts/reproduce.py verify table4` | `results/reproduced/table4/verification.json` |
| Check released evidence | `python scripts/reproduce.py verify table4 --reference` | `reference_verification.json` |
| Plot results | `python scripts/reproduce.py plot table4` | `plots/*.png`, `plots/*.pdf`, `plots/statistics.csv` |

Replace `table4` with any experiment key or `all`. Replays run sequentially.
Figures 6–8 / Tables 3–4 resume complete per-seed tasks; Figures 9–10 resume
row-level CSVs. Other evaluators start a new run. `--output-dir` is per
experiment and cannot overlap packaged references. Rerunning into the same
non-reference output folder can replace its generated files; use a new
directory to retain multiple runs.

The wrapper prepares reference dependencies automatically. Extraction checks
identical shared files and refuses to replace different existing files.

## Common statistics schema

| Column | Meaning |
|---|---|
| `experiment` | Table/figure key |
| `case` | Dataset, size, distribution, seed or scale/cost group |
| `method` | Algorithm; `_I` denotes a self-improved checkpoint |
| `metric` | For example `profit_mean`, `profit_population_std`, `ins`, `oos` |
| `value` | Observed value from the selected results |
| `reference` | Registered comparison value, when applicable |
| `matched` | Cell acceptance; empty for descriptive metrics |

Figures 9–10 export descriptive runtime/classification aggregates and compare
non-runtime keyed rows. Figure 11 exports move counts and plots paths from the
selected replay JSON. Runtime is descriptive, not an acceptance target.
Table 6 uses archived training-loss records with the selected inference replay.

## Native inputs and aggregation

Native solver files remain available. Deterministic `.msgpack` files contain
customer/product counts, `unit_cs`, `ship_cs`, `unit_us`, `Ns`, labels,
`opt_rev` and solver metadata. Random-valuation `.msgpack` files contain setup
and valuation/cost arrays; labels are separate JSONs linked by CSV manifests.

Learned deterministic results use `(seed, sample_file)` identities. Aggregate
seeds 1–10 within each sample, then compute mean/std across samples. Table 6
reports each seed separately. Complete expected sample identities are required.
Figure 6 reads its frozen 29-sample selection from
`provenance/FIGURE6_SAMPLE_SELECTION.json`; it does not search for a new subset.
The full 30-input replay is retained.

Legacy source-level CLIs remain available. The supported public interfaces
are `reproduce.py` and `data_pipeline.py`; old generator main blocks may retain
historical demonstration settings. The wrappers call the same algorithm
functions with explicit paths and recipe parameters.
