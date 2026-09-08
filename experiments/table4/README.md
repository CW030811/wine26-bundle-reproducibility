# Table 4 — Self-improved methods on larger instances

Self-improved FCP-I/PCP-I/FCPLS-I on six larger sizes; 15 reported cells.

```bash
uv run python scripts/reproduce.py prepare table4
uv run python scripts/reproduce.py verify table4 --reference
# Full optimizer replay; can take hours or days:
uv run python scripts/reproduce.py replay table4
uv run python scripts/reproduce.py plot table4
```

Use `--dry-run` to inspect the replay commands. New outputs are under
`results/reproduced/table4/`: native result files, `verification.json`,
`statistics.csv`, and `plots/`. Use `plot table4 --reference` to plot the
packaged replay without solving again. Figure files are regenerated from
data; pixel-identical PDF/PNG rendering across platforms is not required.

Data archive: [bundles/table4.tar.gz](../../bundles/table4.tar.gz).
The complete per-file inventory and SHA-256 values are in
[bundles/manifest.json](../../bundles/manifest.json). Sources, scope and commands
are indexed in [EXPERIMENTS.json](../../EXPERIMENTS.json).

Preparation also extracts the three shared self-improvement training shards (3,000 labeled instances). Base and self-improved checkpoints remain in separate model directories.

See [data and training](../../docs/TRAINING_AND_DATA.md),
[interface contract](../../docs/DATA_INTERFACE.md), and
[acceptance policy](../../docs/ACCEPTANCE.md).
