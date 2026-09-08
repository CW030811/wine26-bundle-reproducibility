# Table 3 — Large-instance base-method comparison

Base FCP/PCP/FCPLS on six larger sizes; 17 reported method/size cells.

```bash
uv run python scripts/reproduce.py prepare table3
uv run python scripts/reproduce.py verify table3 --reference
# Full optimizer replay; can take hours or days:
uv run python scripts/reproduce.py replay table3
uv run python scripts/reproduce.py plot table3
```

Use `--dry-run` to inspect the replay commands. New outputs are under
`results/reproduced/table3/`: native result files, `verification.json`,
`statistics.csv`, and `plots/`. Use `plot table3 --reference` to plot the
packaged replay without solving again. Figure files are regenerated from
data; pixel-identical PDF/PNG rendering across platforms is not required.

Data archive: [bundles/table3.tar.gz](../../bundles/table3.tar.gz).
The complete per-file inventory and SHA-256 values are in
[bundles/manifest.json](../../bundles/manifest.json). Sources, scope and commands
are indexed in [EXPERIMENTS.json](../../EXPERIMENTS.json).

See [data and training](../../docs/TRAINING_AND_DATA.md),
[interface contract](../../docs/DATA_INTERFACE.md), and
[acceptance policy](../../docs/ACCEPTANCE.md).
