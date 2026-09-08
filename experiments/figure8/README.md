# Figure 8 — Base versus self-improved methods

Base and self-improved comparison; all five curves, 20 curve points.

```bash
uv run python scripts/reproduce.py prepare figure8
uv run python scripts/reproduce.py verify figure8 --reference
# Full optimizer replay; can take hours or days:
uv run python scripts/reproduce.py replay figure8
uv run python scripts/reproduce.py plot figure8
```

Use `--dry-run` to inspect the replay commands. New outputs are under
`results/reproduced/figure8/`: native result files, `verification.json`,
`statistics.csv`, and `plots/`. Use `plot figure8 --reference` to plot the
packaged replay without solving again. Figure files are regenerated from
data; pixel-identical PDF/PNG rendering across platforms is not required.

Data archive: [bundles/figure8.tar.gz](../../bundles/figure8.tar.gz).
The complete per-file inventory and SHA-256 values are in
[bundles/manifest.json](../../bundles/manifest.json). Sources, scope and commands
are indexed in [EXPERIMENTS.json](../../EXPERIMENTS.json).

Preparation also extracts the three shared self-improvement training shards (3,000 labeled instances). Base and self-improved checkpoints remain in separate model directories.

See [data and training](../../docs/TRAINING_AND_DATA.md),
[interface contract](../../docs/DATA_INTERFACE.md), and
[acceptance policy](../../docs/ACCEPTANCE.md).
