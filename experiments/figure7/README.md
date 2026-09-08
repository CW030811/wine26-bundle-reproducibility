# Figure 7 — Product-count scalability

Product-count scalability; FCP/PCP/BSP, 12 curve points.

```bash
uv run python scripts/reproduce.py prepare figure7
uv run python scripts/reproduce.py verify figure7 --reference
# Full optimizer replay; can take hours or days:
uv run python scripts/reproduce.py replay figure7
uv run python scripts/reproduce.py plot figure7
```

Use `--dry-run` to inspect the replay commands. New outputs are under
`results/reproduced/figure7/`: native result files, `verification.json`,
`statistics.csv`, and `plots/`. Use `plot figure7 --reference` to plot the
packaged replay without solving again. Figure files are regenerated from
data; pixel-identical PDF/PNG rendering across platforms is not required.

Data archive: [bundles/figure7.tar.gz](../../bundles/figure7.tar.gz).
The complete per-file inventory and SHA-256 values are in
[bundles/manifest.json](../../bundles/manifest.json). Sources, scope and commands
are indexed in [EXPERIMENTS.json](../../EXPERIMENTS.json).

See [data and training](../../docs/TRAINING_AND_DATA.md),
[interface contract](../../docs/DATA_INTERFACE.md), and
[acceptance policy](../../docs/ACCEPTANCE.md).
