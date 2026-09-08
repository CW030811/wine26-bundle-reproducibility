# Figure 6 — Customer-count scalability

Customer-count scalability; fixed 29-sample m=80 evaluation subset and archived BSP normalization.

```bash
uv run python scripts/reproduce.py prepare figure6
uv run python scripts/reproduce.py verify figure6 --reference
# Full optimizer replay; can take hours or days:
uv run python scripts/reproduce.py replay figure6
uv run python scripts/reproduce.py plot figure6
```

Use `--dry-run` to inspect the replay commands. New outputs are under
`results/reproduced/figure6/`: native result files, `verification.json`,
`statistics.csv`, and `plots/`. Use `plot figure6 --reference` to plot the
packaged replay without solving again. Figure files are regenerated from
data; pixel-identical PDF/PNG rendering across platforms is not required.

Data archive: [bundles/figure6.tar.gz](../../bundles/figure6.tar.gz).
The complete per-file inventory and SHA-256 values are in
[bundles/manifest.json](../../bundles/manifest.json). Sources, scope and commands
are indexed in [EXPERIMENTS.json](../../EXPERIMENTS.json).

The m=80 aggregation uses the fixed sample list in [FIGURE6_SAMPLE_SELECTION.json](../../provenance/FIGURE6_SAMPLE_SELECTION.json). All 30 inputs and full per-seed outputs are retained; the same 29 samples are used on every check. BSP is normalized by its archived label.

See [data and training](../../docs/TRAINING_AND_DATA.md),
[interface contract](../../docs/DATA_INTERFACE.md), and
[acceptance policy](../../docs/ACCEPTANCE.md).
