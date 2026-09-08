# Figure 10 — FCPLS candidate-budget K sensitivity

Candidate-budget sensitivity with 630 non-runtime replay rows.

```bash
uv run python scripts/reproduce.py prepare figure10
uv run python scripts/reproduce.py verify figure10 --reference
# Full optimizer replay; can take hours or days:
uv run python scripts/reproduce.py replay figure10
uv run python scripts/reproduce.py plot figure10
```

Use `--dry-run` to inspect the replay commands. New outputs are under
`results/reproduced/figure10/`: native result files, `verification.json`,
`statistics.csv`, and `plots/`. Use `plot figure10 --reference` to plot the
packaged replay without solving again. Figure files are regenerated from
data; pixel-identical PDF/PNG rendering across platforms is not required.

Data archive: [bundles/figure10.tar.gz](../../bundles/figure10.tar.gz).
The complete per-file inventory and SHA-256 values are in
[bundles/manifest.json](../../bundles/manifest.json). Sources, scope and commands
are indexed in [EXPERIMENTS.json](../../EXPERIMENTS.json).

See [data and training](../../docs/TRAINING_AND_DATA.md),
[interface contract](../../docs/DATA_INTERFACE.md), and
[acceptance policy](../../docs/ACCEPTANCE.md).
